"""
Energy Calendar Analysis
Hourly gas consumption (H) + weather (T, G, V) for 4 Czech regions.
Regions: JMP, SMP, SZC, VCP

Pipeline:
  1. Load / cache (parquet)
  2. Impute gaps in H (linear interpolation per region)
  3. Aggregate regions → single time series
       H_total = Σ H_r
       T/G/V   = weighted average (weights = region share of annual consumption)
  4. Feature engineering (calendar features, holidays)
  5. Extended weather model — optimise nonlinear parameter vector θ:
       T_eff    = T + αG·G − αV·V           (apparent temperature)
       T_smooth = β·EMA(T_eff,τ₁) + (1−β)·EMA(T_eff,τ₂)  (thermal inertia)
       HDD_sat  = clip(T_off − T_smooth, 0, T_off − T_min) (saturation)
       Linear part: OLS with HDD_sat + dow_group×hour dummies
     Objective: MAPE on eval years (default 2024–2025)
  6. Holiday index (working-day offset to nearest Czech public holiday)
  7. Energy calendar coefficients: median(H/H_pred) per scenario × offset
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import holidays
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH     = Path("Data/OnlineToky.xlsx")
NORMAL_PATH   = Path("Data/Normal.xlsx")
CACHE_DIR     = Path("cache")
CACHE_PATH    = CACHE_DIR / "raw_long.parquet"
RAW_H_PATH    = CACHE_DIR / "raw_consumption.parquet"
RAW_METEO_PATH= CACHE_DIR / "raw_meteo.parquet"
AGG_PATH      = CACHE_DIR / "agg_series.parquet"
FIT_PATH      = CACHE_DIR / "model_fit.parquet"
CALENDAR_PATH = CACHE_DIR / "calendar_coefficients.parquet"

REGIONS         = ["JMP", "SMP", "SZC", "VCP"]
T_BASE          = 15.0
TIMEZONE        = "Europe/Prague"
CALENDAR_WINDOW = 5

# Training period: stable recent data (2023-2025), avoids COVID and older anomalies
TRAIN_YEARS    = [2023, 2024, 2025]
# Evaluation period for optimising θ (most recent, reliable data)
EVAL_YEARS     = [2024, 2025]
# Calendar period: extend back to 2021 for Tue/Sat/Sun scenario coverage
# (2020 excluded — COVID lockdown distorted holiday behaviour)
CALENDAR_YEARS = [2021, 2022, 2023, 2024, 2025]

# Summer months excluded from heating features (HDD_sat ≈ 0, no heating)
SUMMER_MONTHS  = {6, 7, 8}
# Ridge regularisation alpha for OLS (stabilises near-zero HDD months)
RIDGE_ALPHA    = 1e4

# Bounds for θ = [αG, αV, τ1, τ2, β, T_off, T_min]
THETA_BOUNDS = [
    (0.0,  5.0),    # αG   solar contribution to apparent T [°C per unit G]
    (0.0,  3.0),    # αV   wind contribution to apparent T [°C per m/s]
    (1.0,  48.0),   # τ1   short EMA time constant [hours]
    (24.0, 240.0),  # τ2   long EMA time constant [hours]
    (0.0,  1.0),    # β    weight of short EMA
    (10.0, 18.0),   # T_off heating cutoff temperature [°C]
    (-25.0, 0.0),   # T_min saturation temperature [°C]
]
THETA0 = np.array([1.0, 0.5, 6.0, 72.0, 0.5, 15.0, -15.0])
THETA_NAMES = ["αG", "αV", "τ1[h]", "τ2[h]", "β", "T_off[°C]", "T_min[°C]"]

# ---------------------------------------------------------------------------
# 1. LOAD / CACHE
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load raw data. Returns long-format DataFrame (timestamp × region).
    On first run parses Excel and saves:
      raw_consumption.parquet  — wide: JMP_H / SMP_H / SZC_H / VCP_H (pre-imputation)
      raw_meteo.parquet        — wide: T/G/V per region (no gaps)
      raw_long.parquet         — internal long format
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if CACHE_PATH.exists():
        print(f"Loading from cache: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print(f"Parsing Excel: {DATA_PATH}")
    raw = pd.read_excel(DATA_PATH, index_col=0, engine="openpyxl")

    # OnlineToky.xlsx timestamps are UTC (verified: G peaks at 10h UTC ≈ solar noon)
    # Normal.xlsx timestamps are Prague local time — both end up in TIMEZONE after conversion
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("UTC").tz_convert(TIMEZONE)
    raw.index.name = "timestamp"
    raw.columns = raw.columns.str.strip()

    h_cols     = [f"{r}_H" for r in REGIONS]
    meteo_cols = [c for c in raw.columns if c not in h_cols]
    raw[h_cols].to_parquet(RAW_H_PATH);     print(f"  Saved: {RAW_H_PATH}")
    raw[meteo_cols].to_parquet(RAW_METEO_PATH); print(f"  Saved: {RAW_METEO_PATH}")

    frames = []
    for region in REGIONS:
        part = raw[[f"{region}_{v}" for v in ("H","T","G","V")]].copy()
        part.columns = ["H","T","G","V"]
        part.insert(0, "region", region)
        frames.append(part)

    df = pd.concat(frames).sort_index()
    df.to_parquet(CACHE_PATH)
    print(f"  Saved: {CACHE_PATH}  ({len(df):,} rows)")
    return df

# ---------------------------------------------------------------------------
# 2. IMPUTE GAPS in H
# ---------------------------------------------------------------------------

def impute_consumption(df: pd.DataFrame, max_gap_hours: int = 24) -> pd.DataFrame:
    result = []
    for region, grp in df.groupby("region", sort=False):
        grp = grp.copy()
        n_before = grp["H"].isna().sum()
        grp["H"] = grp["H"].interpolate(
            method="time", limit=max_gap_hours, limit_direction="both"
        )
        n_after = grp["H"].isna().sum()
        if n_before > n_after:
            print(f"  {region}: imputed {n_before-n_after} gaps  "
                  f"({n_after} remaining NaN = future)")
        result.append(grp)
    return pd.concat(result).sort_index()

# ---------------------------------------------------------------------------
# 3. AGGREGATE REGIONS
# ---------------------------------------------------------------------------

def aggregate_regions(df: pd.DataFrame, weight_by: str = "year") -> pd.DataFrame:
    """
    H_total = Σ H_r
    T/G/V   = consumption-weighted average per region.
    weight_by: "year" | "total"
    """
    wide  = df.pivot_table(index=df.index, columns="region",
                           values=["H","T","G","V"])
    H, T, G, V = wide["H"], wide["T"], wide["G"], wide["V"]
    hist_H = H.dropna(how="all")

    if weight_by == "year":
        annual_sum = hist_H.groupby(hist_H.index.year).sum()
        annual_w   = annual_sum.div(annual_sum.sum(axis=1), axis=0)
        overall_w  = annual_w.mean()  # fallback for years not in annual_w (e.g. future)
        year_map   = pd.Series(H.index.year, index=H.index)
        w_lookup   = annual_w.to_dict(orient="index")
        weights    = pd.DataFrame(
            [w_lookup.get(yr, overall_w.to_dict()) for yr in year_map],
            index=H.index, columns=REGIONS
        )
        print("  Region weights (annual share of H_total):")
        for yr in sorted(annual_w.index):
            row = "  ".join(f"{r}={annual_w.loc[yr,r]:.3f}" for r in REGIONS)
            print(f"    {yr}:  {row}")
    else:
        total_w = hist_H.sum() / hist_H.sum().sum()
        weights = pd.DataFrame(
            np.tile(total_w.values, (len(H),1)), index=H.index, columns=REGIONS
        )
        print("  Region weights:", "  ".join(f"{r}={total_w[r]:.3f}" for r in REGIONS))

    agg = pd.DataFrame(index=H.index)
    agg["H"] = H.sum(axis=1, min_count=1)
    agg["T"] = (weights * T).sum(axis=1)
    agg["G"] = (weights * G).sum(axis=1)
    agg["V"] = (weights * V).sum(axis=1)
    return agg

# ---------------------------------------------------------------------------
# 3b. NORMAL METEO  (T_n, G_n from file; V_n computed from history)
# ---------------------------------------------------------------------------

def load_normal_meteo(agg: pd.DataFrame, v_smooth_days: int = 30) -> pd.DataFrame:
    """
    Load T_n and G_n from Normal.xlsx, compute V_n from historical data.

    V_n: for each (day-of-year, hour) compute mean V across available years,
    then smooth with a Gaussian window of ±v_smooth_days days along the
    day-of-year axis to ensure a continuous seasonal profile.

    Returns DataFrame indexed like agg with columns T_n, G_n, V_n.
    Normal.xlsx is essentially periodic; values are looked up by
    (month, day, hour) so they apply to any year.
    """
    # ---- Load T_n, G_n -------------------------------------------------------
    raw_n = pd.read_excel(NORMAL_PATH, index_col=0, engine="openpyxl")
    raw_n.index = pd.DatetimeIndex(raw_n.index).tz_localize(
        TIMEZONE, ambiguous=False, nonexistent="shift_forward"
    )
    raw_n.index.name = "timestamp"
    raw_n.columns = raw_n.columns.str.strip()

    # ---- Compute V_n from historical agg ------------------------------------
    # Use years that have actual V data (all historical rows)
    hist_v = agg[agg["H"].notna()].copy()
    hist_v = hist_v[hist_v.index.year.isin(range(2021, 2026))]
    hist_v["doy"]  = hist_v.index.dayofyear
    hist_v["hour"] = hist_v.index.hour

    # Average V by (doy, hour) across years — 365 × 24 = 8760 cells
    v_profile = hist_v.groupby(["doy", "hour"])["V"].mean()
    # Reindex to ensure all doy/hour combinations exist
    full_idx   = pd.MultiIndex.from_product(
        [range(1, 366), range(24)], names=["doy", "hour"]
    )
    v_profile  = v_profile.reindex(full_idx).interpolate(method="linear")

    # Smooth along doy axis per hour using Gaussian kernel (circular/wrap)
    from scipy.ndimage import gaussian_filter1d
    sigma = v_smooth_days   # std in days
    v_smooth = v_profile.unstack("hour")  # shape (365, 24)
    for h in range(24):
        # Wrap-around Gaussian smoothing (seasonal continuity)
        col = v_smooth.iloc[:, h].values
        # Pad with wrapped data for circular smoothing
        padded = np.concatenate([col[-3*sigma:], col, col[:3*sigma]])
        smoothed = gaussian_filter1d(padded, sigma=sigma)
        v_smooth.iloc[:, h] = smoothed[3*sigma: 3*sigma + 365]
    v_profile_smooth = v_smooth.stack().rename("V_n")

    # ---- Align to agg index -------------------------------------------------
    # Lookup by (month, day, hour) from Normal.xlsx (any year works, use 2023)
    # and V_n by (doy, hour)
    result = pd.DataFrame(index=agg.index)

    # T_n, G_n: match by (month, day-of-month, hour) via a non-leap-year proxy
    def _get_normal_year(ts_index):
        """Return 2023-equivalent timestamps for Normal.xlsx lookup."""
        # Use 2023 as reference (non-leap); Feb-29 maps to Feb-28
        months = ts_index.month
        days   = ts_index.day
        hours  = ts_index.hour
        # Vectorised: build naive strings, fix Feb-29 → Feb-28
        feb29 = (months == 2) & (days == 29)
        days  = np.where(feb29, 28, days)
        naive = pd.DatetimeIndex([
            pd.Timestamp(f"2023-{m:02d}-{d:02d} {h:02d}:00")
            for m, d, h in zip(months, days, hours)
        ])
        return naive.tz_localize(
            TIMEZONE, ambiguous=False, nonexistent="shift_forward"
        )

    ref_idx = _get_normal_year(agg.index)
    # Use direct array lookup to avoid duplicate-label reindex error
    tn_lookup = raw_n["T_n"].to_dict()
    gn_lookup = raw_n["G_n"].to_dict()
    result["T_n"] = [tn_lookup.get(ts, np.nan) for ts in ref_idx]
    result["G_n"] = [gn_lookup.get(ts, np.nan) for ts in ref_idx]
    # Fill any remaining NaN (DST edge cases) by interpolation
    result["T_n"] = result["T_n"].interpolate(method="linear")
    result["G_n"] = result["G_n"].interpolate(method="linear")

    # V_n: match by (doy, hour) — doy from non-leap reference
    doy_arr  = ref_idx.dayofyear
    hour_arr = ref_idx.hour
    v_vals   = [v_profile_smooth.get((int(d), int(h)), np.nan)
                for d, h in zip(doy_arr, hour_arr)]
    result["V_n"] = v_vals

    print(f"  Normal meteo aligned to {len(result):,} timestamps")
    print(f"  T_n: {result['T_n'].min():.1f}…{result['T_n'].max():.1f} °C  "
          f"G_n: {result['G_n'].min():.2f}…{result['G_n'].max():.2f}  "
          f"V_n: {result['V_n'].min():.2f}…{result['V_n'].max():.2f} m/s")
    return result


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]       = df.index.year
    df["month"]      = df.index.month
    df["hour"]       = df.index.hour
    df["dow"]        = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # dow_group: Mon=0, Tue-Thu=1, Fri=2, Sat=3, Sun=4
    df["dow_group"] = df["dow"].map(
        {0:0, 1:1, 2:1, 3:1, 4:2, 5:3, 6:4}
    )

    cz_holidays = set()
    for yr in range(2020, 2027):
        cz_holidays.update(holidays.Czechia(years=yr).keys())
    df["is_holiday"] = pd.Index(df.index.date).isin(cz_holidays).astype(int)
    return df

# ---------------------------------------------------------------------------
# 5. EXTENDED WEATHER MODEL
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential moving average with time constant tau [hours]."""
    alpha = 1.0 - np.exp(-1.0 / max(tau, 0.5))
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    return lfilter(b, a, series).astype(np.float64)


def _nonlinear_features(T: np.ndarray, G: np.ndarray, V: np.ndarray,
                        theta: np.ndarray) -> np.ndarray:
    """
    Given raw T/G/V arrays (full time-ordered series) and θ,
    compute HDD_sat.  Returns 1-D array aligned with input.
    """
    aG, aV, tau1, tau2, beta, T_off, T_min = theta
    T_eff    = T + aG * G - aV * V
    T_smooth = beta * _ema(T_eff, tau1) + (1.0 - beta) * _ema(T_eff, tau2)
    HDD_sat  = np.clip(T_off - T_smooth, 0.0, T_off - T_min)
    return HDD_sat


def _make_fixed_features(df: pd.DataFrame, train_years: list[int]) -> np.ndarray:
    """
    Fixed (θ-independent) feature matrix for rows in df:
      - dow_group × hour dummies  (5×24 = 120 levels, drop_first)
      - month dummies             (11)
      - year dummies              (n_train_years - 1)

    All dummies are fitted on the provided df (which should be the training subset
    so that all expected categories are present).
    """
    label = df["dow_group"].astype(str) + "_" + df["hour"].astype(str)
    dgh   = pd.get_dummies(label,      prefix="dgh", drop_first=True)
    mon   = pd.get_dummies(df["month"], prefix="m",   drop_first=True)
    yr    = pd.get_dummies(df["year"],  prefix="y",   drop_first=True)
    return np.column_stack([dgh.values, mon.values, yr.values]).astype(np.float32)


def _make_heating_features(HDD_hist: np.ndarray, month_arr: np.ndarray) -> np.ndarray:
    """
    Month-varying heating features: one column per non-summer month.
    Summer months (Jun/Jul/Aug) excluded — HDD_sat ≈ 0, no physical heating.
    Returns array of shape (n, 9).
    """
    heating_months = [m for m in range(1, 13) if m not in SUMMER_MONTHS]
    out = np.zeros((len(HDD_hist), len(heating_months)), dtype=np.float64)
    for i, m in enumerate(heating_months):
        mask = month_arr == m
        out[mask, i] = HDD_hist[mask]
    return out


class ProgressBar:
    """Simple ASCII progress tracker for the optimiser."""
    BAR_WIDTH = 36

    def __init__(self, max_iter: int):
        self.max_iter  = max_iter
        self.n_calls   = 0
        self.best_mape = np.inf
        self._t0       = time.time()

    def update(self, mape: float):
        self.n_calls += 1
        if mape < self.best_mape:
            self.best_mape = mape
        pct   = min(self.n_calls / self.max_iter, 1.0)
        filled = int(pct * self.BAR_WIDTH)
        bar   = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self._t0
        sys.stdout.write(
            f"\r  [{bar}] {self.n_calls:4d}/{self.max_iter}"
            f"  best MAPE={self.best_mape:.3f}%  ({elapsed:.0f}s)"
        )
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def fit_extended_model(
    agg: pd.DataFrame,
    hist: pd.DataFrame,
    train_years: list[int] = None,
    eval_years:  list[int] = None,
    max_iter:    int       = 400,
):
    """
    Optimise nonlinear θ to minimise MAPE on eval_years.

    Training subset:  non-holiday ±5D hours  in train_years
    Eval subset:      same filter             in eval_years ⊆ train_years

    Linear features per θ:
      - 12 month×HDD_sat columns  (seasonal heating sensitivity)
      - fixed: dow_group×hour + month + year dummies

    Returns (best_theta, best_model, fixed_features_array, HDD_full_array)
    """
    if train_years is None:
        train_years = TRAIN_YEARS
    if eval_years is None:
        eval_years = EVAL_YEARS

    T_full   = agg["T"].values.astype(np.float64)
    G_full   = agg["G"].values.astype(np.float64)
    V_full   = agg["V"].values.astype(np.float64)
    hist_pos = agg.index.get_indexer(hist.index)

    # ---- restrict to train_years -------------------------------------------
    year_mask  = hist["year"].isin(train_years)
    train_mask = year_mask & hist["H"].notna() & hist["holiday_offset"].isna()
    eval_mask  = train_mask & hist["year"].isin(eval_years)

    # Build fixed features over ALL train_years rows together so that dummy
    # encoding is consistent (same reference category, same columns).
    hist_ty     = hist[year_mask]          # all rows in train_years
    F_ty        = _make_fixed_features(hist_ty, train_years)   # (n_ty, n_fixed)
    # Map back to positions within hist_ty
    ty_idx      = np.where(year_mask)[0]   # positions of train_year rows in hist
    # Build local (within hist_ty) masks
    local_train = train_mask[year_mask].values
    local_eval  = eval_mask[year_mask].values

    F_train     = F_ty[local_train]
    F_eval      = F_ty[local_eval]

    month_train = hist_ty["month"].values.astype(int)[local_train]
    month_eval  = hist_ty["month"].values.astype(int)[local_eval]
    y_train     = hist_ty["H"].values.astype(np.float64)[local_train]
    y_eval      = hist_ty["H"].values.astype(np.float64)[local_eval]

    # HDD positions within hist_ty (for the full-series EMA lookup)
    hist_ty_pos = agg.index.get_indexer(hist_ty.index)

    print(f"  Train rows: {train_mask.sum():,}  ({train_years})   "
          f"Eval rows: {eval_mask.sum():,}  ({eval_years})")

    pbar       = ProgressBar(max_iter)
    best_state = {"mape": np.inf, "theta": THETA0.copy(), "model": None}

    def objective(theta):
        for i, (lo, hi) in enumerate(THETA_BOUNDS):
            theta[i] = np.clip(theta[i], lo, hi)

        HDD_full  = _nonlinear_features(T_full, G_full, V_full, theta)
        HDD_ty    = HDD_full[hist_ty_pos]

        H_train = _make_heating_features(HDD_ty[local_train], month_train)
        H_eval  = _make_heating_features(HDD_ty[local_eval],  month_eval)

        X_train = np.column_stack([H_train, F_train])
        X_eval  = np.column_stack([H_eval,  F_eval])

        mdl   = LinearRegression(fit_intercept=True).fit(X_train, y_train)
        y_hat = np.maximum(mdl.predict(X_eval), 1.0)

        mape = mean_absolute_percentage_error(y_eval, y_hat) * 100.0
        pbar.update(mape)

        if mape < best_state["mape"]:
            best_state.update({"mape": mape, "theta": theta.copy(), "model": mdl})
        return mape

    minimize(
        objective, THETA0,
        method="L-BFGS-B",
        bounds=THETA_BOUNDS,
        options={"maxiter": max_iter, "ftol": 1e-5, "gtol": 1e-4},
    )
    pbar.done()

    best_theta = best_state["theta"]
    best_model = best_state["model"]

    print("\n  Optimised θ:")
    for name, val in zip(THETA_NAMES, best_theta):
        print(f"    {name:<12} = {val:.4f}")

    # Diagnostics
    HDD_full  = _nonlinear_features(T_full, G_full, V_full, best_theta)
    HDD_ty    = HDD_full[hist_ty_pos]
    H_tr      = _make_heating_features(HDD_ty[local_train], month_train)
    X_tr      = np.column_stack([H_tr, F_train])
    H_ev      = _make_heating_features(HDD_ty[local_eval], month_eval)
    X_ev      = np.column_stack([H_ev, F_eval])
    y_hat_tr  = np.maximum(best_model.predict(X_tr), 1.0)
    y_hat_ev  = np.maximum(best_model.predict(X_ev), 1.0)

    print(f"\n  Train ({train_years}):  "
          f"R²={r2_score(y_train,y_hat_tr):.4f}  "
          f"MAPE={mean_absolute_percentage_error(y_train,y_hat_tr)*100:.2f}%")
    print(f"  Eval  ({eval_years}):  "
          f"R²={r2_score(y_eval,y_hat_ev):.4f}  "
          f"MAPE={mean_absolute_percentage_error(y_eval,y_hat_ev)*100:.2f}%")

    # Month heating coefficients (first 9 linear params = non-summer months)
    heating_months = [m for m in range(1, 13) if m not in SUMMER_MONTHS]
    months_cz_all  = ["Led","Úno","Bře","Dub","Kvě","Čvn",
                      "Čvc","Srp","Zář","Říj","Lis","Pro"]
    coefs_heat = best_model.coef_[:len(heating_months)]
    max_c = max(coefs_heat) if max(coefs_heat) > 0 else 1
    print("\n  Heating sensitivity β_m [consumption per °C·HDD]:")
    hi = iter(coefs_heat)
    for m in range(1, 13):
        nm = months_cz_all[m - 1]
        if m in SUMMER_MONTHS:
            print(f"    {nm}: {'(letní — vyloučeno)':>15s}")
        else:
            c = next(hi)
            bar = "█" * max(0, int(c / max_c * 20))
            print(f"    {nm}: {c:10.1f}  {bar}")

    return best_theta, best_model, F_train, HDD_full, train_mask, month_train, F_ty, hist_ty


# ---------------------------------------------------------------------------
# 5b. SUMMER MODEL (Jun–Aug)  — purely calendar-based OLS, no weather
# ---------------------------------------------------------------------------

def fit_summer_model(
    hist:        pd.DataFrame,
    train_years: list = None,
    eval_years:  list = None,
) -> tuple:
    """
    Fit a simple OLS for summer months (SUMMER_MONTHS = {6,7,8}).

    Features: dow × hour dummies  +  month dummies  +  year dummies
    No weather features — summer consumption is weather-independent.

    Returns (model, col_names) where col_names is the list of dummy columns
    used during training (needed for aligned prediction on unseen years).
    """
    if train_years is None:
        train_years = TRAIN_YEARS
    if eval_years is None:
        eval_years = EVAL_YEARS

    train_mask = (
        hist["month"].isin(SUMMER_MONTHS) &
        hist["year"].isin(train_years) &
        hist["H"].notna() &
        hist["holiday_offset"].isna()
    )
    eval_mask = train_mask & hist["year"].isin(eval_years)

    def _make_X(sub):
        return pd.get_dummies(
            sub[["dow", "hour", "month", "year"]],
            columns=["dow", "hour", "month", "year"],
            drop_first=True,
            dtype=float,
        )

    tr = hist[train_mask]
    ev = hist[eval_mask]
    X_tr = _make_X(tr)
    col_names = X_tr.columns.tolist()
    X_ev = _make_X(ev).reindex(columns=col_names, fill_value=0.0)

    model = LinearRegression(fit_intercept=True).fit(X_tr, tr["H"].values)

    y_hat_tr = np.maximum(model.predict(X_tr), 1.0)
    y_hat_ev = np.maximum(model.predict(X_ev), 1.0)

    print(f"  Summer model:  train={train_mask.sum():,} h  eval={eval_mask.sum():,} h")
    print(f"  Train ({train_years}):  "
          f"R²={r2_score(tr['H'].values, y_hat_tr):.4f}  "
          f"MAPE={mean_absolute_percentage_error(tr['H'].values, y_hat_tr)*100:.2f}%")
    print(f"  Eval  ({eval_years}):  "
          f"R²={r2_score(ev['H'].values, y_hat_ev):.4f}  "
          f"MAPE={mean_absolute_percentage_error(ev['H'].values, y_hat_ev)*100:.2f}%")

    return model, col_names


def predict_summer(
    summer_model,
    col_names: list,
    hist: pd.DataFrame,
) -> pd.Series:
    """
    Apply the summer calendar model to ALL summer months in hist.

    Years outside training are handled by reindex(fill_value=0) so that
    unknown year dummies default to the reference-year level.
    """
    summer_mask = hist["month"].isin(SUMMER_MONTHS)
    result = pd.Series(np.nan, index=hist.index, name="H_pred_summer")
    if not summer_mask.any():
        return result

    sub = hist[summer_mask]
    X = pd.get_dummies(
        sub[["dow", "hour", "month", "year"]],
        columns=["dow", "hour", "month", "year"],
        drop_first=True,
        dtype=float,
    ).reindex(columns=col_names, fill_value=0.0)

    result[summer_mask] = np.maximum(summer_model.predict(X), 1.0)
    return result


def predict_extended(
    best_theta: np.ndarray,
    best_model,
    agg:        pd.DataFrame,
    hist:       pd.DataFrame,
    F_ty:       np.ndarray,
    hist_ty:    pd.DataFrame,
    HDD_full:   np.ndarray,
) -> pd.Series:
    """
    Apply best model to ALL hist rows.

    For rows inside train_years: use the consistent F_ty encoding.
    For rows outside train_years (e.g. 2020-2022): reconstruct fixed features
    aligned to the training column layout (year dummies for out-of-range
    years are zeroed → treated as the reference year).
    """
    hist_pos  = agg.index.get_indexer(hist.index)
    HDD_hist  = HDD_full[hist_pos]
    month_arr = hist["month"].values.astype(int)
    n_fixed   = F_ty.shape[1]

    # Build fixed features for the full hist, aligned to training columns
    # Strategy: encode over hist_ty (which defines the column set) +
    # concatenate out-of-training rows with zeroed year dummies.
    ty_in_hist = hist.index.isin(hist_ty.index)

    F_all = np.zeros((len(hist), n_fixed), dtype=np.float32)
    # Fill train_years rows from the precomputed F_ty
    ty_local_idx = np.where(ty_in_hist)[0]
    F_all[ty_local_idx] = F_ty

    # For non-train_years rows: encode dgh + month dummies (year → 0)
    out_mask = ~ty_in_hist
    if out_mask.any():
        hist_out = hist[out_mask]
        label_o  = hist_out["dow_group"].astype(str) + "_" + hist_out["hour"].astype(str)
        # Use same dgh + month structure; year dummies stay 0
        dgh_o = pd.get_dummies(label_o,          prefix="dgh", drop_first=True)
        mon_o = pd.get_dummies(hist_out["month"], prefix="m",   drop_first=True)
        # Align to training column count (dgh: cols 0..118, mon: 119..129)
        n_dgh = 119   # 5*24 - 1
        n_mon = 11
        F_out = np.zeros((out_mask.sum(), n_fixed), dtype=np.float32)
        for i, col in enumerate(dgh_o.columns):
            if i < n_dgh:
                F_out[:, i] = dgh_o[col].values
        for i, col in enumerate(mon_o.columns):
            if n_dgh + i < n_dgh + n_mon:
                F_out[:, n_dgh + i] = mon_o[col].values
        F_all[np.where(out_mask)[0]] = F_out

    H_all = _make_heating_features(HDD_hist, month_arr)
    X_all = np.column_stack([H_all, F_all])
    pred  = np.maximum(best_model.predict(X_all), 1.0)
    return pd.Series(pred, index=hist.index, name="H_pred")

def predict_normal(
    best_theta: np.ndarray,
    best_model,
    agg:        pd.DataFrame,
    hist:       pd.DataFrame,
    normal:     pd.DataFrame,
    F_ty:       np.ndarray,
    hist_ty:    pd.DataFrame,
) -> pd.Series:
    """
    Apply the fitted model to normal meteo (T_n, G_n, V_n) for each
    historical timestamp, using the same calendar structure (dow, hour,
    month, year) as the actual data.

    This answers: "what would consumption have been on this day if
    meteo had been normal?"
    """
    # Build a copy of agg with normal meteo substituted
    agg_n = agg.copy()
    # Align normal meteo to agg index (normal is already indexed like agg)
    agg_n["T"] = normal["T_n"].reindex(agg.index).values
    agg_n["G"] = normal["G_n"].reindex(agg.index).values
    agg_n["V"] = normal["V_n"].reindex(agg.index).values

    # Compute HDD_sat_n on the full series with normal meteo
    T_n = agg_n["T"].values.astype(np.float64)
    G_n = agg_n["G"].values.astype(np.float64)
    V_n = agg_n["V"].values.astype(np.float64)
    HDD_full_n = _nonlinear_features(T_n, G_n, V_n, best_theta)

    return predict_extended(best_theta, best_model, agg_n, hist, F_ty, hist_ty, HDD_full_n)


# ---------------------------------------------------------------------------
# 6. HOLIDAY INDEX
# ---------------------------------------------------------------------------

DOW_NAMES    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
OFFSET_LABELS = {
    -5:"WD-5",-4:"WD-4",-3:"WD-3",-2:"WD-2",-1:"WD-1",
     0:"HOL",
     1:"WD+1", 2:"WD+2", 3:"WD+3", 4:"WD+4", 5:"WD+5",
}


def build_holiday_index(df: pd.DataFrame) -> pd.DataFrame:
    years     = df.index.year.unique()
    hol_dates = sorted({d for yr in years for d in holidays.Czechia(years=yr)})
    records   = {}

    for hdate in hol_dates:
        hts     = pd.Timestamp(hdate)
        hol_dow = hts.dayofweek
        for delta in range(-CALENDAR_WINDOW*2, CALENDAR_WINDOW*2+1):
            cdate = hdate + pd.Timedelta(days=delta)
            cts   = pd.Timestamp(cdate)
            if delta == 0:
                wd_offset = 0
            else:
                sign      = 1 if delta > 0 else -1
                bdays     = pd.bdate_range(min(hts,cts), max(hts,cts), freq="B")
                wd_offset = sign * (len(bdays) - 1)
            if abs(wd_offset) > CALENDAR_WINDOW:
                continue
            prev = records.get(cdate)
            if prev is None or abs(wd_offset) < prev[0]:
                records[cdate] = (abs(wd_offset), wd_offset, hol_dow, hdate)

    if not records:
        df[["holiday_offset","holiday_dow","holiday_date"]] = np.nan
        return df

    lookup = pd.DataFrame.from_dict(
        {d: v[1:] for d,v in records.items()}, orient="index",
        columns=["holiday_offset","holiday_dow","holiday_date"]
    )
    lookup.index = pd.to_datetime(lookup.index)

    df = df.copy()
    date_idx = df.index.normalize().tz_localize(None)
    df["holiday_offset"] = date_idx.map(lookup["holiday_offset"])
    df["holiday_dow"]    = date_idx.map(lookup["holiday_dow"])
    df["holiday_date"]   = date_idx.map(lookup["holiday_date"])
    return df

# ---------------------------------------------------------------------------
# 7. ENERGY CALENDAR COEFFICIENTS
# ---------------------------------------------------------------------------

def print_holiday_coverage(calendar_years: list[int]):
    """Print how many holidays fall on each weekday per year."""
    rows = []
    for yr in range(min(calendar_years), max(calendar_years) + 1):
        for date, name in holidays.Czechia(years=yr).items():
            rows.append({"year": yr, "dow": date.weekday(),
                         "scenario": DOW_NAMES[date.weekday()], "name": name})
    df = pd.DataFrame(rows)

    pivot = (df.groupby(["scenario", "year"]).size()
               .unstack(fill_value=0)
               .reindex(index=DOW_NAMES))
    pivot["TOTAL"] = pivot.sum(axis=1)

    print(f"\n  Holiday coverage (n events per scenario × year):")
    print(f"  Calendar years: {calendar_years}")
    cal_cols = [y for y in pivot.columns if y in calendar_years] + ["TOTAL"]
    print(pivot[cal_cols].to_string())

    thin = pivot.loc[pivot["TOTAL"] < 5, "TOTAL"]
    if not thin.empty:
        print(f"\n  ⚠  Thin scenarios (TOTAL < 5): {list(thin.index)}")
    return df


def compute_calendar_coefficients(
    df: pd.DataFrame,
    train_years: list[int] = None,
) -> pd.DataFrame:
    """
    Compute median(H/H_pred) per (holiday scenario × working-day offset).
    Only uses rows from train_years (consistent with model fit period).
    """
    if train_years is None:
        train_years = TRAIN_YEARS
    df = df[df["year"].isin(train_years)]

    results = []
    for hol_dow in range(7):
        for offset, label in OFFSET_LABELS.items():
            mask = (
                (df["holiday_dow"]    == hol_dow) &
                (df["holiday_offset"] == offset)  &
                df["ratio"].notna() & np.isfinite(df["ratio"]) &
                (df["ratio"] > 0.05) & (df["ratio"] < 5.0)
            )
            sub = df[mask]
            coef = sub["ratio"].median() if len(sub) else np.nan
            n    = sub["holiday_date"].nunique() if len(sub) else 0
            results.append({
                "holiday_dow": hol_dow, "scenario": DOW_NAMES[hol_dow],
                "offset": offset, "offset_label": label,
                "coefficient": coef, "n_events": n,
            })
    return pd.DataFrame(results)


def print_calendar_table(coef_df: pd.DataFrame):
    ordered = [OFFSET_LABELS[k] for k in sorted(OFFSET_LABELS)]
    pivot = (
        coef_df
        .pivot_table(index="scenario", columns="offset_label",
                     values="coefficient", aggfunc="first")
        .reindex(columns=[c for c in ordered if c in coef_df["offset_label"].values])
        .reindex(index=DOW_NAMES)
    )
    n_pivot = (
        coef_df
        .pivot_table(index="scenario", columns="offset_label",
                     values="n_events", aggfunc="first")
        .reindex(columns=[c for c in ordered if c in coef_df["offset_label"].values])
        .reindex(index=DOW_NAMES)
    )
    print("\n=== ENERGY CALENDAR COEFFICIENTS ===")
    print("  (median H_actual/H_predicted,  1.0 = normal consumption)")
    print(pivot.round(3).to_string())
    print("\n  (n_events per cell)")
    print(n_pivot.to_string())
    return pivot

# ---------------------------------------------------------------------------
# 8. GAS-DAY (DAILY) MODEL
# ---------------------------------------------------------------------------

DAILY_FIT_PATH      = CACHE_DIR / "model_fit_daily.parquet"
DAILY_CALENDAR_PATH = CACHE_DIR / "calendar_coefficients_daily.parquet"

# θ = [αG, αV, τ1(days), τ2(days), β, T_off, T_min]
DAILY_THETA_BOUNDS = [
    (0.0,  5.0),   # αG
    (0.0,  3.0),   # αV
    (0.2, 10.0),   # τ1 [days]
    (1.0, 60.0),   # τ2 [days]
    (0.0,  1.0),   # β
    (10.0, 18.0),  # T_off [°C]
    (-25.0, 0.0),  # T_min [°C]
]
DAILY_THETA0 = np.array([1.0, 0.5, 0.5, 5.0, 0.5, 15.0, -15.0])


def _gas_day_weights(hist: pd.DataFrame) -> np.ndarray:
    """
    24-element weight vector for T/G/V aggregation indexed by gas-hour
    position k (k=0 → 06:00, k=23 → 05:00 next day).
    Weights = mean H in heating months, normalised to sum=1.
    """
    heating = hist[hist["month"].isin([m for m in range(1, 13)
                                       if m not in SUMMER_MONTHS])].copy()
    heating["_gas_hour"] = (heating.index.hour - 6) % 24
    w = heating.groupby("_gas_hour")["H"].mean()
    w = w.reindex(range(24), fill_value=w.mean())
    return (w / w.sum()).values


def aggregate_to_gas_day(hist: pd.DataFrame, weights: np.ndarray = None) -> pd.DataFrame:
    """
    Aggregate hourly data to gas-day level.

    Gas day D: 06:00 on date D → 05:59 on date D+1.
    Hours 00–05 belong to the PREVIOUS gas day (previous calendar date).

    H    : sum of all hours in the gas day
    T/G/V: consumption-weighted mean (weights from _gas_day_weights).
            Falls back to simple mean on DST days (≠24 h).
    """
    if weights is None:
        weights = _gas_day_weights(hist)

    df = hist[["H", "T", "G", "V"]].copy()
    local_hours = df.index.hour

    # Gas day label as naive date (tz stripped for groupby)
    cal_dates = pd.to_datetime(df.index.date)  # naive midnight timestamps
    gas_day_labels = np.where(
        local_hours < 6,
        cal_dates - pd.Timedelta(days=1),
        cal_dates,
    )
    df["_gas_day"]  = gas_day_labels
    df["_gas_hour"] = (local_hours - 6) % 24
    df["_weight"]   = weights[df["_gas_hour"].values]

    # Weighted sum components
    for col in ["T", "G", "V"]:
        df[f"_{col}w"] = df[col] * df["_weight"]

    agg = df.groupby("_gas_day").agg(
        H      =("H",        "sum"),
        n_hours=("H",        "count"),
        _Tw    =("_Tw",      "sum"),
        _Gw    =("_Gw",      "sum"),
        _Vw    =("_Vw",      "sum"),
        _wsum  =("_weight",  "sum"),
    )
    # Weighted mean (w_sum ≈ 1.0 for normal 24-h days; ≠1 on DST)
    for col in ["T", "G", "V"]:
        agg[col] = agg[f"_{col}w"] / agg["_wsum"]
    agg = agg[["H", "T", "G", "V", "n_hours"]].copy()

    # Re-attach tz at midnight (calendar date, Europe/Prague)
    agg.index = pd.DatetimeIndex(agg.index).tz_localize(
        TIMEZONE, ambiguous=False, nonexistent="shift_forward"
    )
    agg.index.name = "timestamp"
    agg["year"]  = agg.index.year
    agg["month"] = agg.index.month
    agg["dow"]   = agg.index.dayofweek
    return agg


def aggregate_normal_to_gas_day(normal: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Aggregate hourly normal meteo to gas-day level using the same weights."""
    df = normal[["T_n", "G_n", "V_n"]].copy()
    local_hours = df.index.hour
    cal_dates   = pd.to_datetime(df.index.date)
    gas_day_labels = np.where(
        local_hours < 6,
        cal_dates - pd.Timedelta(days=1),
        cal_dates,
    )
    df["_gas_day"]  = gas_day_labels
    df["_gas_hour"] = (local_hours - 6) % 24
    df["_weight"]   = weights[df["_gas_hour"].values]
    for col in ["T_n", "G_n", "V_n"]:
        df[f"_{col}w"] = df[col] * df["_weight"]

    agg = df.groupby("_gas_day").agg(
        _T_nw =("_T_nw", "sum"),
        _G_nw =("_G_nw", "sum"),
        _V_nw =("_V_nw", "sum"),
        _wsum =("_weight", "sum"),
    )
    for col in ["T_n", "G_n", "V_n"]:
        agg[col] = agg[f"_{col}w"] / agg["_wsum"]
    agg = agg[["T_n", "G_n", "V_n"]].copy()
    agg.index = pd.DatetimeIndex(agg.index).tz_localize(
        TIMEZONE, ambiguous=False, nonexistent="shift_forward"
    )
    agg.index.name = "timestamp"
    return agg


def build_daily_holiday_index(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Holiday index for daily (gas-day) data.
    Holiday assignment: calendar midnight — gas day label date is used as-is.
    Also adds is_holiday (True if the gas day date is a Czech public holiday).
    """
    df = build_holiday_index(daily)   # reuse hourly logic; works on date index
    years    = df.index.year.unique()
    hol_set  = {d for yr in years for d in holidays.Czechia(years=yr)}
    df["is_holiday"] = pd.to_datetime(df.index.date).isin(hol_set)
    return df


def _make_daily_fixed_features(daily_ty: pd.DataFrame, train_years: list) -> np.ndarray:
    """
    Fixed calendar dummies for daily model:
      dow (6) + month (11) + year (n-1)
    Total: ~20 columns (no hour dimension).
    """
    dow_d  = pd.get_dummies(daily_ty["dow"],   prefix="dow", drop_first=True, dtype=float)
    mon_d  = pd.get_dummies(daily_ty["month"], prefix="m",   drop_first=True, dtype=float)
    yr_d   = pd.get_dummies(daily_ty["year"],  prefix="yr",  drop_first=True, dtype=float)
    return np.column_stack([dow_d.values, mon_d.values, yr_d.values]).astype(np.float32)


def fit_daily_model(
    daily:       pd.DataFrame,      # full daily series (all years, incl. future NaN)
    daily_hist:  pd.DataFrame,      # historical rows with H + holiday index
    train_years: list = None,
    eval_years:  list = None,
    max_iter:    int  = 400,
) -> tuple:
    """
    Fit the daily extended weather model.
    Structure mirrors fit_extended_model but operates on gas-day aggregates.
    τ1, τ2 are in units of days.
    """
    if train_years is None:
        train_years = TRAIN_YEARS
    if eval_years is None:
        eval_years = EVAL_YEARS

    T_full = daily["T"].values.astype(np.float64)
    G_full = daily["G"].values.astype(np.float64)
    V_full = daily["V"].values.astype(np.float64)

    hist     = daily_hist
    hist_pos = daily.index.get_indexer(hist.index)

    year_mask  = hist["year"].isin(train_years)
    train_mask = year_mask & hist["H"].notna() & hist["holiday_offset"].isna()
    eval_mask  = train_mask & hist["year"].isin(eval_years)

    hist_ty   = hist[year_mask]
    F_ty      = _make_daily_fixed_features(hist_ty, train_years)
    ty_idx    = np.where(year_mask)[0]
    local_train = train_mask[year_mask].values
    local_eval  = eval_mask[year_mask].values

    F_train     = F_ty[local_train]
    F_eval      = F_ty[local_eval]
    month_train = hist_ty["month"].values.astype(int)[local_train]
    month_eval  = hist_ty["month"].values.astype(int)[local_eval]
    y_train     = hist_ty["H"].values.astype(np.float64)[local_train]
    y_eval      = hist_ty["H"].values.astype(np.float64)[local_eval]
    hist_ty_pos = daily.index.get_indexer(hist_ty.index)

    print(f"  Train rows: {train_mask.sum():,}  ({train_years})   "
          f"Eval rows: {eval_mask.sum():,}  ({eval_years})")

    pbar       = ProgressBar(max_iter)
    best_state = {"mape": np.inf, "theta": DAILY_THETA0.copy(), "model": None}

    def objective(theta):
        for i, (lo, hi) in enumerate(DAILY_THETA_BOUNDS):
            theta[i] = np.clip(theta[i], lo, hi)
        HDD_full = _nonlinear_features(T_full, G_full, V_full, theta)
        HDD_ty   = HDD_full[hist_ty_pos]
        H_train  = _make_heating_features(HDD_ty[local_train], month_train)
        H_eval   = _make_heating_features(HDD_ty[local_eval],  month_eval)
        X_train  = np.column_stack([H_train, F_train])
        X_eval   = np.column_stack([H_eval,  F_eval])
        mdl      = LinearRegression(fit_intercept=True).fit(X_train, y_train)
        y_hat    = np.maximum(mdl.predict(X_eval), 1.0)
        mape     = mean_absolute_percentage_error(y_eval, y_hat) * 100.0
        pbar.update(mape)
        if mape < best_state["mape"]:
            best_state.update({"mape": mape, "theta": theta.copy(), "model": mdl})
        return mape

    minimize(
        objective, DAILY_THETA0,
        method="L-BFGS-B",
        bounds=DAILY_THETA_BOUNDS,
        options={"maxiter": max_iter, "ftol": 1e-5, "gtol": 1e-4},
    )
    pbar.done()

    best_theta = best_state["theta"]
    best_model = best_state["model"]

    print("\n  Optimised θ:")
    for name, val in zip(THETA_NAMES, best_theta):
        print(f"    {name:<12} = {val:.4f}")

    HDD_full = _nonlinear_features(T_full, G_full, V_full, best_theta)
    HDD_ty   = HDD_full[hist_ty_pos]
    H_tr     = _make_heating_features(HDD_ty[local_train], month_train)
    X_tr     = np.column_stack([H_tr, F_train])
    H_ev     = _make_heating_features(HDD_ty[local_eval], month_eval)
    X_ev     = np.column_stack([H_ev, F_eval])
    y_hat_tr = np.maximum(best_model.predict(X_tr), 1.0)
    y_hat_ev = np.maximum(best_model.predict(X_ev), 1.0)

    print(f"\n  Train ({train_years}):  "
          f"R²={r2_score(y_train, y_hat_tr):.4f}  "
          f"MAPE={mean_absolute_percentage_error(y_train, y_hat_tr)*100:.2f}%")
    print(f"  Eval  ({eval_years}):  "
          f"R²={r2_score(y_eval, y_hat_ev):.4f}  "
          f"MAPE={mean_absolute_percentage_error(y_eval, y_hat_ev)*100:.2f}%")

    heating_months = [m for m in range(1, 13) if m not in SUMMER_MONTHS]
    months_cz_all  = ["Led","Úno","Bře","Dub","Kvě","Čvn",
                      "Čvc","Srp","Zář","Říj","Lis","Pro"]
    coefs_heat = best_model.coef_[:len(heating_months)]
    max_c      = max(coefs_heat) if max(coefs_heat) > 0 else 1
    print("\n  Heating sensitivity β_m [MWh per °C·HDD·day]:")
    hi = iter(coefs_heat)
    for m in range(1, 13):
        nm = months_cz_all[m - 1]
        if m in SUMMER_MONTHS:
            print(f"    {nm}: {'(letní — vyloučeno)':>15s}")
        else:
            c   = next(hi)
            bar = "█" * max(0, int(c / max_c * 20))
            print(f"    {nm}: {c:12.1f}  {bar}")

    return best_theta, best_model, HDD_full, train_mask, F_ty, hist_ty


def predict_daily_model(
    best_theta: np.ndarray,
    best_model,
    daily:      pd.DataFrame,   # full daily series (all years)
    daily_hist: pd.DataFrame,   # rows with actual H
    F_ty:       np.ndarray,
    hist_ty:    pd.DataFrame,
    HDD_full:   np.ndarray,
) -> pd.Series:
    """Apply fitted daily model to all historical rows."""
    hist_pos  = daily.index.get_indexer(daily_hist.index)
    HDD_hist  = HDD_full[hist_pos]
    month_arr = daily_hist["month"].values.astype(int)
    n_fixed   = F_ty.shape[1]

    ty_in_hist = daily_hist.index.isin(hist_ty.index)
    F_all      = np.zeros((len(daily_hist), n_fixed), dtype=np.float32)
    ty_local_idx = np.where(ty_in_hist)[0]
    F_all[ty_local_idx] = F_ty

    # Out-of-training rows: dow + month dummies, year dummies → 0
    out_mask = ~ty_in_hist
    if out_mask.any():
        hist_out = daily_hist[out_mask]
        dow_o = pd.get_dummies(hist_out["dow"],   prefix="dow", drop_first=True, dtype=float)
        mon_o = pd.get_dummies(hist_out["month"], prefix="m",   drop_first=True, dtype=float)
        n_dow = 6; n_mon = 11
        F_out = np.zeros((out_mask.sum(), n_fixed), dtype=np.float32)
        for i, col in enumerate(dow_o.columns):
            if i < n_dow:
                F_out[:, i] = dow_o[col].values
        for i, col in enumerate(mon_o.columns):
            if n_dow + i < n_dow + n_mon:
                F_out[:, n_dow + i] = mon_o[col].values
        F_all[np.where(out_mask)[0]] = F_out

    H_heat = _make_heating_features(HDD_hist, month_arr)
    X_all  = np.column_stack([H_heat, F_all])
    pred   = np.maximum(best_model.predict(X_all), 1.0)
    return pd.Series(pred, index=daily_hist.index, name="H_pred")


def predict_daily_normal(
    best_theta:  np.ndarray,
    best_model,
    daily:       pd.DataFrame,
    daily_hist:  pd.DataFrame,
    normal_daily: pd.DataFrame,
    F_ty:        np.ndarray,
    hist_ty:     pd.DataFrame,
) -> pd.Series:
    """Apply daily model to climatological normal meteo."""
    daily_n = daily.copy()
    daily_n["T"] = normal_daily["T_n"].reindex(daily.index).values
    daily_n["G"] = normal_daily["G_n"].reindex(daily.index).values
    daily_n["V"] = normal_daily["V_n"].reindex(daily.index).values
    # Interpolate the few NaN edge cases (same DST issue as hourly)
    for col in ["T", "G", "V"]:
        daily_n[col] = pd.Series(daily_n[col], index=daily.index).interpolate()

    T_n = daily_n["T"].values.astype(np.float64)
    G_n = daily_n["G"].values.astype(np.float64)
    V_n = daily_n["V"].values.astype(np.float64)
    HDD_full_n = _nonlinear_features(T_n, G_n, V_n, best_theta)
    return predict_daily_model(best_theta, best_model, daily_n, daily_hist,
                               F_ty, hist_ty, HDD_full_n)


def fit_daily_summer_model(
    daily_hist:  pd.DataFrame,
    train_years: list = None,
    eval_years:  list = None,
) -> tuple:
    """
    Simple OLS for summer months at daily level.
    Features: dow + month + year dummies (no weather).
    """
    if train_years is None:
        train_years = TRAIN_YEARS
    if eval_years is None:
        eval_years = EVAL_YEARS

    train_mask = (
        daily_hist["month"].isin(SUMMER_MONTHS) &
        daily_hist["year"].isin(train_years) &
        daily_hist["H"].notna() &
        daily_hist["holiday_offset"].isna()
    )
    eval_mask = train_mask & daily_hist["year"].isin(eval_years)

    def _make_X(sub):
        return pd.get_dummies(
            sub[["dow", "month", "year"]],
            columns=["dow", "month", "year"],
            drop_first=True, dtype=float,
        )

    tr    = daily_hist[train_mask]
    ev    = daily_hist[eval_mask]
    X_tr  = _make_X(tr)
    cols  = X_tr.columns.tolist()
    X_ev  = _make_X(ev).reindex(columns=cols, fill_value=0.0)

    model    = LinearRegression(fit_intercept=True).fit(X_tr, tr["H"].values)
    yhat_tr  = np.maximum(model.predict(X_tr), 1.0)
    yhat_ev  = np.maximum(model.predict(X_ev), 1.0)

    print(f"  Daily summer model:  train={train_mask.sum()} d  eval={eval_mask.sum()} d")
    print(f"  Train ({train_years}):  "
          f"R²={r2_score(tr['H'].values, yhat_tr):.4f}  "
          f"MAPE={mean_absolute_percentage_error(tr['H'].values, yhat_tr)*100:.2f}%")
    print(f"  Eval  ({eval_years}):  "
          f"R²={r2_score(ev['H'].values, yhat_ev):.4f}  "
          f"MAPE={mean_absolute_percentage_error(ev['H'].values, yhat_ev)*100:.2f}%")
    return model, cols


def predict_daily_summer(model, col_names: list, daily_hist: pd.DataFrame) -> pd.Series:
    """Apply summer calendar model to all summer months in daily_hist."""
    summer_mask = daily_hist["month"].isin(SUMMER_MONTHS)
    result = pd.Series(np.nan, index=daily_hist.index, name="H_pred_summer")
    if not summer_mask.any():
        return result
    sub = daily_hist[summer_mask]
    X = pd.get_dummies(
        sub[["dow", "month", "year"]],
        columns=["dow", "month", "year"],
        drop_first=True, dtype=float,
    ).reindex(columns=col_names, fill_value=0.0)
    result[summer_mask] = np.maximum(model.predict(X), 1.0)
    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)

    print("STEP 1: Load data")
    df_long = load_data()

    print("\nSTEP 2: Impute gaps in H")
    df_long = impute_consumption(df_long)

    print("\nSTEP 3: Aggregate regions")
    agg = aggregate_regions(df_long, weight_by="year")

    print("\nSTEP 4: Feature engineering")
    agg = build_features(agg)
    agg.to_parquet(AGG_PATH)
    print(f"  Saved: {AGG_PATH}")

    hist = agg[agg["H"].notna()].copy()
    print(f"  Historical rows: {len(hist):,}  "
          f"({hist.index.min().date()} → {hist.index.max().date()})")

    print("\nSTEP 5: Holiday index")
    hist = build_holiday_index(hist)
    n_tagged = hist["holiday_offset"].notna().sum()
    print(f"  Hours tagged with holiday context: {n_tagged:,}  "
          f"({n_tagged/24:.0f} days)")

    print(f"\nSTEP 6a: Fit extended weather model  (train={TRAIN_YEARS})")
    best_theta, best_model, F_train, HDD_full, train_mask, month_train, F_ty, hist_ty = \
        fit_extended_model(agg, hist, train_years=TRAIN_YEARS, eval_years=EVAL_YEARS, max_iter=400)

    print(f"\nSTEP 6b: Fit summer calendar model  (train={TRAIN_YEARS})")
    summer_model, summer_cols = fit_summer_model(
        hist, train_years=TRAIN_YEARS, eval_years=EVAL_YEARS
    )

    print("\nSTEP 7: Normal meteo")
    normal = load_normal_meteo(agg)
    print(f"  Normal meteo loaded: {len(normal):,} rows")

    print("\nSTEP 8: Energy calendar coefficients")
    print_holiday_coverage(CALENDAR_YEARS)

    # Heating model prediction (all months)
    H_pred_heating = predict_extended(
        best_theta, best_model, agg, hist, F_ty, hist_ty, HDD_full
    )
    H_norm_heating = predict_normal(
        best_theta, best_model, agg, hist, normal, F_ty, hist_ty
    )
    # Summer model prediction (summer months only)
    H_pred_summer = predict_summer(summer_model, summer_cols, hist)

    # Merge: summer months use summer model, rest use heating model
    is_summer = hist["month"].isin(SUMMER_MONTHS)
    hist["H_pred"] = np.where(is_summer, H_pred_summer, H_pred_heating)
    # H_norm in summer = H_pred (calendar model is already weather-independent)
    hist["H_norm"] = np.where(is_summer, H_pred_summer, H_norm_heating)
    hist["ratio"]  = hist["H"] / hist["H_pred"]

    # ---- MAPE diagnostics by season ----------------------------------------
    eval_mask = hist["year"].isin(EVAL_YEARS) & hist["H"].notna()
    def _mape(mask):
        sub = hist[eval_mask & mask]
        return mean_absolute_percentage_error(sub["H"], sub["H_pred"]) * 100

    mape_heat   = _mape(~is_summer)
    mape_summer = _mape(is_summer)
    mape_total  = _mape(pd.Series(True, index=hist.index))
    print(f"\n  MAPE by season (eval {EVAL_YEARS}):")
    print(f"    Topná (mimo 6–8): {mape_heat:.2f}%")
    print(f"    Letní  (6–8):     {mape_summer:.2f}%")
    print(f"    Celkem:           {mape_total:.2f}%")

    coef_df = compute_calendar_coefficients(hist, train_years=CALENDAR_YEARS)
    pivot   = print_calendar_table(coef_df)

    # ---- Save results -------------------------------------------------------
    # Attach normal meteo columns to hist for output
    hist["T_n"] = normal["T_n"].reindex(hist.index).values
    hist["G_n"] = normal["G_n"].reindex(hist.index).values
    hist["V_n"] = normal["V_n"].reindex(hist.index).values

    fit_out = hist[["H", "H_pred", "H_norm",
                    "T", "G", "V", "T_n", "G_n", "V_n",
                    "year", "month", "hour", "dow",
                    "is_holiday", "holiday_offset", "holiday_dow"]].copy()

    fit_out.to_parquet(FIT_PATH)
    print(f"\n  Saved: {FIT_PATH}")

    coef_df.to_parquet(CALENDAR_PATH)
    print(f"  Saved: {CALENDAR_PATH}")

    # =========================================================================
    # DAILY (GAS-DAY) MODEL
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Aggregate to gas days")
    gdw = _gas_day_weights(hist)
    daily = aggregate_to_gas_day(hist, gdw)
    print(f"  Gas days: {len(daily):,}  "
          f"({daily.index.min().date()} → {daily.index.max().date()})")
    print(f"  DST days (≠24 h): {(daily['n_hours'] != 24).sum()}")

    daily_hist = daily[daily["H"].notna()].copy()
    print(f"  Historical gas days: {len(daily_hist):,}")

    print("\nSTEP 10: Daily holiday index")
    daily_hist = build_daily_holiday_index(daily_hist)
    n_tagged_d = daily_hist["holiday_offset"].notna().sum()
    print(f"  Days tagged: {n_tagged_d:,}")

    print(f"\nSTEP 11a: Fit daily weather model  (train={TRAIN_YEARS})")
    (d_theta, d_model, HDD_full_d,
     d_train_mask, F_ty_d, hist_ty_d) = fit_daily_model(
        daily, daily_hist, train_years=TRAIN_YEARS, eval_years=EVAL_YEARS, max_iter=400
    )

    print(f"\nSTEP 11b: Fit daily summer model  (train={TRAIN_YEARS})")
    d_summer_model, d_summer_cols = fit_daily_summer_model(
        daily_hist, train_years=TRAIN_YEARS, eval_years=EVAL_YEARS
    )

    print("\nSTEP 12: Normal meteo (daily)")
    normal_daily = aggregate_normal_to_gas_day(normal, gdw)
    print(f"  Normal daily: {len(normal_daily):,} gas days")

    print("\nSTEP 13: Daily predictions & calendar coefficients")
    H_pred_d_heat = predict_daily_model(
        d_theta, d_model, daily, daily_hist, F_ty_d, hist_ty_d, HDD_full_d
    )
    H_norm_d_heat = predict_daily_normal(
        d_theta, d_model, daily, daily_hist, normal_daily, F_ty_d, hist_ty_d
    )
    H_pred_d_sum  = predict_daily_summer(d_summer_model, d_summer_cols, daily_hist)

    is_summer_d = daily_hist["month"].isin(SUMMER_MONTHS)
    daily_hist["H_pred"] = np.where(is_summer_d, H_pred_d_sum, H_pred_d_heat)
    daily_hist["H_norm"] = np.where(is_summer_d, H_pred_d_sum, H_norm_d_heat)
    daily_hist["ratio"]  = daily_hist["H"] / daily_hist["H_pred"]

    # MAPE by season
    eval_mask_d = daily_hist["year"].isin(EVAL_YEARS) & daily_hist["H"].notna()
    def _mape_d(mask):
        sub = daily_hist[eval_mask_d & mask]
        return mean_absolute_percentage_error(sub["H"], sub["H_pred"]) * 100

    print(f"\n  MAPE by season (eval {EVAL_YEARS}):")
    print(f"    Topná (mimo 6–8): {_mape_d(~is_summer_d):.2f}%")
    print(f"    Letní  (6–8):     {_mape_d(is_summer_d):.2f}%")
    print(f"    Celkem:           {_mape_d(pd.Series(True, index=daily_hist.index)):.2f}%")

    coef_df_d = compute_calendar_coefficients(daily_hist, train_years=CALENDAR_YEARS)
    print_calendar_table(coef_df_d)

    # ---- Save daily results -------------------------------------------------
    daily_hist["T_n"] = normal_daily["T_n"].reindex(daily_hist.index).values
    daily_hist["G_n"] = normal_daily["G_n"].reindex(daily_hist.index).values
    daily_hist["V_n"] = normal_daily["V_n"].reindex(daily_hist.index).values

    fit_out_d = daily_hist[["H", "H_pred", "H_norm",
                             "T", "G", "V", "T_n", "G_n", "V_n",
                             "year", "month", "dow",
                             "is_holiday", "holiday_offset", "holiday_dow"]].copy()
    fit_out_d.to_parquet(DAILY_FIT_PATH)
    print(f"\n  Saved: {DAILY_FIT_PATH}")

    coef_df_d.to_parquet(DAILY_CALENDAR_PATH)
    print(f"  Saved: {DAILY_CALENDAR_PATH}")

    return hist, best_theta, best_model, coef_df, pivot


if __name__ == "__main__":
    main()
