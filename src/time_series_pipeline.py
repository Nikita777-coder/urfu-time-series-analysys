from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


RAW_PATH = ROOT / "data" / "raw" / "brisbane_water_quality.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "brisbane_temperature_daily.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "model_comparison.csv"
FORECASTS_PATH = REPORTS_DIR / "forecast_predictions.csv"
BACKTESTING_PATH = REPORTS_DIR / "backtesting_summary.csv"
PREDICTION_INTERVALS_PATH = REPORTS_DIR / "prediction_intervals.csv"
RELIABILITY_PATH = REPORTS_DIR / "reliability.json"
ANOMALIES_PATH = REPORTS_DIR / "anomaly_summary.csv"
ANOMALY_DETAILS_PATH = REPORTS_DIR / "anomaly_details.csv"
DIAGNOSTICS_PATH = REPORTS_DIR / "diagnostics.json"
SUMMARY_PATH = REPORTS_DIR / "summary.json"

TARGET = "Temperature"
DATE_COL = "date"
TEST_DAYS = 60
FORECAST_HORIZON_DAYS = 14
SEASON_LENGTH = 7
RANDOM_STATE = 42
BACKTESTING_HORIZON = 14
BACKTESTING_WINDOWS = 4


QUALITY_COLUMNS_SUFFIX = " [quality]"
LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [3, 7, 14]


@dataclass(frozen=True)
class ForecastResult:
    family: str
    model: str
    forecast: np.ndarray
    notes: str


def _ensure_dirs() -> None:
    for path in [PROCESSED_PATH.parent, REPORTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _numeric_measure_columns(columns: pd.Index) -> list[str]:
    ignored = {"Timestamp", "Record number"}
    return [
        col
        for col in columns
        if col not in ignored and not col.endswith(QUALITY_COLUMNS_SUFFIX)
    ]


def load_and_prepare(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load raw water-quality measurements and create a regular daily time series."""
    _ensure_dirs()
    raw = pd.read_csv(raw_path, parse_dates=["Timestamp"])
    value_columns = _numeric_measure_columns(raw.columns)

    for col in value_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    daily = (
        raw.set_index("Timestamp")[value_columns]
        .sort_index()
        .groupby(level=0)
        .mean()
        .resample("D")
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    daily.index.name = DATE_COL
    daily.to_csv(processed_path)
    return daily


def train_test_split(series: pd.Series, test_days: int = TEST_DAYS) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-test_days], series.iloc[-test_days:]


def mase(y_true: np.ndarray, y_pred: np.ndarray, train: pd.Series, season: int = 1) -> float:
    denom = np.mean(np.abs(train.to_numpy()[season:] - train.to_numpy()[:-season]))
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    score = np.where(denom == 0, 0, 2 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(score) * 100)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    train: pd.Series,
    family: str,
    model: str,
    notes: str,
) -> dict[str, float | str]:
    return {
        "family": family,
        "model": model,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "sMAPE_%": smape(y_true.to_numpy(), y_pred),
        "MASE": mase(y_true.to_numpy(), y_pred, train, season=SEASON_LENGTH),
        "notes": notes,
    }


def _safe_forecast(
    family: str,
    model: str,
    notes: str,
    fn: Callable[[], np.ndarray],
    horizon: int,
) -> ForecastResult:
    try:
        forecast = np.asarray(fn(), dtype=float)
        if len(forecast) != horizon:
            raise ValueError(f"Expected horizon {horizon}, got {len(forecast)}")
        return ForecastResult(family, model, forecast, notes)
    except Exception as exc:  # pragma: no cover - saved in report for reproducibility
        warnings.warn(f"{model} failed: {exc}")
        return ForecastResult(family, model, np.full(horizon, np.nan), f"FAILED: {exc}")


def statistical_forecasts(train: pd.Series, horizon: int) -> list[ForecastResult]:
    seasonal_mean = train.groupby(train.index.dayofweek).mean()
    day_codes = [(train.index[-1] + pd.Timedelta(days=i)).dayofweek for i in range(1, horizon + 1)]

    candidates = [
        ForecastResult(
            "baseline/statistical",
            "Naive",
            np.repeat(train.iloc[-1], horizon),
            "Последнее наблюдение переносится на весь горизонт.",
        ),
        ForecastResult(
            "baseline/statistical",
            "Seasonal naive, lag 7",
            np.resize(train.iloc[-SEASON_LENGTH:].to_numpy(), horizon),
            "Недельный сезонный бейзлайн.",
        ),
        ForecastResult(
            "statistical",
            "Day-of-week mean",
            np.asarray([seasonal_mean.loc[code] for code in day_codes]),
            "Средняя температура по дню недели на обучающей истории.",
        ),
    ]

    candidates.extend(
        [
            _safe_forecast(
                "statistical",
                "Moving average, window 7",
                "Сглаженный локальный уровень за последнюю неделю.",
                lambda: np.repeat(train.rolling(SEASON_LENGTH).mean().iloc[-1], horizon),
                horizon,
            ),
            _safe_forecast(
                "statistical",
                "Simple exponential smoothing",
                "Автоподбор параметра сглаживания через statsmodels.",
                lambda: SimpleExpSmoothing(train).fit(optimized=True).forecast(horizon),
                horizon,
            ),
            _safe_forecast(
                "statistical",
                "Holt linear trend",
                "Линейный тренд без сезонной компоненты.",
                lambda: Holt(train).fit(optimized=True).forecast(horizon),
                horizon,
            ),
            _safe_forecast(
                "statistical",
                "ETS additive trend + weekly seasonality",
                "ETS(A,A,A) с недельной сезонностью.",
                lambda: ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=SEASON_LENGTH,
                )
                .fit(optimized=True)
                .forecast(horizon),
                horizon,
            ),
            _safe_forecast(
                "statistical",
                "Theta",
                "ThetaModel с недельной сезонностью.",
                lambda: ThetaModel(train, period=SEASON_LENGTH).fit().forecast(horizon),
                horizon,
            ),
            _safe_forecast(
                "statistical",
                "SARIMAX (1,1,1)x(1,0,1,7)",
                "Ручная SARIMA-спецификация для тренда и недельной структуры.",
                lambda: SARIMAX(
                    train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, SEASON_LENGTH),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                .fit(disp=False)
                .forecast(horizon),
                horizon,
            ),
        ]
    )

    auto_sarima = _auto_sarimax(train, horizon)
    if auto_sarima is not None:
        candidates.append(auto_sarima)

    return candidates


def _auto_sarimax(train: pd.Series, horizon: int) -> ForecastResult | None:
    best_aic = np.inf
    best_order: tuple[int, int, int] | None = None
    best_seasonal: tuple[int, int, int, int] | None = None
    best_result = None

    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1]:
                for seasonal in [(0, 0, 0, 0), (1, 0, 0, SEASON_LENGTH), (1, 0, 1, SEASON_LENGTH)]:
                    try:
                        fitted = SARIMAX(
                            train,
                            order=(p, d, q),
                            seasonal_order=seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False, maxiter=100)
                    except Exception:
                        continue
                    if fitted.aic < best_aic:
                        best_aic = float(fitted.aic)
                        best_order = (p, d, q)
                        best_seasonal = seasonal
                        best_result = fitted

    if best_result is None or best_order is None or best_seasonal is None:
        return None

    return ForecastResult(
        "statistical",
        f"Auto SARIMAX {best_order}x{best_seasonal}",
        np.asarray(best_result.forecast(horizon), dtype=float),
        f"Мини-grid search по AIC, лучший AIC={best_aic:.2f}.",
    )


def _make_features(values: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": values})
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = df["y"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    day_of_year = values.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["dayofweek"] = values.index.dayofweek
    return df.dropna()


def _recursive_forecast(model, history: pd.Series, horizon: int) -> np.ndarray:
    history = history.copy()
    preds: list[float] = []
    lower = float(history.quantile(0.01) - 2.0)
    upper = float(history.quantile(0.99) + 2.0)

    for _ in range(horizon):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        extended = pd.concat([history, pd.Series([np.nan], index=[next_date])])
        row = _make_features(extended).iloc[[-1]].drop(columns=["y"])
        pred = float(np.clip(model.predict(row)[0], lower, upper))
        preds.append(pred)
        history.loc[next_date] = pred

    return np.asarray(preds)


def ml_forecasts(train: pd.Series, horizon: int) -> list[ForecastResult]:
    supervised = _make_features(train)
    x_train = supervised.drop(columns=["y"])
    y_train = supervised["y"]

    models = [
        (
            "ML",
            "Ridge regression with lag features",
            make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
            "Лаги, rolling-признаки и календарные признаки; L2-регуляризация.",
        ),
        (
            "ML",
            "Random forest with lag features",
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "Нелинейная модель на лаговых и календарных признаках.",
        ),
        (
            "ML",
            "ExtraTrees with lag features",
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "Ансамбль randomized trees для устойчивости на малой истории.",
        ),
        (
            "ML",
            "HistGradientBoosting with lag features",
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.04,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
            "Градиентный бустинг по лаговым признакам.",
        ),
        (
            "DL",
            "MLP shallow (32)",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32,),
                    alpha=0.01,
                    early_stopping=True,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
            "Однослойная нейросеть по лаговым признакам.",
        ),
        (
            "DL",
            "MLP deep (64,32)",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    alpha=0.005,
                    early_stopping=True,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
            "Двухслойная MLP как data-driven DL-бейзлайн.",
        ),
        (
            "DL",
            "MLP deep regularized (64,32,16)",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    alpha=0.05,
                    early_stopping=True,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
            "Более глубокая MLP с усиленной L2-регуляризацией.",
        ),
    ]

    results: list[ForecastResult] = []
    for family, name, model, notes in models:
        try:
            model.fit(x_train, y_train)
            forecast = _recursive_forecast(model, train, horizon)
            results.append(ForecastResult(family, name, forecast, notes))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"{name} failed: {exc}")
            results.append(ForecastResult(family, name, np.full(horizon, np.nan), f"FAILED: {exc}"))

    return results


def detect_anomalies(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = daily[TARGET].copy()
    stl = STL(y, period=SEASON_LENGTH, robust=True).fit()
    resid = pd.Series(stl.resid, index=y.index)
    mad = stats.median_abs_deviation(resid.dropna(), scale="normal")
    robust_z = np.abs((resid - resid.median()) / mad) if mad else pd.Series(0, index=y.index)

    rolling_median = y.rolling(14, center=True, min_periods=7).median()
    q1 = y.rolling(14, center=True, min_periods=7).quantile(0.25)
    q3 = y.rolling(14, center=True, min_periods=7).quantile(0.75)
    iqr = q3 - q1
    iqr_anomaly = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)

    features = _make_features(y).drop(columns=["y"])
    iso = IsolationForest(contamination=0.04, random_state=RANDOM_STATE)
    iso_flags = pd.Series(False, index=y.index)
    iso_flags.loc[features.index] = iso.fit_predict(features) == -1

    details = pd.DataFrame(
        {
            DATE_COL: y.index,
            TARGET: y.to_numpy(),
            "stl_residual": resid.to_numpy(),
            "STL robust z-score": (robust_z > 3.5).to_numpy(),
            "Rolling IQR": iqr_anomaly.fillna(False).to_numpy(),
            "IsolationForest": iso_flags.to_numpy(),
            "rolling_median_14": rolling_median.to_numpy(),
        }
    )
    anomaly_cols = ["STL robust z-score", "Rolling IQR", "IsolationForest"]
    details["anomaly_votes"] = details[anomaly_cols].sum(axis=1)
    details["is_anomaly_consensus"] = details["anomaly_votes"] >= 2

    summary = pd.DataFrame(
        {
            "method": anomaly_cols + ["is_anomaly_consensus"],
            "count": [int(details[col].sum()) for col in anomaly_cols + ["is_anomaly_consensus"]],
        }
    )
    details.to_csv(ANOMALY_DETAILS_PATH, index=False)
    summary.to_csv(ANOMALIES_PATH, index=False)
    return summary, details


def diagnostics(series: pd.Series, best_errors: pd.Series) -> dict[str, object]:
    adf_original = adfuller(series.dropna(), autolag="AIC")
    adf_diff = adfuller(series.diff().dropna(), autolag="AIC")
    ljung = acorr_ljungbox(best_errors.dropna(), lags=[7, 14], return_df=True)
    return {
        "adf_original": {
            "statistic": float(adf_original[0]),
            "p_value": float(adf_original[1]),
            "used_lag": int(adf_original[2]),
            "nobs": int(adf_original[3]),
        },
        "adf_first_difference": {
            "statistic": float(adf_diff[0]),
            "p_value": float(adf_diff[1]),
            "used_lag": int(adf_diff[2]),
            "nobs": int(adf_diff[3]),
        },
        "best_model_ljung_box": {
            str(int(lag)): {
                "lb_stat": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
            }
            for lag, row in ljung.iterrows()
        },
    }


def rolling_origin_backtest(
    series: pd.Series,
    selected_models: list[str],
    horizon: int = BACKTESTING_HORIZON,
    windows: int = BACKTESTING_WINDOWS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    first_cutoff = len(series) - TEST_DAYS
    cutoffs = [first_cutoff + i * horizon for i in range(windows)]
    cutoffs = [cutoff for cutoff in cutoffs if cutoff + horizon <= len(series)]

    for fold, cutoff in enumerate(cutoffs, start=1):
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff : cutoff + horizon]
        results = statistical_forecasts(train, horizon) + ml_forecasts(train, horizon)

        for result in results:
            if result.model not in selected_models or np.isnan(result.forecast).all():
                continue
            metric_row = compute_metrics(test, result.forecast, train, result.family, result.model, result.notes)
            metric_row["fold"] = fold
            metric_row["train_end"] = str(train.index[-1].date())
            metric_row["test_start"] = str(test.index[0].date())
            metric_row["test_end"] = str(test.index[-1].date())
            rows.append(metric_row)

    summary = (
        pd.DataFrame(rows)
        .groupby(["family", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            sMAPE_mean=("sMAPE_%", "mean"),
            MASE_mean=("MASE", "mean"),
        )
        .sort_values(["MASE_mean", "RMSE_mean"])
        .reset_index(drop=True)
    )
    summary.to_csv(BACKTESTING_PATH, index=False)
    return summary


def prediction_intervals(
    forecasts: pd.DataFrame,
    best_model_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    best = forecasts[forecasts["model"] == best_model_name].copy()
    absolute_errors = best["error"].abs()
    q80 = float(absolute_errors.quantile(0.80))
    q95 = float(absolute_errors.quantile(0.95))

    best["lower_80"] = best["forecast"] - q80
    best["upper_80"] = best["forecast"] + q80
    best["lower_95"] = best["forecast"] - q95
    best["upper_95"] = best["forecast"] + q95
    best["covered_80"] = best["actual"].between(best["lower_80"], best["upper_80"])
    best["covered_95"] = best["actual"].between(best["lower_95"], best["upper_95"])
    best.to_csv(PREDICTION_INTERVALS_PATH, index=False)

    report = {
        "method": "Empirical conformal-style intervals from holdout absolute errors",
        "model": best_model_name,
        "absolute_error_q80": q80,
        "absolute_error_q95": q95,
        "coverage_80": float(best["covered_80"].mean()),
        "coverage_95": float(best["covered_95"].mean()),
        "mean_interval_width_80": float((best["upper_80"] - best["lower_80"]).mean()),
        "mean_interval_width_95": float((best["upper_95"] - best["lower_95"]).mean()),
    }
    with RELIABILITY_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return best, report


def plot_report_figures(
    daily: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    forecasts: pd.DataFrame,
    anomaly_details: pd.DataFrame,
    backtesting: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    _ensure_dirs()
    y = daily[TARGET]

    plt.figure(figsize=(12, 4))
    plt.plot(y.index, y, label=TARGET)
    plt.title("Daily Brisbane water temperature")
    plt.ylabel("Temperature")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature_series.png", dpi=160)
    plt.close()

    stl = STL(y, period=SEASON_LENGTH, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stl_decomposition.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y, ax=axes[0], lags=45)
    plot_pacf(y, ax=axes[1], lags=45, method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "acf_pacf.png", dpi=160)
    plt.close(fig)

    top_models = forecasts.attrs.get("top_models", [])
    plt.figure(figsize=(12, 5))
    plt.plot(train.index[-90:], train.iloc[-90:], label="train", color="tab:blue")
    plt.plot(test.index, test, label="test", color="black", linewidth=2)
    for model in top_models:
        model_frame = forecasts[forecasts["model"] == model]
        plt.plot(model_frame[DATE_COL], model_frame["forecast"], label=model, alpha=0.85)
    plt.title("Forecast comparison on 60-day holdout")
    plt.ylabel("Temperature")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast_comparison.png", dpi=160)
    plt.close()

    if not backtesting.empty:
        plt.figure(figsize=(12, 4))
        ordered = backtesting.sort_values("MASE_mean")
        plt.barh(ordered["model"], ordered["MASE_mean"], xerr=ordered["RMSE_std"].fillna(0) * 0)
        plt.gca().invert_yaxis()
        plt.title("Rolling-origin backtesting: mean MASE")
        plt.xlabel("MASE")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "backtesting_mase.png", dpi=160)
        plt.close()

    if not intervals.empty:
        interval_dates = pd.to_datetime(intervals[DATE_COL])
        plt.figure(figsize=(12, 5))
        plt.plot(interval_dates, intervals["actual"], label="actual", color="black", linewidth=2)
        plt.plot(interval_dates, intervals["forecast"], label="forecast", color="tab:blue")
        plt.fill_between(
            interval_dates,
            intervals["lower_95"].astype(float),
            intervals["upper_95"].astype(float),
            alpha=0.20,
            label="95% empirical interval",
        )
        plt.fill_between(
            interval_dates,
            intervals["lower_80"].astype(float),
            intervals["upper_80"].astype(float),
            alpha=0.30,
            label="80% empirical interval",
        )
        plt.title("Best-model forecast with empirical prediction intervals")
        plt.ylabel("Temperature")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "prediction_intervals.png", dpi=160)
        plt.close()

    anomaly_dates = pd.to_datetime(
        anomaly_details.loc[anomaly_details["is_anomaly_consensus"], DATE_COL]
    )
    anomaly_values = y.loc[anomaly_dates]
    plt.figure(figsize=(12, 4))
    plt.plot(y.index, y, label=TARGET)
    plt.scatter(anomaly_values.index, anomaly_values, color="red", label="consensus anomaly", zorder=3)
    plt.title("Consensus anomalies in daily water temperature")
    plt.ylabel("Temperature")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anomalies.png", dpi=160)
    plt.close()


def run_pipeline() -> dict[str, object]:
    started = time.perf_counter()
    _ensure_dirs()
    daily = load_and_prepare()
    series = daily[TARGET].asfreq("D")
    train, test = train_test_split(series)
    horizon = len(test)

    forecast_results = statistical_forecasts(train, horizon) + ml_forecasts(train, horizon)
    valid_results = [result for result in forecast_results if not np.isnan(result.forecast).all()]

    metrics = [
        compute_metrics(test, result.forecast, train, result.family, result.model, result.notes)
        for result in valid_results
    ]
    metrics_df = pd.DataFrame(metrics).sort_values(["MASE", "RMSE"]).reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    forecast_frames = []
    for result in valid_results:
        forecast_frames.append(
            pd.DataFrame(
                {
                    DATE_COL: test.index,
                    "family": result.family,
                    "model": result.model,
                    "forecast": result.forecast,
                    "actual": test.to_numpy(),
                    "error": test.to_numpy() - result.forecast,
                }
            )
        )
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    top_models = metrics_df.head(5)["model"].tolist()
    forecasts.attrs["top_models"] = top_models
    forecasts.to_csv(FORECASTS_PATH, index=False)

    anomaly_summary, anomaly_details = detect_anomalies(daily)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_errors = forecasts.loc[forecasts["model"] == best_model_name].set_index(DATE_COL)["error"]
    diagnostic_report = diagnostics(series, best_errors)
    with DIAGNOSTICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnostic_report, f, ensure_ascii=False, indent=2)

    selected_backtest_models = [
        "Naive",
        "Seasonal naive, lag 7",
        "Holt linear trend",
        "SARIMAX (1,1,1)x(1,0,1,7)",
        "Ridge regression with lag features",
        "HistGradientBoosting with lag features",
        "Random forest with lag features",
        "MLP shallow (32)",
    ]
    backtesting = rolling_origin_backtest(series, selected_backtest_models)
    intervals, reliability_report = prediction_intervals(forecasts, best_model_name)

    plot_report_figures(daily, train, test, forecasts, anomaly_details, backtesting, intervals)

    elapsed = time.perf_counter() - started
    summary = {
        "raw_rows": int(pd.read_csv(RAW_PATH, usecols=["Timestamp"]).shape[0]),
        "start_date": str(series.index.min().date()),
        "end_date": str(series.index.max().date()),
        "observations_daily": int(series.shape[0]),
        "target": TARGET,
        "frequency": "D",
        "test_days": TEST_DAYS,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "model_count": int(metrics_df.shape[0]),
        "best_model": metrics_df.iloc[0].to_dict(),
        "top_5_models": metrics_df.head(5).to_dict(orient="records"),
        "consensus_anomalies": int(anomaly_details["is_anomaly_consensus"].sum()),
        "anomaly_dates": anomaly_details.loc[
            anomaly_details["is_anomaly_consensus"], DATE_COL
        ].astype(str).tolist(),
        "diagnostics": diagnostic_report,
        "backtesting": {
            "windows": int(backtesting["folds"].max()) if not backtesting.empty else 0,
            "best_model": backtesting.iloc[0].to_dict() if not backtesting.empty else {},
        },
        "reliability": reliability_report,
        "pipeline_runtime_seconds": round(elapsed, 3),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
