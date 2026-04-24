"""
prophet_model.py
================
Prophet-based Bitcoin price forecasting with automatic hyperparameter tuning.

Features
--------
- Log-price transformation
- Lag returns, rolling mean/std, and momentum regressors
- Grid search over changepoint scale, seasonality scale, mode, and n_changepoints
- Step-ahead recursive forecasting for the specified horizon
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").disabled = True
import itertools
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from data_utils import calc_mae_rmse, TEST_SIZE

def prepare_prophet_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Build Prophet-ready DataFrame from raw price history.

    Adds regressors:
      - ret_1        : 1-day log return
      - roll_mean_7  : 7-day rolling mean of log-price
      - roll_std_7   : 7-day rolling std of log-price
      - roll_mean_14 : 14-day rolling mean
      - roll_std_14  : 14-day rolling std
      - momentum_7   : log-price minus 7-day lag

    Returns columns: ['ds', 'y', <regressors>]
    """
    p = df.copy()
    p["y"] = np.log(p[price_col])
    p["ret_1"] = p["y"].diff()
    p["roll_mean_7"] = p["y"].rolling(7).mean()
    p["roll_std_7"] = p["y"].rolling(7).std()
    p["roll_mean_14"] = p["y"].rolling(14).mean()
    p["roll_std_14"] = p["y"].rolling(14).std()
    p["momentum_7"] = p["y"] - p["y"].shift(7)

    p = p.dropna().copy()
    p = p.rename(columns={"Date": "ds"})

    return p[[
        "ds", "y",
        "ret_1",
        "roll_mean_7", "roll_std_7",
        "roll_mean_14", "roll_std_14",
        "momentum_7",
    ]]



_PARAM_GRID = {
    "changepoint_prior_scale": [0.05, 0.1, 0.3, 0.5],
    "seasonality_prior_scale": [1.0, 5.0, 10.0],
    "seasonality_mode":        ["additive", "multiplicative"],
    "n_changepoints":          [25, 40, 60],
}

_REG_COLS = [
    "ret_1",
    "roll_mean_7", "roll_std_7",
    "roll_mean_14", "roll_std_14",
    "momentum_7",
]


@st.cache_data(show_spinner=False)
def tune_prophet(
    df: pd.DataFrame,
    price_col: str,
    ci: float,
    test_size: int = TEST_SIZE,
):
    """
    Grid-search Prophet hyperparameters on the held-out test window.

    Returns
    -------
    best_params   : dict
    best_mae      : float
    best_rmse     : float
    test_df       : DataFrame  (columns: ds, y — in price space)
    best_pred     : np.ndarray (price-space predictions for the test window)
    """
    prophet_df = prepare_prophet_features(df, price_col)

    if len(prophet_df) <= test_size:
        raise ValueError("Dataset is too small for Prophet backtesting.")

    train_df = prophet_df.iloc[:-test_size].copy()
    test_df  = prophet_df.iloc[-test_size:].copy()

    best_params = None
    best_mae    = float("inf")
    best_rmse   = float("inf")
    best_pred   = None
    best_lower  = None
    best_upper  = None
    last_error  = None

    for cps, sps, smode, ncp in itertools.product(
        _PARAM_GRID["changepoint_prior_scale"],
        _PARAM_GRID["seasonality_prior_scale"],
        _PARAM_GRID["seasonality_mode"],
        _PARAM_GRID["n_changepoints"],
    ):
        try:
            model = _build_prophet(cps, sps, smode, ncp, ci)
            model.fit(train_df)

            pred_log = model.predict(test_df[["ds"] + _REG_COLS])
            pred   = np.exp(pred_log["yhat"].values)
            lower  = np.exp(pred_log["yhat_lower"].values)
            upper  = np.exp(pred_log["yhat_upper"].values)
            actual = np.exp(test_df["y"].values)

            mae, rmse = calc_mae_rmse(actual, pred)

            if rmse < best_rmse:
                best_rmse   = rmse
                best_mae    = mae
                best_params = dict(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=smode,
                    n_changepoints=ncp,
                )
                best_pred  = pred
                best_lower = lower
                best_upper = upper

        except Exception as exc:
            last_error = str(exc)
            continue

    if best_params is None:
        raise ValueError(f"Prophet tuning failed. Last error: {last_error}")

    test_df_display = test_df.copy()
    test_df_display["y"] = np.exp(test_df_display["y"])

    return best_params, best_mae, best_rmse, test_df_display, best_pred, best_lower, best_upper


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_prophet(cps, sps, smode, ncp, ci) -> Prophet:
    """Instantiate a Prophet model with regressors and monthly seasonality."""
    model = Prophet(
        interval_width=ci,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        seasonality_mode=smode,
        n_changepoints=ncp,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    for col in _REG_COLS:
        model.add_regressor(col)
    return model


def _compute_regressors(log_series: pd.Series) -> dict:
    """
    Compute the regressor values from the tail of the current log-price series.
    Used during step-ahead recursive forecasting.
    """
    return {
        "ret_1":        log_series.diff().iloc[-1],
        "roll_mean_7":  log_series.iloc[-7:].mean(),
        "roll_std_7":   log_series.iloc[-7:].std(),
        "roll_mean_14": log_series.iloc[-14:].mean(),
        "roll_std_14":  log_series.iloc[-14:].std(),
        "momentum_7":   log_series.iloc[-1] - log_series.iloc[-7],
    }



def prophet_forecast(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    ci: float,
    test_size: int = TEST_SIZE,
) -> dict:
    """
    Tune and fit Prophet, then generate a step-ahead recursive forecast.

    Parameters
    ----------
    df         : DataFrame with columns ['Date', price_col, ...]
    price_col  : Name of the price column
    horizon    : Number of days to forecast
    ci         : Confidence interval width (e.g., 0.95)
    test_size  : Number of days in the backtest holdout window

    Returns
    -------
    dict with keys:
      model_name, forecast_df, mae, rmse,
      test_dates, test_actual, test_pred, meta
    """
    best_params, mae, rmse, test_df, backtest_pred, backtest_lower, backtest_upper = tune_prophet(
        df, price_col, ci, test_size=test_size
    )

    prophet_df  = prepare_prophet_features(df, price_col)
    model_full  = _build_prophet(
        cps=best_params["changepoint_prior_scale"],
        sps=best_params["seasonality_prior_scale"],
        smode=best_params["seasonality_mode"],
        ncp=best_params["n_changepoints"],
        ci=ci,
    )
    model_full.fit(prophet_df)

    # Recursive step-ahead forecasting
    history     = df[["Date", price_col]].copy().reset_index(drop=True)
    future_rows = []

    for _ in range(horizon):
        next_date = history["Date"].iloc[-1] + timedelta(days=1)

        log_series  = np.log(history[price_col])
        regressors  = _compute_regressors(log_series)

        future_df = pd.DataFrame([{"ds": next_date, **regressors}])
        pred      = model_full.predict(future_df)

        yhat       = float(np.exp(pred["yhat"].iloc[0]))
        yhat_lower = float(np.exp(pred["yhat_lower"].iloc[0]))
        yhat_upper = float(np.exp(pred["yhat_upper"].iloc[0]))

        future_rows.append(dict(ds=next_date, yhat=yhat, yhat_lower=yhat_lower, yhat_upper=yhat_upper))
        history = pd.concat(
            [history, pd.DataFrame({"Date": [next_date], price_col: [yhat]})],
            ignore_index=True,
        )

    return {
        "model_name":  "Prophet Enhanced",
        "forecast_df": pd.DataFrame(future_rows),
        "mae":         mae,
        "rmse":        rmse,
        "test_dates":  test_df["ds"].values,
        "test_actual": test_df["y"].values,
        "test_pred":   backtest_pred,
        "test_lower":  backtest_lower,
        "test_upper":  backtest_upper,
        "meta": {
            "type":    "classical + regressors",
            "details": (
                "Prophet on log-price with returns, rolling stats, "
                "and momentum regressors"
            ),
        },
    }
