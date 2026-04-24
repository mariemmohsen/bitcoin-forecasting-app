
import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
import numpy as np
import pandas as pd
from pmdarima import auto_arima

from data_utils import calc_mae_rmse, TEST_SIZE


def arima_forecast(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    ci: float,
    test_size: int = TEST_SIZE,
) -> dict:
    """
    Fit an Auto-ARIMA model on log-price and generate a forecast.

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
    series = df.set_index("Date")[price_col].asfreq("D").ffill()

    if len(series) <= test_size:
        raise ValueError("Dataset is too small for ARIMA backtesting.")

    log_series = np.log(series)
    train = log_series.iloc[:-test_size]
    test = log_series.iloc[-test_size:]

    model = auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    alpha_bt = 1.0 - ci
    pred_log, conf_bt_log = model.predict(
        n_periods=test_size,
        return_conf_int=True,
        alpha=alpha_bt,
    )
    pred   = np.exp(pred_log)
    lower_bt = np.exp(conf_bt_log[:, 0])
    upper_bt = np.exp(conf_bt_log[:, 1])
    actual = np.exp(test.values)

    mae, rmse = calc_mae_rmse(actual, pred)

    model.fit(log_series)

    alpha = 1.0 - ci
    future_pred_log, conf_int_log = model.predict(
        n_periods=horizon,
        return_conf_int=True,
        alpha=alpha,
    )

    future_pred = np.exp(future_pred_log)
    lower       = np.exp(conf_int_log[:, 0])
    upper       = np.exp(conf_int_log[:, 1])

    last_date  = series.index[-1]
    future_idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")

    fc_df = pd.DataFrame({
        "ds":         future_idx,
        "yhat":       future_pred,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

    return {
        "model_name":  "ARIMA (Auto)",
        "forecast_df": fc_df,
        "mae":         mae,
        "rmse":        rmse,
        "test_dates":  test.index,
        "test_actual": actual,
        "test_pred":   pred,
        "test_lower":  lower_bt,
        "test_upper":  upper_bt,
        "meta": {
            "type":    "classical",
            "details": f"Auto-ARIMA on log-price, selected order = {model.order}",
        },
    }
