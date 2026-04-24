import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data_utils import calc_mae_rmse, TEST_SIZE
_FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "roll_mean_7", "roll_std_7",
    "roll_mean_14", "roll_std_14",
    "dayofweek", "month", "dayofmonth",
]

_Z_MAP = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}

_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

def build_ml_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Build the supervised-learning feature matrix from raw price history.

    Features
    --------
    Lag features  : lag_1, lag_2, lag_3, lag_7, lag_14, lag_30
    Rolling stats : roll_mean_7, roll_std_7, roll_mean_14, roll_std_14
    Calendar      : dayofweek, month, dayofmonth

    Target        : log(price)

    Rows with NaN (due to lags/rolling windows) are dropped.
    """
    ml = df.copy()
    ml["target"] = np.log(ml[price_col])

    ml["lag_1"]  = ml["target"].shift(1)
    ml["lag_2"]  = ml["target"].shift(2)
    ml["lag_3"]  = ml["target"].shift(3)
    ml["lag_7"]  = ml["target"].shift(7)
    ml["lag_14"] = ml["target"].shift(14)
    ml["lag_30"] = ml["target"].shift(30)

    ml["roll_mean_7"]  = ml["target"].rolling(7).mean()
    ml["roll_std_7"]   = ml["target"].rolling(7).std()
    ml["roll_mean_14"] = ml["target"].rolling(14).mean()
    ml["roll_std_14"]  = ml["target"].rolling(14).std()

    ml["dayofweek"]  = ml["Date"].dt.dayofweek
    ml["month"]      = ml["Date"].dt.month
    ml["dayofmonth"] = ml["Date"].dt.day

    return ml.dropna().reset_index(drop=True)


def _build_step_features(history: pd.DataFrame, price_col: str, next_date: pd.Timestamp) -> dict:
    """
    Compute feature values for a single future step given the running history.
    """
    log_s = np.log(history[price_col])
    return {
        "lag_1":       log_s.iloc[-1],
        "lag_2":       log_s.iloc[-2],
        "lag_3":       log_s.iloc[-3],
        "lag_7":       log_s.iloc[-7],
        "lag_14":      log_s.iloc[-14],
        "lag_30":      log_s.iloc[-30],
        "roll_mean_7":  log_s.iloc[-7:].mean(),
        "roll_std_7":   log_s.iloc[-7:].std(),
        "roll_mean_14": log_s.iloc[-14:].mean(),
        "roll_std_14":  log_s.iloc[-14:].std(),
        "dayofweek":   next_date.dayofweek,
        "month":       next_date.month,
        "dayofmonth":  next_date.day,
    }

def ml_forecast(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    ci: float,
    test_size: int = TEST_SIZE,
) -> dict:
    """
    Train a Random Forest on log-price features and generate a forecast.

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
    ml_df = build_ml_features(df, price_col)

    if len(ml_df) <= test_size:
        raise ValueError("Dataset is too small for Random Forest backtesting.")

    train_df = ml_df.iloc[:-test_size].copy()
    test_df  = ml_df.iloc[-test_size:].copy()

    X_train, y_train = train_df[_FEATURE_COLS], train_df["target"]
    X_test,  y_test  = test_df[_FEATURE_COLS],  test_df["target"]

    model_bt = RandomForestRegressor(**_RF_PARAMS)
    model_bt.fit(X_train, y_train)

    backtest_pred_log = model_bt.predict(X_test)
    backtest_pred     = np.exp(backtest_pred_log)
    actual_vals       = np.exp(y_test.values)

    mae, rmse    = calc_mae_rmse(actual_vals, backtest_pred)
    residual_std = np.std(actual_vals - backtest_pred)
    z_bt         = _Z_MAP.get(round(ci, 2), 1.96)
    bt_lower     = np.maximum(0.0, backtest_pred - z_bt * residual_std)
    bt_upper     = backtest_pred + z_bt * residual_std

    model_full = RandomForestRegressor(**_RF_PARAMS)
    model_full.fit(ml_df[_FEATURE_COLS], ml_df["target"])

    z       = _Z_MAP.get(round(ci, 2), 1.96)
    history = df[["Date", price_col]].copy().reset_index(drop=True)
    future_rows: list[dict] = []

    for _ in range(horizon):
        next_date = history["Date"].iloc[-1] + timedelta(days=1)

        feat   = _build_step_features(history, price_col, next_date)
        X_next = pd.DataFrame([feat])[_FEATURE_COLS]

        yhat_log = float(model_full.predict(X_next)[0])
        yhat     = float(np.exp(yhat_log))

        future_rows.append({
            "ds":         next_date,
            "yhat":       yhat,
            "yhat_lower": max(0.0, yhat - z * residual_std),
            "yhat_upper": yhat + z * residual_std,
        })

        history = pd.concat(
            [history, pd.DataFrame({"Date": [next_date], price_col: [yhat]})],
            ignore_index=True,
        )

    return {
        "model_name":  "Random Forest",
        "forecast_df": pd.DataFrame(future_rows),
        "mae":         mae,
        "rmse":        rmse,
        "test_dates":  test_df["Date"].values,
        "test_actual": actual_vals,
        "test_pred":   backtest_pred,
        "test_lower":  bt_lower,
        "test_upper":  bt_upper,
        "meta": {
            "type":    "machine learning",
            "details": (
                "RandomForestRegressor on log-price with "
                "lag/rolling/calendar features"
            ),
        },
    }
