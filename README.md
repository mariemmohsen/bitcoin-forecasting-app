# ₿ Bitcoin Price Forecasting Portal

An interactive Streamlit application for financial time-series analysis and
Bitcoin price forecasting, built as part of the AI Engineering practicum.

---

## Dataset

**Kaggle BTC Dataset used for testing:**
[Bitcoin Historical Data 2014–2024](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

Alternative dataset (broader date range):
[Bitcoin Historical Data — Investing.com style](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

Download either dataset, save as a `.csv` file, and upload it directly in the
sidebar. The app auto-detects the date column and all standard Kaggle price
column names (`Close`, `Open`, `High`, `Low`, `Adj Close`).

---

## Project structure

```
btc_forecasting/
├── app.py             # Streamlit UI entry point — run this file
├── data_utils.py      # CSV loading, validation, cleaning, indicators, metrics
├── prophet_model.py   # Prophet forecasting with grid-search tuning
├── rf_model.py        # Random Forest forecasting
├── arima_model.py     # Auto-ARIMA forecasting (pmdarima)
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

> **Note on Prophet installation:** Prophet depends on `cmdstanpy`, which
> compiles a C++ binary on first use. If you see a compilation message during
> the first run, let it complete — it only happens once.

---

## Usage

1. **Upload** a BTC historical CSV (must contain a `Date` or `Timestamp`
   column and at least one numeric price column).
2. **Select** the price column, forecasting model, horizon (7–180 days), and
   confidence interval (80 / 90 / 95 / 99%).
3. Toggle **technical indicators** (SMA-20, SMA-50, EMA-20) on/off in the
   sidebar.
4. Click **🚀 Generate Forecast** to train the model and view results.

The app will show:
- A backtest performance table (MAE and RMSE in USD)
- A forecast chart with confidence interval band
- A backtest chart comparing predicted vs actual prices
- A downloadable forecast data table

---

## Models

| Model | Type | Notes |
|---|---|---|
| **Prophet Enhanced** | Classical + regressors | Log-price with returns, rolling stats, momentum; 72-combination grid search |
| **Random Forest** | Machine Learning | Log-price with lag/rolling/calendar features; Gaussian CI from residual std |
| **ARIMA (Auto)** | Classical | Auto-order selection via pmdarima stepwise search on log-price |
| **Auto Select** | — | Runs all three, picks lowest backtest RMSE |

---

## How each model handles crypto-market volatility

Cryptocurrency prices are characterised by heavy tails, sudden regime shifts,
and volatility clustering — properties that challenge standard forecasting
assumptions. Here is how each model in this app addresses those challenges.

### Prophet Enhanced

Prophet was originally designed for business time-series with strong seasonal
patterns and trend changes. By default it assumes a smooth, slowly-changing
trend — which would badly underfit BTC's sudden price spikes.

This implementation adds three layers to manage crypto volatility:

1. **Log-price transformation.** Taking `log(price)` before fitting compresses
   extreme values, stabilises variance across time, and turns multiplicative
   shocks (e.g. "price doubled") into additive ones that Prophet can model
   linearly. Forecasts are exponentiated back to USD at the end.

2. **Extra regressors.** Six engineered features are added as external
   regressors: 1-day log return, 7-day and 14-day rolling mean and standard
   deviation of log-price, and a 7-day momentum term. These give Prophet
   direct access to recent volatility signals rather than relying solely on
   the trend component.

3. **Grid-searched changepoints.** The `changepoint_prior_scale` parameter
   controls how flexible the trend is. A larger value allows faster trend
   reversals; a smaller value avoids overfitting to noise. This app searches
   across four values (0.05, 0.1, 0.3, 0.5) and selects the combination with
   the lowest backtest RMSE, adapting the model to the specific dataset's
   volatility profile.

### Random Forest

Random Forest is a non-parametric ensemble method that makes no assumptions
about the price distribution — it simply learns a mapping from feature vectors
to the next price. This makes it naturally robust to heavy tails and non-linear
price dynamics.

Key design choices for crypto volatility:

1. **Log-price target.** Same motivation as Prophet: log-space reduces the
   influence of extreme values and makes residuals more homoscedastic.

2. **Lag and rolling features.** Lags at 1, 2, 3, 7, 14, and 30 days capture
   short- and medium-term momentum. Rolling 7-day and 14-day standard deviation
   features give the model explicit volatility context — when recent volatility
   is high, the model can learn to widen its predictions accordingly.

3. **Gaussian confidence intervals.** Because Random Forest provides point
   predictions only, confidence intervals are approximated from the standard
   deviation of backtest residuals. This is an empirical CI: if past prediction
   errors were ±$2 000, the interval for future predictions reflects that
   same magnitude of uncertainty.

4. **Recursive forecasting.** For multi-step horizons, each predicted value is
   fed back as a lag feature for the next step. This means uncertainty compounds
   over the horizon — a known limitation for volatile assets, and the reason
   shorter horizons produce tighter, more reliable intervals.

### ARIMA (Auto)

ARIMA is a classical linear time-series model. It captures autocorrelation
structure (the AR terms) and moving-average effects in the residuals (the MA
terms). For BTC it is the most theoretically conservative of the three models.

Volatility-handling choices:

1. **Log-price transformation.** ARIMA assumes variance-stationarity; raw BTC
   prices are strongly heteroscedastic. Fitting on log-price substantially
   improves stationarity and prevents large price levels from dominating the
   error terms.

2. **Automatic order selection.** `pmdarima.auto_arima` performs a stepwise
   AIC-minimising search over AR, MA, and differencing orders. This means the
   model adapts to the autocorrelation structure of the specific dataset rather
   than using a hand-picked order that may be stale.

3. **True confidence intervals.** Unlike the other two models, ARIMA's
   confidence intervals are analytically derived from the model's parameter
   covariance matrix. They widen correctly as the horizon extends — a property
   that reflects genuine uncertainty growth in a volatile market.

**Known limitation:** ARIMA is a linear model and cannot capture
volatility clustering (the tendency for large moves to follow large moves).
For this reason it often under-performs Prophet and Random Forest on BTC data
over longer horizons. The Auto Select option will typically not choose it for
horizons beyond ~14 days.

---

## CSV format

The app accepts any CSV with:
- A date column named `Date`, `Timestamp`, or `Time`
- A numeric price column (`Close`, `Open`, `High`, `Low`, `Adj Close`, or any
  other numeric column)

Dollar signs and commas in price values are stripped automatically.
Missing trading days are forward-filled.

**Minimum 120 rows** (≈ 4 months of daily data) required for reliable results.

---

## Error handling

The app validates uploaded files before any model training and shows a clear
error message for the following cases:

- Empty or non-CSV file
- Missing date column
- No parseable numeric price column
- Price column with more than 50% non-numeric values
- Non-positive price values (required for log-transform)
- Constant price column (no variation to model)
- Dataset shorter than 120 rows

---

## Module overview

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, sidebar, charts, model dispatcher, error display |
| `data_utils.py` | `validate_csv`, `detect_price_column`, `load_and_clean`, `compute_indicators`, `calc_mae_rmse` |
| `prophet_model.py` | `prepare_prophet_features`, `tune_prophet` (cached grid search), `prophet_forecast` |
| `rf_model.py` | `build_ml_features`, `ml_forecast` |
| `arima_model.py` | `arima_forecast` (Auto-ARIMA via pmdarima) |

---

## Submission

- **Tech stack:** Streamlit · Plotly · Prophet · ARIMA (pmdarima) · scikit-learn · pandas · numpy
- **Deadline:** 24 April 2026
- **Format:** GitHub repository or `.zip` containing all `.py` files, `requirements.txt`, and this `README.md`
