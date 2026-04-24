# Bitcoin Price Forecasting Portal

An interactive Streamlit application for financial time-series analysis and Bitcoin price forecasting.

---

## Dataset

Kaggle BTC Dataset used for testing:

[Bitcoin Historical Data 2014–2025 Yahoo Finance](https://www.kaggle.com/datasets/eldintarofarrandi/bitcoin-historical-data-2014-2025-yahoo-finance)

Download the dataset as a CSV file and upload it directly in the sidebar.

The app auto-detects the date column and standard price column names such as:

- Close
- Open
- High
- Low
- Adj Close

---

## Project structure

```text
btc_forecasting/
├── app.py
├── data_utils.py
├── prophet_model.py
├── rf_model.py
├── arima_model.py
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Usage

1. Upload a Bitcoin historical CSV file.
2. Select the price column.
3. Select the forecasting model.
4. Choose the forecast horizon.
5. Choose the confidence interval.
6. Toggle technical indicators on or off.
7. Click Generate Forecast.

The app displays:

- Historical Bitcoin price chart
- Technical indicators
- Model performance metrics
- Forecast chart
- Backtest chart
- Forecast data table

---

## Models

| Model | Type | Description |
|---|---|---|
| Prophet Enhanced | Classical model with regressors | Uses log-price, returns, rolling statistics, momentum, and tuned changepoints |
| Random Forest | Machine Learning | Uses lag features, rolling features, and calendar features |
| ARIMA Auto | Classical time-series model | Uses automatic ARIMA order selection on log-price |
| Auto Select Best Model | Model comparison | Runs available models and selects the one with the lowest backtest RMSE |

---

## How each model handles crypto-market volatility

Bitcoin prices can be highly volatile, with sudden trend changes and large daily movements. The app uses several techniques to make forecasting more stable.

### Prophet Enhanced

Prophet is used with a log-price transformation to reduce the effect of extreme price values. Additional regressors are added, including returns, rolling mean, rolling standard deviation, and momentum. These features help the model understand recent movement and volatility.

The model also tunes changepoint parameters to better adapt to trend changes in the Bitcoin market.

### Random Forest

Random Forest uses historical lag values and rolling statistics to learn patterns from previous Bitcoin price behavior. It does not assume a linear relationship, which makes it useful for capturing non-linear patterns.

The model forecasts future values recursively, meaning each predicted value is used to help predict the next day.

### ARIMA Auto

ARIMA is a classical time-series forecasting model. The app applies ARIMA to log-transformed Bitcoin prices to improve stability. Automatic order selection is used to choose suitable ARIMA parameters based on the dataset.

ARIMA confidence intervals are generated from the statistical model and usually widen as the forecast horizon increases.

---

## CSV format

The app accepts a CSV file with:

- A date column named Date, Timestamp, or Time
- A numeric price column such as Close, Open, High, Low, or Adj Close

The app automatically removes commas and dollar signs from price values.

Missing days are forward-filled to create a continuous daily time series.

Minimum required data size: 120 rows.

---

## Error handling

The app checks the uploaded file before training any model.

It shows clear error messages for:

- Empty files
- Missing date column
- Missing price column
- Invalid numeric values
- Non-positive price values
- Very small datasets
- Model training errors

---

## Module overview

| File | Responsibility |
|---|---|
| app.py | Streamlit interface, sidebar, charts, model selection, and result display |
| data_utils.py | Data loading, cleaning, validation, indicators, and metrics |
| prophet_model.py | Prophet feature preparation, tuning, and forecasting |
| rf_model.py | Random Forest feature engineering and forecasting |
| arima_model.py | ARIMA forecasting |
| requirements.txt | Project dependencies |

---

## Tech stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Prophet
- ARIMA
- scikit-learn

---

## Run command

```bash
streamlit run app.py
```
