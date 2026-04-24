
import io
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

TEST_SIZE    = 30
MIN_ROWS     = 120


def detect_price_column(columns, preferred_col: str) -> str:
    
    cols_original  = list(columns)
    cols_lower_map = {c.strip().lower(): c for c in cols_original}

    if preferred_col in cols_original:
        return preferred_col

    if preferred_col.strip().lower() in cols_lower_map:
        return cols_lower_map[preferred_col.strip().lower()]

    candidates = ["close", "adj close", "close/last", "price", "open", "high", "low"]
    for cand in candidates:
        if cand in cols_lower_map:
            return cols_lower_map[cand]

    numeric_like = [
        c for c in cols_original
        if c.strip().lower() not in ("date", "timestamp", "time", "volume")
    ]
    if numeric_like:
        return numeric_like[0]

    raise ValueError(
        "No suitable price column found in the CSV. "
        "Please ensure the file contains a column named Close, Open, High, or Low."
    )



def validate_csv(raw_bytes: bytes) -> None:
    """
    Run lightweight pre-checks on the raw CSV bytes and raise a descriptive
    ValueError for any common incompatibility, so the UI can show a clear
    error message before attempting a full parse.

    Checks
    ------
    1. File is not empty
    2. File parses as valid CSV (non-zero rows, at least 2 columns)
    3. A date-like column exists (Date / Timestamp / Time)
    4. At least one numeric-looking column exists beyond the date column
    """
    if not raw_bytes:
        raise ValueError("The uploaded file is empty. Please upload a valid BTC CSV.")

    try:
        df_peek = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(
            f"Could not read the file as a CSV. "
            f"Make sure it is a valid comma-separated file. Detail: {exc}"
        )

    if df_peek.empty:
        raise ValueError("The CSV file has no data rows.")

    if len(df_peek.columns) < 2:
        raise ValueError(
            "The CSV must have at least two columns: a date column and a price column. "
            f"Only {len(df_peek.columns)} column(s) found."
        )

    date_cols = [c for c in df_peek.columns if c.strip().lower() in ("date", "timestamp", "time")]
    if not date_cols:
        raise ValueError(
            "No date column found. The file must contain a column named 'Date', "
            "'Timestamp', or 'Time'."
        )

    non_date_cols = [c for c in df_peek.columns if c not in date_cols]
    numeric_candidates = []
    for col in non_date_cols:
        cleaned = (
            df_peek[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        numeric_count = pd.to_numeric(cleaned, errors="coerce").notna().sum()
        if numeric_count > 0:
            numeric_candidates.append(col)

    if not numeric_candidates:
        raise ValueError(
            "No numeric price column found in the CSV. "
            "Expected columns like 'Close', 'Open', 'High', or 'Low' with numeric values."
        )


@st.cache_data(show_spinner=False)
def load_and_clean(raw_bytes: bytes, price_col: str) -> pd.DataFrame:
    """
    Parse raw CSV bytes into a clean, daily-frequency DataFrame.

    Steps
    -----
    1. Detect and parse date column
    2. Clean price column (strip $ / commas, coerce to float)
    3. Validate that numeric conversion succeeded for enough rows
    4. Sort, deduplicate, and reindex to a full daily range (forward-fill gaps)
    5. Guard against non-positive prices (required for log-transform)

    Returns a DataFrame with columns: ['Date', price_col]
    """
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # ── Find date column ───────────────────────────────────────────────────────
    date_col = None
    for c in df.columns:
        if c.strip().lower() in ("date", "timestamp", "time"):
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            "No date column found. The file must contain a column named "
            "'Date', 'Timestamp', or 'Time'."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        df = df.dropna(subset=[date_col]).copy()
        if len(df) == 0:
            raise ValueError(
                f"All {invalid_dates} date values in '{date_col}' are unparseable. "
                "Please check the date format in your CSV."
            )

    df = df.rename(columns={date_col: "Date"})

    if price_col not in df.columns:
        raise ValueError(
            f"Price column '{price_col}' not found. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    bad_prices = df[price_col].isna().sum()
    total_rows = len(df)
    df = df.dropna(subset=[price_col]).copy()

    if len(df) == 0:
        raise ValueError(
            f"The price column '{price_col}' contains no valid numeric values. "
            "Please check you have selected the correct price column."
        )

    if bad_prices / total_rows > 0.5:
        raise ValueError(
            f"{bad_prices} of {total_rows} rows ({bad_prices/total_rows:.0%}) in "
            f"'{price_col}' could not be converted to numbers. "
            "Please verify this is a price column."
        )

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last").copy()

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range).rename_axis("Date")
    df[price_col] = df[price_col].ffill()
    df = df.reset_index()[["Date", price_col]]

    if (df[price_col] <= 0).any():
        n_bad = (df[price_col] <= 0).sum()
        raise ValueError(
            f"{n_bad} row(s) in '{price_col}' contain zero or negative values. "
            "Log-transform requires all prices to be strictly positive."
        )

    price_range = df[price_col].max() - df[price_col].min()
    if price_range < 1e-6:
        raise ValueError(
            f"The price column '{price_col}' appears constant (all values are identical). "
            "Forecasting requires historical variation in price."
        )

    return df



def compute_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Append SMA-20, SMA-50, and EMA-20 columns to the DataFrame."""
    df = df.copy()
    df["SMA_20"] = df[price_col].rolling(20).mean()
    df["SMA_50"] = df[price_col].rolling(50).mean()
    df["EMA_20"] = df[price_col].ewm(span=20, adjust=False).mean()
    return df



def calc_mae_rmse(actual, pred):
    """Return (MAE, RMSE) for two array-likes."""
    mae  = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    return mae, rmse
