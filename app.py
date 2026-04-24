"""
app.py
======
Bitcoin Price Forecasting Portal — Streamlit entry point.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    detect_price_column,
    load_and_clean,
    compute_indicators,
    validate_csv,
    TEST_SIZE,
    MIN_ROWS,
)
from prophet_model import prophet_forecast
from rf_model      import ml_forecast
from arima_model   import arima_forecast


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bitcoin Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f7931a; }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f7931a;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .stButton > button {
        background: linear-gradient(135deg, #f7931a, #e07b10);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 10px;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #e07b10, #c96b00); }
    .error-box {
        background: #2d1117;
        border: 1px solid #6e1a1a;
        border-radius: 8px;
        padding: 12px 16px;
        color: #ff7b72;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Model runner ───────────────────────────────────────────────────────────────

def run_selected_model(
    model_choice: str,
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    ci: float,
):
    """
    Dispatch to the selected model, or run all three and return the best by RMSE.

    Returns
    -------
    result             : dict  — best / selected model result
    comparison_results : list | None — populated only for Auto Select
    """
    if model_choice == "Prophet Enhanced":
        return prophet_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE), None

    if model_choice == "Random Forest":
        return ml_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE), None

    if model_choice == "ARIMA":
        return arima_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE), None

    # ── Auto Select: run all three, pick lowest RMSE ───────────────────────────
    results = []
    errors  = []

    candidates = [
        ("Prophet Enhanced", lambda: prophet_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE)),
        ("Random Forest",    lambda: ml_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE)),
        ("ARIMA",            lambda: arima_forecast(df, price_col, horizon, ci, test_size=TEST_SIZE)),
    ]

    for model_name, fn in candidates:
        try:
            results.append(fn())
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    if not results:
        raise ValueError("All models failed. " + " | ".join(errors))

    best = min(results, key=lambda r: r["rmse"])
    return best, results


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ₿ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload BTC CSV",
        type=["csv"],
        help="Kaggle-style BTC historical CSV (must contain Date and price columns).",
    )

    price_col_option = st.selectbox(
        "Price Column",
        ["Close", "Open", "High", "Low", "Adj Close"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📐 Forecast Settings")

    model_choice = st.selectbox(
        "Forecasting Model",
        ["Auto Select Best Model", "Prophet Enhanced", "Random Forest", "ARIMA"],
        index=0,
    )

    horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7, max_value=180, value=30, step=7,
    )

    ci_pct = st.select_slider(
        "Confidence Interval",
        options=[80, 90, 95, 99],
        value=95,
    )
    ci = ci_pct / 100.0

    st.markdown("---")
    st.markdown("### 📊 Technical Indicators")
    show_sma20 = st.checkbox("SMA 20", value=True)
    show_sma50 = st.checkbox("SMA 50", value=True)
    show_ema20 = st.checkbox("EMA 20", value=False)

    st.markdown("---")
    st.caption(f"Backtest window: last {TEST_SIZE} days")
    run_btn = st.button("🚀 Generate Forecast")


# ── Main ───────────────────────────────────────────────────────────────────────

st.markdown("# ₿ Bitcoin Price Forecasting Portal")
st.markdown(
    "Upload your BTC historical dataset, configure the model, "
    "and generate an interactive forecast."
)

if uploaded is None:
    st.info("👈 Upload a BTC CSV file from the sidebar to begin.")
    st.stop()

# ── Step 1: pre-validate the raw file before any heavy parsing ─────────────────

try:
    raw_bytes = uploaded.read()
    validate_csv(raw_bytes)
except ValueError as exc:
    st.error(f"❌ Invalid file: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"❌ Unexpected error reading file: {exc}")
    st.stop()

# ── Step 2: detect price column ────────────────────────────────────────────────

try:
    df_peek   = pd.read_csv(io.BytesIO(raw_bytes), nrows=5)
    price_col = detect_price_column(df_peek.columns, price_col_option)
except ValueError as exc:
    st.error(f"❌ Column error: {exc}")
    st.stop()

# ── Step 3: full load and clean ────────────────────────────────────────────────

try:
    df = load_and_clean(raw_bytes, price_col)
except ValueError as exc:
    st.error(f"❌ Data error: {exc}")
    st.stop()
except Exception as exc:
    st.error(
        f"❌ Could not process the CSV. "
        f"Please ensure it is a standard Kaggle BTC historical dataset. "
        f"Detail: {exc}"
    )
    st.stop()

# ── Step 4: minimum row check ──────────────────────────────────────────────────

if len(df) < MIN_ROWS:
    st.error(
        f"❌ Dataset too small — {len(df)} rows found, at least {MIN_ROWS} required. "
        "Please upload a longer BTC history (minimum ~4 months of daily data)."
    )
    st.stop()

# ── Step 5: compute indicators ─────────────────────────────────────────────────

df = compute_indicators(df, price_col)

# ── Summary metrics ────────────────────────────────────────────────────────────

current_price = df[price_col].iloc[-1]
start_price   = df[price_col].iloc[0]
pct_change    = (current_price - start_price) / start_price * 100 if start_price != 0 else 0
date_range    = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-value">${current_price:,.0f}</div>
              <div class="metric-label">Latest BTC Price</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col2:
    arrow  = "▲" if pct_change > 0 else "▼"
    colour = "#27c93f" if pct_change > 0 else "#ff5f57"
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-value" style="color:{colour}">{arrow} {pct_change:.1f}%</div>
              <div class="metric-label">Total Return</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-value">{len(df):,}</div>
              <div class="metric-label">Days of Data</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-value" style="font-size:1rem">{date_range}</div>
              <div class="metric-label">Date Range</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Historical price chart ─────────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Historical Price</div>', unsafe_allow_html=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df["Date"], y=df[price_col],
    mode="lines", name="BTC Price",
    line=dict(color="#f7931a", width=2),
    fill="tozeroy", fillcolor="rgba(247,147,26,0.06)",
))
if show_sma20:
    fig_hist.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA_20"], mode="lines",
        name="SMA 20", line=dict(color="#58a6ff", width=1.5, dash="dot"),
    ))
if show_sma50:
    fig_hist.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA_50"], mode="lines",
        name="SMA 50", line=dict(color="#bc8cff", width=1.5, dash="dash"),
    ))
if show_ema20:
    fig_hist.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA_20"], mode="lines",
        name="EMA 20", line=dict(color="#3fb950", width=1.5, dash="longdash"),
    ))

fig_hist.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    height=420,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True, tickprefix="$", tickformat=",.0f"),
    hovermode="x unified",
)
st.plotly_chart(fig_hist, use_container_width=True)


# ── Forecast ───────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Training {model_choice}… this may take a minute for Prophet."):
        try:
            result, comparison_results = run_selected_model(
                model_choice, df, price_col, horizon, ci
            )
        except Exception as exc:
            st.error(f"❌ Model training failed: {exc}")
            st.stop()

    forecast_df  = result["forecast_df"].copy()
    mae          = result["mae"]
    rmse         = result["rmse"]
    used_model   = result["model_name"]
    test_dates   = result["test_dates"]
    test_actual  = result["test_actual"]
    test_pred    = result["test_pred"]

    # ARIMA returns real CI bands; RF uses residual std; Prophet uses its own CI.
    # All three store them in test_lower / test_upper when available.
    test_lower   = result.get("test_lower")
    test_upper   = result.get("test_upper")

    # ── Backtest metrics ───────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        f'<div class="section-header">🎯 Model Performance (Backtest — last {TEST_SIZE} days)</div>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"""<div class="metric-card">
                  <div class="metric-value" style="color:#58a6ff">${mae:,.0f}</div>
                  <div class="metric-label">MAE (USD)</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"""<div class="metric-card">
                  <div class="metric-value" style="color:#bc8cff">${rmse:,.0f}</div>
                  <div class="metric-label">RMSE (USD)</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with mc3:
        last_fc   = forecast_df["yhat"].iloc[-1]
        direction = "▲" if last_fc > current_price else "▼"
        dir_col   = "#27c93f" if last_fc > current_price else "#ff5f57"
        st.markdown(
            f"""<div class="metric-card">
                  <div class="metric-value" style="color:{dir_col}">{direction} ${last_fc:,.0f}</div>
                  <div class="metric-label">{horizon}-Day Forecast</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Selected model: **{used_model}** | "
        f"Type: {result['meta']['type']} | "
        f"{result['meta']['details']}"
    )

    if comparison_results is not None:
        comp_df = (
            pd.DataFrame([
                {
                    "Model": r["model_name"],
                    "MAE ($)":  round(r["mae"], 2),
                    "RMSE ($)": round(r["rmse"], 2),
                    "Selected": "✅" if r["model_name"] == used_model else "",
                }
                for r in comparison_results
            ])
            .sort_values("RMSE ($)")
            .reset_index(drop=True)
        )
        st.dataframe(comp_df, use_container_width=True)

    # ── Forecast chart ─────────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-header">🔮 {used_model} Forecast — {horizon} Days ({ci_pct}% CI)</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    hist_tail = df.tail(180)
    hist_end  = df["Date"].max()

    fig.add_trace(go.Scatter(
        x=hist_tail["Date"], y=hist_tail[price_col],
        name="Historical", mode="lines",
        line=dict(color="#f7931a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{ci_pct}% Confidence Band",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        name=f"{used_model} Forecast",
        mode="lines",
        line=dict(color="#58a6ff", width=2.5, dash="dash"),
    ))
    fig.add_shape(
        type="line",
        x0=hist_end, x1=hist_end, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#f7931a", width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=hist_end, y=1,
        xref="x", yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="#f7931a", size=12),
        xanchor="left", yanchor="top",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=480,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickprefix="$", tickformat=",.0f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Backtest chart — predicted vs actual with CI band ─────────────────────

    st.markdown(
        '<div class="section-header">🔬 Backtest — Predicted vs Actual</div>',
        unsafe_allow_html=True,
    )

    fig_bt = go.Figure()

    # CI band on the backtest chart (all three models now expose lower/upper)
    if test_lower is not None and test_upper is not None:
        test_dates_list = list(test_dates)
        fig_bt.add_trace(go.Scatter(
            x=list(test_dates_list) + list(reversed(test_dates_list)),
            y=list(test_upper) + list(reversed(test_lower)),
            fill="toself",
            fillcolor="rgba(88,166,255,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{ci_pct}% Backtest CI",
            showlegend=True,
        ))

    fig_bt.add_trace(go.Scatter(
        x=test_dates, y=test_actual,
        name="Actual", line=dict(color="#f7931a", width=2),
    ))
    fig_bt.add_trace(go.Scatter(
        x=test_dates, y=test_pred,
        name="Predicted", line=dict(color="#58a6ff", width=2, dash="dash"),
    ))
    fig_bt.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickprefix="$", tickformat=",.0f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Forecast data table ────────────────────────────────────────────────────

    st.markdown(
        '<div class="section-header">📋 Forecast Data</div>',
        unsafe_allow_html=True,
    )

    fc_display = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc_display.columns = ["Date", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
    fc_display["Date"] = pd.to_datetime(fc_display["Date"]).dt.strftime("%Y-%m-%d")

    for c in ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]:
        fc_display[c] = fc_display[c].apply(lambda x: f"${x:,.2f}")

    st.dataframe(fc_display.reset_index(drop=True), use_container_width=True, height=280)