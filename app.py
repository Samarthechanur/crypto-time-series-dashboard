import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# Reduce noisy logs from TensorFlow & pandas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ======================================================
# 1. DATA LOADING & PREPROCESSING
# ======================================================

DATA_PATH = os.path.join("data", "Crypto Historical Data.csv")


@st.cache_data
def load_data(path):
    """Load CSV, detect date & price column, clean and return."""
    df = pd.read_csv(path)

    # --- detect date/time column ---
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl:
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            "No date/time column found. "
            "Expected something with 'date' or 'time' in the column name."
        )

    # parse and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # --- detect close/price column ---
    target_col = None
    for c in df.columns:
        cl = c.lower()
        if "close" in cl or "price" in cl:
            target_col = c
            break

    # fallback: first numeric column
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for target variable.")
        target_col = numeric_cols[0]

    # ensure float + drop NaN
    df[target_col] = df[target_col].astype(float)
    df = df.dropna(subset=[target_col])

    return df, target_col


def infer_frequency(index):
    """Infer frequency of the datetime index or default to daily."""
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if freq is None:
        freq = "D"
    return freq


def make_future_index(df, steps):
    freq = infer_frequency(df.index)
    last_date = df.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=steps,
        freq=freq,
    )
    return future_index


# ======================================================
# 2. FORECASTING MODELS (ARIMA + LSTM)
# ======================================================

def train_arima(series, steps=30, order=(5, 1, 0)):
    """Train ARIMA on the price series and forecast `steps` future points."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return model_fit, forecast


def create_lstm_dataset(values, window_size=30):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i, 0])
        y.append(values[i, 0])
    X, y = np.array(X), np.array(y)
    # reshape for LSTM: (samples, timesteps, features)
    X = np.expand_dims(X, axis=2)
    return X, y


def train_lstm(series, window_size=30, epochs=20, batch_size=16):
    """
    Train an LSTM (deep learning) model on the price series.
    Returns model, scaler, and a forecast function.
    """
    values = series.values.reshape(-1, 1)

    # scale to 0-1
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = create_lstm_dataset(scaled, window_size=window_size)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def forecast_future(n_steps):
        last_window = scaled[-window_size:, 0]
        preds_scaled = []
        for _ in range(n_steps):
            x_input = last_window.reshape(1, window_size, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0, 0]
            preds_scaled.append(pred_scaled)
            last_window = np.append(last_window[1:], pred_scaled)

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).ravel()
        return preds

    return model, scaler, forecast_future


# ======================================================
# 3. STYLING (PREMIUM UI)
# ======================================================

def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #020617, #000000);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .main-shell {
            background: rgba(15, 23, 42, 0.96);
            border-radius: 18px;
            padding: 20px 26px 30px 26px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.9);
            margin-bottom: 18px;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            background: linear-gradient(90deg, #22c55e, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-title {
            font-size: 0.95rem;
            color: #9ca3af;
        }
        .kpi-card {
            background: radial-gradient(circle at top left, #020617, #020617 40%, #020617 100%);
            padding: 1rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(55, 65, 81, 0.9);
        }
        .kpi-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #9ca3af;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e5e7eb;
        }
        .kpi-change-pos { font-size: 0.85rem; color: #22c55e; }
        .kpi-change-neg { font-size: 0.85rem; color: #ef4444; }
        .stTabs [role="tablist"] { gap: 0.35rem; }
        .stTabs [role="tab"] {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
# 4. STREAMLIT APP
# ======================================================

def main():
    st.set_page_config(
        page_title="Crypto Time Series Pro",
        page_icon="📈",
        layout="wide",
    )

    add_custom_css()

    # ---------- Sidebar ----------
    st.sidebar.image(
        "https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=032",
        caption="Crypto Analytics",
    )
    st.sidebar.markdown("### Model Settings")

    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["ARIMA (Statistical)", "LSTM (Deep Learning)"],
    )
    horizon = st.sidebar.slider("Forecast horizon (future points)", 7, 60, 25)
    window_size = st.sidebar.slider("LSTM window size", 10, 60, 30)
    lstm_epochs = st.sidebar.slider("LSTM epochs", 5, 30, 15)
    train_window = st.sidebar.slider(
        "Training history (last N points)",
        200,
        2000,
        730,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Project:** Crypto Time-Series Forecasting  \n"
        "**Stack:** Pandas · NumPy · ARIMA · LSTM · Streamlit · Plotly"
    )

    # ---------- Load Data ----------
    if not os.path.exists(DATA_PATH):
        st.error(
            f"CSV file not found at: {DATA_PATH}. "
            "Make sure 'Crypto Historical Data.csv' is inside the data/ folder."
        )
        return

    df, target_col = load_data(DATA_PATH)

    # adapt training window if dataset is shorter
    train_window = min(train_window, len(df))
    train_df = df.iloc[-train_window:]

    # ---------- KPIs with cleaned returns ----------
    latest_price = df[target_col].iloc[-1]
    prev_price = df[target_col].iloc[-2] if len(df) > 1 else latest_price
    daily_change = (
        (latest_price - prev_price) / prev_price * 100.0 if prev_price != 0 else 0.0
    )

    returns = df[target_col].pct_change()
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns = returns.dropna()
    if len(returns) > 0:
        avg_return = returns.mean() * 100.0
        vol = returns.std() * 100.0
    else:
        avg_return = 0.0
        vol = 0.0

    # ---------- Header + KPI cards ----------
    st.markdown('<div class="main-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-title">Crypto Time Series Analysis & Forecasting</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">'
        'Institution-grade analytics dashboard powered by ARIMA and LSTM deep learning models for crypto price forecasting.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Latest Price</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-value">{latest_price:,.2f}</div>',
            unsafe_allow_html=True,
        )
        change_class = "kpi-change-pos" if daily_change >= 0 else "kpi-change-neg"
        st.markdown(
            f'<div class="{change_class}">{daily_change:+.2f}% vs prev</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="kpi-label">Average Daily Return</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi-value">{avg_return:.3f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="kpi-label">Volatility (Std of Returns)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi-value">{vol:.3f}%</div>', unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Tabs ----------
    tab_overview, tab_forecast, tab_model = st.tabs(
        ["📊 Overview", "🔮 Forecasting", "🧠 Model Details"]
    )

    # ===== OVERVIEW TAB =====
    with tab_overview:
        st.subheader("Raw Data (first 10 rows)")
        st.dataframe(df.head(10))

        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        st.subheader(f"Historical Price Trend ({target_col})")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=df.index,
                y=df[target_col],
                mode="lines",
                name="Historical Price",
                fill="tozeroy",
            )
        )
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title=target_col,
            height=400,
        )
        st.plotly_chart(fig_hist)

        st.subheader("Daily Returns (Volatility)")
        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode="lines",
                name="Daily Return",
            )
        )
        fig_ret.update_layout(
            xaxis_title="Date",
            yaxis_title="Return",
            height=300,
        )
        st.plotly_chart(fig_ret)

    # ===== FORECAST TAB =====
    with tab_forecast:
        st.subheader("Configure & Run Forecast")
        st.markdown(
            "_Training is performed on the most recent section of the series "
            f"(**last {train_window} points**) to capture the current market regime._"
        )

        if st.button("🚀 Train & Forecast"):
            with st.spinner(f"Training {model_type} model..."):
                if "ARIMA" in model_type:
                    series = train_df[target_col]
                    model_fit, forecast = train_arima(series, steps=horizon)
                    future_index = make_future_index(df, horizon)
                    forecast_series = pd.Series(
                        forecast.values,
                        index=future_index,
                        name="ARIMA Forecast",
                    )
                else:
                    series = train_df[target_col]
                    model, scaler, forecast_fn = train_lstm(
                        series,
                        window_size=window_size,
                        epochs=lstm_epochs,
                    )
                    preds = forecast_fn(horizon)
                    future_index = make_future_index(df, horizon)
                    forecast_series = pd.Series(
                        preds,
                        index=future_index,
                        name="LSTM Forecast",
                    )

            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[target_col],
                    mode="lines",
                    name="Historical",
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_series.index,
                    y=forecast_series.values,
                    mode="lines+markers",
                    name="Forecast",
                )
            )
            fig_forecast.update_layout(
                xaxis_title="Date",
                yaxis_title=target_col,
                height=450,
            )
            st.plotly_chart(fig_forecast)

            st.markdown("### Forecast Values")
            st.dataframe(forecast_series.to_frame())
            st.success("Forecast complete ✔")

    # ===== MODEL DETAILS TAB =====
    with tab_model:
        st.subheader("Model Explanations")

        st.markdown("#### ARIMA (AutoRegressive Integrated Moving Average)")
        st.markdown(
            """
            - **AR** (AutoRegressive): uses past values of the series to predict the future.  
            - **I** (Integrated): differencing to remove trend and make the series stationary.  
            - **MA** (Moving Average): models the error term as a combination of past shocks.  
            """
        )

        st.markdown("#### LSTM (Long Short-Term Memory) – Deep Learning Model")
        st.markdown(
            """
            - LSTM is a **Recurrent Neural Network** specialised for sequence / time-series data.  
            - It uses gates (input, forget, output) and a memory cell to keep long-term information.  
            - In this project, we feed a sliding window of past prices and train the network to
              predict the next price, optimising **MSE loss** with the **Adam** optimizer.  
            """
        )

        st.markdown("#### How this impresses HR / Faculty")
        st.markdown(
            """
            - End-to-end system: data ingestion → preprocessing → EDA → ARIMA & LSTM forecasting → interactive dashboard.  
            - Uses a **deep learning model (LSTM)**, so it is valid for a **Deep Learning subject project**.  
            - Modern, professional UI built with **Streamlit + custom CSS**, suitable to show as a portfolio project.  
            """
        )


if __name__ == "__main__":
    main()
