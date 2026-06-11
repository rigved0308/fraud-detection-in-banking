import numpy as np
import pandas as pd
import streamlit as st

from model import (
    fetch_stock_data,
    add_technical_indicators,
    prepare_lstm_data,
    build_lstm_model,
    train_model,
    predict_and_evaluate,
    forecast_future,
)
from charts import (
    plot_price_with_mas,
    plot_volume,
    plot_rsi,
    plot_bollinger,
    plot_macd,
    plot_prediction,
    plot_forecast,
)
from report import build_pdf

st.set_page_config(page_title="Stock Price Analysis", layout="wide", page_icon="📈")

with st.sidebar:
    st.title("📈 Stock Analyser")
    st.markdown("---")
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, GOOGL, MSFT, RELIANCE.NS"
    )
    period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=2)
    n_days = st.slider("Forecast Days", 7, 60, 30)
    epochs = st.slider("LSTM Training Epochs", 5, 50, 10)
    run = st.button("🚀 Analyse & Predict", use_container_width=True)
    st.markdown("---")
    st.caption("Data via Yahoo Finance (yfinance)")

st.title("📈 Stock Price Analysis & Trend Prediction")
st.markdown("Real-time data, technical indicators, LSTM prediction, and future forecast.")
st.markdown("---")

if not run:
    st.info("Set your ticker and options in the sidebar, then click **Analyse & Predict**.")
    st.stop()

with st.spinner(f"Fetching data for **{ticker.upper()}**..."):
    try:
        df_raw = fetch_stock_data(ticker.upper(), period)
        if df_raw.empty:
            st.error("No data found. Check the ticker symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

df = add_technical_indicators(df_raw)
df_clean = df.dropna()

st.success(f"Loaded **{len(df_clean)}** trading days for **{ticker.upper()}**")

with st.expander("📋 Step 1: Data Overview", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    close = df_clean["Close"]
    c1.metric("Current Price", f"${close.iloc[-1]:.2f}")
    c2.metric("52-Week High", f"${close.max():.2f}")
    c3.metric("52-Week Low", f"${close.min():.2f}")
    pct = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    c4.metric("Period Return", f"{pct:.1f}%", delta=f"{pct:.1f}%")
    st.dataframe(
        df_clean[["Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(10),
        use_container_width=True
    )

st.markdown("### 📊 Step 2: Technical Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price + MAs", "Volume", "Bollinger Bands", "RSI", "MACD"]
)

with tab1:
    fig_ma = plot_price_with_mas(df_clean, ticker.upper())
    st.pyplot(fig_ma)

with tab2:
    fig_vol = plot_volume(df_clean, ticker.upper())
    st.pyplot(fig_vol)

with tab3:
    fig_bb = plot_bollinger(df_clean, ticker.upper())
    st.pyplot(fig_bb)
    st.caption("Price touching Upper Band → overbought. Price touching Lower Band → oversold.")

with tab4:
    fig_rsi = plot_rsi(df_clean)
    st.pyplot(fig_rsi)
    st.caption("RSI > 70 = Overbought. RSI < 30 = Oversold.")

with tab5:
    fig_macd = plot_macd(df_clean)
    st.pyplot(fig_macd)
    st.caption("MACD crossing above Signal = Bullish. MACD crossing below = Bearish.")

st.markdown("### 🔎 Step 3: Trend Signal")
rsi_now = df_clean["RSI"].iloc[-1]
ma20_now = df_clean["MA20"].iloc[-1]
ma50_now = df_clean["MA50"].iloc[-1]
price_now = float(df_clean["Close"].iloc[-1])

if price_now > float(ma20_now) and float(ma20_now) > float(ma50_now):
    signal = "🟢 Bullish – Price above MA20 and MA20 above MA50"
elif price_now < float(ma20_now) and float(ma20_now) < float(ma50_now):
    signal = "🔴 Bearish – Price below MA20 and MA20 below MA50"
else:
    signal = "🟡 Neutral – Mixed signals, no clear trend"

rsi_signal = (
    "🔴 Overbought (RSI > 70)"
    if rsi_now > 70
    else ("🟢 Oversold (RSI < 30)" if rsi_now < 30 else "🟡 Neutral RSI")
)

col1, col2 = st.columns(2)
col1.info(f"**Trend Signal:** {signal}")
col2.info(f"**RSI Signal:** {rsi_signal} — RSI = {rsi_now:.1f}")

st.markdown("### 🤖 Step 4: LSTM Price Prediction")
prices = df_clean["Close"].values

with st.spinner("Training LSTM model... this takes a moment ⏳"):
    X_train, X_test, y_train, y_test, scaler, test_start = prepare_lstm_data(prices)
    lstm_model = build_lstm_model(window=60)
    train_model(lstm_model, X_train, y_train, epochs=epochs, verbose=0)
    preds, actual, metrics = predict_and_evaluate(lstm_model, X_test, y_test, scaler)

test_dates = df_clean.index[test_start:]

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"${metrics['RMSE']}")
col2.metric("MAE", f"${metrics['MAE']}")
col3.metric("R² Score", f"{metrics['R2']}")

fig_pred = plot_prediction(actual, preds, test_dates, ticker.upper())
st.pyplot(fig_pred)

st.markdown(f"### 🔮 Step 5: {n_days}-Day Future Price Forecast")

with st.spinner("Generating forecast..."):
    last_60 = prices[-60:]
    forecast = forecast_future(lstm_model, last_60, scaler, n_days)

last_date = df_clean.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=n_days + 1)[1:]

col1, col2, col3 = st.columns(3)
col1.metric("Forecast Start", f"${prices[-1]:.2f}")
col2.metric("Forecast End", f"${forecast[-1]:.2f}")
change = ((forecast[-1] - prices[-1]) / prices[-1]) * 100
col3.metric("Expected Change", f"{change:.1f}%", delta=f"{change:.1f}%")

fig_forecast = plot_forecast(df_clean, forecast, n_days, ticker.upper())
st.pyplot(fig_forecast)

st.markdown("### 💾 Step 6: Export")

df_export = df_clean[["Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].copy()
df_export["LSTM_Pred"] = np.nan
df_export.iloc[-len(preds):, df_export.columns.get_loc("LSTM_Pred")] = preds

csv_bytes = df_export.to_csv().encode("utf-8")

summary_stats = {
    "Ticker": ticker.upper(),
    "Period": period,
    "Trading Days": len(df_clean),
    "Current Price": f"${price_now:.2f}",
    "52-Week High": f"${close.max():.2f}",
    "52-Week Low": f"${close.min():.2f}",
    "Period Return": f"{pct:.1f}%",
    "RSI Now": f"{rsi_now:.1f}",
    "Trend Signal": signal,
}

pdf_figs = {
    "price_ma": fig_ma,
    "volume": fig_vol,
    "bollinger": fig_bb,
    "rsi": fig_rsi,
    "macd": fig_macd,
    "prediction": fig_pred,
    "forecast": fig_forecast,
}

pdf_bytes = build_pdf(
    ticker=ticker.upper(),
    period=period,
    metrics=metrics,
    summary_stats=summary_stats,
    figs=pdf_figs,
)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "📥 Download CSV",
        csv_bytes,
        f"{ticker}_analysis.csv",
        "text/csv",
        use_container_width=True
    )
with d2:
    st.download_button(
        "📄 Download PDF Report",
        pdf_bytes,
        f"{ticker}_report.pdf",
        "application/pdf",
        use_container_width=True
    )

st.caption("Stock Price Analysis | Python · yfinance · TensorFlow · Streamlit")