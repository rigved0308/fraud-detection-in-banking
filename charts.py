import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_price_with_mas(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close Price", color="#1f77b4", linewidth=1.5)

    for col, color in [("MA20", "orange"), ("MA50", "green"), ("MA200", "red")]:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, color=color, linestyle="--", linewidth=1)

    ax.set_title(f"{ticker} – Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_volume(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(df.index, df["Volume"], color="#aec7e8", width=1.5)
    ax.set_title(f"{ticker} – Volume")
    ax.set_ylabel("Volume")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_rsi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["RSI"], color="purple", linewidth=1)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax.fill_between(
        df.index,
        df["RSI"],
        70,
        where=(df["RSI"] >= 70),
        color="red",
        alpha=0.2,
        label="Overbought"
    )
    ax.fill_between(
        df.index,
        df["RSI"],
        30,
        where=(df["RSI"] <= 30),
        color="green",
        alpha=0.2,
        label="Oversold"
    )
    ax.set_title("RSI (14)")
    ax.set_ylabel("RSI")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_bollinger(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", color="#1f77b4", linewidth=1)
    ax.plot(df.index, df["BB_Upper"], label="Upper Band", color="red", linestyle="--", linewidth=0.8)
    ax.plot(df.index, df["BB_Lower"], label="Lower Band", color="green", linestyle="--", linewidth=0.8)
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.1, color="gray")
    ax.set_title(f"{ticker} – Bollinger Bands")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_macd(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1)
    ax.plot(df.index, df["MACD_Signal"], label="Signal", color="orange", linewidth=1)
    hist = df["MACD"] - df["MACD_Signal"]
    ax.bar(
        df.index,
        hist,
        color=["green" if v >= 0 else "red" for v in hist],
        width=1.5,
        alpha=0.5,
        label="Histogram"
    )
    ax.set_title("MACD")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_prediction(actual, preds, dates, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual Price", color="blue", linewidth=1.5)
    ax.plot(dates, preds, label="LSTM Predicted", color="orange", linestyle="--", linewidth=1.5)
    ax.set_title(f"{ticker} – LSTM Prediction vs Actual")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_forecast(df: pd.DataFrame, forecast: np.ndarray, n_days: int, ticker: str):
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=n_days + 1)[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index[-100:], df["Close"].values[-100:], label="Historical", color="blue", linewidth=1.5)
    ax.plot(future_dates, forecast, label=f"Forecast ({n_days}d)", color="red", linestyle="--", linewidth=2)
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"{ticker} – {n_days}-Day Price Forecast")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig