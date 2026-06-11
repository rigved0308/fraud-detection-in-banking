import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    df.dropna(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    if "Close" not in df.columns:
        raise ValueError("Close column not found in downloaded data.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + 2 * std
    df["BB_Lower"] = bb_mid - 2 * std

    df["Daily_Return"] = close.pct_change() * 100
    df["Volatility"] = df["Daily_Return"].rolling(20).std()
    return df


def prepare_lstm_data(prices: np.ndarray, window: int = 60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, y_train, y_test, scaler, split + window


def build_lstm_model(window: int = 60):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32, verbose=0):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=verbose,
    )
    return history


def predict_and_evaluate(model, X_test, y_test, scaler):
    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    return preds, actual, {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
    }


def forecast_future(model, last_window: np.ndarray, scaler, n_days: int = 30):
    window = last_window.copy().reshape(-1, 1)
    scaled = scaler.transform(window)

    preds = []
    seq = list(scaled.flatten())

    for _ in range(n_days):
        inp = np.array(seq[-60:]).reshape(1, 60, 1)
        p = model.predict(inp, verbose=0)[0][0]
        preds.append(p)
        seq.append(p)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()