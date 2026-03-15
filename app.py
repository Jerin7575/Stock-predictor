import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from datetime import datetime

# =============================
# API KEY
# =============================
API_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# =============================
# Stock List
# =============================
stocks = {
    "Apple": {"symbol": "AAPL", "logo": "https://logo.clearbit.com/apple.com"},
    "Tesla": {"symbol": "TSLA", "logo": "https://logo.clearbit.com/tesla.com"},
    "Microsoft": {"symbol": "MSFT", "logo": "https://logo.clearbit.com/microsoft.com"},
    "Amazon": {"symbol": "AMZN", "logo": "https://logo.clearbit.com/amazon.com"},
    "Google": {"symbol": "GOOGL", "logo": "https://logo.clearbit.com/google.com"},
    "NVIDIA": {"symbol": "NVDA", "logo": "https://logo.clearbit.com/nvidia.com"},
    "Meta": {"symbol": "META", "logo": "https://logo.clearbit.com/meta.com"},
    "Netflix": {"symbol": "NFLX", "logo": "https://logo.clearbit.com/netflix.com"}
}

# =============================
# Fetch Stock Data
# =============================
def get_stock_data(symbol):

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={API_KEY}"

    r = requests.get(url)
    data = r.json()

    if "Time Series (Daily)" not in data:
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")

    df = df.rename(columns={
        "1. open":"open",
        "2. high":"high",
        "3. low":"low",
        "4. close":"close",
        "5. volume":"volume"
    })

    df = df.astype(float)

    df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    return df


# =============================
# Feature Engineering
# =============================
def create_features(df):

    df["sma5"] = df["close"].rolling(5).mean()

    df["sma10"] = df["close"].rolling(10).mean()

    df["returns"] = df["close"].pct_change()

    df["volatility"] = df["returns"].rolling(5).std()

    df = df.dropna()

    X = []
    y = []

    for i in range(10, len(df)-1):

        past = df.iloc[i-10:i]

        features = [
            past["returns"].mean(),
            past["returns"].std(),
            df.iloc[i]["sma5"],
            df.iloc[i]["sma10"],
            df.iloc[i]["volatility"],
            df.iloc[i]["close"]
        ]

        target = df.iloc[i+1]["close"]

        X.append(features)

        y.append(target)

    return np.array(X), np.array(y), df


# =============================
# Train Model
# =============================
def predict_price(df):

    X, y, df = create_features(df)

    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(n_estimators=100)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    latest = X[-1].reshape(1,-1)

    predicted = model.predict(latest)[0]

    current = df["close"].iloc[-1]

    change = ((predicted-current)/current)*100

    return current, predicted, change, mae, df


# =============================
# Recommendation
# =============================
def recommendation(change):

    if change > 3:
        return "STRONG BUY 🟢"

    elif change > 1:
        return "BUY 🟢"

    elif change < -3:
        return "STRONG SELL 🔴"

    elif change < -1:
        return "SELL 🔴"

    else:
        return "HOLD ⚪"


# =============================
# UI
# =============================

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")

stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

symbol = stocks[stock_name]["symbol"]

logo = stocks[stock_name]["logo"]

st.image(logo, width=80)

st.subheader(f"{stock_name} ({symbol})")

if st.button("Analyze Stock"):

    with st.spinner("Fetching data..."):

        df = get_stock_data(symbol)

    if df is None:

        st.error("Failed to fetch stock data")

    else:

        current, predicted, change, mae, df = predict_price(df)

        action = recommendation(change)

        col1, col2, col3 = st.columns(3)

        col1.metric("Current Price", f"${current:.2f}")

        col2.metric("Predicted Price", f"${predicted:.2f}")

        col3.metric("Expected Change", f"{change:.2f}%")

        st.subheader("Trading Signal")

        st.success(action)

        st.write("Model Error (MAE):", round(mae,2))

        st.subheader("Stock Price Chart")

        fig = px.line(df.tail(120), y="close", title=f"{symbol} Price")

        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Last updated: {datetime.now()}")
