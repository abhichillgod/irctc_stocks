import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model & scaler
model = joblib.load("irctc_rf_model.pkl")
scaler = joblib.load("irctc_scaler.pkl")

def main():
    st.title("📈 Stock Market Prediction App")
    st.write("Predict whether a stock will go **Up** or **Down** using historical data.")

    # User Inputs
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., IRCTC.NS)", value="IRCTC.NS")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-06-01"))

    # Graph selection
    graph_choice = st.selectbox("Select Graph Type", [
        "Line Plot", "Scatter Plot", "Histogram", "Correlation Heatmap"
    ])

    if st.button("Fetch & Predict"):
        # Fetch stock data
        data = yf.download(stock_symbol, start=start_date, end=end_date)

        if data.empty:
            st.error("⚠️ No data found for the selected period.")
            return

        # Ensure Adj Close exists
        if "Adj Close" not in data.columns:
            data["Adj Close"] = data["Close"]

        # Feature engineering
        df = data.copy()
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
        df['MA10'] = df['Close'].rolling(window=10).mean().fillna(method='bfill')
        df['MA20'] = df['Close'].rolling(window=20).mean().fillna(method='bfill')
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean().fillna(method='bfill')

        # Features list
        features = ['Open','High','Low','Close','Adj Close','Volume','Return','MA5','MA10','MA20','Vol_MA5']

        # Check if all features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required columns in the dataset: {missing_features}")
            return

        # Prepare data
        X = df[features]
        X_scaled = scaler.transform(X)

        # Predictions
        predictions = model.predict(X_scaled)
        df['Prediction'] = predictions

        st.subheader("Prediction Summary")
        st.write(df[['Close', 'Prediction']])

        # Show last day's prediction
        last_pred = predictions[-1]
        if last_pred == 1:
            st.success("✅ Model predicts the stock will go **UP** next day.")
        else:
            st.error("⚠️ Model predicts the stock will go **DOWN** next day.")

        # Graph rendering
        st.subheader(f"📊 {graph_choice}")
        fig, ax = plt.subplots(figsize=(10, 5))

        if graph_choice == "Line Plot":
            sns.lineplot(x=df.index, y=df['Close'], ax=ax)
            ax.set_title("Closing Price Over Time")

        elif graph_choice == "Scatter Plot":
            sns.scatterplot(x=df['Volume'], y=df['Close'], hue=df['Prediction'], ax=ax)
            ax.set_title("Volume vs Close Price")

        elif graph_choice == "Histogram":
            sns.histplot(df['Close'], kde=True, ax=ax)
            ax.set_title("Distribution of Closing Prices")

        elif graph_choice == "Correlation Heatmap":
            corr = df[features].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlation Heatmap")

        st.pyplot(fig)

if __name__ == "__main__":
    main()
