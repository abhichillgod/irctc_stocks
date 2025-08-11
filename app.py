import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")

APP_DIR = os.path.dirname(__file__) if '__file__' in globals() else '.'
MODEL_PATH = os.path.join(APP_DIR, "irctc_rf_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "irctc_scaler.pkl")

# ------------------ Load model & scaler ------------------ #
def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model = None
    scaler = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load model from '{model_path}': {e}")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Could not load scaler from '{scaler_path}': {e}")
    return model, scaler

# ------------------ Flatten columns ------------------ #
def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if len(col) > 1 and col[1] else col[0] for col in df.columns]
    return df

# ------------------ Feature creation ------------------ #
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
        df["Adj Close"] = df["Adj_Close"]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Technical indicators if Close exists
    if "Close" in df.columns:
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    # Volume MA
    if "Volume" in df.columns:
        df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()

    # Fill NaNs only for existing cols
    cols_to_fill = [c for c in ['MA5', 'MA10', 'MA20', 'Vol_MA5'] if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].fillna(method='bfill').fillna(method='ffill')

    return df

# ------------------ Main app ------------------ #
def main():
    st.title("📈 IRCTC / Stock Direction Prediction (Error-Resilient)")
    st.write("Fetch historical data, build features, and predict Up/Down using the pre-trained model.")

    model, scaler = load_artifacts()

    stock_symbol = st.text_input("Stock symbol (Yahoo Finance format)", value="IRCTC.NS")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime("2024-06-01"))

    graph_choice = st.selectbox("Select graph", ["Line Plot", "Scatter Plot", "Histogram", "Correlation Heatmap"])
    debug = st.checkbox("Show debug info", value=False)

    if st.button("Fetch data & run prediction"):
        if model is None or scaler is None:
            st.error("Model or scaler is missing. Please upload 'irctc_rf_model.pkl' and 'irctc_scaler.pkl'.")
            return

        # Fetch from Yahoo Finance
        try:
            raw = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, threads=False)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

        if raw is None or raw.empty:
            st.error("No data returned. Check symbol and date range.")
            return

        # Flatten MultiIndex & strip ticker prefixes
        raw = flatten_multiindex_columns(raw)
        raw.columns = [col.split("_")[-1] for col in raw.columns]

        # If only Adj Close exists, fill in other OHLC
        if "Adj Close" in raw.columns and "Close" not in raw.columns:
            raw["Close"] = raw["Adj Close"]
        for col in ['Open', 'High', 'Low']:
            if col not in raw.columns and "Close" in raw.columns:
                raw[col] = raw["Close"]
        if "Volume" not in raw.columns:
            raw["Volume"] = 0

        if debug:
            st.write("Columns after preprocessing:", list(raw.columns))

        # Final base OHLCV check
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_base = [c for c in base_cols if c not in raw.columns]
        if missing_base:
            st.error(f"Base OHLCV columns still missing: {missing_base}")
            return

        # Ensure Adj Close exists
        if "Adj Close" not in raw.columns:
            raw["Adj Close"] = raw["Close"]

        # Feature creation
        df = prepare_features(raw)

        # Required features
        features = ['Open','High','Low','Close','Adj Close','Volume','Return','MA5','MA10','MA20','Vol_MA5']
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required columns after preprocessing: {missing}")
            return

        # Model input
        X = df[features].copy()
        valid_mask = X.notna().all(axis=1)
        if valid_mask.sum() == 0:
            st.error("No valid rows for prediction (contains NaNs).")
            return

        try:
            X_scaled = scaler.transform(X.loc[valid_mask])
            preds = model.predict(X_scaled)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        df['Prediction'] = np.nan
        df.loc[valid_mask, 'Prediction'] = preds.astype(int)

        # Results table
        st.subheader("Prediction Results (last 20 rows)")
        st.write(df[['Close','Prediction']].tail(20))

        # Last prediction message
        if not df['Prediction'].dropna().empty:
            last_pred = int(df['Prediction'].dropna().iloc[-1])
            if last_pred == 1:
                st.success("✅ Stock predicted to go UP next day.")
            else:
                st.info("🔻 Stock predicted to go DOWN next day.")

        # CSV download
        st.download_button(
            "Download Predictions CSV",
            df.to_csv().encode('utf-8'),
            f"{stock_symbol}_predictions.csv",
            "text/csv"
        )

        # Graph display
        st.subheader(f"📊 {graph_choice}")
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if graph_choice == "Line Plot":
                sns.lineplot(x=df.index, y=df['Close'], ax=ax)
                ax.set_title("Closing Price Over Time")
            elif graph_choice == "Scatter Plot":
                sns.scatterplot(x=df['Volume'], y=df['Close'], hue=df['Prediction'], ax=ax)
                ax.set_title("Volume vs Close")
            elif graph_choice == "Histogram":
                sns.histplot(df['Close'], kde=True, ax=ax)
                ax.set_title("Close Price Distribution")
            elif graph_choice == "Correlation Heatmap":
                sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()
