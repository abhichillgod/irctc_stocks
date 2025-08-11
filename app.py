# app.py (fixed version) - copy this entire file content and replace your current app.py
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

def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Try to load model and scaler; show helpful errors if missing or incompatible."""
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

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If df.columns is a MultiIndex (e.g. from yfinance when multiple tickers requested),
    flatten it to single-level names using the second level when present (e.g. ('IRCTC.NS','Open') -> 'Open')."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if len(col) >= 2 and col[1] not in (None, ''):
                new_cols.append(col[1])
            else:
                joined = "_".join([str(c) for c in col if c not in (None, '')])
                new_cols.append(joined)
        df = df.copy()
        df.columns = new_cols
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
        df["Adj Close"] = df["Adj_Close"]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Create technical features only if base columns exist
    if "Close" in df.columns:
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    if "Volume" in df.columns:
        df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()

    # Fill NaNs only for columns that were actually created
    cols_to_fill = [c for c in ['MA5','MA10','MA20','Vol_MA5'] if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].fillna(method='bfill').fillna(method='ffill')

    return df

def main():
    st.title("📈 IRCTC / Stock Direction Prediction (Robust)")
    st.write("Fetch historical data, build features, and predict Up/Down using the pre-trained model. Toggle debug to see internal columns.")

    model, scaler = load_artifacts()

    stock_symbol = st.text_input("Stock symbol (Yahoo Finance format)", value="IRCTC.NS")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime("2024-06-01"))

    graph_choice = st.selectbox("Select graph to display", ["Line Plot", "Scatter Plot", "Histogram", "Correlation Heatmap"])
    debug = st.checkbox("Show debug info (columns / missing features)", value=False)

    if st.button("Fetch data & run prediction"):
        if model is None or scaler is None:
            st.error("Model or scaler is not available. Please upload 'irctc_rf_model.pkl' and 'irctc_scaler.pkl' to the app folder.")
            return

        try:
            raw = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, threads=False)
        except Exception as e:
            st.error(f"Error fetching data for '{stock_symbol}': {e}")
            return

        if raw is None or raw.empty:
            st.error("No data returned. Check the symbol and date range.")
            return

        raw = flatten_multiindex_columns(raw)
        if debug:
            st.write("Raw dataframe columns:", list(raw.columns))

        if "Adj Close" not in raw.columns and "Close" in raw.columns:
            raw["Adj Close"] = raw["Close"]

        df = prepare_features(raw)

        features = ['Open','High','Low','Close','Adj Close','Volume','Return','MA5','MA10','MA20','Vol_MA5']
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required columns after preprocessing: {missing}")
            if debug:
                st.write("Available columns:", list(df.columns))
            return

        X = df[features].copy()
        valid_mask = X.notna().all(axis=1)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            st.error("No valid rows available for prediction after feature construction (all rows contain NaNs). Try a longer date range.")
            if debug:
                st.write("X head:", X.head(10))
                st.write("X info:", X.info())
            return

        X_valid = X.loc[valid_mask]
        try:
            X_scaled = scaler.transform(X_valid)
        except Exception as e:
            st.error(f"Scaler.transform failed: {e}. Confirm the scaler was fitted on the same feature columns in the same order.")
            if debug:
                st.write("Expected feature count by scaler:", getattr(scaler, 'n_features_in_', 'unknown'))
                st.write("Feature columns being passed:", X_valid.columns.tolist())
            return

        try:
            preds = model.predict(X_scaled)
        except Exception as e:
            st.error(f"Model.predict failed: {e}. Confirm the loaded model is compatible.")
            return

        df['Prediction'] = np.nan
        df.loc[valid_mask, 'Prediction'] = preds.astype(int)

        st.subheader("Results (tail)")
        st.write(df[['Close','Prediction']].tail(20))

        if df['Prediction'].dropna().empty:
            st.warning("Model produced no predictions (unexpected).")
        else:
            last_pred = int(df['Prediction'].dropna().iloc[-1])
            if last_pred == 1:
                st.success("✅ Model predicts the stock will go **UP** for the next trading day (based on last valid row).")
            else:
                st.info("🔻 Model predicts the stock will go **DOWN** for the next trading day (based on last valid row).")

        csv = df.to_csv(index=True).encode('utf-8')
        st.download_button("Download processed data + predictions (CSV)", data=csv, file_name=f"{stock_symbol}_predictions.csv", mime="text/csv")

        st.subheader(f"📊 {graph_choice}")
        fig, ax = plt.subplots(figsize=(10,5))
        try:
            if graph_choice == "Line Plot":
                ax.plot(df.index, df['Close'], label='Close')
                ax.set_title("Closing Price Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
            elif graph_choice == "Scatter Plot":
                if 'Prediction' in df.columns:
                    scatter_df = df.dropna(subset=['Prediction'])
                    ax.scatter(scatter_df['Volume'], scatter_df['Close'], c=scatter_df['Prediction'].astype(int), cmap='coolwarm', alpha=0.8)
                    ax.set_xlabel("Volume")
                    ax.set_ylabel("Close Price")
                    ax.set_title("Volume vs Close (colored by prediction)")
                else:
                    ax.scatter(df['Volume'], df['Close'])
                    ax.set_title("Volume vs Close Price")
            elif graph_choice == "Histogram":
                ax.hist(df['Close'].dropna(), bins=30)
                ax.set_title("Distribution of Closing Prices")
                ax.set_xlabel("Close Price")
                ax.set_ylabel("Count")
            elif graph_choice == "Correlation Heatmap":
                corr = df[features].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Failed to render plot: {e}")
            if debug:
                import traceback
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
