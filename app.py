import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

sns.set(style="whitegrid")

APP_DIR = os.path.dirname(__file__) if '__file__' in globals() else '.'
MODEL_PATH = os.path.join(APP_DIR, "irctc_rf_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "irctc_scaler.pkl")

# ------------------ Helpers ------------------ #
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

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if len(col) >= 2 and col[1] not in (None, ''):
                new_cols.append(col[1])
            else:
                new_cols.append("_".join([str(c) for c in col if c not in (None, '')]))
        df = df.copy()
        df.columns = new_cols
    return df

def canonical_map_column_name(name: str) -> str | None:
    s = str(name).lower()
    s_norm = re.sub(r'[^a-z0-9]+', '_', s)
    if 'adj' in s_norm and 'close' in s_norm:
        return 'Adj Close'
    if 'adjusted' in s_norm and 'close' in s_norm:
        return 'Adj Close'
    if 'open' in s_norm and 'close' not in s_norm:
        return 'Open'
    if 'high' in s_norm:
        return 'High'
    if 'low' in s_norm and 'close' not in s_norm:
        return 'Low'
    if 'volume' in s_norm or s_norm == 'vol':
        return 'Volume'
    if 'close' in s_norm:
        return 'Close'
    return None

def normalize_columns(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    df = flatten_multiindex_columns(df)
    orig_cols = list(df.columns)
    if debug:
        st.write("Original columns:", orig_cols)

    stripped = []
    for c in orig_cols:
        parts = re.split(r'[_\s\.-]+', str(c))
        stripped.append(parts[-1] if len(parts) > 1 else str(c))

    col_map = {}
    for orig, stripped_name in zip(orig_cols, stripped):
        mapped = canonical_map_column_name(stripped_name)
        if mapped:
            col_map[orig] = mapped

    if not col_map and len(orig_cols) == 1:
        col_map[orig_cols[0]] = 'Close'
        if debug:
            st.write("Single-column response detected — mapping to 'Close'.")

    if not col_map:
        for orig in orig_cols:
            mapped = canonical_map_column_name(orig)
            if mapped:
                col_map[orig] = mapped

    if not col_map:
        for orig in orig_cols:
            low = orig.lower()
            if 'open' in low and orig not in col_map:
                col_map[orig] = 'Open'
            elif 'high' in low and orig not in col_map:
                col_map[orig] = 'High'
            elif 'low' in low and 'close' not in low and orig not in col_map:
                col_map[orig] = 'Low'
            elif 'volume' in low and orig not in col_map:
                col_map[orig] = 'Volume'
            elif 'close' in low and orig not in col_map:
                col_map[orig] = 'Close'

    if debug:
        st.write("Column mapping inferred:", col_map)

    if col_map:
        df = df.rename(columns=col_map)
    return df

def make_unique_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols

def prepare_features(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    df = df.copy()
    if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
        df["Adj Close"] = df["Adj_Close"]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    if "Close" in df.columns:
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    else:
        if debug:
            st.write("prepare_features: 'Close' not found — cannot compute MA/Return.")

    if "Volume" in df.columns:
        df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()

    cols_to_fill = [c for c in ['MA5', 'MA10', 'MA20', 'Vol_MA5'] if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].fillna(method='bfill').fillna(method='ffill')

    return df

# ------------------ App ------------------ #
def main():
    st.title("📈 IRCTC / Stock Direction Prediction (Robust Version)")
    st.write("Fetch historical data, normalize columns, build features, and predict Up/Down.")

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
            st.error("Model or scaler missing. Please upload both files.")
            return

        try:
            raw = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, threads=False)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

        if raw is None or raw.empty:
            st.error("No data returned.")
            return

        raw = normalize_columns(raw, debug=debug)

        # Deduplicate column names
        raw.columns = make_unique_columns(raw.columns)

        if debug:
            st.write("Columns after normalization:", list(raw.columns))
            st.write("Raw head:")
            st.write(raw.head())

        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if 'Close' not in raw.columns and len(numeric_cols) == 1:
            raw.rename(columns={numeric_cols[0]: 'Close'}, inplace=True)

        if "Adj Close" in raw.columns and "Close" not in raw.columns:
            raw["Close"] = raw["Adj Close"]

        for c in ['Open', 'High', 'Low']:
            if c not in raw.columns and 'Close' in raw.columns:
                raw[c] = raw['Close']

        if 'Volume' not in raw.columns:
            raw['Volume'] = 0

        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_base = [c for c in base_cols if c not in raw.columns]
        if missing_base:
            st.error(f"Base OHLCV columns still missing: {missing_base}")
            return

        if "Adj Close" not in raw.columns:
            raw["Adj Close"] = raw["Close"]

        df = prepare_features(raw, debug=debug)

        features = ['Open','High','Low','Close','Adj Close','Volume','Return','MA5','MA10','MA20','Vol_MA5']
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        X = df[features].copy()
        valid_mask = X.notna().all(axis=1)
        if valid_mask.sum() == 0:
            st.error("No valid rows for prediction.")
            return

        X_scaled = scaler.transform(X.loc[valid_mask])
        preds = model.predict(X_scaled)

        df['Prediction'] = np.nan
        df.loc[valid_mask, 'Prediction'] = preds.astype(int)

        st.subheader("Prediction Results (last 20 rows)")
        st.write(df[['Close','Prediction']].tail(20))

        if not df['Prediction'].dropna().empty:
            last_pred = int(df['Prediction'].dropna().iloc[-1])
            if last_pred == 1:
                st.success("✅ Stock predicted to go UP next day.")
            else:
                st.info("🔻 Stock predicted to go DOWN next day.")

        st.download_button(
            "Download Predictions CSV",
            df.to_csv().encode('utf-8'),
            f"{stock_symbol}_predictions.csv",
            "text/csv"
        )

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
