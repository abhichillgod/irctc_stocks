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

# ---------------- Utilities ---------------- #
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

def canonical_map_column_name(name: str):
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
    """Try multiple strategies to rename columns to canonical OHLCV names."""
    df = flatten_multiindex_columns(df)
    orig_cols = list(df.columns)
    if debug:
        st.write("Original columns:", orig_cols)

    # Strategy 1: strip prefixes like "IRCTC.NS_Open" -> "Open"
    stripped = []
    for c in orig_cols:
        parts = re.split(r'[_\s\.-]+', str(c))
        stripped.append(parts[-1] if len(parts) > 1 else str(c))

    col_map = {}
    for orig, stripped_name in zip(orig_cols, stripped):
        mapped = canonical_map_column_name(stripped_name)
        if mapped:
            col_map[orig] = mapped

    # If nothing found, attempt mapping on original names
    if not col_map:
        for orig in orig_cols:
            mapped = canonical_map_column_name(orig)
            if mapped:
                col_map[orig] = mapped

    # Last pass: substring matching
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
        st.write("Inferred column map:", col_map)

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure unique columns
    df.columns = make_unique_columns(df.columns)
    return df

def pick_close_candidate(df: pd.DataFrame, debug: bool=False):
    """Heuristic: choose the most-likely 'Close' column if none exists."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if debug:
        st.write("Numeric columns available for Close selection:", numeric_cols)
    if not numeric_cols:
        return None
    # Prefer columns containing 'close' or 'adj'
    for c in numeric_cols:
        if 'close' in str(c).lower():
            return c
    for c in numeric_cols:
        if 'adj' in str(c).lower():
            return c
    # Otherwise pick the numeric column with largest mean (likely price)
    means = {c: df[c].replace([np.inf, -np.inf], np.nan).dropna().mean() for c in numeric_cols}
    # filter out columns with no data
    means = {c: m for c, m in means.items() if not pd.isna(m)}
    if not means:
        return None
    # choose candidate with max mean
    candidate = max(means.items(), key=lambda x: x[1])[0]
    if debug:
        st.write("Chosen close candidate:", candidate, "mean:", means[candidate])
    return candidate

def prepare_features(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    df = df.copy()
    # Normalize adj-close variants
    if "Adj Close" not in df.columns and "Adj_Close" in df.columns:
        df["Adj Close"] = df["Adj_Close"]
    # Copy Close if Adj only
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    # Calculate indicators if Close exists
    if "Close" in df.columns:
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    else:
        if debug:
            st.write("prepare_features: no 'Close' present yet.")
    if "Volume" in df.columns:
        df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    cols_to_fill = [c for c in ['MA5', 'MA10', 'MA20', 'Vol_MA5'] if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].fillna(method='bfill').fillna(method='ffill')
    return df

# ---------------- Fetch with retries & normalize ---------------- #
def fetch_and_normalize(symbol, start_date, end_date, debug=False):
    attempts = []
    # Attempt 1 - standard download
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False, auto_adjust=False)
        attempts.append(("download", df))
    except Exception as e:
        attempts.append(("download_error", e))
        df = None

    # If df is empty or has zero columns, try ticker.history
    if df is None or df.empty or df.shape[1] == 0:
        try:
            t = yf.Ticker(symbol)
            df2 = t.history(start=start_date, end=end_date, interval='1d', actions=False, auto_adjust=False)
            attempts.append(("ticker_history", df2))
            df = df2 if (df2 is not None and not df2.empty) else df
        except Exception as e:
            attempts.append(("ticker_history_error", e))

    # Another attempt: download with period fallback
    if df is None or df.empty or df.shape[1] == 0:
        try:
            df3 = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False, auto_adjust=True)
            attempts.append(("download_auto_adjust", df3))
            df = df3 if (df3 is not None and not df3.empty) else df
        except Exception as e:
            attempts.append(("download_auto_adjust_error", e))

    if df is None or df.empty or df.shape[1] == 0:
        # give up: return df (likely empty) and attempts for debug
        return df, attempts

    # Normalize columns
    df = normalize_columns(df, debug=debug)

    # Ensure unique columns again (normalizer does but double-check)
    df.columns = make_unique_columns(df.columns)

    # If only numeric single column available, rename to Close
    if 'Close' not in df.columns:
        cand = pick_close_candidate(df, debug=debug)
        if cand:
            df = df.rename(columns={cand: 'Close'})
            if debug:
                st.write(f"Renamed column '{cand}' -> 'Close' (heuristic).")

    # If we have Adj Close but not Close, copy
    if 'Adj Close' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['Adj Close']
        if debug:
            st.write("Copied 'Adj Close' -> 'Close'.")

    # Reconstruct OHLC from Close if missing (best-effort)
    for c in ['Open', 'High', 'Low']:
        if c not in df.columns and 'Close' in df.columns:
            df[c] = df['Close']
            if debug:
                st.write(f"Reconstructed '{c}' from Close (best-effort).")

    # Ensure Volume exists
    if 'Volume' not in df.columns:
        df['Volume'] = 0
        if debug:
            st.write("Inserted Volume = 0 (missing).")

    # Deduplicate columns again after manipulations
    df.columns = make_unique_columns(df.columns)

    return df, attempts

# ---------------- Main app ---------------- #
def main():
    st.title("📈 IRCTC / Stock Direction Prediction (Final robust)")
    st.write("Aggressively normalizes yahoo-finance output and reconstructs missing OHLCV where possible.")

    model, scaler = load_artifacts()

    stock_symbol = st.text_input("Stock symbol (Yahoo Finance)", value="IRCTC.NS")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime("2024-06-01"))

    graph_choice = st.selectbox("Graph", ["Line Plot", "Scatter Plot", "Histogram", "Correlation Heatmap"])
    debug = st.checkbox("Show debug info", value=False)

    if st.button("Fetch & Predict"):
        if model is None or scaler is None:
            st.error("Model/scaler missing. Upload 'irctc_rf_model.pkl' and 'irctc_scaler.pkl'.")
            return

        df_raw, attempts = fetch_and_normalize(stock_symbol, start_date, end_date, debug=debug)

        if df_raw is None or df_raw.empty:
            st.error("No usable data returned from Yahoo Finance. Attempts:")
            if debug:
                st.write(attempts)
            return

        if debug:
            st.write("Columns after fetch/normalize:", list(df_raw.columns))
            # show a small sample
            try:
                st.write(df_raw.head())
            except Exception:
                st.write("Cannot display raw sample (duplicate columns etc). Showing columns only.")

        # Final check for base columns
        base_cols = ['Open', 'High', 'Low', 'Close']
        missing_base = [c for c in base_cols if c not in df_raw.columns]
        if missing_base:
            # At this point we've already tried heuristics. If still missing, show errors & debug.
            st.error(f"Base OHLC columns still missing: {missing_base}")
            st.write("Enable debug and paste the printed columns & attempts here so I can add a direct mapping.")
            if debug:
                st.write("Attempts summary (what we tried):", attempts)
            return

        # Ensure Adj Close and Volume exist
        if 'Adj Close' not in df_raw.columns:
            df_raw['Adj Close'] = df_raw['Close']
        if 'Volume' not in df_raw.columns:
            df_raw['Volume'] = 0

        # Create final features
        df = prepare_features(df_raw, debug=debug)

        features = ['Open','High','Low','Close','Adj Close','Volume','Return','MA5','MA10','MA20','Vol_MA5']
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required features after preprocessing: {missing}")
            if debug:
                st.write("Columns available:", list(df.columns))
            return

        # Prepare X and predict
        X = df[features].copy()
        valid_mask = X.notna().all(axis=1)
        if valid_mask.sum() == 0:
            st.error("No valid rows for prediction.")
            return

        X_valid = X.loc[valid_mask]
        try:
            X_scaled = scaler.transform(X_valid)
            preds = model.predict(X_scaled)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            if debug:
                st.write("Scaler n_features_in_:", getattr(scaler, 'n_features_in_', 'unknown'))
                st.write("Features passed:", X_valid.columns.tolist())
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
                st.success("✅ Model predicts UP next day.")
            else:
                st.info("🔻 Model predicts DOWN next day.")

        st.download_button("Download CSV", df.to_csv().encode('utf-8'),
                           file_name=f"{stock_symbol}_predictions.csv", mime="text/csv")

        # Plot
        st.subheader(f"📊 {graph_choice}")
        fig, ax = plt.subplots(figsize=(10,5))
        try:
            if graph_choice == "Line Plot":
                sns.lineplot(x=df.index, y=df['Close'], ax=ax)
                ax.set_title("Close Over Time")
            elif graph_choice == "Scatter Plot":
                sns.scatterplot(x=df['Volume'], y=df['Close'], hue=df['Prediction'], ax=ax)
                ax.set_title("Volume vs Close")
            elif graph_choice == "Histogram":
                sns.histplot(df['Close'].dropna(), kde=True, ax=ax)
                ax.set_title("Close Distribution")
            elif graph_choice == "Correlation Heatmap":
                sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
