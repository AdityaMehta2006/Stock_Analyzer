import random
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from datetime import timedelta

def fetch_data(ticker, start="2020-01-01", end="2024-01-01", interval="1d"):
    """
    Fetches historical stock data from Yahoo Finance.
    Handles MultiIndex columns if present.
    Adapts start_date for hourly data (max 730 days).
    """
    # API Limitation: Hourly data only available for last 730 days
    if interval == "1h":
        max_start = pd.Timestamp.now() - timedelta(days=729)
        if pd.to_datetime(start) < max_start:
            start = max_start.strftime('%Y-%m-%d')
            
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        raise ValueError(f"Network error while fetching data: {e}")
    
    if df.empty:
        raise ValueError(f"Failed to fetch data for '{ticker}'. Check symbol or Internet.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Standardize Index
    df.index.name = "Date"
    
    # Remove Timezone (fixes issues with some pandas ops)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df

# ... (preprocess and add_features remain same, skipping to keep context small) ...

def train_predict(df, model_type="Random Forest", random_state=1, enable_tuning=False):
    """
    Trains model using Class Weights to fix 0% Precision.
    Supports: RF, XGB, Logistic, Stacking.
    """
    n_rows = len(df)
    
    # 1. Determine Split
    if n_rows > 2000:
        split_idx = -252 
    elif n_rows > 500:
        split_idx = -100
    else:
        split_idx = -int(n_rows * 0.2)
        if abs(split_idx) < 5: split_idx = -5 
            
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    if len(train) < 30:
         raise ValueError(f"Training set too small ({len(train)} rows). Expand date range.")
    
    # 2. Dynamic Predictors
    potential_cols = [
        "Close", "Volume", "Open", "High", "Low", "MA50", "MA200", 
        "RSI", "MACD", "BB_Upper", "BB_Lower", "ATR",
        "Trend_SMA50", "Trend_SMA200", "BB_Width", "Vol_Ratio",
        "Lag_1", "Lag_2", "Lag_5"
    ]
    predictors = [c for c in potential_cols if c in df.columns]
    
    # 3. Model Logic
    # Calculate scale_pos_weight for XGB (Negatives / Positives)
    # This forces the model to care about the minority 'Buy' class
    n_pos = train["Target"].sum()
    n_neg = len(train) - n_pos
    scale_weight = n_neg / n_pos if n_pos > 0 else 1
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=random_state, class_weight="balanced")
        
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=random_state)
        
    elif model_type == "Logistic Regression":
        # Great baseline, often better than Trees for simple trends
        model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
        
    elif model_type == "Stacked Ensemble":
        # User requested "XGBoost on top of Regression".
        # Stacking: L1=RF+XGB, L2(Final)=Logistic
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=random_state, class_weight="balanced")),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=random_state))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)

    model.fit(train[predictors], train["Target"])
    
    # Predictions
    preds = model.predict(test[predictors])
    preds_series = pd.Series(preds, index=test.index)
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test[predictors])[:, 1]
    else:
        # Some models might not have proba (rare for classifiers), fallback
        probs = preds
        
    probs_series = pd.Series(probs, index=test.index)
    precision = precision_score(test["Target"], preds_series, zero_division=0)
    
    return model, precision, preds_series, probs_series, test, predictors

def calculate_monthly_returns(df):
    """
    Resamples daily close data to month-end and calculates % change.
    Returns a pivot table suitable for a heatmap (Year x Month).
    """
    if "Close" not in df.columns:
        return pd.DataFrame()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)
        
    # Resample to month end
    monthly_data = df["Close"].resample("M").last()
    
    # Calculate % Change
    monthly_returns = monthly_data.pct_change() * 100
    
    # Create Year/Month columns
    returns_df = pd.DataFrame(monthly_returns).reset_index()
    
    # The reset_index() might name the column 'Date' or 'index' or 'Datetime'
    # We rename the first column to 'Date' to be safe
    returns_df.columns.values[0] = "Date"
    
    returns_df["Year"] = returns_df["Date"].dt.year
    returns_df["Month"] = returns_df["Date"].dt.month_name()
    returns_df["Month_Num"] = returns_df["Date"].dt.month
    
    # Pivot (Rows=Year, Cols=Month)
    heatmap_data = returns_df.pivot(index="Year", columns="Month_Num", values="Close")
    
    # Rename columns to names
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    heatmap_data.columns = [month_map.get(c, c) for c in heatmap_data.columns]
    
    return heatmap_data
    """
    Standardizes uploads to ensure required columns exist.
    Normalizes headers to 'Close', 'Open', 'High', 'Low', 'Volume'.
    """
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    
    col_map = {
        "date": "Date", "time": "Date",
        "close": "Close", "adj close": "Close", "last": "Close",
        "open": "Open", "high": "High", "low": "Low", "volume": "Volume"
    }
    df = df.rename(columns=col_map)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except:
             raise ValueError("CSV must contain a 'Date' column or a datetime index.")
             
    df = df.sort_index()
    
    if "Close" not in df.columns:
        raise ValueError("CSV missing required 'Close' column.")
    
    # Fill missing columns
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
    
    if "Volume" not in df.columns:
        df["Volume"] = 0
        
    return df

def preprocess_custom_data(uploaded_file):
    """
    Standardizes uploads to ensure required columns exist.
    """
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    
    col_map = {
        "date": "Date", "time": "Date",
        "close": "Close", "adj close": "Close", "last": "Close",
        "open": "Open", "high": "High", "low": "Low", "volume": "Volume"
    }
    df = df.rename(columns=col_map)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except:
             raise ValueError("CSV must contain a 'Date' column or a datetime index.")
             
    df = df.sort_index()
    
    if "Close" not in df.columns:
        raise ValueError("CSV missing required 'Close' column.")
    
    # Fill missing columns
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
    
    if "Volume" not in df.columns:
        df["Volume"] = 0
        
    return df

def add_features(df):
    """
    Generates technical indicators. 
    ADAPTIVE: Skips long-term indicators (MA200) if data is insufficient.
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided.")

    df = df.copy()
    n_rows = len(df)
    
    # Target Setup
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    # --- ADAPTIVE PARAMETERS ---
    # prevents "dropna" from wiping out entire small datasets
    if n_rows < 100:
        ma_short, ma_long = 5, 20
        rsi_window = 7
        bb_window = 10
        atr_window = 7
    else:
        ma_short, ma_long = 20, 50
        rsi_window = 14
        bb_window = 20
        atr_window = 14

    # Moving Averages
    df[f"MA{ma_long}"] = df["Close"].rolling(window=ma_long).mean()
    
    # Only calculate MA200 if we have enough data
    if n_rows > 250:
        df["MA200"] = df["Close"].rolling(window=200).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD (Standard config usually fine, but let's be safe on tiny data)
    if n_rows > 35:
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    else:
        df["MACD"] = 0 # Fallback
        df["Signal_Line"] = 0
    
    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=bb_window).mean()
    std_dev = df["Close"].rolling(window=bb_window).std()
    df["BB_Upper"] = df["BB_Mid"] + (2 * std_dev)
    df["BB_Lower"] = df["BB_Mid"] - (2 * std_dev)
    
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_window).mean()
    
    # --- Advanced Ratios ---
    df[f"Trend_MA{ma_long}"] = df["Close"] / df[f"MA{ma_long}"]
    if "MA200" in df.columns:
        df["Trend_SMA200"] = df["Close"] / df["MA200"]
    
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    
    # Lags: Only if we have decent history (avoid noise in very small datasets)
    if n_rows > 100:
        df["Lag_1"] = df["Close"] / df["Close"].shift(1)
        df["Lag_2"] = df["Close"] / df["Close"].shift(2)
        df["Lag_5"] = df["Close"] / df["Close"].shift(5)
    
    # Clean NaNs
    df_clean = df.dropna()
    
    # FAILSAFE: If dropna killed it, relax the drop (just forward fill)
    if df_clean.empty and n_rows > 10:
         df_clean = df.fillna(method='bfill').fillna(method='ffill').dropna()

    if df_clean.empty:
        raise ValueError(f"Data inadequate. Input: {len(df)} rows. Need at least 25 datapoints.")
        
    return df_clean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- LSTM & GRU Definitions ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return self.sigmoid(out)

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return self.sigmoid(out)

def set_global_seed(seed=42):
    """
    Sets seeds for all libraries to ensure deterministic results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fetch_data(ticker, start="2020-01-01", end="2024-01-01", interval="1d"):
    """
    Fetches historical stock data from Yahoo Finance.
    Handles MultiIndex columns if present.
    Adapts start_date for hourly (730d) and minute (7d) data.
    """
    # API Constraints
    now = pd.Timestamp.now()
    
    if interval == "1h":
        max_start = now - timedelta(days=729)
        if pd.to_datetime(start) < max_start:
            start = max_start.strftime('%Y-%m-%d')
            
    elif interval == "1m":
        # 1m data is very restricted (last 7 days)
        max_start = now - timedelta(days=6) # Safety buffer
        if pd.to_datetime(start) < max_start:
            start = max_start.strftime('%Y-%m-%d')
            
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        raise ValueError(f"Network error while fetching data: {e}")
    
    if df.empty:
        raise ValueError(f"Failed to fetch data for '{ticker}'. Check symbol or Internet.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Standardize Index
    df.index.name = "Date"
    
    # Remove Timezone (fixes issues with some pandas ops)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df

def create_sequences(data, target, seq_length=10):
    """
    Converts tabular data into sequences.
    X: (N, Seq_Len, Features)
    y: (N,)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target.iloc[i + seq_length] 
        xs.append(x) # Data is already numpy array, removed .values field
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_predict_deep(df, predictors, model_arch="LSTM", target_col="Target", seq_len=10, epochs=50, random_state=42):
    """
    Handles training for LSTM and GRU on GPU if available.
    """
    # 0. Determinism
    set_global_seed(random_state)
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 2. Scale Data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[predictors])
    
    # 3. Split
    split_idx = int(len(df) * 0.8)
    train_data = scaled_features[:split_idx]
    test_data = scaled_features[split_idx:]
    
    train_target = df[target_col].iloc[:split_idx]
    test_target = df[target_col].iloc[split_idx:]
    
    # 4. Create Sequences
    X_train, y_train = create_sequences(train_data, train_target, seq_len)
    X_test, y_test = create_sequences(test_data, test_target, seq_len)
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Data too small for sequence generation.")

    # 5. Tensors -> GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 6. Model Selection
    if model_arch == "GRU":
        model = GRUClassifier(input_dim=len(predictors)).to(device)
    else:
        model = LSTMClassifier(input_dim=len(predictors)).to(device)
        
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for __ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    # 7. Predict
    model.eval()
    with torch.no_grad():
        # Move back to CPU for numpy conversion
        probs = model(X_test_t).cpu().squeeze().numpy()
        
    preds = (probs > 0.5).astype(int)
    
    # Index Alignment
    actual_test_idx = df.index[split_idx + seq_len:]
    actual_test_idx = actual_test_idx[:len(preds)]
    
    # Return standard outputs
    preds_series = pd.Series(preds, index=actual_test_idx)
    probs_series = pd.Series(probs, index=actual_test_idx)
    precision = precision_score(df.loc[actual_test_idx, target_col], preds, zero_division=0)
    
    # Return slice of DF
    test_df_sliced = df.loc[actual_test_idx]
    
    return model, precision, preds_series, probs_series, test_df_sliced, predictors

def train_predict(df, model_type="Random Forest", random_state=1, enable_tuning=False):
    """
    Trains model using Class Weights.
    Supports: RF, XGB, Logistic, Stacking, LSTM, GRU, Voting.
    """
    # ... (Pre-setup same) ...
    n_rows = len(df)
    
    # 1. Determine Split (Regular)
    if n_rows > 2000: split_idx = -252
    elif n_rows > 500: split_idx = -100
    else: split_idx = -int(n_rows * 0.2)
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # 2. Predictors
    potential_cols = [
        "Close", "Volume", "Open", "High", "Low", "MA50", "MA200", 
        "RSI", "MACD", "BB_Upper", "BB_Lower", "ATR",
        "Trend_SMA50", "Trend_SMA200", "BB_Width", "Vol_Ratio",
        "Lag_1", "Lag_2", "Lag_5"
    ]
    predictors = [c for c in potential_cols if c in df.columns]
    
    # --- BRANCH: DEEP LEARNING ---
    if "LSTM" in model_type:
        return train_predict_deep(df, predictors, model_arch="LSTM")
    if "GRU" in model_type:
        return train_predict_deep(df, predictors, model_arch="GRU")
        
    # --- BRANCH: VOTING ENSEMBLE (RF + XGB + LSTM) ---
    if model_type == "Voting (RF + XGB + LSTM)":
        # 1. Train RF
        rf_model, _, rf_preds, rf_probs, _, _ = train_predict(df, "Random Forest", random_state, enable_tuning)
        # 2. Train XGB
        xgb_model, _, xgb_preds, xgb_probs, _, _ = train_predict(df, "XGBoost", random_state, enable_tuning)
        # 3. Train LSTM
        lstm_model, _, lstm_preds, lstm_probs, lstm_test_df, _ = train_predict_deep(df, predictors, "LSTM")
        
        # Align Indices (LSTM index is shorter)
        common_idx = rf_probs.index.intersection(lstm_probs.index)
        
        rf_p = rf_probs.loc[common_idx]
        xgb_p = xgb_probs.loc[common_idx]
        lstm_p = lstm_probs.loc[common_idx]
        
        # Average Probabilities (Soft Voing)
        final_probs = (rf_p + xgb_p + lstm_p) / 3
        final_preds = (final_probs > 0.5).astype(int)
        
        precision = precision_score(df.loc[common_idx, "Target"], final_preds, zero_division=0)
        
        # Return LSTM slice as it's the constraint
        return "Voting Ensemble", precision, final_preds, final_probs, lstm_test_df, predictors

    # --- BRANCH: STANDARD MODELS ---
    
    if len(train) < 30:
         raise ValueError(f"Training set too small ({len(train)} rows). Expand date range.")
    
    # 3. Model Logic
    n_pos = train["Target"].sum()
    n_neg = len(train) - n_pos
    scale_weight = n_neg / n_pos if n_pos > 0 else 1
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=random_state, class_weight="balanced")
        
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=random_state)
        
    elif model_type == "Logistic Regression":
        model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
        
    elif model_type == "Stacked Ensemble":
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=random_state, class_weight="balanced")),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=random_state))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)

    model.fit(train[predictors], train["Target"])
    
    # Predictions
    preds = model.predict(test[predictors])
    preds_series = pd.Series(preds, index=test.index)
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test[predictors])[:, 1]
    else:
        probs = preds
        
    probs_series = pd.Series(probs, index=test.index)
    precision = precision_score(test["Target"], preds_series, zero_division=0)
    
    return model, precision, preds_series, probs_series, test, predictors
