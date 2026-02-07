import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# 1. Define the Vibe: "I want Apple stock data for the last 5 years"
# Ticker: AAPL (Apple), NVDA (Nvidia), ^NSEI (Nifty 50)
ticker = "NVDA" 

print(f"Fetching data for {ticker}...")

# 2. The One-Liner (Pandas Magic)
# auto_adjust=True fixes split prices
df = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=True)

# Fix: yfinance returns MultiIndex columns, we just want Price for single ticker
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 3. Create a "Target" (What we want to predict)
# Predict if price will go UP tomorrow.
df['Tomorrow'] = df['Close'].shift(-1)
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

# 4. Clean it up
df = df.dropna()

print(df.head())
print(f"Data Ready! We have {len(df)} days of trading data.")

# 5. Feature Engineering
# Moving Averages
df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# Drop NaNs created by rolling windows
df = df.dropna()
print(f"Data after Feature Engineering: {len(df)} rows")

# 6. Prediction
# Split data
train = df.iloc[:-100]
test = df.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low", "MA50", "MA200"]

# Initialize Model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Train
model.fit(train[predictors], train["Target"])

# Predict
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Evaluate
precision = precision_score(test["Target"], preds)
print(f"Precision Score: {precision:.4f}")

# Save the augmented data
df.to_csv("market_data.csv")