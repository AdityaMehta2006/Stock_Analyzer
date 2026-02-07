import yfinance as yf
import pandas as pd

ticker = "AAPL"
start = "2020-01-01"
end = "2024-01-01"

print(f"Testing download for {ticker}...")
try:
    df = yf.download(ticker, start=start, end=end, progress=False)
    print("Download finished.")
    print("Empty?", df.empty)
    print("Columns:", df.columns)
    print(df.head())
except Exception as e:
    print("Caught exception during download:")
    print(e)
