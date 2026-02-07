import yfinance as yf
import json

def test_indian_news():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    print(f"Testing news retrieval for: {tickers}")
    
    for t in tickers:
        print(f"\n--- {t} ---")
        try:
            ticker = yf.Ticker(t)
            news = ticker.news
            if news:
                print(f"Found {len(news)} news items.")
                # Debug: Print first item structure
                print(json.dumps(news[0], indent=2))
            else:
                print("No news found via yfinance.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_indian_news()
