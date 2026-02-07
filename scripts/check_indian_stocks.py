import yfinance as yf

# Try fetching Reliance Industries (NSE) and TCS (BSE)
tickers = ["RELIANCE.NS", "TCS.BO"]

print("Testing Indian Stock Support...")
for t in tickers:
    try:
        df = yf.download(t, period="5d", progress=False)
        if not df.empty:
            print(f"Successfully fetched {t}: {len(df)} rows")
            print(df.head())
        else:
            print(f"Fetched empty data for {t}")
    except Exception as e:
        print(f"Failed to fetch {t}: {e}")
