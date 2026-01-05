import yfinance as yf
import pandas as pd

# Download SPUS 3-year daily data
spus = yf.download("SPUS", period="3y", interval="1d")

# Organize data
spus.reset_index(inplace=True)
spus.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume"
}, inplace=True)
#from oldedst to newest
spus.sort_values("date", inplace=True)
spus.reset_index(drop=True, inplace=True)

#save the organized data
spus.to_csv("C:/Users/hesha/OneDrive/Desktop/SPUS_cleaned.csv", index=False)
print("CSV saved to Desktop!")


