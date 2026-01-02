import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import vectorbt as vbt

#libraries' versions
print("Pandas TA version: " + ta.version) 
print("Pandas version: " + pd.__version__)
print("Yfinance version: " + yf.__version__)
print("Numpy version: " + np.__version__)

#Import NQ OHLCV data from yfinance
ticker = yf.Ticker("NQ=F",)
nq = ticker.history(period="max",interval="1d")

#EMA, RSI, and VWARP indicators
nq['ema_fast'] = nq.ta.ema(length=21)
nq['ema_slow'] = nq.ta.ema(length=55)
nq['rsi'] = nq.ta.rsi(length=14)

tp = (nq['High'] + nq['Low'] + nq['Close']) / 3
vol = nq['Volume']
nq['session_vwap'] = (
    (tp * vol).groupby(nq.index.date).cumsum()/vol.groupby(nq.index.date).cumsum()
)
def zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate the rolling z-score of a pandas Series."""
    m = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - m) / (std + 1e-8) # Adding a small constant to avoid division by zero
    

nq['ema_spread'] = (nq['ema_fast'] - nq['ema_slow']) / nq['Close']
nq['ema_spread_zscore'] = zscore(nq['ema_spread'], window=100)

nq['vwap_dist'] = (nq['Close'] - nq['session_vwap']) / nq['Close']
nq['vwap_dist_zscore'] = zscore(nq['vwap_dist'], window=100)

nq.dropna(inplace=True) #Gets rid of NaN values

#Requirements for trade entries
entries_long = (
    (nq['ema_spread_zscore'] > .5) & 
    nq['rsi'].vbt.crossed_above(40) & 
    (nq['vwap_dist_zscore'] > 0)
)
exits_long = nq['ema_fast'].vbt.crossed_below(nq['ema_slow'])

entries_short = (
    (nq['ema_spread_zscore'] < -.5) & 
    nq['rsi'].vbt.crossed_below(60) & 
    (nq['vwap_dist_zscore'] < 0)
)
exits_short = nq['ema_fast'].vbt.crossed_above(nq['ema_slow'])

#Backtesting; Returns portfolio detailing winrate, best/worst trades, total trades, etc
portfolio = vbt.Portfolio.from_signals(
    nq['Close'],
    entries=entries_long,
    exits=exits_long,
    short_entries=entries_short,
    short_exits=exits_short,
    freq="1D", #fixed inconsistency 
    fees=0.0005,  
)

print(portfolio.stats()) 


