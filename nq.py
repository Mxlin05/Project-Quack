import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import vectorbt as vbt
import zstandard as zstd

#Import NQ OHLCV data from yfinance (Will be replaced with attached OHLCV data)
ticker = yf.Ticker("NQ=F",)
nq = ticker.history(period="max",interval="1d")

#EMA, RSI, and VWARP indicators
nq['ema_fast'] = nq.ta.ema(length=21)
nq['ema_slow'] = nq.ta.ema(length=55)
nq['rsi'] = nq.ta.rsi(length=14)
nq['vwap'] = nq.ta.vwap(close = nq['Close'], volume = nq['Volume'], anchor = "D")
bbands = nq.ta.bbands(length=20)
nq['bb_upper'] = bbands.iloc[:, 2]  
nq['bb_lower'] = bbands.iloc[:, 0]  

nq.dropna(inplace=True) #Gets rid of NaN values

#Requirements for trade entries
entries_long = (
    (nq['ema_fast'] > nq['ema_slow']) & nq['rsi'].vbt.crossed_above(40) & (nq['Close'] > nq['vwap'])
)
exits_long = nq['ema_fast'].vbt.crossed_below(nq['ema_slow'])

entries_short = (
    (nq['ema_fast'] < nq['ema_slow']) & nq['rsi'].vbt.crossed_below(60) & (nq['Close'] < nq['vwap'])
)
exits_short = nq['ema_fast'].vbt.crossed_above(nq['ema_slow'])

#Backtesting; Returns portfolio detailing winrate, best/worst trades, total trades, etc
portfolio = vbt.Portfolio.from_signals(
    nq['Close'],
    entries=entries_long,
    exits=exits_long,
    short_entries=entries_short,
    short_exits=exits_short,
    freq="15min",
    fees=0.0005,  
)
print(portfolio.stats()) 


