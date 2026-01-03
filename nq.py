import pandas as pd
import pandas_ta_classic as ta
import vectorbt as vbt
import zstandard as zstd
import matplotlib.pyplot as plt
import glob

#Decompress hourly NQ OHLCV data
hourly_data =  'c:/Users/bennn/repos/Project-Quack/NQ_OHLCV_1h/glbx-mdp3-20100606-20251231.ohlcv-1h.csv.zst'
with open(hourly_data, 'rb') as binary:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(binary) as decompressed:
        hourlyOHLCV = pd.read_csv(decompressed)
        hourlyOHLCV['datetime'] = pd.to_datetime(hourlyOHLCV['ts_event'])
        hourlyOHLCV.set_index('datetime', inplace=True)

#Decompress minute NQ OHLCV data
#Contains multiple .zst files so runtime is slow
minute_data = sorted(glob.glob('c:/Users/bennn/repos/Project-Quack/NQ_OHLCV_1m/glbx-mdp3-*.ohlcv-1m.csv.zst'))
dfs= []
for file in minute_data:
    with open(file, 'rb') as binary:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(binary) as decompressed:
            df = pd.read_csv(decompressed)
            dfs.append(df)

minuteOHLCV = pd.concat(dfs, ignore_index=True)
minuteOHLCV['datetime'] = pd.to_datetime(minuteOHLCV['ts_event'])
minuteOHLCV.set_index('datetime', inplace=True)

#Hourly EMA, RSI, VWAP, BBANDS indicators
hourlyOHLCV['ema_fast'] = hourlyOHLCV.ta.ema(length=21)
hourlyOHLCV['ema_slow'] = hourlyOHLCV.ta.ema(length=55)
hourlyOHLCV['rsi'] = hourlyOHLCV.ta.rsi(length=14)
hourlyOHLCV['vwap'] = hourlyOHLCV.ta.vwap(close=hourlyOHLCV['close'], volume=hourlyOHLCV['volume'], anchor="D")

bbands = hourlyOHLCV.ta.bbands(length=20)
hourlyOHLCV['bb_upper'] = bbands.iloc[:, 2]
hourlyOHLCV['bb_lower'] = bbands.iloc[:, 0]

#Minute EMA, RSI, VWAP, BBANDS indicators
minuteOHLCV['ema_fast'] = minuteOHLCV.ta.ema(length=21)
minuteOHLCV['ema_slow'] = minuteOHLCV.ta.ema(length=55)
minuteOHLCV['rsi'] = minuteOHLCV.ta.rsi(length=14)
minuteOHLCV['vwap'] = minuteOHLCV.ta.vwap(close=minuteOHLCV['close'], volume=minuteOHLCV['volume'], anchor="D")

bbands = minuteOHLCV.ta.bbands(length=20)
minuteOHLCV['bb_upper'] = bbands.iloc[:, 2]
minuteOHLCV['bb_lower'] = bbands.iloc[:, 0]

# Remove all NaN values 
hourlyOHLCV = hourlyOHLCV.dropna()
minuteOHLCV = minuteOHLCV.dropna()

#Filter data spikes to keep price at a standard amount
standard_hourly = (hourlyOHLCV['close'] > 500) & (hourlyOHLCV['high'] < 30000) & (hourlyOHLCV['low'] > 100)
hourlyOHLCV = hourlyOHLCV[standard_hourly]

standard_minute = (minuteOHLCV['close'] > 500) & (minuteOHLCV['high'] < 30000) & (minuteOHLCV['low'] > 100)
minuteOHLCV = minuteOHLCV[standard_minute]

#Sort data and remove any duplicates
hourlyOHLCV = hourlyOHLCV.sort_index()
hourlyOHLCV = hourlyOHLCV[~hourlyOHLCV.index.duplicated(keep='first')]
minuteOHLCV = minuteOHLCV.sort_index()
minuteOHLCV = minuteOHLCV[~minuteOHLCV.index.duplicated(keep='first')]

#Make sures that start and end dates match between different timeframes
start_dt = max(hourlyOHLCV.index[0], minuteOHLCV.index[0])
end_dt = min(hourlyOHLCV.index[-1], minuteOHLCV.index[-1])
hourlyOHLCV = hourlyOHLCV.loc[start_dt:end_dt]
minuteOHLCV = minuteOHLCV.loc[start_dt:end_dt]

print("Hourly OHLCV Data")
print(hourlyOHLCV)
print("Minute OHLCV Data")
print(minuteOHLCV)

#Hourly data determine the trend: Bullish/Bearish
hourly_trend = (hourlyOHLCV['ema_fast'] > hourlyOHLCV['ema_slow'])
hourly_trend_minute = hourly_trend.reindex(minuteOHLCV.index, method='ffill')

#Minute data determines if price is oversold/overbought and if it stays that way
rsi_cross_up = (minuteOHLCV['rsi'] > 30) & (minuteOHLCV['rsi'].shift(1) <= 30)
rsi_cross_down = (minuteOHLCV['rsi'] < 70) & (minuteOHLCV['rsi'].shift(1) >= 70)

#Long: Hourly Trend: Bullish and RSI is oversold
entries_long = (
    hourly_trend_minute 
    & rsi_cross_up
)

#Short: Hourly Trend: Bearish and RSI is overbought
entries_short = (
    ~hourly_trend_minute
    & rsi_cross_down
)

#Exits when minute indicates a reversal as volatility is ending
exits_long = (minuteOHLCV['close'] > minuteOHLCV['bb_upper'])

exits_short = (minuteOHLCV['close'] < minuteOHLCV['bb_lower'])

#Backtesting
portfolio = vbt.Portfolio.from_signals(
    close=minuteOHLCV['close'],
    open=minuteOHLCV['open'],
    high=minuteOHLCV['high'],
    low=minuteOHLCV['low'],
    entries=entries_long,
    exits=exits_long,
    short_entries=entries_short,
    short_exits=exits_short,
    init_cash= 50000,
    freq="1min",
    fees=0.00002, 
    sl_trail=0.005, 
)
print(portfolio.stats())
