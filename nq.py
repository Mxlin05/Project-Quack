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
hourlyOHLCV = hourlyOHLCV[hourlyOHLCV['close'] > 500] 
minuteOHLCV = minuteOHLCV.dropna()
minuteOHLCV = minuteOHLCV[minuteOHLCV['close'] > 500]
print(hourlyOHLCV.tail())
print(minuteOHLCV.tail())

#Align 1-minute data to 1-hour timeframe 
minuteOHLCV_hourly = minuteOHLCV.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'ema_fast': 'last',
    'ema_slow': 'last',
    'rsi': 'last',
    'vwap': 'last',
    'bb_upper': 'last',
    'bb_lower': 'last'
})

# Align both dataframes to the same datetime
common_index = hourlyOHLCV.index.intersection(minuteOHLCV_hourly.index)
hourlyOHLCV = hourlyOHLCV.loc[common_index]
minuteOHLCV_hourly = minuteOHLCV_hourly.loc[common_index]

#Requirements for trade entries
#Timeframes must align
entries_long = (
    (hourlyOHLCV['ema_fast'] > hourlyOHLCV['ema_slow']) 
    & (minuteOHLCV_hourly['ema_fast'] > minuteOHLCV_hourly['ema_slow'])
    & (hourlyOHLCV['rsi'] < 50)
    & (minuteOHLCV_hourly['rsi'] < 50)
)

entries_short = (
    (hourlyOHLCV['ema_fast'] < hourlyOHLCV['ema_slow']) 
    & (minuteOHLCV_hourly['ema_fast'] < minuteOHLCV_hourly['ema_slow'])
    & (hourlyOHLCV['rsi'] > 50)
    & (minuteOHLCV_hourly['rsi'] > 50)
)

#Exits when hourly indicates a reversal
exits_long = hourlyOHLCV['ema_fast'] < hourlyOHLCV['ema_slow']
exits_short = hourlyOHLCV['ema_fast'] > hourlyOHLCV['ema_slow']

#Backtesting
portfolio = vbt.Portfolio.from_signals(
    close=hourlyOHLCV['close'],
    open=hourlyOHLCV['open'],
    high=hourlyOHLCV['high'],
    low=hourlyOHLCV['low'],
    entries=entries_long,
    exits=exits_long,
    short_entries=entries_short,
    short_exits=exits_short,
    init_cash= 50000,
    freq="1h",
    fees=0.00002, 
    sl_stop=0.02, 
)
print(portfolio.stats())
