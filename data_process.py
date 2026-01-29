import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import zstandard as zstd
import glob

#Decompress hourly NQ OHLCV data
hourly_data =  'NQ_OHLCV_1h/glbx-mdp3-20100606-20251231.ohlcv-1h.csv.zst'
with open(hourly_data, 'rb') as binary:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(binary) as decompressed:
        hourlyDataFrame = pd.read_csv(decompressed)
        hourlyDataFrame['date_time'] = pd.to_datetime(hourlyDataFrame['ts_event'])
        hourlyDataFrame.drop(columns=['ts_event','rtype','publisher_id','instrument_id','symbol'], inplace=True)
        hourlyDataFrame.set_index('date_time', inplace=True)

#Decompress minute NQ OHLCV data. Contains multiple .zst files so runtime is slow
minute_data = sorted(glob.glob('NQ_OHLCV_1m/glbx-mdp3-*.ohlcv-1m.csv.zst'))
dfs= []
for file in minute_data:
    with open(file, 'rb') as binary:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(binary) as decompressed:
            df = pd.read_csv(decompressed)
            dfs.append(df)

minuteDataFrame = pd.concat(dfs, ignore_index=True)
minuteDataFrame['date_time'] = pd.to_datetime(minuteDataFrame['ts_event'])
minuteDataFrame.drop(columns=['ts_event','rtype','publisher_id','instrument_id','symbol'], inplace=True)
minuteDataFrame.set_index('date_time', inplace=True)

#Hourly EMA, RSI, VWAP, BBANDS indicators
hourlyDataFrame['ema_fast'] = hourlyDataFrame.ta.ema(length=21)
hourlyDataFrame['ema_slow'] = hourlyDataFrame.ta.ema(length=55)
hourlyDataFrame['rsi'] = hourlyDataFrame.ta.rsi(length=14)
hourlyDataFrame['vwap'] = hourlyDataFrame.ta.vwap(close=hourlyDataFrame['close'], volume=hourlyDataFrame['volume'], anchor="D")

bbands = hourlyDataFrame.ta.bbands(length=20)
hourlyDataFrame['bb_upper'] = bbands.iloc[:, 2]
hourlyDataFrame['bb_lower'] = bbands.iloc[:, 0]

#Minute EMA, RSI, VWAP, BBANDS indicators
minuteDataFrame['ema_fast'] = minuteDataFrame.ta.ema(length=21)
minuteDataFrame['ema_slow'] = minuteDataFrame.ta.ema(length=55)
minuteDataFrame['rsi'] = minuteDataFrame.ta.rsi(length=14)
minuteDataFrame['vwap'] = minuteDataFrame.ta.vwap(close=minuteDataFrame['close'], volume=minuteDataFrame['volume'], anchor="D")

bbands = minuteDataFrame.ta.bbands(length=20)
minuteDataFrame['bb_upper'] = bbands.iloc[:, 2]
minuteDataFrame['bb_lower'] = bbands.iloc[:, 0]


#Preprocessing data before normalization
#Remove all NaN values 
hourlyDataFrame = hourlyDataFrame.dropna()
minuteDataFrame = minuteDataFrame.dropna()

#Filter out data spikes 
filtered_hourly = (hourlyDataFrame['close'] > 500) & (hourlyDataFrame['high'] < 30000) & (hourlyDataFrame['low'] > 100)
hourlyDataFrame = hourlyDataFrame[filtered_hourly]

filtered_minute = (minuteDataFrame['close'] > 500) & (minuteDataFrame['high'] < 30000) & (minuteDataFrame['low'] > 100)
minuteDataFrame = minuteDataFrame[filtered_minute]

#Sort data and remove any duplicates
hourlyDataFrame = hourlyDataFrame.sort_index()
hourlyDataFrame = hourlyDataFrame[~hourlyDataFrame.index.duplicated(keep='first')]
minuteDataFrame = minuteDataFrame.sort_index()
minuteDataFrame = minuteDataFrame[~minuteDataFrame.index.duplicated(keep='first')]

#Align datetime between different timeframes
start_dt = max(hourlyDataFrame.index[0], minuteDataFrame.index[0])
end_dt = min(hourlyDataFrame.index[-1], minuteDataFrame.index[-1])
hourlyDataFrame = hourlyDataFrame.loc[start_dt:end_dt]
minuteDataFrame = minuteDataFrame.loc[start_dt:end_dt]

#Transform price and volume to log returns
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
hourlyDataFrame[ohlcv_cols] = np.log(hourlyDataFrame[ohlcv_cols] / hourlyDataFrame[ohlcv_cols].shift(1))
minuteDataFrame[ohlcv_cols] = np.log(minuteDataFrame[ohlcv_cols] / minuteDataFrame[ohlcv_cols].shift(1))
hourlyDataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
minuteDataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
hourlyDataFrame.dropna(inplace=True)
minuteDataFrame.dropna(inplace=True)

print("Hourly Columns:", hourlyDataFrame.columns.tolist())
print("Minute Columns:", minuteDataFrame.columns.tolist())
