import pandas as pd
import pandas_ta_classic as ta
import vectorbt as vbt
import zstandard as zstd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob

#Decompress hourly NQ OHLCV data
hourly_data =  'NQ_OHLCV_1h/glbx-mdp3-20100606-20251231.ohlcv-1h.csv.zst'
with open(hourly_data, 'rb') as binary:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(binary) as decompressed:
        hourlyDataFrame = pd.read_csv(decompressed)
        hourlyDataFrame['datetime'] = pd.to_datetime(hourlyDataFrame['ts_event'])
        hourlyDataFrame.set_index('datetime', inplace=True)

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
minuteDataFrame['datetime'] = pd.to_datetime(minuteDataFrame['ts_event'])
minuteDataFrame.set_index('datetime', inplace=True)

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
hourlyOHLCV = hourlyDataFrame.loc[start_dt:end_dt]
minuteOHLCV = minuteDataFrame.loc[start_dt:end_dt]

#Print data
print("Hourly OHLCV Data")
print(hourlyOHLCV.tail())
print("Minute OHLCV Data")
print(minuteOHLCV.tail())

#Transform price and volume to log returns
#Code not yet implemented

#Normalize the data
#Code not yet implemented
scaler = MinMaxScaler()

