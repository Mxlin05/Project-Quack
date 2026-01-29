import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import vectorbt as vbt
import zstandard as zstd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
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
hourlyDataFrame = hourlyDataFrame.loc[start_dt:end_dt]
minuteDataFrame = minuteDataFrame.loc[start_dt:end_dt]

#Print data
print("Hourly Data:")
print(hourlyDataFrame)
print("Minute Data:")
print(minuteDataFrame)

#Transform price and volume to log returns
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
hourlyDataFrame[ohlcv_cols] = np.log(hourlyDataFrame[ohlcv_cols] / hourlyDataFrame[ohlcv_cols].shift(1))
minuteDataFrame[ohlcv_cols] = np.log(minuteDataFrame[ohlcv_cols] / minuteDataFrame[ohlcv_cols].shift(1))
hourlyDataFrame.dropna(inplace=True)
minuteDataFrame.dropna(inplace=True)

#Splitting data into 3 iterations of 4 years training and 1 year of testing using TimeSeriesSplit
total_days = (hourlyDataFrame.index[-1] - hourlyDataFrame.index[0]).days
samples_per_year = len(hourlyDataFrame) / (total_days / 365.25)
four_years_samples = int(samples_per_year * 4)

tscv = TimeSeriesSplit(n_splits=10, test_size=int(samples_per_year), max_train_size=four_years_samples)

print(f"Total Hourly Samples: {len(hourlyDataFrame)}")
print(f"Approx. 4-Year Samples: {four_years_samples}")

for i, (train_index, test_index) in enumerate(tscv.split(hourlyDataFrame)):
    # Get datetime from hourly index to ensure alignment with minute data
    train_start = hourlyDataFrame.index[train_index[0]]
    train_end = hourlyDataFrame.index[train_index[-1]]
    test_start = hourlyDataFrame.index[test_index[0]]
    test_end = hourlyDataFrame.index[test_index[-1]]

    # Slice DataFrames using datetime 
    hourly_train = hourlyDataFrame.loc[train_start:train_end].copy()
    hourly_test = hourlyDataFrame.loc[test_start:test_end].copy()
    
    minute_train = minuteDataFrame.loc[train_start:train_end].copy()
    minute_test = minuteDataFrame.loc[test_start:test_end].copy()

    # Normalize using MinMaxScaler
    indicator_cols = ['ema_fast', 'ema_slow', 'rsi', 'vwap', 'bb_upper', 'bb_lower']
    features = ohlcv_cols + indicator_cols

    hourly_scaler = MinMaxScaler()
    hourly_train[features] = hourly_scaler.fit_transform(hourly_train[features])
    hourly_test[features] = hourly_scaler.transform(hourly_test[features])
    
    minute_scaler = MinMaxScaler()
    minute_train[features] = minute_scaler.fit_transform(minute_train[features])
    minute_test[features] = minute_scaler.transform(minute_test[features])

    #Print iterations dates and shapes
    print("Iteration: ", i + 1,)
    print(f"Train: {train_start} to {train_end}")
    print(f"Test:  {test_start} to {test_end}")

    print(f"Hourly Train Shape: {hourly_train.shape}, Test Shape: {hourly_test.shape}")
    print(f"Minute Train Shape: {minute_train.shape}, Test Shape: {minute_test.shape}")
    print("Sample Normalized Hourly Train Data:")
    print(hourly_train.head())


