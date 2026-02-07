import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import zstandard as zstd
import glob

#Decompress hourly NQ OHLCV data
hourly_data = './raw_OHLCV_data/NQ_OHLCV_1h/glbx-mdp3-20100606-20251231.ohlcv-1h.csv.zst'
with open(hourly_data, 'rb') as binary:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(binary) as decompressed:
        hourlyDataFrame = pd.read_csv(decompressed)
        hourlyDataFrame['date_time'] = pd.to_datetime(hourlyDataFrame['ts_event'])
        hourlyDataFrame.drop(columns=['ts_event','rtype','publisher_id','instrument_id','symbol'], inplace=True)
        hourlyDataFrame.set_index('date_time', inplace=True)

#Decompress minute NQ OHLCV data. Contains multiple .zst files so runtime is slow
minute_data = sorted(glob.glob('./raw_OHLCV_data/NQ_OHLCV_1m/glbx-mdp3-*.ohlcv-1m.csv.zst'))
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

#Create a copy of the raw data before transformation
raw_hour_dataframe = hourlyDataFrame.copy()
raw_minute_dataframe = minuteDataFrame.copy()

#Transform features to log returns, relative closes, moving averages, etc
def transform(df):
    #Log returns of closing
    close = np.log(df['close'] / df['close'].shift(1))
    
    #Relative to closing price
    high = (df['high'] - df['close']) / df['close']
    low  = (df['low'] - df['close']) / df['close']
    open = (df['open'] - df['close']) / df['close']
    
    # Log change of volume
    new_volume = np.log(df['volume'] + 1).pct_change()
    
    # Distance to price
    ema_fast = (df['close'] - df['ema_fast']) / df['ema_fast']
    ema_slow = (df['close'] - df['ema_slow']) / df['ema_slow']
    vwap     = (df['close'] - df['vwap']) / df['vwap']
    
    #Compress both upper and lower bands and then converted to a percent change
    bbands = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI scaled
    rsi = df['rsi'] / 100.0

    df['close']    = close
    df['high']     = high
    df['low']      = low
    df['open']     = open
    df['volume']   = new_volume
    df['ema_fast'] = ema_fast
    df['ema_slow'] = ema_slow
    df['vwap']     = vwap
    df['rsi']      = rsi
    df['bbands'] = bbands
    
    df.drop(columns=['bb_lower','bb_upper'], inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

minuteDataFrame = transform(minuteDataFrame)
hourlyDataFrame = transform(hourlyDataFrame)

print("Hourly Columns:", hourlyDataFrame.columns.tolist())
print("Minute Columns:", minuteDataFrame.columns.tolist())
