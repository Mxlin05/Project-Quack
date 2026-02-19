import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import tensorflow as tf
import joblib
from numpy.lib.stride_tricks import sliding_window_view
from keras.models import load_model
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import vectorbt as vbt

#Create dataframes for both the minute and hour
data = yf.Ticker("NQ=F")

minute_dataframe = data.history(period="7d",interval="1m")
hour_dataframe = data.history(period="7d",interval = "1h")

#Change column names to lower case and drop unnecessary columns
minute_dataframe = minute_dataframe.rename(columns=str.lower)
hour_dataframe = hour_dataframe.rename(columns=str.lower)

minute_dataframe.drop(columns=['dividends', 'stock splits'], inplace=True)
hour_dataframe.drop(columns=['dividends', 'stock splits'], inplace=True)

#Create minute indicators
minute_dataframe['ema_fast'] = minute_dataframe.ta.ema(close=minute_dataframe['close'], length=21)
minute_dataframe['ema_slow'] = minute_dataframe.ta.ema(length=55)
minute_dataframe['rsi'] = minute_dataframe.ta.rsi(length=14)
minute_dataframe['vwap'] = minute_dataframe.ta.vwap(close=minute_dataframe['close'], volume=minute_dataframe['volume'], anchor="D")

bbands = minute_dataframe.ta.bbands(length=20)
minute_dataframe['bb_upper'] = bbands.iloc[:, 2]
minute_dataframe['bb_lower'] = bbands.iloc[:, 0]

#Create hour indicators
hour_dataframe['ema_fast'] = hour_dataframe.ta.ema(close=hour_dataframe['close'], length=21)
hour_dataframe['ema_slow'] = hour_dataframe.ta.ema(length=55)
hour_dataframe['rsi'] = hour_dataframe.ta.rsi(length=14)
hour_dataframe['vwap'] = hour_dataframe.ta.vwap(close=hour_dataframe['close'], volume=hour_dataframe['volume'], anchor="D")

bbands = hour_dataframe.ta.bbands(length=20)
hour_dataframe['bb_upper'] = bbands.iloc[:, 2]
hour_dataframe['bb_lower'] = bbands.iloc[:, 0]

#Drop NaN values
hour_dataframe = hour_dataframe.dropna()    
minute_dataframe = minute_dataframe.dropna()

#Sort data and remove any duplicates
hour_dataframe = hour_dataframe.sort_index()
hour_dataframe = hour_dataframe[~hour_dataframe.index.duplicated(keep='first')]
minute_dataframe = minute_dataframe.sort_index()
minute_dataframe = minute_dataframe[~minute_dataframe.index.duplicated(keep='first')]

#Align datetime between different timeframes
start_dt = max(hour_dataframe.index[0], hour_dataframe.index[0])
end_dt = min(hour_dataframe.index[-1], hour_dataframe.index[-1])
minute_dataframe = minute_dataframe.loc[start_dt:end_dt]
hour_dataframe = hour_dataframe.loc[start_dt:end_dt]

raw_dataframe = minute_dataframe.copy()

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

minute_dataframe = transform(minute_dataframe)
hour_dataframe = transform(hour_dataframe)

#Normalize features
FEATURES = ['open', 'high', 'low', 'close','volume', 'ema_fast', 'ema_slow','rsi', 'vwap', 'bbands']

minute_scaler = joblib.load('scalers/minute_scaler_10.pkl')
hourly_scaler = joblib.load('scalers/hourly_scaler_10.pkl')

minute_dataframe[FEATURES] = minute_scaler.transform(minute_dataframe[FEATURES])
hour_dataframe[FEATURES] = hourly_scaler.transform(hour_dataframe[FEATURES])

def create_horizon_column(minute_dataframe, HORIZON, PERCENT_CHANGE):
    print("Creating horizons column...")
    
    def add_target(dataframe, HORIZON=HORIZON, PERCENT_CHANGE = PERCENT_CHANGE):
        combined_dataframe = (dataframe
        .join(raw_dataframe['high'].rename('raw_high'), how='left')
        .join(raw_dataframe['low'].rename('raw_low'), how='left')
        .join(raw_dataframe['close'].rename('raw_close'),how='left')
        )
        
        # Vectorized Horizon Calculation
        high = combined_dataframe['raw_high'].to_numpy()
        low = combined_dataframe['raw_low'].to_numpy()
        
        #Uses numpy to create a sliding window
        high_window = sliding_window_view(high, window_shape=HORIZON)[1:]
        low_window  = sliding_window_view(low, window_shape=HORIZON)[1:]

        #Assigns max and min price
        max_price = high_window.max(axis=1)
        min_price = low_window.min(axis=1)

        max_price = np.concatenate([max_price, np.full(HORIZON, np.nan)])
        min_price = np.concatenate([min_price, np.full(HORIZON, np.nan)])

        #Calculate percent change in price
        combined_dataframe['bullish_pct'] = ((max_price - combined_dataframe['raw_close']) / combined_dataframe['raw_close'])
        combined_dataframe['bearish_pct'] = ((combined_dataframe['raw_close'] - min_price) / combined_dataframe['raw_close'])
        
        # Vectorized conditions
        long_condition = (combined_dataframe['bullish_pct'] >= PERCENT_CHANGE) & (combined_dataframe['bearish_pct'] < (PERCENT_CHANGE / 2))
        short_condition = (combined_dataframe['bearish_pct'] >= PERCENT_CHANGE) & (combined_dataframe['bullish_pct'] < (PERCENT_CHANGE / 2))

        combined_dataframe['target'] = 1 
        combined_dataframe.loc[long_condition, 'target'] = 2
        combined_dataframe.loc[short_condition, 'target'] = 0

        combined_dataframe = combined_dataframe.dropna()
        print(combined_dataframe['target'].value_counts(normalize=True))

        dropped_columns = ['raw_high', 'raw_low', 'raw_close', 'bullish_pct', 'bearish_pct']
        return combined_dataframe.drop(columns=dropped_columns, errors='ignore')

    minute_dataframe = add_target(minute_dataframe)

    return minute_dataframe

def create_aligned_sequences(min_dataframe,hourly_dataframe, min_window, hour_window, features, target):
    print("Aligning and windowing data...")
    min_dataframe = min_dataframe.sort_index()
    hourly_dataframe = hourly_dataframe.sort_index()
    
    # Create integer index for hourly dataframe
    hourly_dataframe = hourly_dataframe.copy()
    hourly_dataframe['index'] = np.arange(len(hourly_dataframe))
    
    # Merge to find the corresponding hourly index for each minute and then filters any values out of bounds
    aligned = pd.merge_asof(min_dataframe, hourly_dataframe[['index']], left_index=True, right_index=True, direction='backward')
    aligned = aligned.dropna(subset=['index'])
    aligned = aligned[aligned['index'] >= hour_window - 1]
    
    #Assign features and targets
    min_features = aligned[features].values
    hourly_features = hourly_dataframe[features].values
    targets = aligned[target].values
    hourly_indices = aligned['index'].values.astype(int)

    X_min, X_hour, Y = [], [], []
    
    # Iterate through valid minute points to create sequences
    for i in range(min_window, len(aligned)):
        X_min.append(min_features[i-min_window + 1 : i + 1])
        
        index = hourly_indices[i]
        X_hour.append(hourly_features[index - hour_window + 1 : index + 1])
        
        Y.append(targets[i])
            
    return np.array(X_min), np.array(X_hour), np.array(Y)

def create_tf_dataset(x_min, x_hour, y):
    print("Creating the tensorflow dataset...")
    data_set = tf.data.Dataset.from_tensor_slices((
        # type: ignore
        {"minute": x_min, "hour": x_hour}, y
    ))
    return data_set

TARGET = "target" #Target column name
MIN_LOOKBACK = 60 #Minutes allowed for lookback
HOUR_LOOKBACK = 24 #Hours allowed for lookback
BATCH_SIZE = 256 #Batch size for training
FEATURES = ['open', 'high', 'low', 'close', 'volume','ema_fast', 'ema_slow', 'rsi', 'vwap', 'bbands']
HORIZON = 15 #Minutes look ahead
PERCENT_CHANGE = 0.0005 #Percent move needed for a trade to be considered
THRESHOLD = 0.50

#Create target column using a horizon (15 minutes horizon, 2 = long, 0 = short, 1 = no trade)
minute_data = create_horizon_column(minute_dataframe, HORIZON, PERCENT_CHANGE)

# Align and window data
x_minute_data, x_hour_data, y_minute_data = create_aligned_sequences(minute_data, hour_dataframe, MIN_LOOKBACK, HOUR_LOOKBACK, FEATURES, TARGET)

#Create Tensorflow Dataset
tensor_dataset = create_tf_dataset(x_minute_data,x_hour_data,y_minute_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = load_model('./lstm_models/weights/weights_10.keras')


#Get raw probalitites
raw_predictions = model.predict(tensor_dataset) #type: ignore

# Get the highest probability class for each sample
predictions = np.argmax(raw_predictions, axis=1)
retrospects = y_minute_data.copy()

#Make sure the predictions are above the threshold
strict_predictions = np.ones_like(predictions)
strict_predictions[raw_predictions[:, 0] > THRESHOLD] = 0
strict_predictions[raw_predictions[:, 2] > THRESHOLD] = 2

print("Classification Report:")
print(classification_report(
    retrospects, 
    predictions, 
    target_names=['Short (0)', 'No Trade (1)', 'Long (2)'],
    zero_division=0
))

# Convert strict_predictions into entries and exits
long_entries = (strict_predictions == 2)
short_entries = (strict_predictions == 0)

long_exits = pd.Series(long_entries).shift(HORIZON).fillna(False).to_numpy()
short_exits = pd.Series(short_entries).shift(HORIZON).fillna(False).to_numpy()

aligned_raw_close = raw_dataframe['close'].iloc[-len(strict_predictions):]


CONTRACT_MULTIPLIER = 2 
CONTRACTS = 2
trade_size = CONTRACT_MULTIPLIER * CONTRACTS
FEE = 2.50 / (aligned_raw_close.mean() * trade_size)

trade_size = CONTRACT_MULTIPLIER * CONTRACTS
# Build the portfolio simulation
portfolio = vbt.Portfolio.from_signals(
    close=aligned_raw_close, 
    entries=long_entries, 
    exits=long_exits, 
    short_entries=short_entries, 
    short_exits=short_exits,
    size=trade_size,
    freq='1m',          
    init_cash=1000000, #inflated to account for contract sizing, view it as 50000  
    fees = FEE 
)

print(portfolio.stats())
portfolio.plot().show() #type: ignore