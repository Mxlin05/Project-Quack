import pandas as pd
import pandas_ta as ta
import numpy as np
from mlfinpy.util.frac_diff import frac_diff_ffd
import databento as db
import zipfile
import tempfile
from glob import glob
import os
from training.utils import process_column
import joblib

def create_dataframe(file, columns):
    print("Unzipping Files...")
    dir = unzip_file(file)
    print("Converting to Dataframes...")
    df= to_dataframe(dir,columns)

    return df

def calculate_features(df, columns,config):
    print("Calculating Indicators...") 
    df = calculate_indicators(df,config)
    print(df.shape)
    print("Applying Fractional Differentiation...")
    df = apply_fractional_differentiation(df,config,columns)
    print(df.shape)
    
    return df

def calculate_spreads(nq : pd.DataFrame, es : pd.DataFrame, config):
    #Calculate the lead lag spread using log returns

    nq_log_returns = np.log(nq['close'] / nq['close'].shift(1))
    es_log_returns = np.log(es['close'] / es['close'].shift(1))

    lead_lag_spread = nq_log_returns - es_log_returns
    lead_lag_spread = pd.Series(lead_lag_spread).fillna(0)

    #Calcuate the RSI spread
    rsi_spread = nq[f'RSI_{config['indicators']['rsi_length']}'] - es[f'RSI_{config['indicators']['rsi_length']}']

    return lead_lag_spread, rsi_spread

def unzip_file(path):
    #Create a temporary directory to store unzipped files 
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(path,'r') as zf:
        zf.extractall(temp_dir)

    return temp_dir

def to_dataframe(dir,columns):
    try:
        #Look for .dbn.zst files
        files = glob(os.path.join(dir,"*.dbn.zst"))
        df = [] 

        #For every file decompress and add to a dataframe
        for file in files:
            dbn = db.DBNStore.from_file(file)
            df.append(dbn.to_df())

        df = pd.concat(df)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        #Filter to only OHLCV columns
        df = df[columns]
        
        #Replace 0 values with nan to prevent log function errors and atr explosion
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].where(df[price_cols] > 0, np.nan).ffill()

        for col in price_cols:
            median = df[col].rolling(window=180, min_periods=1, center=True).median()
            df[col] = df[col].where((df[col] / median).between(0.8, 1.2), median)

        df['high'] = df[price_cols].max(axis=1)
        df['low']  = df[price_cols].min(axis=1)

        return df 
    except Exception as e:
        print(e)

def calculate_indicators(df, config):
    #Uses a yaml to dynamically add varying values to the indicator parameters
    parameters = config['indicators']

    #Add indicator values
    df.ta.adx(length=parameters['adx_length'], append=True)
    df.ta.macd(fast=parameters['macd_fast'], slow=parameters['macd_slow'], signal=parameters['macd_signal'], append=True)
    df.ta.rsi(length=parameters['rsi_length'], append=True)
    df.ta.vwap(anchor=parameters['vwap_anchor'], append=True)
    df.ta.bbands(length=parameters['bbands_length'], std=parameters['bbands_std'], append=True)
    df.ta.atr(length=parameters['atr_length'], append=True)
    df['fvg'] = calculate_active_fvg(df,parameters['fvg_timeframe']) 
    return df

def apply_fractional_differentiation(df, config, columns):
    # Load yaml values
    fd = config['fractional_differentiation']
    d = fd['d']
    thresh = fd['threshold']

    new_df = df.copy()

    #Run processes in parallel
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_column)(frac_diff_ffd, new_df, col, d, thresh) for col in columns
    )
    for col, frac_series in results: #type: ignore
        new_df[f'{col}_frac'] = frac_series

    new_df = new_df.dropna(axis=0)
    
    return new_df

def calculate_active_fvg(df : pd.DataFrame, timeframe):
    #Create highs and lows for the first and third candle on the 15 minute interval
    temp_df = df.resample(timeframe).agg({'high': 'max', 'low': 'min', 'close': 'last'})
    c1_high = temp_df['high'].shift(2)
    c1_low = temp_df['low'].shift(2)
    c3_high = temp_df['high']
    c3_low = temp_df['low']

    #Returns the gap size. Positive for bullish and negative for bearish
    bullish_fvg = np.where(c3_low > c1_high, c3_low - c1_high, 0)
    bearish_fvg = np.where(c3_high < c1_low, c3_high - c1_low, 0)
    temp_df['fvg'] = bullish_fvg + bearish_fvg
    temp_df['gap_top'] = np.where(bullish_fvg > 0, c3_low,  np.where(bearish_fvg < 0, c1_low,  np.nan))
    temp_df['gap_bot'] = np.where(bullish_fvg > 0, c1_high, np.where(bearish_fvg < 0, c3_high, np.nan))

    # Carry the active gap forward, zeroing it once price fills it
    active_fvg = 0.0
    active_top = np.nan
    active_bot = np.nan
    result     = np.zeros(len(temp_df))
    
    for i, (idx, row) in enumerate(temp_df.iterrows()):
        if row['fvg'] != 0:
            active_fvg = row['fvg']
            active_top = row['gap_top']
            active_bot = row['gap_bot']

        # Zero out if price has filled the gap
        if active_fvg != 0:
            filled = (active_fvg > 0 and row['low'] <= active_top) or \
                    (active_fvg < 0 and row['high'] >= active_bot)
            if filled:
                active_fvg = 0.0
                active_top = np.nan
                active_bot = np.nan

        # Distance from current close to the nearest gap boundary
        if active_fvg > 0:
            result[i] = row['close'] - active_top
        elif active_fvg < 0:
            result[i] = active_bot - row['close']
        else:
            result[i] = 0.0

    temp_df['active_fvg'] = result
    return temp_df['active_fvg'].reindex(df.index, method='ffill')

def calculate_positional_encoding(df):
    ts_event = df.index

    print(ts_event.hour)
    df['minutes'] = (ts_event.hour * 60) + ts_event.minute
    print(df['minutes'])
    df['day'] = ts_event.dayofweek #type: ignore


    return df