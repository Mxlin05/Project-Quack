import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import urllib.parse
import sqlalchemy as sqla
import dotenv
import time
from tqdm.keras import TqdmCallback

#Gets login data to access database
dotenv.load_dotenv("database.env")

def connect_to_database():
    while True:
        try:
            print("Attempting to connect to the database")
            conn_str = (
            "DRIVER=ODBC Driver 18 for SQL Server;"
            f"SERVER={os.getenv('Server')};"
            "DATABASE=Project Quack;"
            f"UID={os.getenv('UserId')};"
            f"PWD={os.getenv('Password')};"
            "TrustServerCertificate=yes;"
            )
            conn_url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
            engine = sqla.create_engine(
                conn_url,
                fast_executemany=True,  
                connect_args={'timeout': 30},
                pool_pre_ping=True
            )

            with engine.connect():
                pass

            print("Database connection successful")
            return engine
        except:
            print(f"Connection failed, retrying...")
            time.sleep(30)

def get_data(i,engine):
    #Select all data from both training and testing sets
    print(f"Attempting to get iteration {i}'s data...")
    minute_train = pd.read_sql(f"SELECT * FROM minute_training_data_{i}", engine, index_col="date_time")
    minute_test = pd.read_sql(f"SELECT * FROM minute_testing_data_{i}", engine, index_col="date_time")
    hourly_train = pd.read_sql(f"SELECT * FROM hourly_training_data_{i}", engine, index_col="date_time")
    hourly_test = pd.read_sql(f"SELECT * FROM hourly_testing_data_{i}", engine, index_col="date_time")
    return minute_train, minute_test, hourly_train, hourly_test

def create_horizon_column(minute_train, minute_test, HORIZON, engine, PERCENT_CHANGE):
    print("Creating horizons column...")
    
    def add_target(dataframe, HORIZON=HORIZON, engine=engine, PERCENT_CHANGE = PERCENT_CHANGE):
        start_dt = dataframe.index.min()
        end_dt = dataframe.index.max()
        
        # Fetch raw close prices for this range
        query = f"""
        SELECT [date_time], [high], [low], [close] 
        FROM raw_minute_data 
        WHERE date_time >= '{start_dt}' 
        AND date_time <= '{end_dt}' 
        ORDER BY date_time ASC"""
        raw_dataframe = pd.read_sql(query, engine, index_col='date_time')

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

    minute_train = add_target(minute_train)
    minute_test = add_target(minute_test)

    return minute_train, minute_test

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
        {"minute": x_min, "hour": x_hour}, y
    ))
    return data_set

def create_model(i,MIN_LOOKBACK,HOUR_LOOKBACK,FEATURES, train_dataset, test_dataset, EPOCHS):
    print("Creating LSTM Model...")
    
    #Stops training if no improvement in validation loss after 3 epochs
    early_stop_monitor = EarlyStopping(patience=3, verbose=1, restore_best_weights=True )

    #Creating both minute and hourly LSTM inputs
    minute_input = Input(shape=(MIN_LOOKBACK,len(FEATURES)), name ="minute")
    x_min = LSTM(64,return_sequences=True)(minute_input)
    x_min = LSTM(64,return_sequences=False)(x_min)

    hour_input = Input(shape=(HOUR_LOOKBACK,len(FEATURES)), name ="hour")
    x_hour = LSTM(32,return_sequences=True)(hour_input)
    x_hour = LSTM(32,return_sequences=False)(x_hour)

    #Combine both LSTM outputs. Uses leaky relu activation and dropout for regularization
    x = Concatenate()([x_min,x_hour])
    x = Dense(128, activation="leaky_relu")(x)
    x = Dropout(0.33)(x)

    #Final output: 3 classes (long, short, no trade) with softmax activation
    output = Dense(3,activation="softmax")(x)

    #Create and compile model
    model = Model(inputs=[minute_input, hour_input], outputs=output)
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        verbose="0",
        epochs=EPOCHS, 
        callbacks=[early_stop_monitor, TqdmCallback(verbose=1)])
    
    model.save(f"lstm_models/lstm_model_{i}.keras") #Saves model after training

    return history

if __name__ == "__main__":
    ITERATIONS = 10 #10 Walk-Forward Iterations
    TARGET = "target" #Target column name
    MIN_LOOKBACK = 60 #60 minutes
    HOUR_LOOKBACK = 24 #24 hours
    BATCH_SIZE = 512 #Batch size for training
    FEATURES = ['open', 'high', 'low', 'close', 'volume','ema_fast', 'ema_slow', 'rsi', 'vwap', 'bbands']
    HORIZON = 15 #15 minutes look ahead
    EPOCHS = 10 #Maximum epochs for training
    PERCENT_CHANGE = 0.0005 #0.05 percent move

    for i in range(1,ITERATIONS+1):
        #Connect to the database
        engine = connect_to_database()

        #Get sql data from database
        minute_train, minute_test, hourly_train, hourly_test = get_data(i,engine)

        #Create target column using a horizon (15 minutes horizon, 2 = long, 0 = short, 1 = no trade)
        minute_train, minute_test = create_horizon_column(minute_train, minute_test, HORIZON, engine, PERCENT_CHANGE)

        # Align and window data
        x_min_train, x_hr_train, y_min_train = create_aligned_sequences(minute_train, hourly_train, MIN_LOOKBACK, HOUR_LOOKBACK, FEATURES, TARGET)
        x_min_test, x_hr_test, y_min_test = create_aligned_sequences(minute_test, hourly_test, MIN_LOOKBACK, HOUR_LOOKBACK, FEATURES, TARGET)

        #Create Tensorflow Dataset
        train_dataset = create_tf_dataset(x_min_train,x_hr_train,y_min_train).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = create_tf_dataset(x_min_test,x_hr_test,y_min_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        #Create Model
        history = create_model(i,MIN_LOOKBACK,HOUR_LOOKBACK,FEATURES,train_dataset,test_dataset, EPOCHS)