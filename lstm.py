import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import urllib.parse
import sqlalchemy as sqla
import dotenv
import os
import time

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
            url_object = urllib.parse.urlparse(conn_url)
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
        except Exception as e:
            print(f"Connection failed, retrying...")
            time.sleep(30)

def get_data(i,engine):
    print(f"Attempting to get iteration {i}'s data...")
    minute_train = pd.read_sql(f"SELECT * FROM minute_training_data_{i}", engine)
    minute_test = pd.read_sql(f"SELECT * FROM minute_testing_data_{i}", engine)
    hourly_train = pd.read_sql(f"SELECT * FROM hourly_training_data_{i}", engine)
    hourly_test = pd.read_sql(f"SELECT * FROM hourly_testing_data_{i}", engine)
    print(f"Recieved iteration {i}'s data")
    return minute_train, minute_test, hourly_train, hourly_test

def prep_data(minute_train, minute_test, hourly_train, hourly_test, FEATURES, TARGET):
    print("Preparing the data...")
    x_min_train = minute_train[FEATURES].values
    y_min_train = minute_train[TARGET].values

    x_min_test  = minute_test[FEATURES].values
    y_min_test  = minute_test[TARGET].values

    x_hr_train = hourly_train[FEATURES].values
    x_hr_test  = hourly_test[FEATURES].values
    print("Finished preparing the data")
    return x_min_train, y_min_train, x_min_test, y_min_test, x_hr_train, x_hr_test

def create_sequences(x,y,window):
    print("Windowing the data...")
    X, Y = [], []
    for i in range(len(x) - window):
        X.append(x[i:i+window])
        if y is not None:
            Y.append(y[i+window])
    print("Finished windowing the data")
    return np.array(X), np.array(Y)

def create_tf_dataset(x_min, x_hour, y):
    print("Creating the tensorflow dataset...")
    data_set = tf.data.Dataset.from_tensor_slices((
        {"minute": x_min, "hour": x_hour}, y
    ))
    print("Finished creating the tensorflow dataset...")
    return data_set

def create_model():
    print("Creating LSTM Model...")

if __name__ == "__main__":
    ITERATIONS = 10
    TARGET = "target"
    MIN_WINDOW = 60
    HOUR_WINDOW = 24
    BATCH_SIZE = 64
    FEATURES = ['open', 'high', 'low', 'close', 'volume','ema_fast', 'ema_slow', 'rsi', 'vwap', 'bb_upper', 'bb_lower']

    for i in range(1,ITERATIONS+1):
        #Connect to the database
        engine = connect_to_database()

        #Get sql data from database
        minute_train, minute_test, hourly_train, hourly_test = get_data(i,engine)

        #Seperate the features and the target
        x_min_train, y_min_train, x_min_test, y_min_test, x_hr_train, x_hr_test = prep_data( minute_train, minute_test, hourly_train, hourly_test, FEATURES, TARGET )
        
        #Window our data
        x_min_train, y_min_train = create_sequences(x_min_train, y_min_train, MIN_WINDOW)
        x_min_test, y_min_test = create_sequences(x_min_test, y_min_test, MIN_WINDOW)
        x_hr_train, _ = create_sequences(x_hr_train, None, HOUR_WINDOW)
        x_hr_test, _  = create_sequences(x_hr_test, None, HOUR_WINDOW)

        #Align the timeframes (Hourly data is repeated 60 times to match data shapes)
        x_hr_train = np.repeat(x_hr_train, 60, axis=0)
        x_hr_test  = np.repeat(x_hr_test, 60, axis=0)

        #Trim to the shortest length
        train_len = min(len(x_min_train), len(x_hr_train))
        x_min_train = x_min_train[:train_len]
        y_min_train = y_min_train[:train_len]
        x_hr_train  = x_hr_train[:train_len]

        test_len = min(len(x_min_test), len(x_hr_test))
        x_min_test = x_min_test[:test_len]
        y_min_test = y_min_test[:test_len]
        x_hr_test  = x_hr_test[:test_len]

        #Create Tensorflow Dataset
        train_dataset = create_tf_dataset(x_min_train,x_hr_train,y_min_train).shuffle(10,000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = create_tf_dataset(x_min_test,x_hr_test,y_min_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        #Create Model
        create_model()