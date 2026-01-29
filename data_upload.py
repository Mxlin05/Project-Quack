from data_process import hourlyDataFrame, minuteDataFrame, ohlcv_cols
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import urllib.parse
import sqlalchemy as sqla
import dotenv
import os

'''
Updating sql database
1. Download the libraries (py -m pip install sqlalchemy pyodbc dotenv)
2. Download ODBC driver 18 from microsoft
'''

#Gets login data to access database
dotenv.load_dotenv("database.env")

#Connecting to the database
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
    connect_args={'timeout': 30} 
)

#Splitting data into 3 iterations of 4 years training and 1 year of testing using TimeSeriesSplit
total_days = (hourlyDataFrame.index[-1] - hourlyDataFrame.index[0]).days
samples_per_year = len(hourlyDataFrame) / (total_days / 365.25)
four_years_samples = int(samples_per_year * 4)

tscv = TimeSeriesSplit(n_splits=10, test_size=int(samples_per_year), max_train_size=four_years_samples)

print(f"Total Hourly Samples: {len(hourlyDataFrame)}")
print(f"Approx. 4-Year Samples: {four_years_samples}")

training_data = []
testing_data = []

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
    
    #Uploading Dataframe to database
    hourly_train.to_sql(
        f"hourly_training_data_{i + 1}",
        engine,
        index=True,
        if_exists="replace",
        chunksize=5000,
        dtype={'date_time': sqla.types.DateTime}
    )
    minute_train.to_sql(
        f"minute_training_data_{i + 1}",
        engine,
        index=True,
        if_exists="replace",
        chunksize=5000,
        dtype={'date_time': sqla.types.DateTime}
    )
    hourly_test.to_sql(
        f"hourly_testing_data_{i + 1}",
        engine,
        index=True,
        if_exists="replace",
        chunksize=5000,
        dtype={'date_time': sqla.types.DateTime}
    )
    minute_test.to_sql(
        f"minute_testing_data_{i + 1}",
        engine,
        index=True,
        if_exists="replace",
        chunksize=5000,
        dtype={'date_time': sqla.types.DateTime}
    )

