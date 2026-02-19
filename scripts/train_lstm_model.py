from data_postprocess import create_aligned_sequences, create_tf_dataset, create_model, compute_class_weights, create_horizon_column, get_data, connect_to_database
import gc
from keras.backend import clear_session
import tensorflow as tf

if __name__ == "__main__":
    ITERATIONS = 10 #Walk-Forward Iterations
    TARGET = "target" #Target column name
    MIN_LOOKBACK = 60 #Minutes allowed for lookback
    HOUR_LOOKBACK = 24 #Hours allowed for lookback
    BATCH_SIZE = 256 #Batch size for training
    FEATURES = ['open', 'high', 'low', 'close', 'volume','ema_fast', 'ema_slow', 'rsi', 'vwap', 'bbands']
    HORIZON = 15 #Minutes look ahead
    EPOCHS = 50 #Maximum epochs for training
    PERCENT_CHANGE = 0.0005 #Percent move needed for a trade to be considered


    for i in range(1,ITERATIONS+1):
        #Connect to the database
        engine = connect_to_database()

        #Get sql data from database
        minute_train, minute_test, hourly_train, hourly_test = get_data(i,engine)

        #Create target column using a horizon (15 minutes horizon, 2 = long, 0 = short, 1 = no trade)
        minute_train, minute_test = create_horizon_column(minute_train, minute_test, HORIZON, engine, PERCENT_CHANGE)

        #Creating weight classes to place emphasis on longs/shorts
        class_weights = compute_class_weights(minute_train)

        # Align and window data
        x_min_train, x_hr_train, y_min_train = create_aligned_sequences(minute_train, hourly_train, MIN_LOOKBACK, HOUR_LOOKBACK, FEATURES, TARGET)
        x_min_test, x_hr_test, y_min_test = create_aligned_sequences(minute_test, hourly_test, MIN_LOOKBACK, HOUR_LOOKBACK, FEATURES, TARGET)

        #Create Tensorflow Dataset
        train_dataset = create_tf_dataset(x_min_train,x_hr_train,y_min_train).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = create_tf_dataset(x_min_test,x_hr_test,y_min_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        #Create Model
        model = create_model(i,MIN_LOOKBACK,HOUR_LOOKBACK,FEATURES,train_dataset,test_dataset, EPOCHS, class_weights)

        #Clears memory before restarting training
        clear_session()
        del train_dataset, test_dataset, x_min_train, minute_train
        gc.collect()


