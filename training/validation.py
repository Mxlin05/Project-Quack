from sklearn.preprocessing import MinMaxScaler
from architecture import TimeSeriesDataset
import pandas as pd
import joblib
from training.utils import create_cv

def calculate_folds(config, df, labels):
    x = df
    y = labels
    cv = create_cv(x, config)

    folds = []

    #Run a loop using the CombinatorialPurgedKFold object to split up features and target to train and validate
    for fold, (train_indices, val_indices) in enumerate(cv.split(x)):
        x_train = x.iloc[train_indices]
        y_train = y.iloc[train_indices]

        x_val = x.iloc[val_indices]
        y_val = y.iloc[val_indices]

        #Scale every value to fit between 0 and 1
        scaler = MinMaxScaler() 
        scaler.fit(x_train)
        joblib.dump(scaler, f"./model/scalers/scaler_{fold+1}.pkl")

        x_trained_scaled = pd.DataFrame(scaler.transform(x_train), index =x_train.index, columns=x_train.columns)
        x_val_scaled = pd.DataFrame(scaler.transform(x_val), index =x_val.index, columns=x_val.columns)
        
        #Creates the tensors needed for training and validating by * using sequence length
        train_dataset = TimeSeriesDataset(x_trained_scaled, y_train, sequence_length=config['testing']['sequence_length'])
        val_dataset = TimeSeriesDataset(x_val_scaled, y_val, sequence_length=config['testing']['sequence_length'])

        folds.append({
            'fold': fold + 1,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        })

    return folds

