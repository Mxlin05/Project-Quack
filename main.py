import yaml
from training.features import create_dataframe, calculate_features, calculate_spreads, calculate_positional_encoding
from training.labeling import calculate_labels
from training.utils import align_dataframes
from training.validation import calculate_folds
from training.architecture import TimeSeriesDataset, LSTMModel, PositiveEncodingModel, TransformerModel
import pandas as pd
import glob

def preprocessing(config):
    nq_path = config['data']['nq_path']
    es_path = config['data']['es_path']
    columns = config['data']['columns']

    #Create Dataframes
    nq_df = create_dataframe(nq_path, columns)
    es_df = create_dataframe(es_path, columns)

    print(nq_df.shape)  #type: ignore
    print(es_df.shape)  #type: ignore

    #Add features as the datasets'x
    nq_df = calculate_features(nq_df,columns,config)   
    es_df = calculate_features(es_df,columns,config)

    nq_df.to_parquet('nq_features.parquet')
    es_df.to_parquet('es_features.parquet')

    nq_df, es_df = align_dataframes(nq_df, es_df) #type: ignore

    print(nq_df.shape)  #type: ignore
    print(es_df.shape)  #type: ignore

    #Calculates the divergence between the two correlated assets
    print("Calculating Spreads...")
    nq_df['lead_lag_spread'], nq_df['rsi_spread'] = calculate_spreads(nq_df, es_df, config)
    
    #Add labels as the dataset's y
    print("Calculating Labels...")
    labels = calculate_labels(nq_df,config) 
    print(labels.shape)

    nq_df, _ = align_dataframes(nq_df, labels)
    es_df, _ = align_dataframes(es_df, labels)
    
    #Calculate positional encodings which will be used for the transformers
    print("Calculating Postitional Encodings")
    nq_df = calculate_positional_encoding(nq_df)
    es_df = calculate_positional_encoding(es_df)

    #Save as a parquet file for ease of storage and fast lookup
    print("Saving as a .parquet File ...")
    labels.to_parquet('data/processed/labels.parquet',engine='pyarrow')
    nq_df.to_parquet('data/processed/nq_dataframe.parquet',engine='pyarrow')
    es_df.to_parquet('data/processed/es_dataframe.parquetx',engine='pyarrow')

def training(config):
    nq_df = pd.read_parquet('nq_df.parquet')
    es_df = pd.read_parquet('es_df.parquet')
    labels = pd.read_parquet('labels.parquet')
    
def main():
    #Access the config
    with open('config.yaml') as file:
        config = yaml.safe_load(file)

    #Check if parquet files exist, if not then create them
    if not glob.glob("data/processed/*.parquet"):
        preprocessing(config)

if __name__ == "__main__":
    main()

    
    