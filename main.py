import yaml
from training.features import create_dataframe, calculate_features, calculate_spreads
from training.labeling import calculate_labels
from training.utils import merge_features_and_labels, align_dataframes
import glob

def preprocessing(config):
    nq_path = config['data']['nq_path']
    es_path = config['data']['es_path']
    columns = config['data']['columns']

    #Create Dataframes
    nq_df = create_dataframe(nq_path, columns)
    es_df = create_dataframe(es_path, columns)

    nq_df, es_df = align_dataframes(nq_df, es_df)

    #Add features as the datasets'x
    nq_df = calculate_features(nq_df,columns,config)   
    es_df = calculate_features(es_df,columns,config)

    #Calculates the divergence between the two correlated assets
    print("Calculating Spreads...")
    lead_lag_spread, rsi_spread = calculate_spreads(es_df, nq_df, config)
    nq_df['lead_lag_spread'] = lead_lag_spread
    nq_df['rsi_spread'] = rsi_spread

    #Add labels as the dataset's y
    print("Calculating Labels...")
    nq_labels = calculate_labels(nq_df,config) 

    #Self explanatory
    print("Merging Features and Labels...")
    nq_merged = merge_features_and_labels(nq_df,nq_labels)

    nq_merged, es_df = align_dataframes(nq_merged, es_df)
    
    #Save as a parquet file for ease of storage and fast lookup
    print("Saving as a .parquet File ...")
    nq_merged.to_parquet('data/processed/nq_dataframe.parquet',engine='pyarrow')
    es_df.to_parquet('data/processed/es_dataframe.parquet',engine='pyarrow')

def main():
    #Access and assign the config
    with open('config.yaml') as file:
        config = yaml.safe_load(file)

    #Check if parquet files exist, if not then create them
    if not glob.glob("data/processed/*.parquet"):
        preprocessing(config)

if __name__ == "__main__":
    main()