import pandas as pd
from mlfinpy.cross_validation.combinatorial import CombinatorialPurgedKFold

def calculate_vertical_barrier(df, config):
    max_trade_time = config['testing']['max_trade_time']
    t1 = df.index + pd.Timedelta(minutes=max_trade_time)
    t1 = pd.Series(t1,index=df.index) 

    return t1

def create_cv(df, config):
    validation = config['testing']['validation']
    t1 = calculate_vertical_barrier(df, config)
    cv = CombinatorialPurgedKFold(
        n_splits=validation['splits'],
        n_test_splits=validation['test_splits'],
        samples_info_sets= t1, #type: ignore
        embargo=config['testing']['max_trade_time'] + 15
    )
    return cv

def align_dataframes(df1, df2):
    #Align the two dataframes
    df1, df2 = df1.align(df2, join='inner', axis=0)
    return df1, df2

def process_column(frac_diff_ffd, df, col, d, thresh):
    #Helper function to process a column
    diff_df = frac_diff_ffd(series=df[[col]], diff_amt=d, thresh=thresh)
    return col, diff_df.iloc[:, 0]