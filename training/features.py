import pandas as pd
import pandas_ta as ta
import numpy as np
import mlfinpy as fin
import databento as db
import zipfile
import tempfile
from glob import glob
import os



def unzip_file(path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(path,'r') as zf:
        zf.extractall(temp_dir)

    return temp_dir

def to_dataframe(dir):
    try:
        files = glob(os.path.join(dir,"*.dbn.zst"))
        df = []
        for file in files:
            df = db.DBNStore.from_file(file).to_df()

        return df
    except:
        raise ValueError("Dataframe conversion failed")
        

raw_zip = 'data/raw/NQ_OHLCV.zip'
extracted_path = unzip_file(raw_zip)
print(f"Extracted to: {extracted_path}")

df = to_dataframe(extracted_path)
print(df.head())

