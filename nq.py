import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import numpy as np

print("Pandas TA version: " + ta.version)
print("Pandas version: " + pd.__version__)
print("Yfinance version: " + yf.__version__)
print("Numpy version: " + np.__version__)

nq = yf.Ticker("NQ=F")