import pandas as pd
import pandas_ta_classic as ta
import vectorbt as vbt
import zstandard as zstd
import matplotlib.pyplot as plt

#Import NQ OHLCV data
compressed_data =  'c:/Users/bennn/repos/Project-Quack/NQ_OHLCV_1h/glbx-mdp3-20100606-20251231.ohlcv-1h.csv.zst'
with open(compressed_data, 'rb') as binary:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(binary) as reader:
        nq = pd.read_csv(reader)
        nq['datetime'] = pd.to_datetime(nq['ts_event'])
        nq.set_index('datetime', inplace=True)

#EMA, RSI, VWAP, BBANDS indicators
nq['ema_fast'] = nq.ta.ema(length=21)
nq['ema_slow'] = nq.ta.ema(length=55)
nq['rsi'] = nq.ta.rsi(length=14)
nq['vwap'] = nq.ta.vwap(close=nq['close'], volume=nq['volume'], anchor="D")

bbands = nq.ta.bbands(length=20)
nq['bb_upper'] = bbands.iloc[:, 2]
nq['bb_lower'] = bbands.iloc[:, 0]

# Remove all NaN values 
nq = nq.dropna()
nq = nq[nq['close'] > 0]
print(nq.tail())

#Requirements for trade entries
entries_long = (
    (nq['ema_fast'] > nq['ema_slow']) 
    & (nq['rsi'] < 50)
)

exits_long = nq['ema_fast'] < nq['ema_slow']

entries_short = (
    (nq['ema_fast'] < nq['ema_slow']) 
    & (nq['rsi'] > 50)
)

exits_short = nq['ema_fast'] > nq['ema_slow']


#Backtesting
portfolio = vbt.Portfolio.from_signals(
    nq['close'],
    entries=entries_long,
    exits=exits_long,
    short_entries=entries_short,
    short_exits=exits_short,
    init_cash= 100,
    freq="1h",
    fees=0.0005,  
)
print(portfolio.stats())
