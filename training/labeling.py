from mlfinpy.labeling import get_events, get_bins
from mlfinpy.filters import cusum_filter
from training.utils import calculate_vertical_barrier


def calculate_labels(df, config):
    #Barriers are based off the ATR indicator values as well as a set max trade time
    volatility = df[f"ATRr_{config['indicators']['atr_length']}"]
    close = df['close']
    target = volatility/close

    #First need to generate the events using the triple barrier method
    def calculate_events(df = df, config = config):
        labeling = config['testing']['labeling']

        threshold = target
        t1 = calculate_vertical_barrier(df, config)
        cumsum_events = cusum_filter(close, threshold=threshold)

        events = get_events(
            close=df['close'],
            t_events= cumsum_events, # type: ignore
            pt_sl= labeling['pt_sl'],
            target=target,
            min_ret=0.0001, 
            vertical_barrier_times=t1, # type: ignore
            num_threads=1
        )

        return events

    #Calculates the exact returns from every barrier hit
    events = calculate_events()
    labels = get_bins(events,df['close'])   

    targets = events['trgt'].loc[labels.index]
    vertical_hits = labels['ret'].abs() < targets
    labels.loc[vertical_hits, 'bin'] = 0

    return labels


