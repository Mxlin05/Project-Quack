from mlfinpy.labeling import get_events, get_bins
from mlfinpy.filters import cusum_filter
from training.utils import calculate_vertical_barrier


def calculate_labels(df, config):
    #First need to generate the events using the triple barrier method
    def calculate_events(df = df, config = config):
        #Barriers are based off the ATR indicator values as well as a set max trade time
        volatility = df[f"ATRr_{config['indicators']['atr_length']}"]

        labeling = config['testing']['labeling']
        t1 = calculate_vertical_barrier(df, config)
        cumsum_events = cusum_filter(df['close'], threshold=volatility)

        events = get_events(
            close=df['close'],
            t_events= cumsum_events, # type: ignore
            pt_sl= labeling['pt_sl'],
            target=volatility,
            min_ret=0.0001, 
            vertical_barrier_times=t1, # type: ignore
            num_threads=1
        )

        return events

    #Calculates the exact returns from every barrier hit
    events = calculate_events()
    labels = get_bins(events,df['close'])

    #Rows where the returns was less than target are turned to 0 as they hit the vertical barrier
    vertical_hits = labels['ret'].abs() < events['trgt']
    labels.loc[vertical_hits, 'bin'] = 0

    return labels


