import numpy as np
import pandas as pd
import random
np.random.seed(42)
random.seed(42)

from merlion.utils import TimeSeries
from merlion.models.anomaly.lstm_ed import LSTMEDConfig, LSTMED

def run_LSTMED(train_data, test_data, params):
    config = LSTMEDConfig(lr=params.lr,batch_size=params.batch_size,num_epochs=params.num_epochs,sequence_len = params.sequence_len)
    model = LSTMED(config)
    model.train(train_data)
    train_scores = model.get_anomaly_label(test_data)
    preds = (train_scores.to_pd()>0).astype(int)
    return preds

class LSTMED_parameters:
    lr = 0.0003
    batch_size = 64
    plot = True

if __name__=='__main__':
    my_df = pd.read_csv('../../masters_project/Datasets/synthetic/ARIMA1_ber_1.csv')
    pred = run_LSTMED(my_df,LSTMED_parameters)


