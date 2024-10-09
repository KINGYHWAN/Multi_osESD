import numpy as np
import pandas as pd
import random
np.random.seed(42)
random.seed(42)

from merlion.utils import TimeSeries
from merlion.models.anomaly.vae import VAEConfig, VAE

# def run_VAE(x_data, y_data, parameters, train_percent):
#     # train_data = data['value']
#     # train_labels =  data['anomaly']
#     train_data = TimeSeries.from_pd(x_data)
#     train_labels = TimeSeries.from_pd(y_data)
#     config = VAEConfig(lr=parameters.lr,batch_size=parameters.batch_size)
#     model = VAE(config)
#     model.train(train_data=train_data, anomaly_labels=train_labels)
#     # train_scores = model.get_anomaly_label(train_data[int(len(train_data) * train_percent):])
#     if train_percent<1:
#         train_scores = model.get_anomaly_label(train_data[int(len(train_data)*train_percent):])
#     else:
#         train_scores = model.get_anomaly_label(train_data[train_percent:])
#     preds= (train_scores.to_pd()>0).astype(int)
#     return preds


def run_VAE(train_data, test_data, params):
    config = VAEConfig(lr=params.lr,batch_size=params.batch_size,num_epochs=params.num_epochs,sequence_len = params.sequence_len)
    model = VAE(config)
    model.train(train_data)
    train_scores = model.get_anomaly_label(test_data)
    preds = (train_scores.to_pd()>0).astype(int)
    return preds


class VAE_parameters:
    lr = 0.0003
    batch_size = 64
    plot = True

if __name__=='__main__':
    my_df = pd.read_csv('../../masters_project/Datasets/synthetic/ARIMA1_ber_1.csv')
    pred = run_VAE(my_df,VAE_parameters)

