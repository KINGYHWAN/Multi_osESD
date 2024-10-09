
import numpy as np
import pandas as pd
import random
np.random.seed(42)
random.seed(42)

from merlion.utils import TimeSeries
from merlion.models.anomaly.autoencoder import AutoEncoderConfig, AutoEncoder

# def run_AE(data, parameters):
#     train_data = data['value']
#     train_labels =  data['anomaly']
#     train_data = TimeSeries.from_pd(train_data)
#     train_labels = TimeSeries.from_pd(train_labels)
#     config = AutoEncoderConfig(lr=parameters.lr,batch_size=parameters.batch_size)
#     model = AutoEncoder(config)
#     model.train(train_data=train_data, anomaly_labels=train_labels)
#     train_scores = model.get_anomaly_label(train_data)
#     pred_index = list(np.where(train_scores.to_pd()>0)[0])
#     return pred_index

# def run_AE(x_data, y_data, parameters, train_percent):
#     # train_data = data['value']
#     # train_labels =  data['anomaly']

#     train_data = TimeSeries.from_pd(x_data)
#     train_labels = TimeSeries.from_pd(y_data)

#     config = AutoEncoderConfig(lr=parameters.lr,batch_size=parameters.batch_size)
#     model = AutoEncoder(config)
#     model.train(train_data=train_data, anomaly_labels=train_labels)
#     # train_scores = model.get_anomaly_label(train_data[int(len(train_data) * train_percent):])
#     print(len(model.get_anomaly_label(train_data[int(len(train_data)*train_percent):])))
#     # adasdsd
#     if train_percent<1:
#         train_scores = model.get_anomaly_label(train_data[int(len(train_data)*train_percent):])
#     else:
#         train_scores = model.get_anomaly_label(train_data[train_percent:])
#     preds= (train_scores.to_pd()>0).astype(int)
#     return preds

def run_AE(train_data, test_data, params):
    config = AutoEncoderConfig(lr=params.lr,batch_size=params.batch_size,num_epochs=params.num_epochs,sequence_len = params.sequence_len)
    model = AutoEncoder(config)
    model.train(train_data)
    train_scores = model.get_anomaly_label(test_data)
    preds = (train_scores.to_pd()>0).astype(int)
    return preds


class AE_parameters:
    lr = 0.0003
    batch_size = 64
    plot = True

if __name__=='__main__':
    my_df = pd.read_csv('../../masters_project/Datasets/synthetic/ARIMA1_ber_1.csv')
    pred = run_AE(my_df,AE_parameters)

