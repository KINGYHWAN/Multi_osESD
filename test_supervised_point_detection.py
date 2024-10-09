import pandas as pd
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from time import time
from merlion.utils import TimeSeries

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report,roc_auc_score
from models import gov_CNN
from utils import parameters
from utils import plotting_modules
from utils import call_datasets
from utils import data_config
from utils import scoring

from models import main_1_multi_osESD
from models import main_2_osESD_point_anomaly_detector
from models import main_3_osESD_CNN_classifier
from models import compare_KNN
from models import compare_AE
from models import compare_isolation_forest
from models import compare_LSTMED
from models import compare_rrcf
from models import compare_VAE
from models import gov_CNN

from models import classifier_CNN
from models import classifier_MLP
from models import classifier_Tabnet


from models import main_4_1_supervised_point_anomaly
from models import main_4_2_unsupervised_point_anomaly
from models import main_4_3_supervised_point_anomaly_batch_not_1_폐기
# from models import main_4_4_supervised_point_anomaly_batch_not_1_copied_폐기





'''
supervised point anomaly tests.
Will run on 7 types of datasets.
1. synthetic ARIMA (make 10, average)
2. synthetic seasonal (make 10, average)
3. NAB (choose 10, average)
4. yahoo (A1,A2,A3,A4, all, each average)
That makes 7.
Show average recall, precision, f1score, accuracy, and time on testing dataset.
'''



def supervised_multivariate_tests():
    print("RUNNING SUPERVISED POINT ANOMALY TESTS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_directory = 'results_tuning'
    data_name = 'art_synth_quad'
    prototype_df = call_datasets.call_ARIMA_datasets_prototype()


    # adasdasd
    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names =   ['MOSESD','KNN','IF','AE','VAE','LSTM']
    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)
    ### Preparing ARIMA dataset
    ### is univariate, has no labels.

    data_len = 10000
    test_runs = 5
    train_percent = 0.8
    std_val = 4
    for i in range(test_runs):
        dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, std_val)
        ALL_RESULTS.append(run_comparing_tests(ARIMA_df, 'anomaly', train_percent, parameters.osESD_supervised_ARIMA))


    for i in range(test_runs):
        dataset_names.append('seasonal_'+str(i+1))
        # Enhanced seasonal amplitude and modified noise levels
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, std_val)
        ALL_RESULTS.append(run_comparing_tests(seasonal_df, 'anomaly', train_percent, parameters.osESD_supervised_seasonal))

    Yahoos = call_datasets.call_yahoo_datasets()
    for df_type in (Yahoos):
        for idx in range(test_runs):
            dataset_names.append('yahoo_' + str(idx + 1))
            yahoo_df = data_config.sum_then_inject([df_type[0],df_type[1],df_type[2]], 0.03, std_val)
            ALL_RESULTS.append(run_comparing_tests(yahoo_df, 'anomaly',train_percent, parameters.osESD_supervised_yahoo))

    std_val = 2
    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()

    exchange_1624 = []
    for df in exchange_df:
        if len(df) in [1624, 1643]:
            exchange_1624.append(df['value'][:1620])

    for i in range(test_runs):
        dataset_names.append('NABrealTraffic_'+str(i+1))
        mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, std_val)
        ALL_RESULTS.append(run_comparing_tests(mixed_exchange_1624_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB1))


    # print(ALL_RESULTS)
    # adassd

    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'][:4030])

    for i in range(test_runs):
        dataset_names.append('NABCloudWatch_'+str(i+1))
        mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
        ALL_RESULTS.append(run_comparing_tests(mixed_awscloudwatch_4032_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB2))

    ALL_RESULTS = pd.DataFrame(ALL_RESULTS)
    ALL_RESULTS.columns = col_names[1:]
    ALL_RESULTS['Dataset'] = dataset_names
    ALL_RESULTS = ALL_RESULTS[col_names]
    print(ALL_RESULTS)

    ALL_RESULTS.to_csv('results//single_point_algorithms_sup.csv')
    grouped_df = data_config.average_according_to_dataset(ALL_RESULTS,'Dataset')
    grouped_df.to_csv('results//single_point_algorithms_sup_average.csv')


    ### Preparing NAB dataset
    ### Each is a list of datasets, has anomalies but is not labeled.
    ### Should merge and inject anomalies.

    # traffic_df, exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    #
    # traffic_2500 = []
    # for df in traffic_df:
    #     if len(df)==2500:
    #         traffic_2500.append(df['value'])
    # mixed_traffic_2500_df = data_config.sum_then_inject(traffic_2500,0.03,10)
    #
    # exchange_1624 = []
    # for df in exchange_df:
    #     if len(df) in [1624, 1643]:
    #         exchange_1624.append(df['value'][:1624])
    # mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, 10)
    #
    # awscloudwatch_4032 = []
    # for df in awscloudwatch_df:
    #     if len(df)==4032:
    #         awscloudwatch_4032.append(df['value'])
    # mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, 10)




def run_comparing_tests(data, y_col, train_percent, MOSESD_params):

    y_data = data[y_col]
    cols = [col for col in data.columns if col != y_col]
    x_data = data[cols]


    n_train = int(len(x_data) * train_percent)
    x_train, x_test = x_data[:n_train], x_data[n_train:]
    y_train, y_test = y_data[:n_train], y_data[n_train:]
    
    train_data = pd.concat([x_train, y_train], axis=1)
    train_ts = TimeSeries.from_pd(train_data)

    test_data = pd.concat([x_test, y_test], axis=1)
    test_ts = TimeSeries.from_pd(test_data)
    y_true = y_test.values

    one_data_return_values = []
    
    osESD_start = time()
    osESD_preds = main_4_1_supervised_point_anomaly.multi_osESD_supervised(x_data, y_data, MOSESD_params, 0.2, train_percent)
    # osESD_f1 = scoring.f1_score(y_true, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,3)
    # print(len(y_true),len(osESD_preds))
    # print(scoring.all_scores(y_true,osESD_preds))
    # print(len(y_true),len(osESD_preds))
    one_data_return_values += scoring.all_scores(y_true,osESD_preds)
    one_data_return_values.append(osESD_time)

    # return one_data_return_values
    # print(scoring.all_scores(y_true, osESD_preds))
    # adasds
    # print(scoring.all_scores(y_true,osESD_preds))
    # adsad    
    # adasdsad

    # print(scoring.all_scores(testing_reals,osESD_preds))
    # ADADASD
    KNN_start = time()
    KNN_pred = compare_KNN.run_KNN([x_train,y_train] ,[x_test,y_test], parameters.KNN_parameters)
    # KNN_f1 = scoring.f1_score(testing_reals, KNN_pred)
    KNN_end = time()
    KNN_time = round(KNN_end -KNN_start,3)
    # print(scoring.all_scores(testing_reals,KNN_pred))
    one_data_return_values += scoring.all_scores(y_true,KNN_pred)
    # print(scoring.all_scores(y_true,KNN_pred))
    one_data_return_values.append(KNN_time)

    iso_start = time()
    iso_pred = compare_isolation_forest.run_IF([x_train,y_train] ,[x_test,y_test], parameters.IF_parameters)
    # iso_f1 = scoring.f1_score(testing_reals, iso_pred)
    iso_end = time()
    iso_time = round(iso_end -iso_start,3)
    # print(scoring.all_scores(testing_reals,iso_pred))
    one_data_return_values += scoring.all_scores(y_true,iso_pred)
    # print(scoring.all_scores(y_true,iso_pred))
    one_data_return_values.append(iso_time)

    AE_start = time()
    # AE_pred = compare_AE.run_AE(train_data ,y_data ,parameters.ae_parameters ,train_percent)
    AE_pred = compare_AE.run_AE(train_ts, test_ts, parameters.ae_parameters)
    # AE_f1 = scoring.f1_score(testing_reals ,AE_pred)
    AE_end = time()
    AE_time = round(AE_end -AE_start,3)
    # print(scoring.all_scores(testing_reals,AE_pred))
    one_data_return_values += scoring.all_scores(y_true,AE_pred)
    # print(scoring.all_scores(y_true,AE_pred))
    one_data_return_values.append(AE_time)

    VAE_start = time()
    # VAE_pred = compare_VAE.run_VAE(train_data ,y_data ,parameters.vae_parameters ,train_percent)
    VAE_pred = compare_VAE.run_VAE(train_ts, test_ts, parameters.vae_parameters)
    # VAE_f1 = scoring.f1_score(testing_reals ,VAE_pred)
    VAE_end = time()
    VAE_time = round(VAE_end -VAE_start,3)
    # print(scoring.all_scores(testing_reals,VAE_pred))
    one_data_return_values += scoring.all_scores(y_true,VAE_pred)
    # print(scoring.all_scores(y_true,VAE_pred))
    one_data_return_values.append(VAE_time)

    LSTM_start = time()
    LSTM_pred = compare_LSTMED.run_LSTMED(train_ts, test_ts, parameters.lstmed_parameters)
    # LSTM_f1 = scoring.f1_score(testing_reals ,LSTM_pred)
    LSTM_end = time()
    LSTM_time = round(LSTM_end -LSTM_start,3)
    # print(scoring.all_scores(testing_reals,LSTM_pred))
    one_data_return_values += scoring.all_scores(y_true,LSTM_pred)

    # print(scoring.all_scores(y_true,osESD_preds))
    # print(scoring.all_scores(y_true,KNN_pred))
    # print(scoring.all_scores(y_true,iso_pred))
    # print(scoring.all_scores(y_true,AE_pred))
    # print(scoring.all_scores(y_true,VAE_pred))
    # print(scoring.all_scores(y_true,LSTM_pred))
    # adasdsad
    one_data_return_values.append(LSTM_time)

    # print(one_data_return_values)
    # adsasdsd

    return one_data_return_values




