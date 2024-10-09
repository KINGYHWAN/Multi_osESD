import pandas as pd
import torch
import numpy as np
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
import random
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
from models import compare_AE
from models import compare_isolation_forest
from models import compare_LSTMED
from models import compare_rrcf
from models import compare_VAE
from models import gov_CNN
from models import compare_KNN

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


'''
TRAIN ON 15%, TEST ON NEXT 5%.
THEN 'UPDATE' ON 20%, MAKING IT TRAINED ON 35%. THEN TEST ON NEXT 5%.
SAME 3 TIMES. 55%, 75%, 95%.
SET THE TIME AS SUBTRACTED FROM THE NEXT ONE. -> IMPLEMENT INTO CODE, MAKE PAST_TIME VARIABLE
'''



def supervised_sequential_multivariate_tests():
    print("RUNNING SEQUENTIAL SUPERVISED POINT ANOMALY TESTS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_directory = 'results_tuning'
    data_name = 'art_synth_quad'
    prototype_df = call_datasets.call_ARIMA_datasets_prototype()


    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names = ['MOSESD','KNN','IF','AE','VAE','LSTM']
    col_names = ['Dataset','train_percent']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)
    ### Preparing ARIMA dataset
    ### is univariate, has no labels.

    data_len = 10000
    test_runs = 5
    std_val = 4
    seq_percent = 0.2
    # ㅁㅇㄴㅁㅇㄴㅇㄴ
    for i in range(test_runs):
        # dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, std_val)
        ALL_RESULTS+=(run_sequential_comparing_tests('ARIMA_'+str(i+1),ARIMA_df, 'anomaly',seq_percent,parameters.osESD_supervised_ARIMA))


    seq_percent = 0.2
    for i in range(test_runs):
        dataset_names.append('seasonal_'+str(i+1))
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, std_val)
        # ALL_RESULTS.append(run_comparing_tests(seasonal_df, 'anomaly', train_percent, parameters.osESD_supervised_seasonal))
        ALL_RESULTS+=(run_sequential_comparing_tests('seasonal_'+str(i+1),seasonal_df, 'anomaly',seq_percent,parameters.osESD_supervised_seasonal))
    

    std_val = 2
    seq_percent = 0.2
    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'][:4030])
            
    for i in range(test_runs):
        dataset_names.append('NABCloudwatch_'+str(i+1))
        mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
        ALL_RESULTS+=(run_sequential_comparing_tests('NABCloudWatch_',mixed_awscloudwatch_4032_df, 'anomaly', seq_percent, parameters.osESD_supervised_NAB2))

    ALL_RESULTS = pd.DataFrame(ALL_RESULTS)
    ALL_RESULTS.columns = col_names
    ALL_RESULTS.to_csv('results//sequential_single_point_algorithms.csv')
    # grouped_df = data_config.average_according_to_dataset(ALL_RESULTS,'Dataset')
    grouped_df = data_config.average_according_to_dataset_sequential(ALL_RESULTS, 'Dataset', 'train_percent')
    grouped_df.to_csv('results//sequential_single_point_algorithms_average.csv')






def run_sequential_comparing_tests(df_name,testing_df,y_col,sequential_percent, MOSESD_params):

    if sequential_percent not in [0.05,0.1,0.2,0.25,0.5]:
        raise ValueError("Please set sequential error to one of the following: 0.05,0.1,0.2,0.25,0.5")

    print(sequential_percent)
    one_data_return_values = []
    cols = [col for col in testing_df.columns if col != y_col]
    y_full_data = testing_df[y_col]
    train_full_data = testing_df[cols]
    L = len(y_full_data)

    current_percent = sequential_percent
    flag=False

    while True:

        cur_results = [df_name,np.round(current_percent,4)]

        test_start_idx = int(L*(current_percent-0.05))
        test_end_idx = int(L*current_percent)

        train_data = train_full_data[:test_end_idx]
        y_data = y_full_data[:test_end_idx]

        # test_data = train_full_data[test_start_idx:test_end_idx]
        # testing_reals = y_full_data[test_start_idx:test_end_idx]
        train_percent = (test_start_idx)/(test_end_idx)

        x_train, x_test = train_full_data[:test_start_idx], train_full_data[test_start_idx:test_end_idx]
        y_train, y_test = y_full_data[:test_start_idx], y_full_data[test_start_idx:test_end_idx]
        # print(len(x_train),len(train_data))

        y_true = y_test.values

        # train_data = 
        train_ts = TimeSeries.from_pd(pd.concat([x_train, y_train], axis=1))

        # test_data = 
        test_ts = TimeSeries.from_pd(pd.concat([x_test, y_test], axis=1))
        y_true = y_test.values

        # print(len(train_data),len(y_data))
        # print(train_percent)
        osESD_start = time()
        # print(len(y_true),len(osESD_preds))
        print(len(train_data),len(y_data))
        osESD_preds = main_4_1_supervised_point_anomaly.multi_osESD_supervised(train_data, y_data, MOSESD_params, 0.2, test_start_idx)
        print(len(y_true),len(osESD_preds), current_percent, train_percent)
        osESD_f1 = scoring.f1_score(y_true, osESD_preds)
        
        osESD_end = time()
        osESD_time = round(osESD_end - osESD_start,3)
        # print(len(y_true),len(osESD_preds))
        # print(scoring.all_scores(y_true,osESD_preds))
        # print("LEN : ", len(y_true), len(osESD_preds))
        # print(len(y_true),len(osESD_preds)) # 201, 201
        # print(y_true.shape, osESD_preds.shape)
        cur_results += scoring.all_scores(y_true,osESD_preds)
        # print(cur_results)
        cur_results.append(osESD_time)
        # print(cur_results)
        # sadd
        # print(scoring.all_scores(y_true,osESD_preds))
    # return one_data_return_values

        KNN_start = time()
        # print(train_data)
        # print(y_data)
        # print(len(train_data), len(y_data))
        KNN_pred = compare_KNN.run_KNN([x_train,y_train] ,[x_test,y_test] , parameters.KNN_parameters)
        # KNN_f1 = scoring.f1_score(y_true, KNN_pred)
        KNN_end = time()
        KNN_time = round(KNN_end -KNN_start,3)
        # print(scoring.all_scores(y_true,KNN_pred))
        cur_results += scoring.all_scores(y_true,KNN_pred)
        cur_results.append(KNN_time)

        iso_start = time()
        iso_pred = compare_isolation_forest.run_IF([x_train,y_train] ,[x_test,y_test] , parameters.IF_parameters)
        # iso_f1 = scoring.f1_score(y_true, iso_pred)
        iso_end = time()
        iso_time = round(iso_end -iso_start,3)
        # print(scoring.all_scores(y_true,iso_pred))
        # print("LEN : ",len(y_true),len(iso_pred))
        # print(len(y_true),len(iso_pred)) # 201, 202
        # print(y_true.shape, iso_pred.shape)
        cur_results += scoring.all_scores(y_true,iso_pred)
        cur_results.append(iso_time)


        AE_start = time()
        # print(len(y_true))
        AE_pred = compare_AE.run_AE(train_ts, test_ts ,parameters.ae_parameters)
        # AE_f1 = scoring.f1_score(y_true ,AE_pred)
        AE_end = time()
        AE_time = round(AE_end -AE_start,3)
        # print("LEN : ", len(y_true), len(AE_pred))
        # print(scoring.all_scores(y_true,AE_pred))
        # print(len(y_true),len(AE_pred))
        # print(y_true.shape, AE_pred.shape)
        cur_results += scoring.all_scores(y_true,AE_pred)
        cur_results.append(AE_time)
        # adadasd

        VAE_start = time()
        VAE_pred = compare_VAE.run_VAE(train_ts, test_ts ,parameters.vae_parameters)
        # VAE_f1 = scoring.f1_score(y_true ,VAE_pred)
        VAE_end = time()
        VAE_time = round(VAE_end -VAE_start,3)
        # print("LEN : ", len(y_true), len(VAE_pred))
        # print(scoring.all_scores(y_true,VAE_pred))
        cur_results += scoring.all_scores(y_true,VAE_pred)
        cur_results.append(VAE_time)


        LSTM_start = time()
        LSTM_pred = compare_LSTMED.run_LSTMED(train_ts, test_ts ,parameters.lstmed_parameters)
        # LSTM_f1 = scoring.f1_score(y_true ,LSTM_pred)
        LSTM_end = time()
        LSTM_time = round(LSTM_end -LSTM_start,4)
        # print("LEN : ", len(y_true), len(LSTM_pred))
        # print(scoring.all_scores(y_true,LSTM_pred))
        cur_results += scoring.all_scores(y_true,LSTM_pred)
        # print(y_true,LSTM_pred)
        # adadsad
        cur_results.append(LSTM_time)

        one_data_return_values.append(cur_results)
        if flag:
            current_percent = 1
            break

        current_percent+=sequential_percent


        if current_percent+0.05>=1:
            current_percent=0.95
            flag=True


    return one_data_return_values




