import pandas as pd
import random
import torch
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



def supervised_multivariate_compare_tests():

    print("RUNNING SUPERVISED POINT ANOMALY TESTS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_directory = 'results_tuning'
    data_name = 'art_synth_quad'
    prototype_df = call_datasets.call_ARIMA_datasets_prototype()


    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names = ['MOSESD','iso','AE','VAE','LSTM']
    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)

    arima_dfs = 1
    data_len = 10000
    std_val = 4
    train_size = 0.8
    for i in range(arima_dfs):
        print(i)
        print("Tuning ARIMA dataset number %d out of %d datasets"%(i+1,arima_dfs))
        dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, std_val)
        run_KNN_tuning(ARIMA_df,'anomaly',train_size,'ARIMA'+str(i+1))
        run_IF_tuning(ARIMA_df,'anomaly',train_size,'ARIMA'+str(i+1))
        run_AE_tuning(ARIMA_df,'anomaly',train_size,'ARIMA'+str(i+1))
        run_VAE_tuning(ARIMA_df,'anomaly',train_size,'ARIMA'+str(i+1))
        run_LSTM_tuning(ARIMA_df,'anomaly',train_size,'ARIMA'+str(i+1))
        # adsads
        # ALL_RESULTS.append(run_comparing_tests(ARIMA_df, 'anomaly', 0.2))
    # # adsadsa
    # asdadadsad

    for i in range(1):
        dataset_names.append('seasonal_'+str(i+1))
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, std_val)
        # ALL_RESULTS.append(run_comparing_tests(seasonal_df, 'anomaly', 0.2))
        run_KNN_tuning(seasonal_df,'anomaly',train_size,'seasonal'+str(i+1))
        run_IF_tuning(seasonal_df,'anomaly',train_size,'seasonal'+str(i+1))
        run_AE_tuning(seasonal_df,'anomaly',train_size,'seasonal'+str(i+1))
        run_VAE_tuning(seasonal_df,'anomaly',train_size,'seasonal'+str(i+1))
        run_LSTM_tuning(seasonal_df,'anomaly',train_size,'seasonal'+str(i+1))



    Yahoos = call_datasets.call_yahoo_datasets()
    for i,df_type in enumerate(Yahoos):
        dataset_names.append('yahoo_' + str(i + 1))
        yahoo_df = data_config.sum_then_inject([df_type[0],df_type[1],df_type[2]], 0.03, std_val)
        # ALL_RESULTS.append(run_comparing_tests(yahoo_df, 'anomaly', 0.2))
        run_KNN_tuning(yahoo_df,'anomaly',train_size,'yahoo'+str(i+1))
        run_IF_tuning(yahoo_df,'anomaly',train_size,'yahoo'+str(i+1))
        run_AE_tuning(yahoo_df,'anomaly',train_size,'yahoo'+str(i+1))
        run_VAE_tuning(yahoo_df,'anomaly',train_size,'yahoo'+str(i+1))
        run_LSTM_tuning(yahoo_df,'anomaly',train_size,'yahoo'+str(i+1))

    std_val = 2
    train_size = 0.5
    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    exchange_1624 = []
    for df in exchange_df:
        if len(df) in [1624, 1643]:
            exchange_1624.append(df['value'][:1624])
    dataset_names.append('NAB1')
    mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, std_val)
    # ALL_RESULTS.append(run_comparing_tests(mixed_exchange_1624_df, 'anomaly', 0.2, parameters.osESD_supervised_NAB1))
    run_KNN_tuning(mixed_exchange_1624_df,'anomaly',train_size,'NAB1')
    run_IF_tuning(mixed_exchange_1624_df,'anomaly',train_size,'NAB1')
    run_AE_tuning(mixed_exchange_1624_df,'anomaly',train_size,'NAB1')
    run_VAE_tuning(mixed_exchange_1624_df,'anomaly',train_size,'NAB1')
    run_LSTM_tuning(mixed_exchange_1624_df,'anomaly',train_size,'NAB1')


    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'])
    dataset_names.append('NAB2')
    mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
    # ALL_RESULTS.append(run_comparing_tests(mixed_awscloudwatch_4032_df, 'anomaly', 0.2, parameters.osESD_supervised_NAB2))

    run_KNN_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_size,'NAB2')
    run_IF_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_size,'NAB2')
    run_AE_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_size,'NAB2')
    run_VAE_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_size,'NAB2')
    run_LSTM_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_size,'NAB2')







def run_KNN_tuning(testing_df,y_col,train_percent,df_name):

    neighborss = [3,5,7,10,20]

    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['n_estimators']
    col_names += metrics
    tuned_values = []
    for neighbors in neighborss:
        print('running neighbors size ',neighbors)
        parameters.KNN_parameters.neighbors=neighbors
        tuned_results = [neighbors]

        cols = [col for col in testing_df.columns if col != y_col]
        y_data = testing_df[y_col]
        x_data = testing_df[cols]
        
        n_train = int(len(x_data) * train_percent)
        x_train, x_test = x_data[:n_train], x_data[n_train:]
        y_train, y_test = y_data[:n_train], y_data[n_train:]

        train_data = pd.concat([x_train, y_train], axis=1)
        train_ts = TimeSeries.from_pd(train_data)
        test_data = pd.concat([x_test, y_test], axis=1)
        test_ts = TimeSeries.from_pd(test_data)
        y_true = y_test.values

        KNN_start = time()
        KNN_preds = compare_KNN.run_KNN(
            [x_train, y_train], [x_test, y_test], parameters.KNN_parameters)
        # KNN_f1 = scoring.f1_score(testing_reals, KNN_preds)
        KNN_end = time()
        KNN_time = round(KNN_end - KNN_start,3)
        # print(scoring.all_scores(testing_reals,KNN_preds))
        tuned_results+=scoring.all_scores(y_true,KNN_preds)
        tuned_results.append(KNN_time)
        tuned_values.append(tuned_results)

    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_KNN_'+df_name+'.csv')



def run_IF_tuning(testing_df,y_col,train_percent,df_name):

    n_estimatorss = [500,1000,1500]
    max_sampless = [50,100,150]

    # n_estimatorss = [500]
    # max_sampless = [50]

    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['n_estimators','max_samples']
    col_names += metrics

    tuned_values = []
    for n_estimators in n_estimatorss:
        print('running lr size ',n_estimators)
        parameters.IF_parameters.n_estimators=n_estimators
        for max_samples in max_sampless:
            parameters.IF_parameters.max_samples = max_samples
            tuned_results = [n_estimators,max_samples]

            cols = [col for col in testing_df.columns if col != y_col]
            y_data = testing_df[y_col]
            x_data = testing_df[cols]
            
            n_train = int(len(x_data) * train_percent)
            x_train, x_test = x_data[:n_train], x_data[n_train:]
            y_train, y_test = y_data[:n_train], y_data[n_train:]

            train_data = pd.concat([x_train, y_train], axis=1)
            train_ts = TimeSeries.from_pd(train_data)
            test_data = pd.concat([x_test, y_test], axis=1)
            test_ts = TimeSeries.from_pd(test_data)
            y_true = y_test.values
            # print(y_data)
            IF_start = time()
            IF_preds = compare_isolation_forest.run_IF(
                [x_train, y_train], [x_test, y_test], parameters.IF_parameters)
            # IF_f1 = scoring.f1_score(testing_reals, IF_preds)
            IF_end = time()
            IF_time = round(IF_end - IF_start,4)
            # print(scoring.all_scores(testing_reals,IF_preds))
            tuned_results+=scoring.all_scores(y_true,IF_preds)
            tuned_results.append(IF_time)
            tuned_values.append(tuned_results)

    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_IF_'+df_name+'.csv')



def run_AE_tuning(testing_df,y_col,train_percent,df_name):

    lrs = [0.0001,0.0003,0.0005,0.001,0.005]
    batch_sizes = [32,64,128]
    num_epochss = [50,100,200]
    sequence_lens = [5,10,20,50]


    # lrs = [0.0005]
    # batch_sizes = [64]
    # num_epochss = [10]
    # sequence_lens = [5]

    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['LR','batch_size','num_epochs','seq_len']
    col_names += metrics

    tuned_values = []
    for lr in lrs:
        print('running lr size ',lr)
        parameters.ae_parameters.lr=lr
        for batch_size in batch_sizes:
            parameters.ae_parameters.batch_size = batch_size
            for num_epochs in num_epochss:
                parameters.ae_parameters.num_epochs = num_epochs
                for sequence_len in sequence_lens:
                    parameters.ae_parameters.sequence_len = sequence_len
                    tuned_results = [lr,batch_size,num_epochs, sequence_len]

                    cols = [col for col in testing_df.columns if col != y_col]
                    y_data = testing_df[y_col]
                    x_data = testing_df[cols]
                    
                    n_train = int(len(x_data) * train_percent)
                    x_train, x_test = x_data[:n_train], x_data[n_train:]
                    y_train, y_test = y_data[:n_train], y_data[n_train:]

                    train_data = pd.concat([x_train, y_train], axis=1)
                    train_ts = TimeSeries.from_pd(train_data)
                    test_data = pd.concat([x_test, y_test], axis=1)
                    test_ts = TimeSeries.from_pd(test_data)
                    y_true = y_test.values

                    AE_start = time()
                    AE_preds = compare_AE.run_AE(train_ts, test_ts, parameters.ae_parameters)
                    # AE_f1 = scoring.f1_score(testing_reals, AE_preds)
                    AE_end = time()
                    AE_time = round(AE_end - AE_start,4)
                    # print(scoring.all_scores(testing_reals,AE_preds))
                    tuned_results+=scoring.all_scores(y_true,AE_preds)
                    tuned_results.append(AE_time)
                    tuned_values.append(tuned_results)

    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_AE_'+df_name+'.csv')


def run_VAE_tuning(testing_df,y_col,train_percent,df_name):

    lrs = [0.0001,0.0003,0.0005,0.001,0.005]
    batch_sizes = [32,64,128]
    num_epochss = [50,100,200]
    sequence_lens = [5,10,20,50]

    # lrs = [0.0005]
    # batch_sizes = [64]
    # num_epochss = [10]
    # sequence_lens = [5]


    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['LR','batch_size','num_epochs','seq_len']
    col_names += metrics


    tuned_values = []
    for lr in lrs:
        print('running lr size ',lr)
        parameters.vae_parameters.lr=lr
        for batch_size in batch_sizes:
            parameters.vae_parameters.batch_size = batch_size
            for num_epochs in num_epochss:
                parameters.vae_parameters.num_epochs = num_epochs
                for sequence_len in sequence_lens:
                    parameters.vae_parameters.sequence_len = sequence_len

                    tuned_results = [lr,batch_size,num_epochs, sequence_len]

                    cols = [col for col in testing_df.columns if col != y_col]
                    y_data = testing_df[y_col]
                    x_data = testing_df[cols]

                    n_train = int(len(x_data) * train_percent)
                    x_train, x_test = x_data[:n_train], x_data[n_train:]
                    y_train, y_test = y_data[:n_train], y_data[n_train:]

                    train_data = pd.concat([x_train, y_train], axis=1)
                    train_ts = TimeSeries.from_pd(train_data)
                    test_data = pd.concat([x_test, y_test], axis=1)
                    test_ts = TimeSeries.from_pd(test_data)
                    y_true = y_test.values

                    VAE_start = time()
                    VAE_preds = compare_VAE.run_VAE(train_ts, test_ts, parameters.vae_parameters)
                    # VAE_f1 = scoring.f1_score(testing_reals, VAE_preds)
                    VAE_end = time()
                    VAE_time = round(VAE_end - VAE_start,4)
                    # print(scoring.all_scores(testing_reals,VAE_preds))
                    tuned_results+=scoring.all_scores(y_true,VAE_preds)
                    tuned_results.append(VAE_time)
                    tuned_values.append(tuned_results)

    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_VAE_'+df_name+'.csv')


def run_LSTM_tuning(testing_df,y_col,train_percent,df_name):

    lrs = [0.0001,0.0003,0.0005,0.001,0.005]
    batch_sizes = [32,64,128]
    num_epochss = [50,100,200]
    sequence_lens = [5,10,20,50]

    # lrs = [0.0005]
    # batch_sizes = [64]
    # num_epochss = [10]
    # sequence_lens = [5]


    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['LR','batch_size','num_epochs','seq_len']
    col_names += metrics

    tuned_values = []
    for lr in lrs:
        print('running lr size ',lr)
        parameters.lstmed_parameters.lr=lr
        for batch_size in batch_sizes:
            parameters.lstmed_parameters.batch_size = batch_size
            for num_epochs in num_epochss:
                parameters.lstmed_parameters.num_epochs = num_epochs
                for sequence_len in sequence_lens:
                    parameters.lstmed_parameters.sequence_len = sequence_len
                    tuned_results = [lr,batch_size,num_epochs, sequence_len]
                    cols = [col for col in testing_df.columns if col != y_col]
                    y_data = testing_df[y_col]
                    x_data = testing_df[cols]
                    
                    n_train = int(len(x_data) * train_percent)
                    x_train, x_test = x_data[:n_train], x_data[n_train:]
                    y_train, y_test = y_data[:n_train], y_data[n_train:]

                    train_data = pd.concat([x_train, y_train], axis=1)
                    train_ts = TimeSeries.from_pd(train_data)
                    test_data = pd.concat([x_test, y_test], axis=1)
                    test_ts = TimeSeries.from_pd(test_data)
                    y_true = y_test.values

                    LSTM_start = time()
                    LSTM_preds = compare_LSTMED.run_LSTMED(train_ts, test_ts, parameters.lstmed_parameters)
                    # LSTM_f1 = scoring.f1_score(testing_reals, LSTM_preds)
                    LSTM_end = time()
                    LSTM_time = round(LSTM_end - LSTM_start,4)
                    # print(scoring.all_scores(testing_reals,LSTM_preds))
                    tuned_results+=scoring.all_scores(y_true,LSTM_preds)
                    tuned_results.append(LSTM_time)
                    tuned_values.append(tuned_results)

    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_LSTM_'+df_name+'.csv')






