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

from models import classifier_CNN
from models import classifier_MLP
from models import classifier_Tabnet


from models import main_4_1_supervised_point_anomaly
from models import main_4_2_unsupervised_point_anomaly



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


    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names = ['MOSESD','KNN','iso','AE','VAE','LSTM']
    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)

    data_len = 10000
    arima_dfs = 3
    train_percent = 0.8
    std_val = 4


    for i in range(1):
        print(i)
        print("Tuning ARIMA dataset number %d out of %d datasets"%(i+1,arima_dfs))
        dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, std_val)
        run_osESD_tuning(ARIMA_df,'anomaly',train_percent,'ARIMA'+str(i+1))

    seasonal_dfs = 3
    for i in range(1):
        dataset_names.append('seasonal_'+str(i+1))
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, std_val)
        run_osESD_tuning(seasonal_df,'anomaly',train_percent,'seasonal'+str(i+1))

    Yahoos = call_datasets.call_yahoo_datasets()
    for df_type in Yahoos:
        yahoo_df = data_config.sum_then_inject([df_type[0],df_type[1],df_type[2]], 0.03, std_val)


    ### Preparing NAB dataset
    ### Each is a list of datasets, has anomalies but is not labeled.
    ### Should merge and inject anomalies.

    # traffic_df, exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    std_val = 2
    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    exchange_1624 = []
    for df in exchange_df:
        if len(df) in [1624, 1643]:
            exchange_1624.append(df['value'][:1620])
    mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, std_val)
    run_osESD_tuning(mixed_exchange_1624_df,'anomaly',train_percent,'NAB_1')
    
    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'][:4030])
    mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
    run_osESD_tuning(mixed_awscloudwatch_4032_df,'anomaly',train_percent,'NAB_2')




    # traffic_2500 = []
    # for df in traffic_df:
    #     if len(df)==2500:
    #         traffic_2500.append(df['value'])
    # mixed_traffic_2500_df = data_config.sum_then_inject(traffic_2500,0.03,10)
    # run_osESD_tuning(mixed_traffic_2500_df,'anomaly',0.2,'NAB_3')

def run_osESD_tuning(testing_df,y_col,train_percent,df_name):


    # for yahoo, they need small windows
    rwins = [10,20]
    dwins = [10,20]
    # rwins = [20]
    # dwins = [20]
    init_sizes = [50,100]
    alphas = [0.001,0.01,0.05]
    maxrs = [10]

    total_change_rates = [0.00001,0.0001,0.0005]
    total_o_change_rates = [0.0001,0.001,0.005,0.001]

    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'Time']
    col_names = ['rwin','dwin','init_size','alpha','maxr','total_change_rate','o_change_rate',]
    col_names += metrics

    tuned_values = []
    for rwin in rwins:
        print('running rwin size ',rwin)
        # print(rwins)
        parameters.osESD_supervised.rwin_size=rwin
        for dwin in dwins:
            parameters.osESD_supervised.dwin_size = dwin
            for init_size in init_sizes:
                parameters.osESD_supervised.init_size = init_size
                for alpha in alphas:
                    parameters.osESD_supervised.alpha = alpha
                    for maxr in maxrs:
                        parameters.osESD_supervised.maxr = maxr
                        for change_rate in total_change_rates:
                            parameters.osESD_supervised.total_change_rate = change_rate
                            for o_change_rate in total_o_change_rates:
                                parameters.osESD_supervised.total_o_change_rate = o_change_rate
                                tuned_results = [rwin,dwin,init_size,alpha,maxr,change_rate,o_change_rate]
                                cols = [col for col in testing_df.columns if col != y_col]
                                y_data = testing_df[y_col]
                                train_data = testing_df[cols]

                                test_start_idx = int(len(testing_df ) *train_percent)
                                testing_values = train_data[test_start_idx:]
                                testing_reals = y_data[test_start_idx:]

                                osESD_start = time()
                                osESD_preds = main_4_1_supervised_point_anomaly.multi_osESD_supervised(
                                    train_data, y_data, parameters.osESD_supervised, 0.2, train_percent)
                                osESD_end = time()
                                osESD_time = round(osESD_end - osESD_start,4)
                                tuned_results+=scoring.all_scores(testing_reals,osESD_preds)
                                tuned_results.append(osESD_time)
                                tuned_values.append(tuned_results)


    tuned_values = pd.DataFrame(tuned_values)
    tuned_values.columns = col_names
    print(tuned_values)
    tuned_values.to_csv('results_tuning//tuning_single_point_algorithms_M_OSESD_'+df_name+'.csv')




### 아래는 기존 tuning 때 쓰던 것.
def run_comparing_tests(testing_df,y_col,train_percent):

    one_data_return_values = []
    cols = [col for col in testing_df.columns if col != y_col]
    y_data = testing_df[y_col]
    train_data = testing_df[cols]

    test_start_idx = int(len(testing_df ) *train_percent)
    testing_values = train_data[test_start_idx:]
    testing_reals = y_data[test_start_idx:]


    osESD_start = time()
    osESD_preds = main_4_1_supervised_point_anomaly.multi_osESD_supervised(
        train_data, y_data, parameters.osESD_supervised, train_percent)
    osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,4)
    print(scoring.all_scores(testing_reals,osESD_preds))
    one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    one_data_return_values.append(osESD_time)

    iso_start = time()
    iso_pred = compare_isolation_forest.run_isolation_forest(train_data ,y_data, parameters.isolation_forest_parameters,train_percent)
    # iso_f1 = scoring.f1_score(testing_reals, iso_pred)
    iso_end = time()
    iso_time = round(iso_end -iso_start,4)
    # print(scoring.all_scores(testing_reals,iso_pred))
    one_data_return_values += scoring.all_scores(testing_reals,iso_pred)
    one_data_return_values.append(iso_time)


    AE_start = time()
    AE_pred = compare_AE.run_AE(train_data ,parameters.ae_parameters ,y_data ,train_percent)
    # AE_f1 = scoring.f1_score(testing_reals ,AE_pred)
    AE_end = time()
    AE_time = round(AE_end -AE_start,4)
    # print(scoring.all_scores(testing_reals,AE_pred))
    one_data_return_values += scoring.all_scores(testing_reals,AE_pred)
    one_data_return_values.append(AE_time)


    VAE_start = time()
    VAE_pred = compare_VAE.run_VAE(train_data ,parameters.vae_parameters ,y_data ,train_percent)
    # VAE_f1 = scoring.f1_score(testing_reals ,VAE_pred)
    VAE_end = time()
    VAE_time = round(VAE_end -VAE_start,4)
    # print(scoring.all_scores(testing_reals,VAE_pred))
    one_data_return_values += scoring.all_scores(testing_reals,VAE_pred)
    one_data_return_values.append(VAE_time)


    LSTM_start = time()
    LSTM_pred = compare_LSTMED.run_LSTMED(train_data ,parameters.lstmed_parameters ,y_data ,train_percent)
    # LSTM_f1 = scoring.f1_score(testing_reals ,LSTM_pred)
    LSTM_end = time()
    LSTM_time = round(LSTM_end -LSTM_start,4)
    # print(scoring.all_scores(testing_reals,LSTM_pred))
    one_data_return_values += scoring.all_scores(testing_reals,LSTM_pred)
    one_data_return_values.append(LSTM_time)
    return one_data_return_values

    # print("osESD scores : {:.4f}, ISO scores : {:.4f}, AE scores : {:.4f}, VAE scores : {:.4f}, LSTM scores : {:.4f}".
    #       format(osESD_f1, iso_f1, AE_f1, VAE_f1, LSTM_f1))
    # print("osESD time : {:.4f}, ISO time : {:.4f}, AE time : {:.4f}, VAE time : {:.4f}, LSTM time : {:.4f}".
    #       format(osESD_time, iso_time, AE_time, VAE_time, LSTM_time))






