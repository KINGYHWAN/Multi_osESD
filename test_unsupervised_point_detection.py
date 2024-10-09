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



def unsupervised_multivariate_tests():
    print("RUNNING UNSUPERVISED POINT ANOMALY TESTS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_directory = 'results_tuning'
    data_name = 'art_synth_quad'
    prototype_df = call_datasets.call_ARIMA_datasets_prototype()


    # adasdasd
    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names = ['osESD','iso','AE','VAE','LSTM']
    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)
    ### Preparing ARIMA dataset
    ### is univariate, has no labels.


    data_len = 10000
    for i in range(3):
        dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, 5)
        ALL_RESULTS.append(run_unsupervised_comparing_tests(ARIMA_df, 'anomaly', 0.2,parameters.osESD_supervised_ARIMA))
    
    for i in range(3):
        dataset_names.append('seasonal_'+str(i+1))
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90],  [0.1, 0.2, 0.3], 3)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90],  [0.1, 0.2, 0.3], 3)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90],  [0.1, 0.2, 0.3], 3)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, 5)
        ALL_RESULTS.append(run_unsupervised_comparing_tests(seasonal_df, 'anomaly', 0.2, parameters.osESD_supervised_seasonal))

    Yahoos = call_datasets.call_yahoo_datasets()
    for idx,df_type in enumerate(Yahoos):
        dataset_names.append('yahoo_' + str(idx + 1))
        df_num = len(df_type)
        randoms = random.sample([i for i in range(df_num)], 3)
        len_ = min(len(df_type[randoms[0]]),len(df_type[randoms[1]]),len(df_type[randoms[2]]))
        yahoo_df = data_config.multi_df([df_type[randoms[0]][:len_],df_type[randoms[1]][:len_],df_type[randoms[2]][:len_]],'anomaly')
        ALL_RESULTS.append(run_unsupervised_comparing_tests(yahoo_df, 'anomaly', 0.2, parameters.osESD_supervised_yahoo))


    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    exchange_1624 = []
    for df in exchange_df:
        if len(df) in [1624, 1643]:
            exchange_1624.append(df['value'][:1624])
    dataset_names.append('NAB1')
    mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, 10)
    ALL_RESULTS.append(run_unsupervised_comparing_tests(mixed_exchange_1624_df, 'anomaly', 0.2, parameters.osESD_supervised_NAB1))

    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'])
    dataset_names.append('NAB2')
    mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, 10)
    ALL_RESULTS.append(run_unsupervised_comparing_tests(mixed_awscloudwatch_4032_df, 'anomaly', 0.2, parameters.osESD_supervised_NAB2))

    ALL_RESULTS = pd.DataFrame(ALL_RESULTS)
    ALL_RESULTS.columns = col_names[1:]
    ALL_RESULTS['Dataset'] = dataset_names
    ALL_RESULTS = ALL_RESULTS[col_names]
    print(ALL_RESULTS)

    ALL_RESULTS.to_csv('results//single_point_algorithms.csv')
    grouped_df = data_config.average_according_to_dataset(ALL_RESULTS,'Dataset')
    grouped_df.to_csv('results//single_point_algorithms_average.csv')


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




def run_unsupervised_comparing_tests(testing_df,y_col,train_percent,osESD_params):

    one_data_return_values = []
    cols = [col for col in testing_df.columns if col != y_col]
    y_data = testing_df[y_col]
    train_data = testing_df[cols]

    test_start_idx = int(len(testing_df ) *train_percent)
    testing_values = train_data[test_start_idx:]
    testing_reals = y_data[test_start_idx:]

    print(sum(y_data))
    # osESD_start = time()
    # osESD_preds = main_4_3_supervised_point_anomaly_batch_not_1.multi_osESD_supervised(
    #     train_data, y_data, parameters.osESD_supervised, train_percent)
    # osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    # osESD_end = time()
    # osESD_time = osESD_end - osESD_start
    # # print(sum(osESD_preds))
    # # print(osESD_f1)
    # # print(classification_report(testing_reals, osESD_preds))
    # one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    # print(scoring.)


    osESD_start = time()
    osESD_preds = main_4_2_unsupervised_point_anomaly.multi_oseSD_unsupervised(
        train_data, osESD_params)
    osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,4)
    print(len(testing_reals),len(osESD_preds))
    print(scoring.all_scores(testing_reals,osESD_preds))
    one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    one_data_return_values.append(osESD_time)

    # return one_data_return_values

    print(scoring.all_scores(testing_reals,osESD_preds))
    # adsadsads

    iso_start = time()
    iso_pred = compare_isolation_forest.run_IF(train_data ,y_data, parameters.IF_parameters,train_percent)
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






