
import torch
from time import time
import pandas as pd
import numpy as np
import random
from models import ablation_no_regression_replace
from models import ablation_no_decaying_lr
from models import ablation_no_f1_backpropagation

from models import main_4_1_supervised_point_anomaly
from utils import call_datasets
from utils import data_config
from utils import scoring
from utils import parameters



def ablation_test():

    print("RUNNING ABLATION TESTS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_directory = 'results_tuning'
    data_name = 'art_synth_quad'
    prototype_df = call_datasets.call_ARIMA_datasets_prototype()


    ALL_RESULTS = []
    dataset_names = [] ### Used in rows.
    model_names = ['full_MOSESD','no_back','no_replace']

    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']
    for model_name in model_names:
        for metric in metrics:
            col_names.append(model_name + "_" + metric)
    ### Preparing ARIMA dataset
    ### is univariate, has no labels.

    data_len = 10000
    test_runs = 5
    std_val = 4
    train_percent = 0.8
    for i in range(test_runs):
        dataset_names.append('ARIMA_'+str(i+1))
        ARIMA_1 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_2 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_3 = call_datasets.call_ARIMA_dataset(data_len, [0.6, -0.5, 0.4, -0.4, 0.3], [0.3, -0.2], 1)
        ARIMA_df = data_config.sum_then_inject([ARIMA_1 ,ARIMA_2 ,ARIMA_3], 0.03, std_val)
        ALL_RESULTS.append(run_ablation_tests(ARIMA_df, 'anomaly', train_percent, parameters.osESD_supervised_ARIMA))

    for i in range(test_runs):
        dataset_names.append('seasonal_'+str(i+1))
        seasonal_1 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_2 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_3 = call_datasets.call_seasonal_dataset(data_len, [365, 180, 90] ,  [0.1, 0.2, 0.3], 1)
        seasonal_df = data_config.sum_then_inject([seasonal_1, seasonal_2, seasonal_3], 0.03, std_val)
        # ALL_RESULTS.append(run_comparing_tests(seasonal_df, 'anomaly', train_percent, parameters.osESD_supervised_seasonal))
        ALL_RESULTS.append(run_ablation_tests(seasonal_df, 'anomaly', train_percent, parameters.osESD_supervised_seasonal))

    Yahoos = call_datasets.call_yahoo_datasets()
    for df_type in (Yahoos):
        for idx in range(test_runs):
            dataset_names.append('yahoo_' + str(idx + 1))
            # yahoo_df = data_config.multi_df([df_type[0],df_type[1],df_type[2]],'anomaly')
            yahoo_df = data_config.sum_then_inject([df_type[0],df_type[1],df_type[2]], 0.03, std_val)
            ALL_RESULTS.append(run_ablation_tests(yahoo_df, 'anomaly', train_percent, parameters.osESD_supervised_yahoo))
    
    
    std_val = 2
    exchange_df, awscloudwatch_df = call_datasets.call_NAB_dataset()
    exchange_1624 = []
    for df in exchange_df:
        if len(df) in [1624, 1643]:
            exchange_1624.append(df['value'][:1620])
            
    for i in range(test_runs):
        dataset_names.append('NABrealTraffic_'+str(i+1))
        mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, std_val)
        ALL_RESULTS.append(run_ablation_tests(mixed_exchange_1624_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB1))

    awscloudwatch_4032 = []
    for df in awscloudwatch_df:
        if len(df)==4032:
            awscloudwatch_4032.append(df['value'][:4030])
    for i in range(test_runs):
        dataset_names.append('NABCloudWatch_'+str(i+1))
        mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
        ALL_RESULTS.append(run_ablation_tests(mixed_awscloudwatch_4032_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB2))



    # exchange_1624 = []
    # for df in exchange_df:
    #     if len(df) in [1624, 1643]:
    #         exchange_1624.append(df['value'][:1620])
    # dataset_names.append('NABrealTraffic')
    # mixed_exchange_1624_df = data_config.sum_then_inject(exchange_1624, 0.03, std_val)
    # ALL_RESULTS.append(run_ablation_tests(mixed_exchange_1624_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB1))

    # awscloudwatch_4032 = []
    # for df in awscloudwatch_df:
    #     if len(df)==4032:
    #         awscloudwatch_4032.append(df['value'][:4030])
    # dataset_names.append('NABCloudWatch')
    # mixed_awscloudwatch_4032_df = data_config.sum_then_inject(awscloudwatch_4032, 0.03, std_val)
    # ALL_RESULTS.append(run_ablation_tests(mixed_awscloudwatch_4032_df, 'anomaly', train_percent, parameters.osESD_supervised_NAB2))
  
    ALL_RESULTS = pd.DataFrame(ALL_RESULTS)
    ALL_RESULTS.columns = col_names[1:]
    ALL_RESULTS['Dataset'] = dataset_names
    ALL_RESULTS = ALL_RESULTS[col_names]
    print(ALL_RESULTS)

    ALL_RESULTS.to_csv('results//ablation_single_point_algorithms.csv')
    grouped_df = data_config.average_according_to_dataset(ALL_RESULTS,'Dataset')
    grouped_df.to_csv('results//ablation_single_point_algorithms_average.csv')


def run_ablation_tests(testing_df, y_col, train_percent, osESD_params):

    one_data_return_values = []
    cols = [col for col in testing_df.columns if col != y_col]
    y_data = testing_df[y_col]
    train_data = testing_df[cols]

    test_start_idx = int(len(testing_df ) *train_percent)
    testing_values = train_data[test_start_idx:]
    testing_reals = y_data[test_start_idx:]

    # print(sum(y_data))

    osESD_start = time()
    osESD_preds = main_4_1_supervised_point_anomaly.multi_osESD_supervised(train_data, y_data, osESD_params, 0.2, train_percent)
    osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,3)
    # print(scoring.all_scores(testing_reals,osESD_preds))
    one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    one_data_return_values.append(osESD_time)


    osESD_start = time()
    osESD_preds = ablation_no_f1_backpropagation.multi_osESD_supervised_no_backpropogation(train_data, y_data, osESD_params, 0.2, train_percent)
    osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,3)
    # print(scoring.all_scores(testing_reals,osESD_preds))
    one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    one_data_return_values.append(osESD_time)
    # asdasdssd

    osESD_start = time()
    osESD_preds = ablation_no_regression_replace.multi_osESD_supervised_no_replace(train_data, y_data, osESD_params, 0.2, train_percent)
    osESD_f1 = scoring.f1_score(testing_reals, osESD_preds)
    osESD_end = time()
    osESD_time = round(osESD_end - osESD_start,3)
    # print(scoring.all_scores(testing_reals,osESD_preds))
    one_data_return_values += scoring.all_scores(testing_reals,osESD_preds)
    one_data_return_values.append(osESD_time)

    return one_data_return_values

