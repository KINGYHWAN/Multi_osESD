from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report,roc_auc_score
from models import gov_CNN
from utils import parameters
from utils import plotting_modules
from utils import call_datasets
from utils import data_config
from utils import scoring
from time import time
import pandas as pd
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
from models import classifier_Utime

from utils import scoring

from models import main_4_1_supervised_point_anomaly
from models import main_4_2_unsupervised_point_anomaly
from models import main_4_3_supervised_point_anomaly_batch_not_1_폐기
# from models import main_4_4_supervised_point_anomaly_batch_not_1_copied_폐기




def unsupervised_classification_tests():
    print("RUNNING UNSUPERVISED CLASSIFICATION TESTS")

    ALL_RESULTS = []
    df_names = ['JUM','SBE','BSC1','BSC2']
    
    # dataset_names = df_names+[i+"_shortened" for i in df_names]

    model_names = ['CNN','MLP','TabNet']
    col_names = ['Dataset']
    metrics = ['accuracy', 'recall', 'precision', 'f1-score', 'time']

    for model_name in model_names:
        for metric in metrics:
            col_names.append("Before_"+model_name + "_" + metric)
        
    for model_name in model_names:    
        for metric in metrics:
            col_names.append("After_"+model_name + "_" + metric)

    train_per = 0.8
    data_len = 10000
    mob_df_1 = call_datasets.call_MobiAct('JUM',1,2)[:data_len]
    mob_df_2 = call_datasets.call_MobiAct('SBE',5,1)[:data_len]
    mob_df_3 = call_datasets.call_MobiAct('BSC',1,1)[:data_len]
    mob_df_4 = call_datasets.call_MobiAct('BSC',5,1)[:data_len]

    print(mob_df_1['label'].value_counts())
    print(mob_df_2['label'].value_counts())
    print(mob_df_3['label'].value_counts())
    print(mob_df_4['label'].value_counts())

    # stopstopstopstop
    dataset_names = []
    test_times = 5 # 5
    print("Testing 1")
    for i in range(test_times):
        dataset_names.append(df_names[0]+"_"+str(i+1))
        ALL_RESULTS.append(run_classification_tests(mob_df_1,'label',train_per))

    print("Testing 2")
    for i in range(test_times):
        dataset_names.append(df_names[1]+"_"+str(i+1))
        ALL_RESULTS.append(run_classification_tests(mob_df_2,'label',train_per))

    print("Testing 3")
    for i in range(test_times):
        dataset_names.append(df_names[2]+"_"+str(i+1))
        ALL_RESULTS.append(run_classification_tests(mob_df_3,'label',train_per))

    print("Testing 4")
    for i in range(test_times):
        dataset_names.append(df_names[3]+"_"+str(i+1))
        ALL_RESULTS.append(run_classification_tests(mob_df_4,'label',train_per))

    ALL_RESULTS = pd.DataFrame(ALL_RESULTS)
    ALL_RESULTS.columns = col_names[1:]
    ALL_RESULTS['Dataset'] = dataset_names
    ALL_RESULTS = ALL_RESULTS[col_names]
    # print(ALL_RESULTS)

    ALL_RESULTS.to_csv('results//classifier_algorithms_unsup.csv')
    grouped_df = data_config.average_according_to_dataset(ALL_RESULTS,'Dataset')
    grouped_df.to_csv('results//classifier_algorithms_unsup_average.csv')





def run_classification_tests(testing_df,y_label, train_percent):

    # parameters.CNN_parameters.epochs = 2
    # parameters.MLP_parameters.epochs = 2
    # parameters.TabNet_params.num_epochs = 2


    one_data_return_values = []

    #### RUNNING TEST 1 ON MOBIACT JUMP DATASET
    # print("RUNNING TEST 1 CLASSIFICATION HUMAN ACTION")
    # x_cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','azimuth','pitch','roll']
    # # y_label = 'label'
    columns_list = testing_df.columns.tolist()
    columns_list.remove(y_label)
    x_cols = columns_list

    x_cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','azimuth','pitch','roll']


    x_data = testing_df[x_cols]
    y_data = testing_df[y_label]
    y_data = data_config.convert_labels_to_int(y_data)

    # print(y_data.value_counts())
    if len(y_data.unique())==1:
        raise ValueError ("Only one label in dataset, choose another one")

    # print("TRAINING CLASSIFIERS ON ORIGINAL DATA")
    input_data = x_data
    input_y = y_data


    CNN_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_CNN.run_CNN(
        input_data, input_y, train_percent, parameters.CNN_parameters)
    CNN_end = time()
    CNN_time = round(CNN_end-CNN_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    # print(scoring.all_scores(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(CNN_time)


    MLP_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_MLP.run_MLP(
        input_data, input_y, train_percent, parameters.MLP_parameters)
    MLP_end = time()
    MLP_time = round(MLP_end-MLP_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(MLP_time)


    Tab_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_Tabnet.run_tabNet(
        input_data, input_y, train_percent, parameters.TabNet_params)
    Tab_end = time()
    Tab_time = round(Tab_end-Tab_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(Tab_time)



    # print("TRAINING MULTI OSESD")
    train_data = x_data
    clean_mobiact_df, mobiact_predicted_anoms = main_4_2_unsupervised_point_anomaly.multi_oseSD_unsupervised(
        train_data, parameters.osESD_unsupervised)


    # print("TRAINING CLASSIFIERS ON CLEANED DATA")
    input_data = clean_mobiact_df
    input_y = y_data


    CNN_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_CNN.run_CNN(
        input_data, input_y, train_percent, parameters.CNN_parameters)
    CNN_end = time()
    CNN_time = round(CNN_end-CNN_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(CNN_time)


    MLP_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_MLP.run_MLP(
        input_data, input_y, train_percent, parameters.MLP_parameters)
    MLP_end = time()
    MLP_time = round(MLP_end-MLP_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(MLP_time)


    Tab_start = time()
    train_labels, train_preds, test_labels, test_preds, trained_model = classifier_Tabnet.run_tabNet(
        input_data, input_y, train_percent, parameters.TabNet_params)
    Tab_end = time()
    Tab_time = round(Tab_end-Tab_start,3)
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    one_data_return_values += scoring.all_scores(test_labels, test_preds)
    one_data_return_values.append(Tab_time)


    return one_data_return_values




