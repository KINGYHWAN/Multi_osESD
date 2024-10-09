
import pandas as pd
import numpy as np
import os

from .data_config import multi_df

def call_MobiAct(name,idx_1,idx_2):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    traffic_directory = '..//datasets//MobiAct_dataset//Annotated Data//'
    temp_dir = os.path.join(current_directory, traffic_directory)
    path = name+"//"+name+"_"+str(idx_1)+"_"+str(idx_2)+"_annotated.csv"
    df_dir = os.path.join(temp_dir,path)
    df = pd.read_csv(df_dir)
    return df


def call_ARIMA_dataset(length, alphas, betas, noise_std=1):
    # ARIMA_df = call_ARIMA_dataset(1000, [0.2, 0.27, 0.1], [-0.3, 0.1])
    # plt.plot(ARIMA_df['value'])
    noise = np.random.normal(0, noise_std, length)
    ts = np.zeros(length)
    ts[0] = np.mean(noise[:max(len(alphas), len(betas))])
    alpha_len = len(alphas)
    beta_len = len(betas)
    for t in range(1, length):
        AR_part = sum(alphas[i] * ts[t - i - 1] for i in range(min(alpha_len, t)))
        MA_part = sum(betas[i] * noise[t - i - 1] for i in range(min(beta_len, t)))
        ts[t] = AR_part + MA_part + noise[t]
    ts_df = pd.DataFrame({'value': ts})
    return ts_df

def call_seasonal_dataset(length, frequencies, intercepts, noise_std=1, slope=0.01):
    time = np.arange(0, length)
    linear_trend = slope * time + intercepts[0]
    ts = linear_trend
    for freq, intercept in zip(frequencies, intercepts[1:]):
        phase = np.random.rand() * 2 * np.pi
        seasonality = intercept + np.sin(2 * np.pi * time / freq + phase) * 10
        ts += seasonality
    noise = noise_std * np.random.normal(size=length)
    ts += noise
    return pd.DataFrame({'value': ts})



def call_yahoo_datasets():
    ### Will use only datasets with known anomalies.
    ### They are not labeled.
    current_directory = os.path.dirname(os.path.realpath(__file__))

    A3 = []
    traffic_directory = '..//datasets//yahoo_dataset//A3Benchmark'
    temp_dir = os.path.join(current_directory,traffic_directory)
    for data in os.listdir(temp_dir):
        if data.endswith('.csv'):
            sec_dir = os.path.join(temp_dir,data)
            df = pd.read_csv(sec_dir)
            A3.append(df['value'])

    return [A3]


def call_NAB_dataset():
    ### Will use only datasets with known anomalies.
    ### They are not labeled.
    current_directory = os.path.dirname(os.path.realpath(__file__))


    real_Traffic = []
    traffic_directory = '..//datasets//NAB_benchmark//realTraffic'
    temp_dir = os.path.join(current_directory,traffic_directory)
    for data in os.listdir(temp_dir):
        if data.endswith('.csv'):
            sec_dir = os.path.join(temp_dir,data)
            df = pd.read_csv(sec_dir)
            real_Traffic.append(df)


    real_Adexchange = []
    current_directory = os.path.dirname(os.path.realpath(__file__))
    traffic_directory = '..//datasets//NAB_benchmark//realAdExchange'
    temp_dir = os.path.join(current_directory,traffic_directory)
    for data in os.listdir(temp_dir):
        if data.endswith('.csv'):
            sec_dir = os.path.join(temp_dir,data)
            df = pd.read_csv(sec_dir)
            real_Adexchange.append(df)

    realAWSCloudwatch = []

    current_directory = os.path.dirname(os.path.realpath(__file__))
    traffic_directory = '..//datasets//NAB_benchmark//realAWSCloudwatch'
    temp_dir = os.path.join(current_directory,traffic_directory)
    for data in os.listdir(temp_dir):
        if data.endswith('.csv'):
            sec_dir = os.path.join(temp_dir,data)
            df = pd.read_csv(sec_dir)
            realAWSCloudwatch.append(df)

    return real_Adexchange, realAWSCloudwatch
    return real_Traffic, real_Adexchange, realAWSCloudwatch


def call_ARIMA_datasets_prototype():
    df1 = pd.read_csv('datasets//ARIMA_datasets//ARIMA1_quad_1.csv')
    df2 = pd.read_csv('datasets//ARIMA_datasets//ARIMA1_quad_2.csv')
    df3 = pd.read_csv('datasets//ARIMA_datasets//ARIMA1_quad_3.csv')
    df4 = pd.read_csv('datasets//ARIMA_datasets//ARIMA1_quad_4.csv')
    y_col = 'anomaly'
    new_df = multi_df([df1,df2,df3,df4],y_col)
    return new_df

def call_gov_dataset():
    df = pd.read_csv('datasets//project_datasets//government//new_testing_dataset.csv')
    return df




def call_ECG5000_dataset():

    # ### Call in ECG5000 dataset.
    # ### ECG5000은 내 문제랑 맞지는 않는다. 이거는 140개 value 받고
    # ### classification을 하는거지, point anomaly를 찾는 데이터셋은 아니다.
    # train, test = call_datasets.call_ECG5000_dataset()
    # print(len(train), len(test))
    # print(train.head())
    # print(test.tail())
    # print(sum(train['anomaly']))
    # print(sum(test['anomaly']))

    column_names = ['X' + str(i) for i in range(140)]
    column_names.append('anomaly')

    with open('datasets//ECG5000//ECG5000_TRAIN.txt', 'r') as file:
        lines = file.readlines()
    temp_df = [[0 for _ in range(141)] for _ in range(len(lines))]
    for idx,line in enumerate(lines):
        data = line.split()
        temp_df[idx][:140] = [float(value) for value in data[1:]]
        temp_df[idx][140] = int(float(data[0]))


    ecg_train = pd.DataFrame(temp_df,columns=column_names)

    # ecg_train['anomaly'] = ecg_train['anomaly'].apply(lambda x: 1 if x == 0 else 0)



    with open('datasets//ECG5000//ECG5000_TEST.txt', 'r') as file:
        lines = file.readlines()
    temp_df = [[0 for _ in range(141)] for _ in range(len(lines))]
    for idx,line in enumerate(lines):
        data = line.split()
        temp_df[idx][:140] = [float(value) for value in data[1:]]
        temp_df[idx][140] = int(float(data[0]))
    ecg_test = pd.DataFrame(temp_df,columns=column_names)
    # ecg_test['anomaly'] = ecg_test['anomaly'].apply(lambda x: 1 if x == 0 else 0)
    return ecg_train, ecg_test





