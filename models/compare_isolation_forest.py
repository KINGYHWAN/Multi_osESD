
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def run_IF(train_df, test_df, params):
    # values = x_data.values.reshape(-1, 1)
    # values = x_data.iloc[:int(len(x_data)*train_percent)].values
    # y_train = y_data.iloc[:int(len(x_data)*train_percent)]

    anom_percent = sum(train_df[1])/len(train_df[1])
    # if anom_percent==0:
    #     anom_percent = 0.01
    # else:
    #     anom_percent *= 0.75
    
    values = train_df[0].values
    y_train = train_df[1]

    params.contamination = anom_percent
    isolation_forest = IsolationForest(n_estimators = params.n_estimators,
                                       contamination = params.contamination,
                                       max_samples = params.max_samples,
                                       random_state = 42)
    isolation_forest.fit(values)
    # anom_preds = isolation_forest.predict(x_data.values)
    anom_preds = isolation_forest.predict(test_df[0].values)
    anom_preds[anom_preds == 1] = 0
    anom_preds[anom_preds == -1] = 1
    # pred_index = list(np.where(anom_preds == 1)[0])
    # test_preds = anom_preds[int(len(x_data)*train_percent):]
    # print(train_percent)
    # print(len(anom_preds))
    # if train_percent<1:
    #     test_preds = anom_preds[int(len(x_data)*train_percent):]
    # else:
    #     test_preds = anom_preds[train_percent:]
    # print(test_preds,len(test_preds))
    return anom_preds.reshape((len(anom_preds), 1))

    # if train_percent<1:
    #     train_scores = model.get_anomaly_label(train_data[int(len(train_data)*train_percent):])
    # else:
    #     train_scores = model.get_anomaly_label(train_data[train_percent:])
    # preds= (train_scores.to_pd()>0).astype(int)
    # return preds



class isolation_forest_parameters:
    n_estimators = 100
    max_samples = 100
    contamination = 0.01
    plot = False

if __name__=='__main__':
    my_df = pd.read_csv('../../masters_project/Datasets/synthetic/ARIMA1_ber_1.csv')
    pred = run_IF(my_df,isolation_forest_parameters)


