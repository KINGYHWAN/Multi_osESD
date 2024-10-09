from pytorch_tabnet.tab_model import TabNetClassifier
from utils import data_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from utils import deep_learning_utils

def run_tabNet(x_data,y_data,train_percent, params):
    # print("Training TabNet")
    scaler = StandardScaler()
    scaled_x_data = scaler.fit_transform(x_data)

    window_x, window_y = deep_learning_utils.make_lin_window(scaled_x_data, y_data, params.window_length, y_index=params.y_index)
    shuffled_window_x, shuffled_window_y = deep_learning_utils.shuffle_windows(window_x, window_y)
    train_x, train_y, test_x, test_y = deep_learning_utils.split_train_test(shuffled_window_x, shuffled_window_y, train_ratio=train_percent)
    tabnet_model = TabNetClassifier(verbose=0, seed=42)

    tabnet_model.fit(
        train_x, train_y,
        eval_set=[(test_x, test_y)],
        eval_name=['eval'],
        eval_metric=[params.eval_metric], #
        max_epochs=params.num_epochs,  # You can set it lower if the training is slow
        patience=50,  # How many epochs to wait after last time validation loss improved.
        # Useful to prevent overfitting
        batch_size=params.batch_size,  # Adjust based on your dataset size and memory capacity
        virtual_batch_size=128,  # Size of the mini batches used for "Ghost Batch Normalization"
        num_workers=0,  # Number of workers used in DataLoader - can adjust based on your system
        drop_last=False  # Whether to drop the last incomplete batch if it's not divisible by the batch size
    )
    train_preds = tabnet_model.predict(train_x)
    test_preds = tabnet_model.predict(test_x)
    return train_y, train_preds, test_y, test_preds, tabnet_model


class TabNet_params:
    window_length = 20
    y_index = 1
    num_epochs = 100
    eval_metric = 'accuracy'
    batch_size = 128

'''
binary classification metrics : ‘auc’, ‘accuracy’, ‘balanced_accuracy’, ‘logloss’
multiclass classification : ‘accuracy’, ‘balanced_accuracy’, ‘logloss’
regression: ‘mse’, ‘mae’, ‘rmse’, ‘rmsle’
'''

if __name__=='__main__':

    mobiact_1 = pd.read_csv('..//datasets//MobiAct_dataset//Annotated Data//JUM//JUM_1_2_annotated.csv')
    x_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
    y_col = 'label'
    x_data = mobiact_1[x_cols]
    y_data = mobiact_1[y_col]
    y_data = data_config.convert_labels_to_int(y_data)

    train_labels, train_preds, test_labels, test_preds, trained_model = run_tabNet(x_data, y_data, 0.8, TabNet_params)

    print(classification_report(train_labels, train_preds))
    print(classification_report(test_labels, test_preds))


