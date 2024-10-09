import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import torch.optim as optim
from utils import deep_learning_utils
from utils import data_config

# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, classification_report
# import torch.optim as optim

class Classifier_CNN_network(nn.Module):
    def __init__(self, num_channels, height, width, num_classes):
        super(Classifier_CNN_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.flattened_features = self._get_conv_output((num_channels, height, width))
        self.fc1 = nn.Linear(self.flattened_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.dropout(self.flatten(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Classifier_CNN:

    def __init__(self, x_data, y_data, train_percentage, params):
        ### RECEIVE JUST DATAFRAME FORMAT.
        self.x_data = x_data.values
        self.y_data = y_data
        self.train_percentage = train_percentage
        self.params = params
        self.y_classes = len(y_data.unique())

    def initiate(self):
        window_x, window_y = deep_learning_utils.make_window(self.x_data, self.y_data, self.params.window_length, self.params.delay_y)
        shuffled_window_x, shuffled_window_y = deep_learning_utils.shuffle_windows(window_x, window_y)
        train_x, train_y, test_x, test_y = deep_learning_utils.split_train_test(shuffled_window_x, shuffled_window_y, train_ratio=self.train_percentage)

        train_tensor_x, train_tensor_y = torch.Tensor(train_x), torch.Tensor(train_y).long()
        self.x_width = train_tensor_x.shape[3]
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.params.batch_size, shuffle=True)

        test_tensor_x, test_tensor_y = torch.Tensor(test_x), torch.Tensor(test_y).long()
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.params.batch_size, shuffle=True)

    def train(self):
        # print("CNN Training...")
        num_channels = 1
        model = Classifier_CNN_network(num_channels, self.params.window_length, self.x_width, self.y_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr)
        for epoch in range(self.params.epochs):
            for inputs, labels in self.train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        self.trained_model = model

    def predict_train(self):
        model = self.trained_model
        model.eval()
        train_preds = []
        train_labels = []
        with torch.no_grad():
            for inputs, labels in self.train_dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        # print(f1_score(train_labels, train_preds))
        return train_labels, train_preds

    def predict_test(self):
        model = self.trained_model
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                # Forward pass to get outputs
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        # print(f1_score(test_labels, test_preds))
        return test_labels, test_preds

    def online_test(self, new_x_data, new_y_data):  ### receive data until batch is full. predict and return predictions

        ### 근데 해당 x_data y_data 을 기존 데이터셋에 추가도 하자.
        ### 나중에 다시 train도 가능하게 -> retrain 함수
        new_x_data = new_x_data.values
        # print(len(self.x_data))

        # print(len(new_x_data))

        self.x_data = np.vstack((self.x_data,new_x_data))
        # print(len(self.x_data))
        # print("X DONE")
        # self.y_data = np.concatenate((self.y_data,new_y_data))
        self.y_data = pd.concat([self.y_data, new_y_data], axis=0).reset_index(drop=True)
        # print("Y DONE")

# np.vstack((arr1, arr2))
        # print(new_x_data)
        # print(new_y_data)
        window_x, window_y = deep_learning_utils.make_window(new_x_data, new_y_data, self.params.window_length, self.params.delay_y)
        shuffled_window_x, shuffled_window_y = deep_learning_utils.shuffle_windows(window_x, window_y)
        # train_x, train_y, test_x, test_y = split_train_test(shuffled_window_x, shuffled_window_y, )

        # print("MADE WINDOWS")
        test_tensor_x, test_tensor_y = torch.Tensor(shuffled_window_x), torch.Tensor(shuffled_window_y).long()
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
        online_test_dataloader = DataLoader(test_dataset, batch_size=self.params.batch_size, shuffle=True)

        model = self.trained_model
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in online_test_dataloader:
                # Forward pass to get outputs
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        # print("END TRAINING")
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        # print(f1_score(test_labels, test_preds))
        return test_labels, test_preds

    def retrain(self):
        self.initiate()
        self.train()
        pass


def run_CNN(x,y,train_percentage, cnn_parameters):
    ### Take x and y as dataframe.
    testing_model = Classifier_CNN(x, y, train_percentage, cnn_parameters)
    testing_model.initiate()
    testing_model.train()
    train_labels, train_preds = testing_model.predict_train()
    test_labels, test_preds = testing_model.predict_test()
    # print(classification_report(train_labels, train_preds))
    # print(classification_report(test_labels, test_preds))
    return train_labels, train_preds, test_labels, test_preds, testing_model


def online_train(x_data, y_data, model, params):

    training_idx = 0
    retrain_idx = 1
    end_flag = False

    while training_idx < len(x_data):
        data_queue = []
        y_queue = []
        for queue_idx in range(params.window_length*2):
            if training_idx == len(x_data):
                end_flag = True
                break
            data_queue.append(x_data.iloc[training_idx].values)
            y_queue.append(y_data.iloc[training_idx])
            training_idx+=1
        if end_flag:
            break
        # print(classification_report(online_labels, online_preds))
        online_labels, online_preds = model.online_test(pd.DataFrame(data_queue), pd.Series(y_queue))
        if training_idx>retrain_idx*params.retrain_frequency :
            model.retrain()
            retrain_idx+=1

    train_labels, train_preds = model.predict_train()
    test_labels, test_preds = model.predict_test()
    return train_labels, train_preds, test_labels, test_preds, model
    ### IF CONTINUOUS ONLINE LEARNING, THERE IS NO RETURN

### Example usage
class CNN_parameters:
    window_length = 50
    batch_size = 32
    epochs = 10
    lr = 0.0003
    delay_y = 10
    retrain_frequency = 5000


if __name__=='__main__':
    mobiact_1 = pd.read_csv('..//datasets//MobiAct_dataset//Annotated Data//JUM//JUM_1_2_annotated.csv')
    x_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
    y_col = 'label'
    x_data = mobiact_1[x_cols]
    y_data = mobiact_1[y_col]
    y_data = data_config.convert_labels_to_int(y_data)

    scaler = StandardScaler()
    x_data = pd.DataFrame(scaler.fit_transform(x_data))  ### 이거 나중에 좀 수정해라.

    input_data = x_data
    input_y = y_data

    train_labels, train_preds, test_labels, test_preds, trained_model = run_CNN(input_data, input_y, 0.8,
                                                                                CNN_parameters)

    print(classification_report(train_labels, train_preds))
    print(classification_report(test_labels, test_preds))

    mobiact_1 = pd.read_csv('..//datasets//MobiAct_dataset//Annotated Data//JUM//JUM_1_2_annotated.csv')
    x_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
    y_col = 'label'
    x_data = mobiact_1[x_cols]
    y_data = mobiact_1[y_col]
    y_data = data_config.convert_labels_to_int(y_data)

    online_train_labels, online_train_preds, online_test_labels, online_test_preds, trained_model = online_train(x_data,
                                                                                                                 y_data,
                                                                                                                 trained_model,
                                                                                                                 CNN_parameters)

    print(classification_report(online_train_labels, online_train_preds))
    print(classification_report(test_labels, test_preds))
