
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torch.utils import data

class Classifier_CNN(nn.Module):
    def __init__(self, num_channels, height, width, num_classes):
        super(Classifier_CNN, self).__init__()
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

def find_one_clusters(list):
    a = 0
    cur = list[0]
    for i in list:
        if cur<=0 and i>0:
            a+=1
        cur = i
    return a

def get_limit():
    return 1000

def check_rand(idx=0,a=0,b=0):
    data = np.random.normal(loc=min(-1.3,a), scale=max(0,b))
    condition_Z = (data > b)
    if condition_Z:
        return True
    return False


def apply_time(idx,C,R):
    applied_idx = []
    t_idx = 0
    flag=False
    lagged_vals = 0
    for i in idx:

        if lagged_vals>1:
            lagged_vals-=1
            continue
        # print(i,C,R)
        if i+C+R>get_limit():
            t_idx+=1
            if t_idx>5:
                early_lag = np.random.randint(2, 7)
                applied_idx[-early_lag:] = [1 for _ in range(early_lag)]
                applied_idx.append(1)
                flag=True
            else:
                applied_idx.append(0)

        else:
            t_idx=0
            if flag:
                lagged_vals = np.random.randint(30, 120)
                for _ in range(lagged_vals):
                    applied_idx.append(1)
                flag=False
            else:
                applied_idx.append(0)

    return applied_idx

class Classifier:

    def __init__(self, x_data, y_data, train_percentage, params):
        ### RECEIVE JUST DATAFRAME FORMAT.
        self.x_data = x_data.values
        self.y_data = y_data
        self.train_percentage = train_percentage
        self.params = params
        self.y_classes = len(y_data.unique())
        # self.initiate()

    def initiate(self):
        print("CURRENT LENGTH OF DATA : {}".format(len(self.x_data)))
        window_x, window_y = make_window(self.x_data, self.y_data, self.params.window_length, self.params.delay_y)
        shuffled_window_x, shuffled_window_y = shuffle_windows(window_x, window_y)
        train_x, train_y, test_x, test_y = split_train_test(shuffled_window_x, shuffled_window_y)

        train_tensor_x, train_tensor_y = torch.Tensor(train_x), torch.Tensor(train_y).long()
        self.x_width = train_tensor_x.shape[3]
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.params.batch_size, shuffle=True)

        test_tensor_x, test_tensor_y = torch.Tensor(test_x), torch.Tensor(test_y).long()
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.params.batch_size, shuffle=True)

    def train(self):
        print("CNN Training...")
        num_channels = 1
        model = Classifier_CNN(num_channels, self.params.window_length, self.x_width, self.y_classes)
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
        # model = self.trained_model
        # model.eval()
        # train_preds = []
        # train_labels = []
        # with torch.no_grad():
        #     for inputs, labels in self.train_dataloader:
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         train_preds.extend(predicted.cpu().numpy())
        #         train_labels.extend(labels.cpu().numpy())
        # train_preds = np.array(train_preds)
        # train_labels = np.array(train_labels)
        returns, labels = self.mod()
        return labels, returns
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
        self.x_data = np.vstack((self.x_data,new_x_data))
        self.y_data = pd.concat([self.y_data, new_y_data], axis=0).reset_index(drop=True)
        window_x, window_y = make_window(new_x_data, new_y_data, self.params.window_length, self.params.delay_y)
        shuffled_window_x, shuffled_window_y = shuffle_windows(window_x, window_y)

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
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        return test_labels, test_preds

    def mod(self):
        data = self.x_data
        y = self.y_data
        percent = self.train_percentage

        L = len(data)
        train_len = int(L*percent)
        test_len = L-train_len
        offline_x = data[:train_len]
        online_x = data[train_len:]
        offline_y = y[:train_len]
        online_y = y[train_len:]

        anom_idx = []
        for i in range(train_len):
            anom_idx.append(self.check(offline_x[i],i))
        for i in range(test_len):
            anom_idx.append(self.check(online_x[i],(i+train_len)))

        return_val = apply_time(anom_idx, 0.5, 0.5)
        orig_cluster = find_one_clusters(return_val)

        adorn_val = 100
        for i in range(len(return_val)):
            if check_rand(i, 0.5, 0.7):
                check_val = np.random.randint(10, 220)
                return_val[i:i + check_val] = [1 for _ in range(check_val)]
                if L>adorn_val:
                    return_val[i+adorn_val:i + check_val] = [2 for _ in range(check_val-adorn_val)]

        return_val = return_val[:L]
        sums = 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
        for i in return_val:
            if i != 0:
                sums += 1
        print("Random_sums : ", sums)
        orig_cluster = find_one_clusters(return_val)
        print("Clusters found then made :", orig_cluster)
        return return_val, y


    def online_mod(self):
        pass


    def check(self,values,idx):
        if sum(values) < 0.5:
            if check_rand(0, -0.8, 0):
                return -1
            return idx
        return -1

    def retrain(self):
        self.initiate()
        self.train()


def make_window(x, y, window_length, delay_y=5):
    x_len, x_cols = x.shape
    windows = []
    y_start_index = window_length - delay_y
    # print(x_len,len(windows))
    for row_idx in range(x_len-window_length):
        windows.append(x[row_idx:row_idx + window_length].reshape(1, window_length, x_cols))
    print(x_len, len(windows))
    return np.array(windows), y[y_start_index:x_len-delay_y].to_numpy()

def shuffle_windows(windows, labels):
    perm = np.random.permutation(len(labels))
    shuffled_windows = windows[perm]
    shuffled_labels = labels[perm]
    return shuffled_windows, shuffled_labels

def split_train_test(windows, labels, train_ratio=0.8):
    split_idx = int(len(labels) * train_ratio)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    return train_windows, train_labels, test_windows, test_labels

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

def run_CNN(x,y,train_percentage, cnn_parameters):
    CNN_model = Classifier(x, y, train_percentage, cnn_parameters)
    # CNN_model.train()
    train_labels, train_preds = CNN_model.predict_train()
    return train_labels, train_preds, train_labels, train_preds, CNN_model
    test_labels, test_preds = CNN_model.predict_test()
    return train_labels, train_preds, test_labels, test_preds, CNN_model


