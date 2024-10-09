
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
import time
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import collections
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler

from utils import data_config


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.ReLu = nn.ReLU(inplace=True)
        self.Bnorm = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        h = X
        h = self.conv(h)
        h = self.ReLu(h)
        h = self.Bnorm(h)
        return h


'''create model that segmentates input and based on the segmentation produces classification of the time series'''


class UNet(nn.Module):
    def __init__(self, in_channel, class_num, kernel_size=3, stride=1, padding=None, pooling=2) -> None:
        super(UNet, self).__init__()
        self.Encoder = nn.ModuleList()
        self.Decoder = nn.ModuleList()
        enc_in_channels = [in_channel, 16, None, 16, 32, None, 32, 64, None, 64, 128, None, 128,
                           256]  ## two convolution and one maxpooling
        enc_out_channels = [16, 16, None, 32, 32, None, 64, 64, None, 128, 128, None, 256, 256]

        dec_in_channels = [1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64]
        dec_out_channels = [512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, class_num]
        self.Dropout = nn.Dropout1d(p=0.5)
        for i in range(len(enc_in_channels)):
            if enc_in_channels[i] == None:
                self.Encoder.append(nn.MaxPool1d(2, 2))
            else:
                self.Encoder.append(
                    ConvBlock(enc_in_channels[i], enc_out_channels[i], kernel_size=kernel_size, stride=stride))

        for i in range(len(dec_out_channels)):
            if i % 3 == 0 and i != len(dec_out_channels) - 1:
                self.Decoder.append(nn.ConvTranspose1d(dec_in_channels[i], dec_out_channels[i], 2, 2))
            elif i == len(dec_out_channels) - 1:
                self.Decoder.append(ConvBlock(dec_in_channels[i], dec_out_channels[i], kernel_size=1, stride=1))
            else:
                self.Decoder.append(
                    ConvBlock(dec_in_channels[i], dec_out_channels[i], kernel_size=kernel_size, stride=stride))
        self.deconv1_1 = nn.ConvTranspose1d(256, 128, 3, 3)
        self.deconv1_2 = ConvBlock(256, 128, 3, 1)
        self.deconv1_3 = ConvBlock(128, 128, 3, 1)

        self.deconv2_1 = nn.ConvTranspose1d(128, 64, 3, 3)
        self.deconv2_2 = ConvBlock(128, 64, 3, 1)
        self.deconv2_3 = ConvBlock(64, 64, 3, 1)

        self.deconv3_1 = nn.ConvTranspose1d(64, 32, 3, 3)
        self.deconv3_2 = ConvBlock(64, 32, 3, 1)
        self.deconv3_3 = ConvBlock(32, 32, 3, 1)

        self.deconv4_1 = nn.ConvTranspose1d(32, 16, 3, 3)
        self.deconv4_2 = ConvBlock(32, 16, 3, 1)
        self.deconv4_3 = ConvBlock(16, class_num, kernel_size=1, stride=1)
        # self.zeropad1=nn.ConstantPad1d((90,89),0)
        self.conv1 = nn.Conv1d(in_channels=class_num, out_channels=class_num, kernel_size=1)
        self.sm = nn.Softmax(dim=1)
        self.sm2 = nn.Tanh()

    def forward(self, image):
        inputshape = image.shape

        h = image
        concat_saves = []
        for idx, layer in enumerate(self.Encoder):
            if (idx + 1) % 3 == 0 and idx != len(self.Encoder) - 1:
                concat_saves.append(h)
            h = layer(h)
            # print(h.shape)
            if idx == len(self.Encoder) - 2:
                h = self.Dropout(h)
        cat_idx = len(concat_saves) - 1

        h = self.deconv1_1(h)
        concat_saves[3] = concat_saves[3][:, :, 4:4 + h.shape[2]]
        cat1 = torch.cat([concat_saves[3], h], dim=1)
        h = self.deconv1_2(cat1)
        h = self.deconv1_3(h)

        h = self.deconv2_1(h)
        concat_saves[2] = concat_saves[2][:, :, 4:4 + h.shape[2]]
        cat2 = torch.cat([concat_saves[2], h], dim=1)
        h = self.deconv2_2(cat2)
        h = self.deconv2_3(h)

        h = self.deconv3_1(h)
        c_crop3 = CenterCrop((h.shape[2]))
        concat_saves[1] = concat_saves[1][:, :, 4:4 + h.shape[2]]
        cat3 = torch.cat([concat_saves[1], h], dim=1)
        h = self.deconv3_2(cat3)
        h = self.deconv3_3(h)

        h = self.deconv4_1(h)
        concat_saves[0] = concat_saves[0][:, :, 4:4 + h.shape[2]]
        cat4 = torch.cat([concat_saves[0], h], dim=1)
        h = self.deconv4_2(cat4)
        h = self.deconv4_3(h)

        l_pad = int((inputshape[2] - h.shape[2]) / 2)
        r_pad = inputshape[2] - h.shape[2] - l_pad

        pad = nn.ConstantPad1d((l_pad, r_pad), 0)
        h = pad(h)
        h = self.conv1(h)
        h = self.sm2(h)
        # h=nn.Linear()
        avgpool = nn.AvgPool1d(h.shape[2])
        h = avgpool(h)
        h = self.conv1(h)
        h = self.sm(h)
        return h.squeeze(2)


class Trainer:
    def __init__(self ,data ,net ,epoch ,criterion) :
        self.data =data
        self.net =net
        self.epoch =epoch
        self.criterion =criterion
        if torch.cuda.is_available():
            self.net.cuda()
        self.init()
        pass
    def init(self):
        dataset_size = len(self.data)
        train_size = int(dataset_size * 0.7)
        validation_size = int(dataset_size * 0.05)
        test_size = dataset_size - train_size - validation_size
        train_dataset, validation_dataset, test_dataset = random_split(self.data, [train_size, validation_size, test_size])
        self.train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.validation_dataloader =DataLoader(validation_dataset ,batch_size=64 ,shuffle=True)
    def to_cuda(self ,gs):
        if torch.cuda.is_available():
            if type(gs)==list:
                return [g.cuda() for g  in gs]
            return gs.cuda()
        return gs
    def tester(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss =0
        t_correct =0
        t_size =len(self.test_dataloader.dataset)
        with torch.no_grad():
            self.net.eval()
            for batch ,(X ,y) in enumerate(tqdm(self.test_dataloader ,desc=str(200) ,unit='b')):
                X=X.to(device).float()
                y=y.to(device).long()
                pred = self.net(X).to(device)
            # pred = self.net(X).to(device)
                loss=loss+self.criterion(pred,y)
                t_correct =t_correct+ (pred.argmax(1 ) == y).type(torch.float).sum().item()
                print("predicted: ",(pred.argmax(1)) )
                print("real ", y)
                np.save("predicted",np.array(pred.argmax(1).cpu()))
                np.save("real",np.array(y.cpu()))

        print("Accuracy : {}".format(t_correct/t_size))

    def run_epoch(self,net,data):
        losses,accs=[],[]
        correct=0
        size = len(data.dataset)
        num_batches = len(data)
        for batch,(X,y) in enumerate (tqdm(data,desc=str(200),unit ='b')):
            device = "cuda" if torch.cuda.is_available() else "cpu"

            X = torch.autograd.Variable(X)
            y = torch.autograd.Variable(y)
            X=X.to(device).float()
            y=y.to(device).long()
            pred = self.net(X).to(device)
            # pred = net(X).to(device)
            loss=self.criterion(pred,y)
            loss.requires_grad_(True)
            losses.append(loss)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        l= sum(losses)/num_batches
        correct /= size
        return l,correct


    def train(self):
        max_acc=0.0
        train_str='Train epoch {}: l o ss {} acc {}'
        test_str='Test epoch {}: lo s s {} acc {} maxacc {}'
        line_str="\t \n"
        for e_id in range(self.epoch):
            self.net.train()
            loss,acc=self.run_epoch(self .net,self.train_dataloader)
            print(train_str.format(e_id,loss,acc))

            t_losses,accs=[],[]
            t_correct=0
            t_size = len(self .validation_dataloader.dataset)
            t_num_batches = len(self.validation_dataloader)
            with torch.no_grad():
                self.net.eval()
                for batch,(X,y) in enumerate(tqdm(self.validation_dataloader,desc=str(200),unit='b')):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    X=X.to(device).float()
                    y=y.to(device).long()
                    pred = self.net(X).to(device)
                    # pred = self.net(X).to(device)
                    loss=self.criterion(pred,y)
                    t_losses .append(loss)
                    t_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            t_loss= sum(t_losses) / t_num_batches
            t_correct /= t_size
            if max_acc>t_correct:
                max_acc = max(max_acc, t_correct)
                torch.save(self.net)




class TimeSeries(object):
    def __init__(self,T,label,num_class,num_channel) -> None:
        self.T=T
        self.label=label
        self.num_class=num_class
        self.channel_num=num_channel


class FileLoader(object):
    def __init__(self) -> None:
        pass
    def load_data(self,types,interval,num_feature):
        batch_data=[] ##batch 데이터 저장
        label_info=[] ##라벨 저장
        current_directory = os.path.dirname(os.path.realpath(__file__))

        files="..\\datasets\\MobiAct_dataset\\Annotated Data\\"
        temp_dir = os.path.join(current_directory,files)
        # current_directory = os.getcwd()
        test_types = ['WAL', 'JOG', 'JUM']
        test_types = ['WAL', 'JOG', 'JUM','LYI','STD','FKL','FOL']
        # adl_types = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']
        TESTING_DATASETS = ['']
        testing_actions = ['WAL', 'JOG', 'JUM']
        # for ft in Path(files).iterdir():
        for action in testing_actions:
            ft = action
            sec_dir = os.path.join(temp_dir,action)
            os.chdir(temp_dir)
            print(os.getcwd())
            cur_iter=0
            interval=200
            print(Path(ft).iterdir())

            idx=0
            # for csv in Path(ft).iterdir():
            for csv in os.listdir(sec_dir):
                idx+=1
                if not str(csv).endswith("csv"):
                    continue
                print(csv)
                csv = os.path.join(sec_dir,csv)
                temp=pd.read_csv(csv)
                temp=np.array(temp)
                cur_iter=0
                while cur_iter+interval<temp.shape[0]:
                    lab=temp[cur_iter:cur_iter+interval,-1:]
                    x=temp[cur_iter:cur_iter+interval,2:-1]
                    a=(collections.Counter(lab.flatten()))
                    if len(a)>1:
                        cur_iter+=interval
                        continue

                    #만약, lable이름이 LYI 나 SIT이면 , STP(Stopping)으로 바꿔주고 append-> 아니면 그냥 하던데로 append하세요
                    if lab.flatten()[0] in test_types:
                        cur_iter+=interval
                        label_info.append(lab.flatten()[0])
                        batch_data.append(x)
                        continue
                    else:
                        cur_iter+=10

        le=preprocessing.LabelEncoder() ##label encoding.
        le.fit(test_types)
        label_info=le.transform(label_info)
        # print(batch_data)
        batch_data=np.array(batch_data,dtype=float)
        sh=batch_data.shape
        batch_data=batch_data.reshape(sh[0]*sh[1],sh[2])
        scaler = MinMaxScaler()
        scaler.fit(batch_data)
        data=scaler.transform(batch_data)
        data=data.reshape(sh)

        return TimeSeries(np.array(data,dtype=float),np.array(label_info,dtype=float),len(test_types),num_channel=num_feature) ## 나중에 바꿔주기


class MobiactDataset(Dataset):
    def __init__(self, data, label, feat_dim) -> None:
        super().__init__()

        self.data = data
        self.label = label
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, :, :]
        label = self.label[idx]
        data = data.swapaxes(0, 1)

        return data, label

def app_run(data):
    net = UNet(9,7) ## hard coding이므로 나중에 수정 필요
    # net2=LSTMClassifier(9,32,7)
    # net3=CNN_LSTM(9,32,7)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(data,net, 100,loss_fn)
    trainer.train()
    trainer.tester()

def main():
    data=FileLoader().load_data(["STD"],500,9)
    datum=MobiactDataset(data.T,data.label,9)
    app_run(datum)

if __name__ == "__main__":
    # print(os.curdir)
    main()


#
#
# def UTime(x,y, train_percent, params):
#     net = UNet(9, 7)
#     trainer = Trainer(data,net, 100,loss_fn)
#
#
# def run_Utime(x,y,train_percentage,utime_parameters):
#
#     testing_model = UTime(x, y, train_percentage, utime_parameters)
#     testing_model.initiate()
#     testing_model.train()
#     train_labels, train_preds = testing_model.predict_train()
#     test_labels, test_preds = testing_model.predict_test()
#     # print(classification_report(train_labels, train_preds))
#     # print(classification_report(test_labels, test_preds))
#     return train_labels, train_preds, test_labels, test_preds, testing_model
#
# class utime_parameters:
#     epoch = 10

# def main():

    # mobiact_1 = pd.read_csv('..//datasets//MobiAct_dataset//Annotated Data//JUM//JUM_1_2_annotated.csv')
    # x_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
    # y_col = 'label'
    # x_data = mobiact_1[x_cols]
    # y_data = mobiact_1[y_col]
    # y_data = data_config.convert_labels_to_int(y_data)
    #
    # scaler = StandardScaler()
    # x_data = pd.DataFrame(scaler.fit_transform(x_data))  ### 이거 나중에 좀 수정해라.
    #
    # input_data = x_data
    # input_y = y_data
    #
    # train_labels, train_preds, test_labels, test_preds, trained_model = run_UTime(input_data, input_y, 0.8,
    #                                                                             utime_parameters)
    # data=FileLoader().load_data(["STD"],500,9)
    #
    # datum=MobiactDataset(data.T,data.label,9)
    # app_run(datum)