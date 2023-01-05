import configparser
import os
import socket
import sys
import json

import numpy as np
import torch.optim
import torch.utils.data as data
from torch.utils.data import Dataset

from model1 import LogisticRegressionModel


class ParamsParser:
    parameters = {}

    def __init__(self):
        self.parameters['current_path'] = os.getcwd()
        config = configparser.ConfigParser()
        config.read('initiator-config.ini')

        train_param = config['TRAIN']
        dataset_param = config['DATASET']
        socket_param = config['SOCKET']
        self.parameters['data_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'dataset'
        self.parameters['original'] = self.parameters['data_path'] + os.sep + train_param['DataFile']
        self.parameters['log_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'log'
        self.parameters['logfile'] = self.parameters['log_path'] + os.sep + train_param['LogFile']
        self.parameters['epochs'] = train_param.getint('Epochs')
        self.parameters['target_features'] = train_param.getint('NumOfFeaturesToRecover')
        self.parameters['training_rate'] = dataset_param.getfloat('TrainPortion')
        self.parameters['test_rate'] = dataset_param.getfloat('TestPortion')
        self.parameters['prediction_rate'] = dataset_param.getfloat('PredictPortion')
        self.parameters['host'] = socket_param['host']
        self.parameters['port'] = socket_param.getint('port')

    def getparam(self, param):
        if param in self.parameters.keys():
            return self.parameters[param]
        return "invalid parameter!"

    # params is a list consisting of params
    def getparams(self, params):
        dic = {}
        for param in params:
            if param in self.parameters.keys():
                dic[param] = self.parameters[param]
        return dic


class MyDataset(Dataset):
    def __init__(self, parser: ParamsParser):
        test_rate = parser.getparam('test_rate')
        prediction_rate = parser.getparam('prediction_rate')
        file_path = parser.getparam('original')
        data_path = parser.getparam('data_path')

        original = torch.from_numpy(np.loadtxt(file_path, dtype=float, delimiter=',')).float()
        self.total_samples_num = len(original)
        self.pred_samples_num = int(self.total_samples_num * prediction_rate)  # 6097
        self.test_samples_num = int((1 - prediction_rate) * test_rate * self.total_samples_num)  # 4878
        self.train_samples_num = self.total_samples_num - self.test_samples_num - self.pred_samples_num  # 19513

        # save to file
        # In order to maintain the consistency of ids in prediction sets of both parties,
        # not use torch.utils.data.random_split
        train_file = original[:self.train_samples_num, :]
        test_file = original[self.train_samples_num:self.train_samples_num + self.test_samples_num, :]
        pred_file = original[self.train_samples_num + self.test_samples_num:, :]

        np.savetxt(data_path + os.sep + 'train_set.csv', train_file, fmt='%.3f', delimiter=',')
        np.savetxt(data_path + os.sep + 'test_set.csv', test_file, fmt='%.3f', delimiter=',')
        np.savetxt(data_path + os.sep + 'pred_set.csv', pred_file, fmt='%.3f', delimiter=',')

        self.samples, self.labels = original[:, :-1], original[:, -1]
        self.feature_num = len(self.samples[0])
        self.class_num = len(np.unique(original[:, -1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def train(model, epoch, optimizer, train_loader):
    for data, label in train_loader:
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criteria(y_pred, label.long())
        loss.backward()
        optimizer.step()


def test(model, epoch, test_loader, test_num):
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            y_pred = model(data)
            loss += criteria(y_pred, label.long())
            res = y_pred.argmax(dim=1, keepdim=True)
            correct += res.eq(label.view_as(res)).sum().item()
    print('step{}, accuracy: {}%'.format(epoch, correct / test_num))


client_type = 'initiator'


def send_data(client, cmd, **kv):
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    jd['data'] = kv

    jsonstr = json.dumps(jd)
    print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf-8'))



if __name__ == '__main__':
    pp = ParamsParser()
    # dataset = MyDataset(pp)
    # LR_model = LogisticRegressionModel(dataset.feature_num, dataset.class_num)
    # optimizer = torch.optim.Adam(LR_model.parameters())
    # criteria = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    #
    # train_set = torch.utils.data.Subset(dataset, range(dataset.train_samples_num))
    # test_set = torch.utils.data.Subset(dataset, range(dataset.train_samples_num, dataset.train_samples_num + dataset.test_samples_num))
    # pred_set = torch.utils.data.Subset(dataset, range(dataset.test_samples_num + dataset.train_samples_num, dataset.__len__()))
    # # for i in range(dataset.train_samples_num):
    # #     x, y = train_set.__getitem__(i)
    # #     print(x, y)
    # train_loader = data.DataLoader(train_set, batch_size=64, shuffle=False)
    # test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False)
    #
    # for i in range(1, pp.getparam('epochs') + 1):
    #     train(LR_model, i, optimizer, train_loader)
    #     test(LR_model, i, test_loader, dataset.test_samples_num)
    #     scheduler.step()
    #
    # initiator_param = None
    # for name, param in LR_model.named_parameters():
    #     if param.requires_grad:
    #         initiator_param = param.data

    # send params to coordinator
    client = socket.socket()
    client.connect(('127.0.0.1', 12345))
    print(client.recv(1024).decode(encoding='utf-8'))
    send_data(client, 'CONNECT')

    while True:
        print('send parameters to coordinator')
        b = input('start')
        a = [1, 2, 3, 4, 5]
        send_data(client, 'SEND_DATA', data=a)
