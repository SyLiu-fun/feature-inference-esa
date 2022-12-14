import configparser
import os

import numpy as np
import torch.optim
import torch.utils.data as data
from torch.utils.data import Dataset

from model import LogisticRegressionModel


class ParamsParser:
    parameters = {}

    def __init__(self):
        self.parameters['current_path'] = os.getcwd()
        config = configparser.ConfigParser()
        config.read('initiator-config.ini')

        train_param = config['TRAIN']
        dataset_param = config['DATASET']
        self.parameters['data_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'dataset'
        self.parameters['original'] = self.parameters['data_path'] + os.sep + train_param['DataFile']
        self.parameters['log_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'log'
        self.parameters['logfile'] = self.parameters['log_path'] + os.sep + train_param['LogFile']
        self.parameters['epochs'] = train_param.getint('Epochs')
        self.parameters['target_features'] = train_param.getint('NumOfFeaturesToRecover')
        self.parameters['training_rate'] = dataset_param.getfloat('TrainPortion')
        self.parameters['test_rate'] = dataset_param.getfloat('TestPortion')
        self.parameters['prediction_rate'] = dataset_param.getfloat('PredictPortion')

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
    def __init__(self, parser:ParamsParser):
        test_rate = parser.getparam('test_rate')
        prediction_rate = parser.getparam('prediction_rate')
        file_path = parser.getparam('original')
        data_path = parser.getparam('data_path')

        original = torch.from_numpy(np.loadtxt(file_path, dtype=float, delimiter=',')).float()
        self.total_samples_num = len(original)
        self.pred_samples_num = int(self.total_samples_num * prediction_rate)  # 6097
        self.test_samples_num = int((1 - prediction_rate) * test_rate * self.total_samples_num)  # 4878
        self.train_samples_num = self.total_samples_num - self.test_samples_num -self. pred_samples_num  # 19513

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


def train(model, epochs, optimizer, train_loader):
    for data, label in train_loader:
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criteria(y_pred, label.long())
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    pp = ParamsParser()
    dataset = MyDataset(pp)
    LR_model = LogisticRegressionModel(dataset.feature_num, dataset.class_num)
    optimizer = torch.optim.Adam(LR_model.parameters())
    criteria = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    # TODO: do not use random split method. how to split Dataset sequentially?
    train_set, test, pred = torch.utils.data.random_split(dataset, [19513, 4878, 6097])
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=False)

    train(LR_model, pp.getparam('epochs'), optimizer, train_loader)
