import configparser
import os

import numpy as np


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


# divide the origin dataset into training set, test set and prediction set
def partition(parser: ParamsParser):
    test_rate = parser.getparam('test_rate')
    prediction_rate = parser.getparam('prediction_rate')
    file_path = parser.getparam('original')
    data_path = parser.getparam('data_path')

    original = np.loadtxt(file_path, dtype=float, delimiter=',')
    total_samples_num = len(original)
    pred_samples_num = int(total_samples_num * prediction_rate)  # 6097
    test_samples_num = int((1 - prediction_rate) * test_rate * total_samples_num)  # 4878
    train_samples_num = total_samples_num - test_samples_num - pred_samples_num  # 19513

    train_samples = original[:train_samples_num, :]
    test_samples = original[train_samples_num:train_samples_num + test_samples_num, :]
    pred_samples = original[train_samples_num + test_samples_num:, :]

    np.savetxt(data_path + os.sep + 'train_set.csv', train_samples, fmt='%.3f', delimiter=',')
    np.savetxt(data_path + os.sep + 'test_set.csv', test_samples, fmt='%.3f', delimiter=',')
    np.savetxt(data_path + os.sep + 'pred_set.csv', pred_samples, fmt='%.3f', delimiter=',')
    return train_samples, test_samples, pred_samples




if __name__ == '__main__':
    pp = ParamsParser()
    partition(pp)
