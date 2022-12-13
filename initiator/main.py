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
def partition(parser:ParamsParser):
    training_rate = parser.getparam('training_rate')
    test_rate = parser.getparam('test_rate')
    prediction_rate = parser.getparam('prediction_rate')
    file_path = parser.getparam('original')

    original = np.loadtxt(file_path, dtype=np.float)
    print(original)


if __name__ == '__main__':
    pp = ParamsParser()
    partition(pp)

