import configparser
import os
import socket
import sys
import json
import time

import numpy as np
import torch.optim
import torch.utils.data as data
from torch.utils.data import Dataset

from model1 import LogisticRegressionModel


# TODO: fix the bug that parameters of the model don't change in each training epoch
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
        self.feature_min, _ = self.samples.min(dim=0)
        self.feature_max, _ = self.samples.max(dim=0)

        self.samples = (self.samples - self.feature_min) / (self.feature_max - self.feature_min)
        self.feature_num = len(self.samples[0])
        self.class_num = len(np.unique(original[:, -1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def train(model, epoch, optimizer, train_loader):
    global client
    idx = 0
    while True:
        try:
            for data, label in train_loader:
                optimizer.zero_grad()
                idx += 1
                y_pred = model(data)
                # send data to coordinator
                send_data(client, 'START', data=idx)
                send_data(client, 'SEND_DATA', data=y_pred.tolist())

                # buffer time
                time.sleep(0.01)
                send_data(client, 'BATCH_END')
                param_recv = client.recv(4096).decode('utf-8')
                pred = torch.tensor(eval(param_recv))
                pred.requires_grad = True
                loss = criteria(pred, label.long())
                loss.backward()
                optimizer.step()

            send_data(client, 'ITER_END')
        except ConnectionAbortedError:
            client.close()
            print("iter-{} complete!".format(epoch))
            break


def test(model, epoch, test_loader, test_num):
    loss = 0
    correct = 0
    # model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            y_pred = model(data)
            loss += criteria(y_pred, label.long())
            res = y_pred.argmax(dim=1, keepdim=True)
            correct += res.eq(label.view_as(res)).sum().item()
    print("***************")
    print('step{}, accuracy: {}%'.format(epoch, correct / test_num * 100))
    print("***************")


def attack(ground_truth, input_sample, id):
    # 返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量，例tensor([-1.9996,  1.1006, -1.8737, -1.6735, -1.8565,  0.9147,
    # 1.1568, -0.4748, 0.1593,  0.0408])
    noise = torch.randn(pp.getparam('target_features'))
    # sigmoid激活
    t = noise.sigmoid()
    # parameters['num_target_features'] = 10，假设target的每条数据有10个feature，这10个feature在每个sample的最后10列
    random_mse = ((t - input_sample[total_feature_num - pp.getparam('target_features'):]) ** 2).mean()
    input_sample.resize_((total_feature_num, 1))
    # 前10个数据是adv（攻击者）拥有的数据， 后10个是target所拥有的数据
    input_sample_adv = input_sample[:total_feature_num - parameters['num_target_features'], :]
    input_sample_target = input_sample[total_feature_num - parameters['num_target_features']:, :]

    # 该模块下的tensor不会自动求导，对ground_truth求对数，即求ln(v)
    with torch.no_grad():
        ground_truth_ln = torch.log(ground_truth)
    # left和right size均为1
    ground_truth_ln_left = ground_truth_ln[:class_num - 1]
    ground_truth_ln_right = ground_truth_ln[1:]
    ground_truth_ln_diff = ground_truth_ln_left - ground_truth_ln_right
    ground_truth_ln_diff.resize_(class_num - 1, 1)
    a = torch.matmul(params_adv, input_sample_adv)
    b = ground_truth_ln_diff - a
    # 求出target的训练集
    x_target = torch.matmul(params_target_inv, b)
    attack_mse = ((x_target - input_sample_target) ** 2).mean()
    return attack_mse, random_mse


def send_data(client, cmd, **kv):
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    if cmd == 'SEND_DATA' or cmd == 'START':
        jd['data'] = kv
    elif cmd == 'SEND_LABELS':
        jd['label'] = kv

    jsonstr = json.dumps(jd)
    # print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf-8'))


if __name__ == '__main__':
    client_type = None
    pp = ParamsParser()
    dataset = MyDataset(pp)
    LR_model = LogisticRegressionModel(dataset.feature_num, dataset.class_num)
    optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    criteria = torch.nn.NLLLoss()

    train_set = torch.utils.data.Subset(dataset, range(dataset.train_samples_num))
    test_set = torch.utils.data.Subset(dataset, range(dataset.train_samples_num,
                                                      dataset.train_samples_num + dataset.test_samples_num))
    pred_set = torch.utils.data.Subset(dataset,
                                       range(dataset.test_samples_num + dataset.train_samples_num, dataset.__len__()))

    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False)

    base_name = "initiator"

    for i in range(1, pp.getparam('epochs') + 1):
        client = socket.socket()
        # connect to coordinator
        client_type = base_name + "-it-" + str(i)
        print("initiator is connecting to coordinator in iterator-{}".format(i))
        client.connect(('127.0.0.1', 12345))
        print(client.recv(1024).decode(encoding='utf-8'))
        send_data(client, 'CONNECT')

        train(LR_model, i, optimizer, train_loader)
        #test(LR_model, i, test_loader, dataset.test_samples_num)

        print(optimizer.param_groups)

        scheduler.step()

    linear_layer_params = None
    for name, param in LR_model.named_parameters():
        if param.requires_grad:
            linear_layer_params = param.data

    # compute the required coefficients for attack
    # linear_layer_params 是 2 * 20的tensor
    linear_layer_params_left = linear_layer_params[:1, :]
    # print('params left: ', linear_layer_params_left)
    linear_layer_params_right = linear_layer_params[1:, :]
    # print('params right: ', linear_layer_params_right)
    # 计算相邻参数的差值，即（θk - θk+1）
    linear_layer_params_sub = linear_layer_params_left - linear_layer_params_right
    # print('params sub: ', linear_layer_params_sub)

    # target 拥有的参数为后10列，adv拥有的参数为前10列
    params_target = linear_layer_params_sub[:, total_feature_num - parameters['num_target_features']:]
    params_adv = linear_layer_params_sub[:, :total_feature_num - parameters['num_target_features']]
    # print('params target shape: ', params_target.shape)
    # print('params adversary shape: ', params_adv.shape)

    # Computes the pseudoinverse (Moore-Penrose inverse) of the params_target matrix
    # 计算target参数矩阵的逆元
    params_target_inv = torch.pinverse(params_target)

    total_attack_mse = 0.0
    total_rand_mse = 0.0
    pred_interval = 1000
    # 取predict数据集中的每一条数据（数据，标签）
    for i in range(dataset.pred_samples_num):
        sample, label = pred_set.__getitem__(i)
        # 用model计算ground_truth, 即softmax后的预测概率
        y_ground_truth = LR_model(sample)

        # 使用attack攻击模型计算mse
        back_attack_mse, rand_mse = attack(y_ground_truth, sample, i)
        total_attack_mse += back_attack_mse
        total_rand_mse += rand_mse

    cur_average_attack_mse = total_attack_mse / pred_set_num
    cur_average_rand_mse = total_rand_mse / pred_set_num
    logging.critical('average attack mse: %f', cur_average_attack_mse)
    logging.critical('average random mse: %f', cur_average_rand_mse)
    back_prop_mse.update(cur_average_attack_mse)
    random_guess_mse.update(cur_average_rand_mse)

