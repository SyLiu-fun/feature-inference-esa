import socket
from threading import Thread
import time
import json

import numpy as np
import torch

ADDRESS = ('127.0.0.1', 12345)

g_socket_server = None

g_conn_pool = {}

param_dict = {}

labels = []

# length of data
data_len = 0


def init():
    """
    init server
    """
    global g_socket_server
    g_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    g_socket_server.bind(ADDRESS)
    g_socket_server.listen(5)  # 最大等待数
    print("Coordinator is waiting for client connecting...")


def accept_client():
    """
    接收新连接
    """
    while True:
        client, info = g_socket_server.accept()  # 阻塞，等待客户端连接
        # 给每个客户端创建一个独立的线程进行管理
        thread = Thread(target=message_handle, args=(client, info))
        # 设置成守护线程
        thread.setDaemon(True)
        thread.start()


def message_handle(client, info):
    """
    消息处理
    """
    global data_len, labels
    data = []
    client.sendall("Connection established successfully!".encode(encoding='utf-8'))
    while True:
        try:
            part = client.recv(4096)
            msg = part.decode(encoding='utf-8')
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_type = jd['client_type']
            if 'CONNECT' == cmd:
                g_conn_pool[client_type] = client
                print("current connect num: {}".format(len(g_conn_pool)))
                print('on client connect: ' + client_type, info)
            elif 'SEND_DATA' == cmd:
                # print('recv client msg: ' + client_type, jd['data'])
                data.append(jd['data']['data'])
            elif 'END' == cmd:
                data_len = len(data)
                param_dict[client_type] = data
                # print(param_dict.get(client_type))
            elif 'SEND_LABELS' == cmd:
                labels.append(jd['label']['label'])
        except Exception as e:
            print(e.with_traceback())
            remove_client(client_type)
            break


def remove_client(client_type):
    client = g_conn_pool[client_type]
    if client is not None:
        client.close()
        g_conn_pool.pop(client_type)
        print("client offline: " + client_type)


if __name__ == '__main__':
    init()
    # 新开一个线程，用于接收新连接
    thread = Thread(target=accept_client)
    thread.setDaemon(True)
    thread.start()
    # main thread
    while True:
        if len(param_dict) == 2:
            param_list = [0] * data_len
            loss_list = []
            Softmax = torch.nn.Softmax(dim=1)
            loss = torch.nn.NLLLoss()
            for i in range(data_len):
                for key in param_dict.keys():
                    param_list[i] += torch.tensor(param_dict.get(key)[i])
                # print(Softmax(param_list[i]))
                param_list[i] = Softmax(param_list[i])
                loss_list.append(loss(param_list[i], torch.tensor(labels[i]).long()))
            # print(loss_list[:])
            # TODO: remove clients after loss calculating, init global parameters
            for cli in g_conn_pool:
                print(g_conn_pool.get(cli))
        time.sleep(0.1)

    