import socket
from threading import Thread
import time
import json

ADDRESS = ('127.0.0.1', 12345)

g_socket_server = None

g_conn_pool = {}

param_dict = {}


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
    client.sendall("connect server successfully!".encode(encoding='utf-8'))
    while True:
        try:
            bytes = client.recv(1024)
            msg = bytes.decode(encoding='utf-8')
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_type = jd['client_type']
            if 'CONNECT' == cmd:
                g_conn_pool[client_type] = client
                print('on client connect: ' + client_type, info)
            elif 'SEND_DATA' == cmd:
                print('recv client msg: ' + client_type, jd['data'])
                param_dict[client_type] = jd['data']
        except Exception as e:
            print(e)
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
    # 主线程逻辑
    while True:
        time.sleep(0.1)

    