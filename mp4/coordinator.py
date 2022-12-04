import server
from server import encode_command, decode_command, encode_ping_ack, decode_ping_ack
import threading
import socket
import os
import time
import struct
import json

import torch

# The coordinator must send out queries to the nodes and get results
# The dataset to be evaluated
# Will also instruct nodes to load model (this is equal to "training")
# nodes 1-5 will handle job 1, nodes 6-10 will handle job2
# although the above may need rebalancing to make query times equal
# the rebalancing must happen automatically (or maybe use a mathematical calculation based on batch size and real-world performance)
# also, the number of images per query WILL differ (this is the batch size)

MASTER_PORT = 20086
FILE_PORT = 10086
GET_ADDR_PORT = 10087

COORDINATOR_PORT = 10088
ML_PORT = 10089

# default hyperparameters
model1_batch_size = 10
model2_batch_size = 10

class Coordinator:
    def __init__(self, master_port: int, file_port: int, coordinator_port: int, ml_port: int):
        self.master_port = master_port
        self.host = socket.gethostbyname(socket.gethostname())
        self.file_port = file_port
        self.coordinator_port = coordinator_port
        self.ml_port = ml_port

        # hyperparameters (just batch size for now)
        self.hyperparameter_lock = threading.Lock()
        self.model1_batch_size = 10
        self.model2_batch_size = 10
        self.job_1_vms = 5
        self.job_2_vms = 5

        # query time stats
        self.statistics_lock = threading.Lock()
        self.model_1_query_times = []
        self.model_2_query_times = []

    def repair(self, ip):
        start_time = time.time()
        self.ntf_lock.acquire()
        if ip in self.node_to_file:
            sdfsfileids = self.node_to_file.pop(ip)
        else:
            self.ntf_lock.release()
            return
        self.ntf_lock.release()
        for sdfsfileid in sdfsfileids:
            self.ftn_lock.acquire()
            if sdfsfileid in self.file_to_node:
                ips = list(self.file_to_node[sdfsfileid])
                self.file_to_node[sdfsfileid].remove(ip)
            else:
                self.ftn_lock.release()
                continue
            for ipaddr in ips:
                res = self.issue_repair(sdfsfileid, ipaddr, ips)
                if res == '1':
                    break
            self.ftn_lock.release()
        end_time = time.time()
        print('replication for node: ', ip, " complete")
        if len(sdfsfileids) > 0:
            print('files re-replicated: ')
            for sdfsfileid in sdfsfileids:
                print('  ', sdfsfileid)
        print('time consumed: ', end_time-start_time)


    def issue_repair(self, sdfsfileid, ip, ips):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'repair')
            s.recv(1) # for ack
            s.send(json.dumps({'sdfsfileid': sdfsfileid, 'ips': ips}).encode())
            res = s.recv(1).decode()
            return res

    def background(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.host, self.master_port))
            while True:
                encoded_command, addr = s.recvfrom(4096)
                decoded_command = json.loads(encoded_command.decode())
                command_type = decoded_command['command_type']
                if command_type == 'fail_notice':
                    fail_ip = decoded_command['command_content']
                    for ip in fail_ip:
                        t = threading.Thread(target=self.repair, args=(ip, ))
                        t.start()
                elif command_type == 'put_notice':
                    sdfsfileid, ip = decoded_command['command_content']
                    self.ntf_lock.acquire()
                    self.node_to_file.setdefault(ip, set())
                    self.node_to_file[ip].add(sdfsfileid)
                    self.ntf_lock.release()

                    self.ftn_lock.acquire()
                    self.file_to_node.setdefault(sdfsfileid, set())
                    self.file_to_node[sdfsfileid].add(ip)
                    self.ftn_lock.release()
                elif command_type == 'delete_notice':
                    sdfsfileid, ip = decoded_command['command_content']
                    self.ntf_lock.acquire()
                    if ip in self.node_to_file:
                        self.node_to_file[ip].remove(sdfsfileid)
                    self.ntf_lock.release()

                    self.ftn_lock.acquire()
                    if sdfsfileid in self.file_to_node:
                        self.file_to_node[sdfsfileid].remove(ip)
                    self.ftn_lock.release()

    def get_addr_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, GET_ADDR_PORT))
            s.listen()
            while True:
                conn, addr = s.accept()
                sdfsfileid = conn.recv(4096).decode()
                self.ftn_lock.acquire()
                self.file_to_node.setdefault(sdfsfileid, set())
                res = list(self.file_to_node[sdfsfileid])
                self.ftn_lock.release()
                conn.send(json.dumps(res).encode())

    def run(self):
        t1 = threading.Thread(target=self.background)
        t1.start()
        t2 = threading.Thread(target=self.get_addr_thread)
        t2.start()

        while True:
            command = input('>')
            if command == 'info':
                self.ntf_lock.acquire()
                print(self.node_to_file)
                self.ntf_lock.release()

                self.ftn_lock.acquire()
                print(self.file_to_node)
                self.ftn_lock.release()

if __name__ == '__main__':
    master = FMaster(MASTER_PORT, FILE_PORT)
    master.run()

