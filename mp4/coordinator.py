import server
from server import encode_command, decode_command, encode_ping_ack, decode_ping_ack
import threading
import socket
import os
import time
import struct
import json
import numpy as np

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
total_job = 150
COORDINATOR_PORT = 10088
ML_PORT = 10089

# default hyperparameters
model1_batch_size = 3
model2_batch_size = 3

# use zfill(2) for adding vm number
vm_leg_1 = 'fa22-cs425-88' 
vm_leg_2 = '.cs.illinois.edu'

class Coordinator:
    def __init__(self, coordinator_port: int, ml_port: int):
        self.host = socket.gethostbyname(socket.gethostname())
        self.coordinator_port = coordinator_port
        self.ml_port = ml_port

        # hyperparameters (just batch size for now)
        self.hyperparameter_lock = threading.Lock()
        self.model1_batch_size = 1
        self.model2_batch_size = 1
        self.job_1_vms = [1,2,3,4,5]
        self.job_2_vms = [6,7,8,9,10]
        

        # query time stats
        self.statistics_lock = threading.Lock()
        self.model_1_query_times = []
        self.model_2_query_times = []
        self.model_1_query_endtimes = []
        self.model_2_query_endtimes = []


        self.start_lock = threading.Lock()
        self.start_1 = False
        self.start_2 = False


    # def repair(self, ip):
    #     start_time = time.time()
    #     self.ntf_lock.acquire()
    #     if ip in self.node_to_file:
    #         sdfsfileids = self.node_to_file.pop(ip)
    #     else:
    #         self.ntf_lock.release()
    #         return
    #     self.ntf_lock.release()
    #     for sdfsfileid in sdfsfileids:
    #         self.ftn_lock.acquire()
    #         if sdfsfileid in self.file_to_node:
    #             ips = list(self.file_to_node[sdfsfileid])
    #             self.file_to_node[sdfsfileid].remove(ip)
    #         else:
    #             self.ftn_lock.release()
    #             continue
    #         for ipaddr in ips:
    #             res = self.issue_repair(sdfsfileid, ipaddr, ips)
    #             if res == '1':
    #                 break
    #         self.ftn_lock.release()
    #     end_time = time.time()
    #     print('replication for node: ', ip, " complete")
    #     if len(sdfsfileids) > 0:
    #         print('files re-replicated: ')
    #         for sdfsfileid in sdfsfileids:
    #             print('  ', sdfsfileid)
    #     print('time consumed: ', end_time-start_time)


    # def issue_repair(self, sdfsfileid, ip, ips):
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         try:
    #             s.connect((ip, self.file_port))
    #         except socket.error as e:
    #             return
    #         s.send(b'repair')
    #         s.recv(1) # for ack
    #         s.send(json.dumps({'sdfsfileid': sdfsfileid, 'ips': ips}).encode())
    #         res = s.recv(1).decode()
    #         return res

    def background(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.host, self.coordinator_port))
            while True:
                encoded_command, addr = s.recvfrom(4096)
                decoded_command = json.loads(encoded_command.decode())
                command_type = decoded_command['command_type']
                if command_type == 'set_batch_size':

                    model = int(decoded_command['model_to_change'])
                    new_batch_size = int(decoded_command['new_batch_size'])
                    self.hyperparameter_lock.acquire()
                    if (model == 1):
                        self.model1_batch_size = new_batch_size
                    else:
                        self.model2_batch_size = new_batch_size
                    self.hyperparameter_lock.release()
                elif command_type == 'run_job':
                    model = int(decoded_command['model_to_run'])
                    if(model == 1):
                        self.start_1 = True
                    else:
                        self.start_2 = True
             
                    
                elif command_type == 'get_stats':
                    # jobs per vm
                    self.statistics_lock.acquire()
                    print('job 1 vms:')
                    for the_vm in self.job_1_vms:
                        print(vm_leg_1 + str(the_vm).zfill(2) + vm_leg_2)
                    print('job 2 vms:')
                    for the_vm in self.job_2_vms:
                        print(vm_leg_1 + str(the_vm).zfill(2) + vm_leg_2)

                    # query time stats
                    print()
                    print('job 1 query time stats:')
                    print('queries processed so far:', len(self.model_1_query_times))
                    print('average query time:', np.mean(self.model_1_query_times))
                    print('std dev query time:', np.std(self.model_1_query_times))
                    print('median query time:', np.median(self.model_1_query_times))
                    print('90th, 95th, 99th percentile query time:', np.percentile(self.model_1_query_times, (90, 95, 99)))

                    print('job 2 query time stats:')
                    print('queries processed so far:', len(self.model_2_query_times))
                    if len(self.model_2_query_times) != 0:
                        print('average query time:', np.mean(self.model_2_query_times))
                        print('std dev query time:', np.std(self.model_2_query_times))
                        print('median query time:', np.median(self.model_2_query_times))
                        print('90th, 95th, 99th percentile query time:', np.percentile(self.model_2_query_times, (90, 95, 99)))

                    # query rate stats
                    curr = time.time()
                    index_one = 0
                    for t in self.model_1_query_endtimes[::-1]:
                        if curr - t < 10:
                            index_one += 1
                    print("Average query rate for job 1: ", index_one/10.0)
                    index_two = 0
                    for t in self.model_2_query_endtimes[::-1]:
                        if curr - t < 10:
                            index_two += 1
                    print("Average query rate for job 2: ", index_two/10.0)
                    
                            
                    

                    self.statistics_lock.release()
               
                # used by VMs to send query time stats back to coordinator
                elif command_type == 'query_time':
                    model = int(decoded_command['model'])
                    query_time = float(decoded_command['query_time'])
                    end_time = float(decoded_command['end_time'])
                    self.statistics_lock.acquire()
                    if (model == 1):
                        self.model_1_query_times.append(query_time)
                        self.model_1_query_endtimes.append(end_time)
                    else:
                        self.model_2_query_times.append(query_time)
                        self.model_2_query_endtimes.append(end_time)
                    self.statistics_lock.release()
                


    def run_jobs(self):
        while(True):
            self.start_lock.acquire()
            if(self.start_1 == True):
                self.start_lock.release()
                break
            time.sleep(1)
            self.start_lock.release()
        
        index = 1
        total_vm = self.job_1_vms + self.job_2_vms
        while(index < total_job):
            for the_vm in total_vm:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    if the_vm in self.job_1_vms:
                        s.sendto(json.dumps({'start_index': index, "end_index": index+self.model1_batch_size, "model" : 1}).encode(), (vm_leg_1 + str(the_vm).zfill(2) + vm_leg_2, self.ml_port))
                        index += self.model1_batch_size
                    else:
                        s.sendto(json.dumps({'start_index': index, "end_index": index+self.model2_batch_size, "model" : 2}).encode(), (vm_leg_1 + str(the_vm).zfill(2) + vm_leg_2, self.ml_port))
                        index += self.model2_batch_size
                if(index >= total_job):
                    break
            

    def run(self):
        t1 = threading.Thread(target=self.background)
        t1.start()
        t2 = threading.Thread(target=self.run_jobs)
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
    master = Coordinator(COORDINATOR_PORT, ML_PORT)
    master.run()

