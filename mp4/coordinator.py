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