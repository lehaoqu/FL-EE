import torch
import torch.nn as nn
import importlib

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    parser.add_argument('--T', type=float, default=3, help="kd T")
    return parser


class Client(BaseClient):
    
    def run(self):
        self.train()
    

class Server(BaseServer):
    def run(self):
        self.sample()
        # print('sample')
        self.downlink()
        # print('downlink')
        self.client_update()
        # print('client_update')
        self.uplink()
        # print('unlink')
        self.aggregate()
        # print('aggregate')
        