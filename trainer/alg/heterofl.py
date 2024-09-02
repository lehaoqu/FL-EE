import torch

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
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
        