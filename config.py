import argparse
from functools import partial

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser('config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument = partial(parser.add_argument, help=' ')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
        self.args = parser.parse_args()