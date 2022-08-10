import argparse
from functools import partial

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser('config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument = partial(parser.add_argument, help=' ')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
        parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for architecture')
        parser.add_argument('--unrolled', action='store_true', default=False, help='use one step unrolled validation loss')
        parser.add_argument('--learning_rate_min', type=float, default=0.001, help='learning rate')
        parser.add_argument('--layers', type=int, default=4, help='Number of layers (sequential cells)')
        self.args = parser.parse_args()