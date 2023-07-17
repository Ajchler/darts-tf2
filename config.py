"""
Config module managing user arguments and default configuration

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

import argparse
import tensorflow as tf
from functools import partial

class Config:
    """Config class
    """
    def __init__(self, type):
        """Method which initializes arguments parser

        """
        parser = argparse.ArgumentParser('config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument = partial(parser.add_argument, help=' ')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
        parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for architecture')
        parser.add_argument('--unrolled', action='store_true', default=True, help='use one step unrolled validation loss')
        parser.add_argument('--cutout', action='store_true', default=False, help='use cutout on input images')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--learning_rate_min', type=float, default=0.04, help='learning rate')
        parser.add_argument('--layers', type=int, default=8, help='number of layers (sequential cells)')
        parser.add_argument('--nodes', type=int, default=4, help='number of inner nodes (states)')
        parser.add_argument('--multiplier', type=int, default=4, help='multiplier')
        parser.add_argument('--grad_clip', type=int, default=5, help='gradient clipping')
        parser.add_argument('--approx', action='store_true', default=False, help='use approx convolutions')
        if type == 'train':
            parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary head')
            parser.add_argument('--genotype_file', required=True, help='genotype to build network from')
            parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
            parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
        self.args = parser.parse_args()
