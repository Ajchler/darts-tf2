from collections import namedtuple
import tensorflow as tf

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'conv_3x3',
    'dconv_3x3',
    'conv_1x1',
    'rel_attention',
    'ffn',
    'none'
]