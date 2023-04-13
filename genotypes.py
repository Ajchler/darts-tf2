from collections import namedtuple
import tensorflow as tf

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

#PRIMITIVES = [
#    'none',
#    'avg_pool_3x3',
#    'max_pool_3x3',
#    'skip_connect',
#    'sep_conv_3x3',
#    'sep_conv_5x5',
#    'dil_conv_3x3',
#    'dil_conv_5x5'
#]

PRIMITIVES = [
    'sep_conv_3x3',
    'rel_attention',
    'ffn',
    'none',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3'
]