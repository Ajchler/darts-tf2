from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'conv_3x3',
    'dconv_3x3',
    'conv_1x1',
    'ffn',
    'global_pool',
    'fc'
]