from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    #'none',
    'conv_3x3',
    'dconv_3x3',
    'conv_1x1',
    #'rel_attention',
    'ffn'
]