import tensorflow as tf
from tensorflow_tools.constants import *


def get_all_inputs(graph, prefix=False):
    prefix_func = (lambda x: 'import/' + x) if prefix else (lambda x: x)
    name = prefix_func(INPUT_TENSOR_0)
    inputs = []
    i = 0
    while True:
        try:
            input_tensor = graph.get_tensor_by_name(name)
        except KeyError:
            break
        name = prefix_func(INPUT_TENSOR + '_{}'.format(i) + ':0')
        i += 1
        inputs.append(input_tensor)
    return inputs