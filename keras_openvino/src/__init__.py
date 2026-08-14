from keras.src.backend.common.name_scope import name_scope
from keras_openvino.src import ops
from keras_openvino.src import random
from keras_openvino.src import rnn
from keras_openvino.src.ops.core import compute_output_spec
from keras_openvino.src.ops.core import device_scope
from keras_openvino.src.variable import Variable

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = False
IS_THREAD_SAFE = True

distribution_lib = None
