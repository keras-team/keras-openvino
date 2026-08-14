import openvino as ov
import openvino.opset16 as ov_opset

from keras.src.backend.common import KerasVariable
from keras_openvino.src.ops.core import OpenVINOKerasTensor
from keras_openvino.src.ops.core import convert_to_numpy
from keras_openvino.src.ops.core import convert_to_tensor
from keras_openvino.src.utils import OPENVINO_DTYPES


class Variable(KerasVariable):
    def _initialize(self, value):
        if isinstance(value, OpenVINOKerasTensor):
            self._value = value
        elif isinstance(value, ov.Tensor):
            value_const = ov_opset.constant(
                value.data, dtype=OPENVINO_DTYPES[self._dtype]
            )
            self._value = OpenVINOKerasTensor(value_const.output(0))
        else:
            value_const = ov_opset.constant(
                value, dtype=OPENVINO_DTYPES[self._dtype]
            )
            self._value = OpenVINOKerasTensor(value_const.output(0))

    def _direct_assign(self, value):
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __array__(self):
        return convert_to_numpy(self)

    def __getitem__(self, idx):
        arr = convert_to_numpy(self)
        return arr.__getitem__(idx)

    def __int__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return int(arr)

    def __float__(self):
        arr = convert_to_numpy(self)
        if arr.ndim > 0:
            raise TypeError(
                "Only scalar arrays can be converted to Python scalars. "
                f"Got: shape={arr.shape}"
            )
        return float(arr)
