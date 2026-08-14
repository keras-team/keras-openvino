import numpy as np
import openvino as ov
import openvino.opset16 as ov_opset

from keras.src.backend.common import dtypes

OPENVINO_DTYPES = {
    "float16": ov.Type.f16,
    "float32": ov.Type.f32,
    "float64": ov.Type.f64,
    "uint8": ov.Type.u8,
    "uint16": ov.Type.u16,
    "uint32": ov.Type.u32,
    "uint64": ov.Type.u64,
    "int8": ov.Type.i8,
    "int16": ov.Type.i16,
    "int32": ov.Type.i32,
    "int64": ov.Type.i64,
    "bfloat16": ov.Type.bf16,
    "bool": ov.Type.boolean,
    "float8_e4m3fn": ov.Type.f8e4m3,
    "float8_e5m2": ov.Type.f8e5m2,
    "string": ov.Type.string,
}

DTYPES_MAX = {
    ov.Type.bf16: 3.38953139e38,
    ov.Type.f16: np.finfo(np.float16).max,
    ov.Type.f32: np.finfo(np.float32).max,
    ov.Type.f64: np.finfo(np.float64).max,
    ov.Type.u8: np.iinfo(np.uint8).max,
    ov.Type.u16: np.iinfo(np.uint16).max,
    ov.Type.u32: np.iinfo(np.uint32).max,
    ov.Type.u64: np.iinfo(np.uint64).max,
    ov.Type.i8: np.iinfo(np.int8).max,
    ov.Type.i16: np.iinfo(np.int16).max,
    ov.Type.i32: np.iinfo(np.int32).max,
    ov.Type.i64: np.iinfo(np.int64).max,
    ov.Type.boolean: 1,
}

DTYPES_MIN = {
    ov.Type.bf16: -3.38953139e38,
    ov.Type.f16: np.finfo(np.float16).min,
    ov.Type.f32: np.finfo(np.float32).min,
    ov.Type.f64: np.finfo(np.float64).min,
    ov.Type.u8: np.iinfo(np.uint8).min,
    ov.Type.u16: np.iinfo(np.uint16).min,
    ov.Type.u32: np.iinfo(np.uint32).min,
    ov.Type.u64: np.iinfo(np.uint64).min,
    ov.Type.i8: np.iinfo(np.int8).min,
    ov.Type.i16: np.iinfo(np.int16).min,
    ov.Type.i32: np.iinfo(np.int32).min,
    ov.Type.i64: np.iinfo(np.int64).min,
    ov.Type.boolean: 0,
}


def ov_to_keras_type(ov_type):
    for _keras_type, _ov_type in OPENVINO_DTYPES.items():
        if ov_type == _ov_type:
            return _keras_type
    raise ValueError(
        f"Requested OpenVINO type has no keras analogue '{ov_type.to_string()}'"
    )


def align_operand_types(x1, x2, op_name, force_float=False):
    x1_type = x1.element_type
    x2_type = x2.element_type
    if x1_type.is_dynamic() or x2_type.is_dynamic():
        raise ValueError(
            f"'{op_name}' operation is not supported for dynamic operand type "
            "with openvino backend"
        )
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    if force_float:
        result_type = dtypes.result_type(x1_type, x2_type, float)
    else:
        result_type = dtypes.result_type(x1_type, x2_type)
    result_type = OPENVINO_DTYPES[result_type]
    if x1_type != result_type:
        x1 = ov_opset.convert(x1, result_type).output(0)
    if x2_type != result_type:
        x2 = ov_opset.convert(x2, result_type).output(0)
    return x1, x2


def get_device():
    return "CPU"
