import inspect

import openvino as ov
import openvino.opset16 as ov_opset

from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.saved_model_export_archive import SavedModelExportArchive
from keras_openvino.src.ops.core import OpenVINOKerasTensor
from keras_openvino.src.utils import OPENVINO_DTYPES


class OpenvinoExportArchive(SavedModelExportArchive):
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the openvino backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the openvino backend."
        )


def get_model_for_openvino_export(model, input_signature):
    def parameterize_inputs(inputs, prefix=""):
        if isinstance(inputs, (list, tuple)):
            return [
                parameterize_inputs(e, f"{prefix}{i}")
                for i, e in enumerate(inputs)
            ]
        elif isinstance(inputs, dict):
            return {k: parameterize_inputs(v, k) for k, v in inputs.items()}
        elif isinstance(inputs, OpenVINOKerasTensor):
            ov_type = OPENVINO_DTYPES[str(inputs.dtype)]
            ov_shape = list(inputs.shape)
            param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
            param.set_friendly_name(prefix)
            return OpenVINOKerasTensor(param.output(0))
        else:
            raise TypeError(f"Unknown input type: {type(inputs)}")

    if isinstance(input_signature, list) and len(input_signature) == 1:
        input_signature = input_signature[0]

    sample_inputs = tree.map_structure(
        lambda x: convert_spec_to_tensor(x, replace_none_number=1),
        input_signature,
    )
    params = parameterize_inputs(sample_inputs)
    signature = inspect.signature(model.call)
    if len(signature.parameters) > 1 and isinstance(params, (list, tuple)):
        outputs = model(*params)
    else:
        outputs = model(params)
    parameters = [p.output.get_node() for p in tree.flatten(params)]
    results = [ov_opset.result(r.output) for r in tree.flatten(outputs)]
    ov_model = ov.Model(results=results, parameters=parameters)
    flat_specs = tree.flatten(input_signature)
    for ov_input, spec in zip(ov_model.inputs, flat_specs):
        # Respect the dynamic axes from the original input signature.
        dynamic_shape_dims = [-1 if dim is None else dim for dim in spec.shape]
        dynamic_shape = ov.PartialShape(dynamic_shape_dims)
        ov_input.get_node().set_partial_shape(dynamic_shape)

    return ov_model
