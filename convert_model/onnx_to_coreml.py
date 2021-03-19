# standard libraries
import argparse
import os
# third-party libraries
from coremltools.models.neural_network import quantization_utils
import onnx
from onnx_coreml import convert


def onnx_to_coreml(model_name:str, half_precision:bool, quarter_precision:bool, return_or_save:str='save'):
    """
    Arguments
    ---------
    model_name: str
        filename of the model to convert
    half_precision: bool
        whether to convert the coreml model to half precision
    quarter_precision: bool
        whether to conver the coreml model to quarter precision
    return_or_save: str == "save" or "return"
        if "save", the model will saved as the model_name. if 'return' the model object is returned

    Returns
    -------
    coreml_model: coreml-model-object
        if return_or_save == 'return', the model object is returned. otherwise, None.
    """

    assert return_or_save in ['save', 'return'], \
        f"return_or_save must be 'save' or 'return'. {return_or_save} entered."

    assert not(half_precision and quarter_precision), \
        "half-precision and quarter-precision flags can't both be used during same call."
    
    onnx_path = os.path.join("onnx_models", model_name+"_model.onnx")
    coreml_path = os.path.join("coreml_models", model_name+"_model.mlmodel")

    onnx_model = onnx.load(onnx_path)

    coreml_model = convert(model=onnx_model,
                            minimum_ios_deployment_target = '13')

    if half_precision:
        coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=16)
        print("\n~~~~ Converted CoreML Model to half precision ~~~~\n")
    elif quarter_precision:
        coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=8)
        print("\n~~~~ Converted CoreML Model to quarter precision ~~~~\n")
    else:
        print("\n~~~~ CoreML Model kept at single precision ~~~~\n")

    if return_or_save == 'save':
        coreml_model.save(coreml_path)
        print(f"Onnx model successfully converted to CoreML at: {coreml_path}")
    elif return_or_save == 'return':
        return onnx_model, coreml_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="converts onnx model to coreml.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--half-precision", action='store_true', default=False,  help="converts the model to half precision.")
    parser.add_argument("--quarter-precision", action='store_true', default=False, help="converts the model to quarter precision.")
    args = parser.parse_args()
    print(f"Args in onnx_to_coreml: {args}")

    return_or_save = 'save'
    onnx_to_coreml(args.model_name, args.half_precision, args.quarter_precision, return_or_save)
