# standard libraries
import argparse
from collections import OrderedDict
import json
import os
# third-party libraries
import onnx
import torch
# project libraries
from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export
import speech.loader as loader
from speech.models.ctc_model import CTC as CTC_model
from speech.utils.io import load_config, load_state_dict


def torch_to_coreml(
    model_name:str, 
    num_frames:int, 
    use_state_dict:bool, 
    return_models:bool=False)->None:
    """
    Arg:
        model_name (str): filename of the model
        num_frames (int): number of feature frames that will fix the model's size
        return_models (bool, False): if true, the function will return the torch and onnx model objects
    """  

    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    config = load_config(config_path)
    model_cfg = config['model']

    freq_dim = 257  #freq dimension out of log_spectrogram 
    vocab_size = 39
    
    torch_model = CTC_model(freq_dim, vocab_size, model_cfg) 

    state_dict = load_state_dict(torch_path, device=device)

    torch_model.load_state_dict(state_dict)
    torch_model.to(device)
    print("model on cuda?: ", torch_model.is_cuda)    
    
    torch_model.eval()    

    # create the tracking inputs
    hidden_size = config['model']['encoder']['rnn']['dim'] 
    x, (h_in, c_in) = generate_test_input("pytorch", model_name, 31, hidden_size) 

    traced_model = torch.jit.trace(torch_model, (x, (h_in, c_in)))

    x_46, (h_46, c_46) = generate_test_input("pytorch", model_name, 46, hidden_size)

    out_46, (h_out_46, c_out_46) = traced_model(x_46, (h_46, c_46))

    if return_models:
        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument(
        "--model-name", help="name of the model."
    )
    parser.add_argument(
        "--num-frames", type=int, 
        help="number of input frames in time dimension hard-coded in onnx model"
    )
    args = parser.parse_args()

    return_models = False

    torch_to_coreml(
        args.model_name, 
        args.num_frames,
        return_models
    )