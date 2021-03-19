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



def torch_to_onnx(
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
    time_dim = num_frames
    
    model_cfg.update({'blank_idx': config['preproc']['blank_idx']})
    torch_model = CTC_model(freq_dim, vocab_size, model_cfg) 

    state_dict = load_state_dict(torch_path, device=device)

    torch_model.load_state_dict(state_dict)
    torch_model.to(device)
    print("model on cuda?: ", torch_model.is_cuda)    
    
    torch_model.eval()    

    # create the tracking inputs
    hidden_size = config['model']['encoder']['rnn']['dim'] 
    input_tensor = generate_test_input("pytorch", model_name, time_dim, hidden_size) 

    # export the models to onnx
    torch_onnx_export(torch_model, input_tensor, onnx_path)
    print(f"Torch model sucessfully converted to Onnx at {onnx_path}")

    if return_models:
        onnx_model = onnx.load(onnx_path)
        return torch_model, onnx_model


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

    torch_to_onnx(
        args.model_name, 
        args.num_frames,
        return_models
    )
