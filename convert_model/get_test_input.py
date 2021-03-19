import torch
import numpy


def generate_test_input(
    model_format:str, 
    model_name:str, 
    time_dim: int, 
    hidden_size:int=512,
    half_precision:bool=False):
    """
    outputs a test input based on the model format ("pytorch" or "onnx") and the model name
    Arguments
        time_dim: time_dimension into the model
        hidden_size (int): size of RNN/LSTM cell
    """
    batch_size = 1
    layer_count = 5 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if half_precision else torch.float32

    if model_format == "pytorch":       
        if model_name == "super_resolution":
            return torch.randn(batch_size, 1, 224, 224, requires_grad=True, device=device)
        elif model_name == "resnet18" or model_name == "alexnet":
            return torch.randn(batch_size, 3, 224,224, requires_grad=True, device=device)
        elif model_name == "lstm":
            return (torch.randn(5, 3, 10, device=device), 
                    torch.randn(layer_count * 2, 3, 20, device=device),
                    torch.randn(layer_count * 2, 3, 20, device=device)
                    )
        else:
            return (torch.randn(1,time_dim, 257, device=device).type(dtype),
                    (torch.randn(layer_count * 1, 1, hidden_size, device=device).type(dtype),
                    torch.randn(layer_count * 1, 1, hidden_size, device=device).type(dtype))
                    )
    else: 
        raise ValueError("model_format parameters must be 'pytorch' or 'onnx'")
