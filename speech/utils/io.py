# standard libraries
from collections import OrderedDict
import glob
import json 
import os
import pickle
import yaml
# third-party libraries
import torch
# project libraries


def get_names(path:str, tag:str, get_config:bool=False, model_name:str=''):
    MODEL = "model_state_dict.pth"
    PREPROC = "preproc.pyc"
    
    tag = tag + "_" if tag else ""
    # if model_name is non-empty, reassign the global variable MODEL
    if model_name:      
        MODEL = model_name
    model_path = os.path.join(path, tag + MODEL)
    preproc_path = os.path.join(path, tag + PREPROC)

    if get_config:
        config_path = glob.glob(os.path.join(path, "ctc_config*[.yaml, .json]"))
        assert len(config_path) == 1, \
            f"{len(config_path)} config files found in directory: {path}"
        output = (model_path, preproc_path, config_path[0])
    
    else:
        output = (model_path, preproc_path)

    return output

def save(model, preproc, path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    torch.save(model.state_dict(), model_n)
    with open(preproc_n, 'wb') as fid:
        pickle.dump(preproc, fid)


def load(path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    model = torch.load(model_n, map_location=torch.device('cpu'))
    with open(preproc_n, 'rb') as fid:
        preproc = pickle.load(fid)
    return model, preproc


def load_pretrained(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model


def load_state_dict(model_path:str, device:torch.device)->OrderedDict:
    """
    returns the state dict of the model object or state dict specified
    in `model_path`
    Args:
        model_path: path to model or state_dict object
        device: torch device
    Returns:
        state_dict - OrderedDict: state dict of model
    """

    model_or_state_dict = torch.load(model_path, map_location=device)
    
    if isinstance(model_or_state_dict, OrderedDict):
        state_dict = model_or_state_dict
    elif isinstance(model_or_state_dict, torch.nn.Module):
        model = model_or_state_dict
        state_dict = model.state_dict()
    else:
        raise ValueError(f"model_path {model_path} does not point to model or state_dict object")
    
    return state_dict




def save_dict(dct, path):
    with open(path, 'wb') as fid:
        pickle.dump(dct, fid)


def export_state_dict(model_in_path, params_out_path):
    model = torch.load(model_in_path, map_location=torch.device('cpu'))
    pythtorch.save(model.state_dict(), params_out_path)


def read_data_json(data_path):
    with open(data_path) as fid:
        dataset = [json.loads(l) for l in fid]
        #ulimit = float('inf') #256    # target lengths cannot be longer than 256 for pytorch native loss
        #filtered_dataset = [datum for datum in dataset if len(datum['text']) <= ulimit]
        return dataset


def write_data_json(dataset:list, write_path:str):
    """
    Writes a list of dictionaries in json format to the write_path
    """
    with open(write_path, 'w') as fid:
        for example in dataset:
            json.dump(example, fid)
            fid.write("\n")


def read_pickle(pickle_path:str):
    assert os.path.exists(pickle_path), f"path {pickle_path} doesn't exist"
    with open(pickle_path, 'rb') as fid:
        pickle_object = pickle.load(fid)
    return pickle_object


def write_pickle(pickle_path:str, object_to_pickle):
    assert pickle_path != '', 'pickle_path is empty'
    with open(pickle_path, 'wb') as fid:
        pickle.dump(object_to_pickle, fid) 


def write_json(json_path:str, dict_to_write:dict)->None:
    """This function writes a dictionary to json at `json_path`.
    Args:
        json_path (str): path where json will be written
        dict_to_write (dict): dictionary to be written to json
    Returns:
        None
    """
    assert json_path != '', f'json_path: {json_path} is empty'
    assert isinstance(dict_to_write, dict), \
        f'dict_to_write is type: {type(dict_to_write)}, not dict'

    with open(json_path, 'w') as fid:
        json.dump(dict_to_write, fid)


def load_config(config_path:str)->dict:
    """
    loads the config file in json or yaml format
    """
    _, config_ext = os.path.splitext(config_path)    

    if config_ext == '.json':
        with open(config_path, 'r') as fid:
            config = json.load(fid)
    elif config_ext == '.yaml':
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file) 
    else:
        raise ValueError(f"config file extension {config_ext} not accepted")
    
    return config


def load_from_trained(model, model_cfg):
    """loads the model with pretrained weights from the model in model_cfg["trained_path"]
    Args:
        model (torch model)
        model_cfg (dict): configuration for the model
    """
    trained_model = torch.load(model_cfg["local_trained_path"], map_location=torch.device('cpu'))
    if isinstance(trained_model, dict):
        trained_state_dict = trained_model
    else:
        trained_state_dict = trained_model.state_dict()
    
    trained_state_dict = filter_state_dict(trained_state_dict, remove_layers=model_cfg["remove_layers"])
    model_state_dict = model.state_dict()
    model_state_dict.update(trained_state_dict)
    model.load_state_dict(model_state_dict)
   
    return model


def filter_state_dict(state_dict, remove_layers=[]):
    """
    filters the inputted state_dict by removing the layers specified
    in remove_layers
    Arguments:
        state_dict (OrderedDict): state_dict of pytorch model
        remove_layers (list(str)): list of layers to remove 
    """

    state_dict = OrderedDict(
        {key:value for key,value in state_dict.items()
        if key not in remove_layers}
        )
    return state_dict
