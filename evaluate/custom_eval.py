# standard libraries
import argparse
from collections import defaultdict, OrderedDict
import os
import csv 
from pathlib import Path
from typing import Tuple
# third-party libraries
import editdistance
import torch
import tqdm
# project libraries
from evaluate.eval import run_eval
import speech.loader
from speech.models.ctc_decoder import decode as ctc_decode
from speech.models.ctc_model_train import CTC_train as CTC_train
from speech.utils.data_helpers import lexicon_to_dict, path_to_id, text_to_phonemes
from speech.utils.io import get_names, load_config, load_state_dict, read_data_json, read_pickle
from speech.utils.visual import print_nonsym_table



def eval1(config:dict)->None:
    """
    This function produces a formatted output used to compare the predictions of a set of models 
    against the phonemes in the target and guess for each utterance. This output is used to 
    determine if the models are correctly labelling mispronunciations. 
    Config contains:
        models: contains dict of models with name, path, tag, and model_name for the 
            speech.utils.io.get_names function
        dataset_path (str): path to evaluation dataset
        save_path (str): path where the formatted txt file will be saved
        lexicon_path (str): path to lexicon used to convert words in target and guess to phonemes
        n_top_beams (int): number of beams output from the ctc_decoder
    Return:
        None
    """

    # unpack the config
    model_params = config['models']
    dataset_path = Path(config['dataset_path'])
    output_path = config['output_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the models and preproc objects
    print(f"model_params contains: {model_params}")

    model_preproc = {
        model_name: _load_model(params, device) for model_name, params in model_params.items()
    }

    if dataset_path.suffix == ".tsv":
        output_dict = output_dict_from_tsv(dataset_path, config['lexicon_path'])
    elif dataset_path.suffix == ".json":
        output_dict = output_dict_from_json(dataset_path)
    else:
        raise ValueError(f"dataset extension must be '.tsv' or '.json'. Recieved: {dataset_path.suffix}")

    # directory where audio paths are stored
    audio_dir = dataset_path.parent.joinpath("audio")

    # loop through each output file and perform inference on each model
    for rec_id in tqdm.tqdm(output_dict.keys()):
        audio_path = audio_dir.joinpath(rec_id).with_suffix(".wav")
        dummy_target = []   # dummy target list fed into the preprocessor, not used
        output_dict[rec_id]['infer'] = {}   # initialize a dict for the inference outputs

        for model_name, (model, preproc) in model_preproc.items():
            with torch.no_grad():    # no gradients calculated to speed up inference
                inputs, dummy_target = preproc.preprocess(str(audio_path), dummy_target)
                inputs = torch.FloatTensor(inputs)
                inputs = torch.unsqueeze(inputs, axis=0).to(device)   # add the batch dim and push to `device`
                probs, _ = model(inputs, softmax=True)      # don't need rnn_args output in `_`
                probs = probs.data.cpu().numpy().squeeze() # convert to numpy and remove batch-dim
                top_beams = ctc_decode(probs, 
                                        beam_size=3, 
                                        blank=model.blank, 
                                        n_top_beams=config['n_top_beams']
                )
                top_beams = [(preproc.decode(preds), probs) for preds, probs in top_beams]
                output_dict[rec_id]['infer'].update({model_name: top_beams})


    # write the PER predictions to a txt file
    # sort the dictionary to ease of matching audio file with formatted output
    output_dict = OrderedDict(sorted(output_dict.items()))
    per_counter = defaultdict(lambda : {"total_diff":0, "total_phones":0})
    with open(output_path, 'w') as out_file:  
        for rec_id in output_dict.keys():  
            out_file.write(f"rec_id:\t\t\t{rec_id}\n")
            # write the header
            for name, values in output_dict[rec_id]['header'].items():
                out_file.write(f"{name}:\t\t\t{values}\n")
            # write predictions from each model. writing multiple search-beams, if specified
            for model_name in output_dict[rec_id]['infer'].keys():
                top_beam=True
                for preds, confid in output_dict[rec_id]['infer'][model_name]:
                    per = editdistance.eval(output_dict[rec_id]['reference_phones'], preds)
                    per_counter[model_name]['total_diff'] += per
                    len_phones = len((output_dict[rec_id]['reference_phones']))
                    per_counter[model_name]['total_phones'] += len_phones
                    per /= len_phones
                    # this top-beam if-else is used if multiple top beams in the search decoder
                    # are desired
                    if top_beam:
                        out_file.write(f"{model_name}:\t({round(per, 2)})\t{' '.join(preds)}\n")
                        top_beam = False
                    else:
                        out_file.write(f"\t   \t({round(per, 2)})\t{' '.join(preds)}\n")
            out_file.write("\n\n")

        out_file.write("Dataset PER Values\n")
        out_file.write("------------------\n")
        for model_name, per_dict in per_counter.items():
            per = round(per_dict['total_diff'] / per_dict['total_phones'], 3)
            out_file.write(f"{model_name}\t{per}\n")
        out_file.write("------------------\n")

def output_dict_from_tsv(tsv_dataset_path:str, lexicon_path:str)->dict:
    """This function returns a formatted output dict using a tsv dataset path
    """
    output_dict = {}     # dictionary containing the printed outputs
    lexicon = lexicon_to_dict(lexicon_path)

    # open the tsv file that contains the data on each example
    with open(tsv_dataset_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        tsv_dataset = list(tsv_reader)
        # tsv header is: "id", "target", "guess", "lessonId", "lineId", "uid", "redWords_score", "date"
        for row in tsv_dataset:
            record_id, target, guess = row[0], row[1], row[2]
            # text_to_phonemes returns a list of string-phonemes
            target_phones = text_to_phonemes(target, lexicon, unk_token="<UNK>")
            guess_phones = text_to_phonemes(guess, lexicon, unk_token="<UNK>")
            
            output_dict[record_id] = {
                "header": {
                    "target": target,
                    "guess": guess,
                    "tar_pn": " ".join(target_phones),
                    "ges_pn": " ".join(guess_phones)
                },
                "reference_phones": target_phones # used for PER calculation
            }
            
    return output_dict


def output_dict_from_json(json_dataset_path:str)->dict:
    """This function returns a formatted output dict using a json dataset path
    """
    output_dict = {}     # dictionary containing the printed outputs
    dataset = read_data_json(json_dataset_path)

    for xmpl in dataset:
        record_id = path_to_id(xmpl['audio'])
        
        output_dict[record_id] = {
            "header": {
                "labels": " ".join(xmpl['text'])
            },
            "reference_phones": xmpl['text'] # used for PER calculation
        }
        
    return output_dict


def _load_model(model_params:str, device)->Tuple[torch.nn.Module, speech.loader.Preprocessor]:
    """
    This function will load the model, config, and preprocessing object and prepare the model and preproc for evaluation
    Args:
        model_path (dict): dict containing model path, tag, and filename
        device (torch.device): torch processing device
    Returns:
        torch.nn.Module: torch model
        preprocessing object (speech.loader.Preprocessor): preprocessing object
    """

    model_path, preproc_path, config_path = get_names(
        model_params['path'], 
        tag=model_params['tag'], 
        get_config=True,
        model_name=model_params['filename']
    )
    
    # load and update preproc
    preproc = read_pickle(preproc_path)
    preproc.update()

    # load and assign config
    config = load_config(config_path)
    model_cfg = config['model']
    model_cfg.update({'blank_idx': config['preproc']['blank_idx']}) # creat `blank_idx` in model_cfg section

    # create model
    model = CTC_train(
        preproc.input_dim,
        preproc.vocab_size,
        model_cfg
    )

    state_dict = load_state_dict(model_path, device=device)
    model.load_state_dict(state_dict)

    model.to(device)
    # turn model and preproc to eval_mode
    model.set_eval()
    preproc.set_eval()

    return model, preproc



def eval2(config:dict)->None:
    """This function prints a table with a set of models as rows and a set of datasets as columns
    where the values in the table are the PER's for each row-column pai.
    Config contains:
        models (dict): dict with model names as keys and values of path, tag, and model_name
            for the `get_names` function
        datasets (dict): dict with dataset names as keys and dataset paths as values
        save_path (str): path where the output file will be saved
        lexicon_path (str): path to lexicon
    Return:
        None
    """

    # unpack the config
    model_params = config['models']
    datasets = config['datasets']
    output_path = config['output_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the models and preproc objects
    print(f"model_params contains: {model_params}")

    # dict to contain all of the per values for each model
    per_dict = dict()
    # loop through the models and datasets the calculate a per for each combo
    for model_name, params in model_params.items():
        per_dict[model_name] = dict()   # initialize the new key in the per_dict
        print(f"calculating per for model: {model_name}")
        for data_name, data_path in datasets.items():
            print(f"calculating per for dataset: {data_name}")
            per = run_eval(
                model_path=params['path'],
                dataset_json = data_path,
                batch_size = 1,
                tag = params['tag'],
                model_name = params['filename']
            )
            print(f"PER value is: {per}")
            per_dict[model_name][data_name] = per

    print("full per_dict values: ")
    print(per_dict)
    print_nonsym_table(per_dict, title="PER values", row_name="Data\\Model")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model."
    )
    parser.add_argument(
        "--config", help="Path to config file containing the necessary inputs"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config['eval_type'] == "eval1":
        eval1(config)
    elif config['eval_type'] == 'eval2':
        eval2(config)
    else:
        raise ValueError(f'eval types must be either "eval1" or "eval2", not {config["eval_type"]}')
