# standard libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import json
# third-party libraries
import editdistance
import matplotlib as plt
import numpy as np
import torch
import tqdm
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_decoder import decode
from speech.models.ctc_model_train import CTC_train
from speech.utils.io import get_names, load_config, load_state_dict, read_data_json, read_pickle


def eval_loop(model, ldr, device):
    """Runs the evaluation loop on the input data `ldr`.
    
    Args:
        model (torch.nn.Module): model to be evaluated
        ldr (torch.utils.data.DataLoader): evaluation data loader
        device (torch.device): device inference will be run on

    Returns:
        list: list of labels, predictions, and confidence levels for each example in
            the dataloader
    """
    all_preds = []; all_labels = []; all_preds_dist=[]
    all_confidence = []
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            batch = list(batch)
            inputs, targets, inputs_lens, targets_lens = model.collate(*batch)
            inputs = inputs.to(device)
            probs, rnn_args = model(inputs, softmax=True)
            probs = probs.data.cpu().numpy()
            preds_confidence = [decode(p, beam_size=3, blank=model.blank)[0] for p in probs]
            preds = [x[0] for x in preds_confidence]
            confidence = [x[1] for x in preds_confidence]
            all_preds.extend(preds)
            all_confidence.extend(confidence)
            all_labels.extend(batch[1])
    return list(zip(all_labels, all_preds, all_confidence))


def run_eval(
        model_path, 
        dataset_json, 
        batch_size=8, 
        tag="best", 
        model_name="model_state_dict.pth",
        device = None,
        add_filename=False, 
        add_maxdecode:bool=False, 
        formatted=False, 
        config_path = None, 
        out_file=None)->int:
    """
    calculates the  distance between the predictions from
    the model in model_path and the labels in dataset_json

    Args:
        model_path (str): path to the directory that contains the model,
        dataset_json (str): path to the dataset json file
        batch_size (int): number of examples to be fed into the model at once
        tag (str): string that prefixes the model_name.  if best,  the "best_model" is used
        model_name (str): name of the model, likely either "model_state_dict.pth" or "model"
        device (torch.device): device that the evaluation should run on
        add_filename (bool): if true, the filename is added to each example in `save_json`
        add_maxdecode (bool): if true, the predictions using max decoding will be added in addition 
            to the predictions from the ctc_decoder
        formatted (bool): if true, the `format_save` will be used instead of `json_save` where 
            `format_save` outputs a more human-readable output file
        config_path (bool): specific path to the config file, if the one in `model_path` is not desired
        out_file (str): path where the output file will be saved
    
    Returns:
        (int): returns the computed error rate of the model on the dataset
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path, preproc_path, config_path = get_names(model_path, tag=tag, model_name=model_name, get_config=True)
    
    # load and update preproc
    preproc = read_pickle(preproc_path)
    preproc.update()

    # load and assign config
    config = load_config(config_path)
    model_cfg = config['model']
    model_cfg.update({'blank_idx': config['preproc']['blank_idx']}) # creat `blank_idx` in model_cfg section


    # create model
    model = CTC_train(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)

    state_dict = load_state_dict(model_path, device=device)
    model.load_state_dict(state_dict)
    
    ldr =  loader.make_loader(
        dataset_json,
        preproc, 
        batch_size
    )
    
    model.to(device)
    model.set_eval()
    print(f"preproc train_status before set_eval: {preproc.train_status}")
    preproc.set_eval()
    preproc.use_log = False
    print(f"preproc train_status after set_eval: {preproc.train_status}")


    results = eval_loop(model, ldr, device)
    print(f"number of examples: {len(results)}")
    #results_dist = [[(preproc.decode(pred[0]), preproc.decode(pred[1]), prob)] 
    #                for example_dist in results_dist
    #                for pred, prob in example_dist]
    results = [(preproc.decode(label), preproc.decode(pred), conf)
               for label, pred, conf in results]
    # maxdecode_results = [(preproc.decode(label), preproc.decode(pred))
    #           for label, pred in results]
    cer = speech.compute_cer(results, verbose=True)

    print("PER {:.3f}".format(cer))
    
    if out_file is not None:
        compile_save(results, dataset_json, out_file, formatted, add_filename)
    
    return round(cer, 3)


def compile_save(results, dataset_json, out_file, formatted=False, add_filename=False):
    """This function compiles the results and saved in to two different formats:
        a simple json format or a human-readable output. 
    """
    output_results = []
    if formatted:
        format_save(results, dataset_json, out_file)
    else: 
        json_save(results, dataset_json, out_file, add_filename)
        

def format_save(results, dataset_json, out_file):
    """This function writes the results to a file in a human-readable format.
    """
    out_file = create_filename(out_file, "compare", "txt")
    out_file = os.path.join("predictions", out_file)
    print(f"file saved to: {out_file}")
    with open(out_file, 'w') as fid:
        write_list = list()
        for label, pred, conf in results:
            lower_list = lambda x: list(map(str.lower, x))
            label, pred = lower_list(label), lower_list(pred)
            filepath, order = match_filename(label, dataset_json, return_order=True)
            filename = os.path.splitext(os.path.split(filepath)[1])[0]
            PER, (dist, length) = speech.compute_cer([(label,pred)], verbose=False, dist_len=True)
            write_list.append({"order":order, "filename":filename, "label":label, "preds":pred,
            "metrics":{"PER":round(PER,3), "dist":dist, "len":length, "confidence":round(conf, 3)}})
        write_list = sorted(write_list, key=lambda x: x['order'])
            
        for write_dict in write_list: 
            fid.write(f"{write_dict['filename']}\n") 
            fid.write(f"label: {' '.join(write_dict['label'])}\n") 
            fid.write(f"preds: {' '.join(write_dict['preds'])}\n")
            
            PER, dist = write_dict['metrics']['PER'], write_dict['metrics']['dist'] 
            length, conf = write_dict['metrics']['len'], write_dict['metrics']['confidence']
            fid.write(f"metrics: PER: {PER}, dist: {dist}, len: {length}, conf: {conf}\n")
            fid.write("\n")

        for write_dict in write_list:
            fid.write(f"{write_dict['filename']}, {write_dict['metrics']['PER']}\n")

def json_save(results, dataset_json, out_file, add_filename):
    """This function writes the results into a json format.
    """
    output_results = []
    for label, pred, conf in results: 
        if add_filename:
            filename = match_filename(label, dataset_json)
            PER = speech.compute_cer([(label,pred)], verbose=False)
            res = {'filename': filename,
                'prediction' : pred,
                'label' : label,
                'PER': round(PER, 3)}
        else:   
            res = {'prediction' : pred,
                'label' : label}
        output_results.append(res)

    # if including filename, add the suffix "_fn" before extension
    if add_filename: 
        out_file = create_filename(out_file, "pred-fn", "json")
        output_results = sorted(output_results, key=lambda x: x['PER'], reverse=True) 
    else: 
        out_file = create_filename(out_file, "pred", "json")
    print(f"file saved to: {out_file}") 
    with open(out_file, 'w') as fid:
        for sample in output_results:
            json.dump(sample, fid)
            fid.write("\n") 

def match_filename(label:list, dataset_json:str, return_order=False) -> str:
    """
    returns the filename in dataset_json that matches
    the phonemes in label
    """
    dataset = read_data_json(dataset_json)
    matches = []
    for i, sample in enumerate(dataset):
        if sample['text'] == label:
            matches.append(sample["audio"])
            order = i
    
    assert len(matches) < 2, f"multiple matches found {matches} for label {label}"
    assert len(matches) >0, f"no matches found for {label}"
    if return_order:
        output = (matches[0], order)
    else:
        output = matches[0]
    return output

def create_filename(base_fn, suffix, ext):
    if "." in ext:
        ext = ext.replace(".", "")
    return base_fn + "_" + suffix + os.path.extsep + ext  


def plot_per_historgram(per_path:str, save_path:str=None):
    """
    This function plots PER values as a histogram. The plot is saved to `save_path`.
    Args:
        per_path (str): path to per csv file with one column as the sample_id and the other the per value
        save_path (str): path to save the histrogram plot
    """
    import csv
    import matplotlib.pyplot as plt
    import numpy as np

    with open(per_path, 'r') as fid:
        reader = csv.reader(fid, delimiter=',')
        per_list = [float(row[1]) for row in reader]

    plt.hist(per_list, bins=10, range=(0.0, 1.0))
    plt.title("histogram of 2020-10-29 model PER values")
    plt.xlabel("PER bins")
    plt.ylabel("# of records")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    #plt.yticks(labels=per_list)
    if save_path == None:
        save_path = "PER_histogram.png" 
    plt.savefig(save_path)

def plot_all_hist():
    """
    This code can be run in iPython to produce histogram plots of several model PER values
    """

    per_path_08_06 = "/Users/dustin/Desktop/2020-08-06_PER.txt"
    with open(per_path_08_06, 'r') as fid:
        reader = csv.reader(fid, delimiter=',')
        per_list_08_06 = [float(row[1]) for row in reader]

    per_path_09_25 = "/Users/dustin/Desktop/2020-09-25_PER.txt"
    with open(per_path_09_25, 'r') as fid:
        reader = csv.reader(fid, delimiter=',')
        per_list_09_25 = [float(row[1]) for row in reader]

    per_path_10_29 = "/Users/dustin/Desktop/2020-10-29_PER.txt"
    with open(per_path_10_29, 'r') as fid:
        reader = csv.reader(fid, delimiter=',')
        per_list_10_29 = [float(row[1]) for row in reader]


    fig, axs = plt.subplots(3,1, constrained_layout=True)

    axs[0].hist(per_list_10_29, bins=10, range=(0.0, 1.0), color='b')
    axs[1].hist(per_list_09_25, bins=10, range=(0.0, 1.0), color='orange')
    axs[2].hist(per_list_08_06, bins=10, range=(0.0, 1.0), color='g')


    axs[0].set_yticks(np.arange(0, 40, step=5))
    axs[1].set_yticks(np.arange(0, 40, step=5))
    axs[2].set_yticks(np.arange(0, 40, step=5))

    axs[0].set_xticks(np.arange(0, 1.1, step=0.1))
    axs[1].set_xticks(np.arange(0, 1.1, step=0.1))
    axs[2].set_xticks(np.arange(0, 1.1, step=0.1))

    axs[2].set_xlabel("PER bins")

    axs[0].set_ylabel("# of records")
    axs[1].set_ylabel("# of records")
    axs[2].set_ylabel("# of records")

    axs[0].set_title("2020-10-29 model, 0.168 PER")
    axs[1].set_title("2020-09-25 model, 0.24 PER")
    axs[2].set_title("2020-08-06 model, 0.38 PER")

    plt.savefig("PER_historgrams_2020-11-09.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1,
        help="Batch size during evaluation")
    parser.add_argument("--best", action="store_true", default=False,
        help="Use best model on dev set instead of last saved model.")
    parser.add_argument("--save",
        help="Optional file to save predicted results.")
    parser.add_argument("--filename", action="store_true", default=False,
        help="Include the filename for each sample in the json output.")
    parser.add_argument("--formatted", action="store_true", default=False,
        help="Output will be written to file in a cleaner format.")
    parser.add_argument("--config-path", type=str, default=None,
        help="Replace the preproc from model path a  preproc copy using the config file.")
    args = parser.parse_args()

    run_eval(
        args.model, 
        args.dataset, 
        tag='best' if args.best else None,
        batch_size = args.batch_size, 
        add_filename=args.filename,  
        formatted=args.formatted, 
        config_path=args.config_path, 
        out_file=args.save
    )
