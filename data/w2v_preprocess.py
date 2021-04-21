# Copyright SpeakLabs 2021
# 
# This script will transform the librispeech phoneme labels into the wav2vec manifest files

# standard lib
import argparse
import json
import os
from pathlib import Path
import time
# third-party libs
import soundfile
# project libs
from speech.utils.data_helpers import path_to_id
from speech.utils.io import read_data_json
from speech.utils.wave import wav_duration


def filter_w2v_files(tsv_path, phn_path, filter_path):
    """This function will filter our the audio files specified in `filter_path`
    from the tsv_path and phn_path files. th
    """
    
    with open(filter_path, 'r') as filter_f:
        filter_id_set = {path_to_id(path.strip()) for path in filter_f}
    
    tsv_out_path = tsv_path.replace(".tsv", "-filtered.tsv")
    phn_out_path = phn_path.replace(".phn", "-filtered.phn")

    with open(tsv_path, 'r') as tsv_in_f, open(tsv_out_path, 'w') as tsv_out_f, \
        open(phn_path, 'r') as phn_in_f, open(phn_out_path, 'w') as phn_out_f:
        
        root = next(tsv_in_f).strip()
        tsv_out_f.write(root + '\n')

        tsv_in_f, phn_in_f = list(tsv_in_f), list(phn_in_f)
        assert len(tsv_in_f) == len(phn_in_f), "tsv and phn files are not same length"

        skipped_file_count = 0
        for i, tsv_line in enumerate(tsv_in_f):
            sub_path, _ = tsv_line.strip().split()
            line_id = path_to_id(sub_path)
            if line_id not in filter_id_set:
                tsv_out_f.write(tsv_line+ '\n')
                phn_out_f.write(phn_in_f[i]+ '\n')
            else:
                skipped_file_count += 1

        assert skipped_file_count == len(filter_id_set), \
            "{skipped_file_count} skipped, but should have skipped {len(filter_id_set)}"

def create_w2v_files(json_path:str, data_dir:str, save_dir:str):
    """Creates the tsv, phn, and dict.phn.txt files for the wav2vec from
    from the train.json file.

    Args:
        json_path: path to train.json file
        data_dir: root directory of all audio files
        save_dir: directory where tsv, phn, and dict files will be saved

    """

    tsv_path = create_tsv_file(json_path, data_dir, save_dir)

    dataset = read_data_json(json_path)
    phn_path = create_phn_file(dataset, tsv_path)

    check_manifest(dataset, tsv_path, phn_path)


def create_tsv_file(json_path:str, data_dir:str, save_dir:str)->str:
    """Creates a tsv manifest file for w2v model.
    
    Args:
        json_path: path to train.json file
        data_dir: root directory of all audio files
        save_dir: directory where tsv, phn, and dict files will be saved
    Returns:
        str: path to written tsv file
    """
    dataset = read_data_json(json_path)
    json_path = Path(json_path)
    tsv_filename = json_path.with_suffix(".tsv").name
    tsv_path = os.path.join(save_dir, tsv_filename)
    with open(tsv_path, "w") as out_f:
        print(data_dir, file=out_f)    
        for xmpl in dataset:
            file_path = xmpl['audio']
            frames = soundfile.info(file_path).frames
            print(
                "{}\t{}".format(os.path.relpath(file_path, data_dir), frames), file=out_f
            )

    return tsv_path


def create_phn_file(json_dataset:dict, tsv_path:str)->str:

    phone_path_map = {x['audio']: " ".join(x['text']) for x in json_dataset}
    tsv_path = Path(tsv_path)
    phn_file = tsv_path.with_suffix(".phn")

    # write the set of phones to "dict.phn.txt" in save_dir
    save_dir = tsv_path.parent
    write_phn_set(json_dataset, save_dir)
    
    with open(tsv_path, 'r') as tsv_f, open(phn_file, 'w') as phn_f:
        root = next(tsv_f).strip()    
    
        for line in tsv_f:
            file_path, _ = line.strip().split()
            phones = phone_path_map[os.path.join(root, file_path)]
            print(phones, file=phn_f)

    return phn_file
    
def write_phn_set(json_dataset:dict, save_dir):
    """
    """
    phn_set = set()
    phn_set.update([phn for x in json_dataset for phn in x['text']])
    with open(os.path.join(save_dir, "dict.phn.txt"), 'w') as fid:
        for phn in sorted(phn_set):
            print(f"{phn} 1", file=fid)


def check_manifest(json_dataset:dict, tsv_path:str, phn_path:str)->None:
    """Check that the paths in train.tsv and phoneme labels in train.phn match those in train.json"""

    data_dict = {xmpl['audio']: xmpl['text'] for xmpl in json_dataset}

    with open(tsv_path, 'r') as tsv_f, open(phn_path, 'r') as phn_f:
        root = next(tsv_f).strip()
        tsv_f, phn_f = list(tsv_f), list(phn_f)
        assert len(tsv_f) == len(phn_f), \
            f"number of entries in tsv: {len(tsv_f)} doesn't match number in phn file: {len(phn_f)}"

        for tsv_line, phones in zip(tsv_f, phn_f):
            subdir_path = tsv_line.strip().split()[0]
            audio_path = os.path.join(root, subdir_path)
            phones = phones.strip().split()
            assert data_dict[audio_path] == phones, \
                f"phones for path: {audio_path} in json file: {data_dict[audio_path]} don't match" + \
                f"phones in phn file: {phones}"

        print("All files match. All good!")


def w2v_to_json(w2v_tsv_path, w2v_phn_path, json_out_path)->None:
    """This function takes in the tsv and phn files for the w2v models and writes a training-json
    for the deepspeech models to the `json_out_path` location.
    """ 

    with open(w2v_tsv_path) as tsv_f, open(w2v_phn_path) as phn_f, open(json_out_path, 'w') as out_f:
        root = next(tsv_f).strip()
        
        for tsv_line, phones in zip(tsv_f, phn_f):
            sub_path, _ = tsv_line.strip().split()
            full_path = os.path.join(root, sub_path)
            phones = phones.strip().split()

            duration = wav_duration(full_path)
            
            json.dump({'text': phones, 'duration': duration, 'audio': full_path}, out_f)
            out_f.write('\n')
    
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="creates a data json file"
    )
    parser.add_argument(
        "--action", type=str, help="specifies function to call"
    )  
    parser.add_argument(
        "--data-dir", type=str, help="path root directory of dataset"
    )
    parser.add_argument(
        "--filter-path", type=str, help="path to file with audio recordings to filter out of dataset"
    )
    parser.add_argument(
        "--json-path", type=str, help="path to training json file"
    )
    parser.add_argument(
        "--phn-path", type=str, help="path to .phn file"
    )
    parser.add_argument(
        "--tsv-path", type=str, help="path to wav2vec tsv file"
    )
    parser.add_argument(
        "--save-dir", type=str, help="directory where outfile file will be saved"
    )
    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()
    if args.action == "check-manifest":
        check_manifest(args.tsv_path, args.phn_path, args.json_path)
    elif args.action == "create-tsv-file":
        create_tsv_file(args.json_path, args.save_dir)
    elif args.action == "create-w2v-files":
        create_w2v_files(args.json_path, args.data_dir, args.save_dir)
    elif args.action == "filter-w2v-files":
        filter_w2v_files(args.tsv_path, args.phn_path, args.filter_path)
    elif args.action == "w2v-to-json":
        w2v_to_json(args.tsv_path, args.phn_path, args.json_path)
    print(f"script took: {round((time.time() - start_time)/ 60, 3)} min")  
