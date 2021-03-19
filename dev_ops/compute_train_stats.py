# standard libs
import argparse
import os
from typing import List
# third-party libs
# project libs
from speech.utils.io import read_data_json


def main(data_path:str)-> None:
    """
    Prints various stats like the total audio length of the input dataset in `data_path`
    Args
        data_path (str): path to the dataset
    
    """
    dataset = read_data_json(data_path)
    
    # iterate through the dataset 
    durations = list()
    path_counter = dict()
    data_disk_prefix = "/mnt/disks/data_disk/home"
    for elem in dataset:
        durations.append(elem['duration'])
        path = elem['audio']
        assert path.startswith(data_disk_prefix), f"path {path} is not a data disk path"
        path_counter[path] = path_counter.get(path, 0) + 1
               
    # print out the total time
    total_sec = sum(durations)
    total_hr = round(total_sec / 3600, 2)

    data_name = os.path.basename(data_path)

    print(f"total duration for {data_name}: {total_hr} hrs:")

    # check if any paths occured more than once
    dup_paths = {path: count for path, count in path_counter.items() if count > 1}
    print(f"duplicated paths: {dup_paths}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model."
    )
    parser.add_argument(
        "data_path", type=str, help="path to dataset"
    )
    
    args = parser.parse_args()

    main(args.data_path)
