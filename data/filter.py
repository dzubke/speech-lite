# Copyright 2020 Speak Labs

"""
This script filters the a training.json files further
based on the number of files and certain constraints. This is mostly
used for the speak-train dataset.
"""

# standard libs
import argparse
from collections import defaultdict
import csv
import json
import os
import random
from typing import List
# third-party libs
import numpy as np
import yaml
# project libs
from speech.utils.io import read_data_json
from speech.utils.data_helpers import check_disjoint_filter, check_distribution_filter, check_update_contraints
from speech.utils.data_helpers import get_dataset_ids, get_disjoint_sets, get_record_ids_map
from speech.utils.data_helpers import path_to_id, process_text


def filter_speak_train(config:dict)->None:
    """
    This script filters the dataset in `full_json_path` and write the new dataset to `filter_json_path`.
    The constraints on the filtered dataset are:
        - utterances per speaker, lesson, and line cannot exceed the decimal values 
            as a fraction of the `dataset_size`. 
            Older config files have an absolute value on the `max_speaker_count`
        - the utterances are not also included in the datasets specified in `excluded_datasets`

    Config contents:
        full_json_path (str): path to the source json file that that the output will filter from
        metadata_path (str): path to the tsv file that includes metadata on each recording, 
            like the speaker_id
        filter_json_path (str): path to the filtered, written json file
        dataset_size (int): number of utterances included in the output dataset
        constraints (dict): dict of constraints on the number of utterances per speaker, lesson, 
            and line expressed as decimal fractions of the total dataset.
        disjoint_datasets (Dict[Tuple[str],str]): dict whose keys are a tuple of the ids that will be disjoint
            and whose values are the datasets paths whose examples will be disjiont from the output
    Returns:
        None, only files written.
    """
    # unpacking the config
    # TODO, only unpack what is necessary
    full_json_path = config['full_json_path']
    metadata_path = config['metadata_tsv_path']
    filter_json_path = config['filter_json_path']
    dataset_size = config['dataset_size']



    # re-calculate the constraints as integer counts based on the `dataset_size`
    constraints = {name: int(value * dataset_size) for name, value in config['constraints'].items()}
    print("constraints: ", constraints)

    # read and shuffle the full dataset and convert to iterator to save memory
    full_dataset = read_data_json(full_json_path)
    random.shuffle(full_dataset)
    full_dataset = iter(full_dataset)

    # get the mapping from record_id to other ids (like speaker, lesson, line) for each example
    record_ids_map = get_record_ids_map(metadata_path, list(constraints.keys()))

    # create a defaultdict with set values for each disjoint-id name
    disjoint_id_sets  = get_disjoint_sets(config['disjoint_datasets'])
    print("all disjoint names: ", disjoint_id_sets.keys())

    # id_counter keeps track of the counts for each speaker, lesson, and line ids
    id_counter = {name: dict() for name in constraints}

    examples_written = 0
    # loop until the number of examples in dataset_size has been written
    with open(filter_json_path, 'w') as fid:
        while examples_written < dataset_size:
            if examples_written != 0 and examples_written % config['print_modulus'] == 0:
                print(f"{examples_written} examples written")
            try:
                example = next(full_dataset)
            except StopIteration:
                print(f"Stop encountered {examples_written} examples written")
                break
                
            record_id = path_to_id(example['audio'])
            # check if the ids associated with the record_id are not included in the disjoint_datasets
            pass_filter = check_disjoint_filter(record_id, disjoint_id_sets, record_ids_map)
            if pass_filter:
                # check if the record_id pass the speaker, line, lesson constraints
                pass_constraint = check_update_contraints(
                    record_id, 
                    record_ids_map,
                    id_counter, 
                    constraints
                )
                if pass_constraint:
                    # if you don't want to use distribution filtering, the example always passes
                    if not config['dist_filter']['use']:
                        pass_distribution_filter = True
                    else:
                        # creates a filter based on the params in `dist_filter`
                        pass_distribution_filter = check_distribution_filter(example, config['dist_filter'])
                    if pass_distribution_filter:
                        json.dump(example, fid)
                        fid.write("\n")
                        # increment counters
                        examples_written += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="filters a training dataset")
    parser.add_argument("--config", type=str,
        help="path to preprocessing config.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    print("config: ", config)
    if config['dataset_name'].lower() == "speaktrain":
        filter_speak_train(config) 
