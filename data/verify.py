# This script seeks to verify that certain assumptions about the datasets are true.
# For example that certain speak training and test sets are or are not disjoint.


# standard libraries
import argparse
import os
from typing import Dict, List, Tuple
# third-party libraries

# project libraries
from speech.utils.io import load_config, read_data_json

def verify_disjoint(config:dict)->None:
    """This function will... 
    Args:
        config (dict): the config consists of a str, list, list
            full_dataset (str): path to the dataset with all speak training data
            disjoint_datasets (list): list of paths to datasets that are disjoint
            non_disjoint_datasets (list): list of paths to datasets that are non-disjoint
                these include the spk-25hr, 100-hr, and 250-hr which share recordings with the
                spk-500hr dataset
    Returns:
        None
    """
    full_id_dict = get_id_sets([config['full_dataset']])
    # extracting the set from the dict: Dict[str, Tuple[set, int]]
    full_id_set = list(full_id_dict.values())[0][0]
    # getting the dict of sets for each dataset 
    disjoint_id_sets = get_id_sets(config['disjoint_datasets'])
    n_disjoint_id_sets = get_id_sets(config['non_disjoint_datasets'])

    print("\n ===========================================")
    print("Assessing the intersection of the disjoint_sets")
    print_intersection_stats(disjoint_id_sets, full_id_set)
    
    print("\n\n ===========================================") 
    print("Assessing the intersection of the non_disjoint_sets")
    print_intersection_stats(n_disjoint_id_sets, full_id_set)
        
    #for name, (set, data_len) in n_disjoint_id_sets.items():
        #



def print_intersection_stats(id_dict:Dict[str, Tuple[set, int]], full_id_set:set)->None:
    """Looops through the items in id_dict and prints out the intersections
    Args:
        id_dict (dict): dict of sets and data-length as values and dataset name as a key
        full_id_set (set): set of the ids of the entire speak training dataset (7M examples)
    Returns:
        None
    """
    
    for name, (outer_set, data_len) in id_dict.items():
        print(f"Assessing {name}")
        print(f"length: {data_len}")
        print(f"interesction with full dataset: {len(list(full_id_set.intersection(outer_set)))}")
        for inner_name, (inner_set, inner_len) in id_dict.items():
            # don't count intersection with itself
            if name == inner_name:
                continue
            # count the number of intersecting examples
            intersect_count = len(list(outer_set.intersection(inner_set)))
            if 'test' in inner_name:
                print(f"checking intersect on testset: {inner_name}")
                if intersect_count != 0:
                    print(f"intersecting record is: {outer_set.intersection(inner_set)}")
            print(f"intersection with {inner_name} is {intersect_count} examples")
        print()
 

def get_id_sets(dataset_paths:List[str])->Dict[str, Tuple[set, int]]:
    """
    This function returns a dictionary with the dataset-name as the keys and a set of 
        record-ids as the values
    Args:
        dataset_paths (List[str]): a list of dataset paths (str)
    Returns:
        Dict[str, set]: a dict with the set of ids as values
    """
    data_dict = dict()
    
    for data_path in dataset_paths:
        # _extract_id on the data path will return the dataset name
        data_name = _extract_id(data_path)
        dataset = read_data_json(data_path)
        # set comprehension what extracts the record-id from each audiopath in the dataset
        id_set = {
            _extract_id(xmpl['audio']) for xmpl in dataset
        }
        data_dict.update({data_name: (id_set, len(dataset))})
    
    return data_dict    

def _extract_id(record_path:str)->str:
    #returns the basename of the path without the extension
    return os.path.basename(
        os.path.splitext(record_path)[0]
    )     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model."
    )
    parser.add_argument(
        "--config", help="Path to config file containing the necessary inputs"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    verify_disjoint(config)
