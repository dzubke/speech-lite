# standard libraries
from collections import defaultdict
import csv
from datetime import date
from gc import get_referents
import glob
import json
import os
import re
import string
import sys
from types import ModuleType, FunctionType
from typing import List, Set
# third-party libraries
import tqdm
# project libraries
from speech.utils import convert
from speech.utils.io import read_data_json

UNK_WORD_TOKEN = list()

def today_date()->str:
    """returns a string of todays date formatted as 'YYYY-MM-DD'
    """
    return str(date.today())


def lexicon_to_dict(lexicon_path:str, corpus_name:str=None)->dict:
    """
    This function reads the librispeech-lexicon.txt file which is a mapping of words in the
    librispeech corpus to phoneme labels and represents the file as a dictionary.
    The digit accents are removed from the file name. 
    """
    accepted_corpora = [
        "librispeech", "tedlium", "cmudict", "commonvoice", "voxforge", "tatoeba", "speaktrain", None,
        "speaktrainmetadata", "switchboard", "peoplesspeech"
    ]
    if corpus_name not in accepted_corpora:
        raise ValueError("corpus_name not accepted")
    
    lex_dict = dict()
    with open(lexicon_path, 'r', encoding="ISO-8859-1") as fid:
        lexicon = process_lines(fid, corpus_name)
        for line in lexicon: 
            word, phones = word_phone_split(line, corpus_name)
            phones = clean_phonemes(phones, corpus_name)
            # this if-statement will ignore the second pronunciation like in librispeech
            if lex_dict.get(word, UNK_WORD_TOKEN) == UNK_WORD_TOKEN:
                lex_dict[word] = phones
    lex_dict = remove_alt_pronun(lex_dict, corpus_name)
    #assert type(lex_dict)== defaultdict, "word_phoneme_dict is not defaultdict"
    return lex_dict


def process_lines(file_reader, corpus_name:str)->list:
    """Strips and splits the lexicon lines.
    The case is kept intact for the 'switchboard' corpus, but is lowered for all others

    Args:
        file_reader: file reader object of lexicon
        corpus_name (str)
    Returns:
        (list): split lexicon lines
    """
    if corpus_name == 'switchboard':
        lexicon = [l.strip().split() for l in file_reader]
    else:
        lexicon = [l.strip().lower().split() for l in file_reader]
    return lexicon


def word_phone_split(line:list, corpus_name:str):
    """
    Splits the input list line into the word and phone entry.
    voxforge has a middle column that is not used
    """
    if corpus_name == "voxforge":
        word, phones = line[0], line[2:]
    else:
        try:
            word, phones = line[0], line[1:]
        except IndexError:
            print(f"index error in line: {line}")
            raise IndexError
    return word, phones


def clean_phonemes(phonemes, corpus_name):
    """Librispeech and cmudict have stress digits that need to be removed
    """

    if corpus_name == "librispeech" or corpus_name == "cmudict":
        return [phones.rstrip(string.digits) for phones in phonemes]
    else:
        return phonemes


def remove_alt_pronun(lex_dict, corpus_name):
    """removes alternative pronunciations
    """
    if corpus_name in ["tedlium", "cmudict", "voxforge"]:
        return defaultdict(
            lambda: UNK_WORD_TOKEN, 
            {key: value for key, value in lex_dict.items() if not re.search("\(\d\)$", key)}
        )
    else: # no-op
        return lex_dict


def combine_lexicon_helper(lex1_dict:dict, lex2_dict:dict)->(dict, dict):
    """
    this function takes as input a dictionary representation of the two
    lexicons and outputs a combined dictionary lexicon. it also outputs
    a dict of words with different pronunciations
    Arguments:
        lex1_dict - dict[str:list(str)]: dict representation of the first lexicon
        lex2_dict - dict[str:list(str)]: dict representation of the second lexicon
    Returns:
        combo_dict - dict[str:list(str]
    """
    word_set = set(list(lex1_dict.keys()) + list(lex2_dict.keys()))
    combo_dict = defaultdict(lambda: list())
    diff_labels = dict()

    for word in word_set:
        if  word not in lex1_dict:
            # word has to be in lex2_dict
            combo_dict.update({word:lex2_dict.get(word)})
        elif word not in lex2_dict:
            # word has to be in lex1_dict
            combo_dict.update({word:lex1_dict.get(word)})
        else:
            # word is in both dicts, used lex1_dict
            if lex1_dict.get(word) == lex2_dict.get(word):
                combo_dict.update({word:lex1_dict.get(word)})
            else:   # phoneme labels are not the same
                combo_dict.update({word:lex1_dict.get(word)})
                diff_labels.update({word: {"lex1": lex1_dict.get(word), "lex2": lex2_dict.get(word)}})

    return  combo_dict, diff_labels


def create_master_lexicon(cmu_dict:dict, ted_dict:dict, lib_dict:dict, out_path:str='')->dict:
    """
    Creates a master lexicon using pronuciations from first cmudict, then tedlium
    dictionary and finally librispeech. 
    Arguments:
        cmu_dict - dict[str:list(str)]: cmu dict processed with lexicon_to_dict
        ted_dict - dict[str:list(str)]: tedlium dict processed with lexicon_to_dict
        lib_dict - dict[str:list(str)]: librispeech dict processed with lexicon_to_dict
        out_path - str (optional): output path where the master lexicon will be written to
    Returns:
        master_dict - dict[str:list(str)]
    """

    word_set = set(list(cmu_dict.keys()) + list(ted_dict.keys())+list(lib_dict.keys()))
    master_dict = defaultdict(lambda: UNK_WORD_TOKEN)

    # uses the cmu_dict pronunciation first, then tedlium_dict, and last librispeech_dict
    for word in word_set:
        if  word in cmu_dict:
            master_dict.update({word:cmu_dict.get(word)})
        elif word in ted_dict:
            master_dict.update({word:ted_dict.get(word)})
        elif word in lib_dict:
            master_dict.update({word:lib_dict.get(word)})

    if out_path:
        with open(out_path, 'w') as fid:
            for key in sorted(master_dict):
                fid.write(f"{key} {' '.join(master_dict[key])}\n")
 
    return master_dict


def skip_file(dataset_name:str, audio_path:str)->bool:
    """
    if the audio path is in one of the noted files with errors, return True
    """

    sets_with_errors = ["tatoeba", "voxforge", "speaktrain"]
    # CK is directory name and min, max are the ranges of filenames
    tatoeba_errors = {"CK": {"min":6122903, "max": 6123834}}
    voxforge_errors = {"DermotColeman-20111125-uom": "b0396"}

    skip = False
    if dataset_name not in sets_with_errors:
        # jumping out of function to reduce operations
        return skip
    file_name, ext = os.path.splitext(os.path.basename(audio_path))
    dir_name = os.path.basename(os.path.dirname(audio_path))
    if dataset_name == "tatoeba":
        for tat_dir_name in tatoeba_errors.keys():
            if dir_name == tat_dir_name:
                if tatoeba_errors[tat_dir_name]["min"] <= int(file_name) <=tatoeba_errors[tat_dir_name]["max"]:
                    skip = True
   
    elif dataset_name == "voxforge":
        #example path: ~/data/voxforge/archive/DermotColeman-20111125-uom/wav/b0396.wv
        speaker_dir = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
        if speaker_dir in voxforge_errors.keys():
            file_name, ext = os.path.splitext(os.path.basename(audio_path))
            if file_name in voxforge_errors.values():
                skip = True
    
    elif dataset_name == "speaktrain":
        # the speak files in the test sets cannot go into the training set
        # so they will be skipped based on their firestore record id
        speak_test_ids = set(get_speak_test_ids())
        if file_name in speak_test_ids:
            skip = True

    return skip


def get_files(root_dir:str, pattern:str):
    """
    returns a list of the files in the root_dir that match the pattern
    """
    return glob.glob(os.path.join(root_dir, pattern))


def get_speak_test_ids():
    """
    returns the document ids of the recordings in the old (2019-11-29) and new (2020-05-27) speak test set.
    Two text files containing the ids must existing in the <main>/speech/utils/ directory.
    """
    abs_dir = os.path.dirname(os.path.abspath(__file__))

    file_path_2019 = os.path.join(abs_dir, 'speak-test-ids_2019-11-29.txt')
    file_path_2020 = os.path.join(abs_dir, 'speak-test-ids_2020-05-27.txt')

    assert os.path.exists(file_path_2020), \
        "speak-test-ids_2020-05-27.txt doesn't exist in <main>/speech/utils/"
    assert os.path.exists(file_path_2019), \
        "speak-test-ids_2019-11-29.txt doesn't exist in <main>/speech/utils/"

    with open(file_path_2019, 'r') as id_file:
        ids_2019 = id_file.readlines() 
        ids_2019 = [i.strip() for i in ids_2019]

    with open(file_path_2020, 'r') as id_file: 
        ids_2020 = id_file.readlines() 
        ids_2020 = [i.strip() for i in ids_2020] 

    return ids_2019 + ids_2020


def text_to_phonemes(transcript:str, lexicon:dict, unk_token=list())->list:
    """
    The function takes in a string of text, cleans the text, and outputs a list of phoneme 
    labels from the `lexicon_dict`. 
    Args:
        transcript (str): string of words
        lexicon (dict): lexicon mapping of words (keys) to a list of phonemes (values)
    Returns:
        (list): a list of phoneme strings
    """
    if isinstance(unk_token, str):
        unk_token = [unk_token]
    elif isinstance(unk_token, list):
        pass
    else:
        raise TypeError(f"unk_token has type {type(unk_token)}, not str or list")

    phonemes = list()
    transcript = process_text(transcript)
    transcript = transcript.split(' ')
    for word in transcript:
        phonemes.extend(lexicon.get(word, unk_token))
    
    return phonemes


def process_text(transcript:str, remove_apost:bool = False)->str:
    """This function removes punctuation (except apostrophe's) and extra space
    from the input `transcript` string and lowers the case. 

    Args:
        transcript (str): input string to be processed
        remove_apost (bool): if True, the apostrophe will be removed from `transcript`
    Returns:
        (str): processed string
    """
    # allows for alphanumeric characters, space, and apostrophe
    accepted_char = '[^A-Za-z0-9 \']+'
    # replacing a weird encoding with apostrophe's
    transcript = transcript.replace(chr(8217), "'")
    # filters out unaccepted characters, lowers the case
    try:
        transcript = transcript.strip().lower()
        transcript = re.sub(accepted_char, '', transcript)
    except TypeError:
        print(f"Type Error with: {transcript}")
    # check that all punctuation (minus apostrophe) has been removed 
    punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    for punc in punct_noapost:
        if punc in transcript:
            raise ValueError(f"unwanted punctuation {punc} in transcript")
    # remove apostrophe, if selected
    if remove_apost:
        transcript = transcript.replace("'", "") 

    return transcript


def check_update_contraints(record_id:int, 
                            record_ids_map:dict,
                            id_counter:dict, 
                            constraints:dict)->bool:
    """This function is used by downloading and filtering code primarily on speak data
    to constrain the number of recordings per speaker, line, and/or lesson.
    It checks if the counts for the `record_id` is less than the constraints in `constraints. 
    If the count is less, the constraint passes and the `id_counter` is incremented.
  
    Args:
        record_id (int): id of the record
        record_id_map (dict): dict that maps record_ids to speaker, lesson, and line ids
        id_counter (dict): dict of counts of speaker, lesson, and line ids
        constraints (dict): dict of 3 ints specifying the max number of utterances
            per speaker, line, and lesson
    Returns:
        bool: true if the count of utterances per speaker, lesson, and line are all
            below the max value in `constraints`
    """
    pass_constraint = True
    # constraint_names = ['lesson', 'line', 'speaker']
    constraint_names = list(constraints.keys())

    for name in constraint_names:
        constraint_id = record_ids_map[record_id][name]
        count = id_counter[name].get(constraint_id, 0)
        if count >= constraints[name]:
            pass_constraint = False
            break
    
    # if `record_id` passes the constraint, update the `id_counter`
    if pass_constraint:
        for name in constraint_names:
            constraint_id = record_ids_map[record_id][name]
            id_counter[name][constraint_id] = id_counter[name].get(constraint_id, 0) + 1

    return pass_constraint


def check_disjoint_filter(record_id:str, disjoint_id_sets:dict, record_ids_map:dict)->bool:
    """This function checks if the record_id contains any common ids with the disjoint datasets.
    If a common ids is found, the check fails.

    This function is used in filter.py.

    Args:
        record_ids (str): record id for a recording.
        disjoint_id_sets (Dict[str, Set[str]]): dictionary that maps the ids along which the output dataset
            will be disjoint to the set of ids included in the `disjoint_datasets`. 
        record_ids_map (Dict[str, Dict[str, str]]): dictionary to maps record_id to other ids like
            speaker, lesson, line (or target-sentence).
        
    Returns:
        (bool): True if the ids associated with the record_id are not contained in any of 
            the `disjoint_ids_sets`. Otherwise, False.
    """
    # assumes the check passes (not the safest initial assumption but it makes the logic cleaner)
    pass_check = True
    # names of the ids along which the output dataset will be disjoint
    for id_name, dj_id_set in disjoint_id_sets.items():
        disjoint_id = record_ids_map[record_id][id_name]
        # if the id is contained in the id_set of the disjoint_datasets, the check fails
        if disjoint_id in dj_id_set:
            pass_check = False
            break
    
    return pass_check


def check_distribution_filter(example: dict, filter_params:dict)->bool:
    """This function filters the number of examples in the output dataset based on
    the values in the `dist_filter_params` dict. 

    Args:
        example (dict): dictionary of a single training example
        dist_filter_params (dict): a dictionary with the keys and values below
            key (str): key of the value filtered on the input `example`
            function (str): name of the function to apply to the values in `example[key]`
            threshhod (float): threshhold to filter upon
            above-threshold-percent (float): percent of examples to pass through with values above
                the threshhold
            below-threshold-percent (float): percent of examples to pass through with values below
                the threshold
    Returns:
        (boo): whether the input example should pass through the filter
    """
    assert 0 <= filter_params['percent-above-threshold'] <= 1.0, "probs not between 0 and 1"
    assert 0 <= filter_params['percent-below-threshold'] <= 1.0, "probs not between 0 and 1"

    # fitler fails unless set to true
    pass_filter = False
    # value to pass into the `filter_fn`
    filter_input = example[filter_params['key']]
    # function to evaluate on the `filter_input`
    filter_fn = filter_params['function']
    filter_value = eval(f"{filter_fn}({filter_input})")

    if filter_value >= filter_params['threshold']:
        # make random trial using the `percent-above-threshhold`
        if np.random.binomial(1, filter_params['percent-above-threshold']):
            pass_filter = True
    else:
        # make random trial using the `percent-above-threshhold`
        if np.random.binomial(1, filter_params['percent-below-threshold']):
            pass_filter = True
    
    return pass_filter


def get_disjoint_sets(disjoint_dict:dict, record_ids_map:dict)->dict:
    """Creates a dictionary of sets of ids that will be used to ensure a created datasets
        does not contain the ids included in the Sets in the output dict. 

    Args:
        disjoint_dict (Dict[str, Tuple[str]]): dict of dataset_path as keys and a tuple of id_names
            as values.
        record_ids_map (Dict[str, Dict[str, int]]): mapping from record_id to lesson, speaker
            target_sentence, and record_id
        
    Returns:
        Dict[str, Set[str]]: a dict with `id_names` as keys and a set of ids as values
    """

    disjoint_id_sets = defaultdict(set)
    for dj_data_path, dj_names in disjoint_dict.items():
        # get all the record_ids in the dataset
        record_ids = get_dataset_ids(dj_data_path)
        # loop through the disjoint-id-names in the key-tuple
        for dj_name in dj_names:
            for record_id in record_ids:
                # add the id to the relevant id-set
                disjoint_id_sets[dj_name].add(record_ids_map[record_id][dj_name])
    
    return disjoint_id_sets


def get_dataset_ids(dataset_path:str)->Set[str]:
    """This function reads a dataset path and returns a set of the record ID's
    in that dataset. The record ID's mainly correspond to recordings from the speak dataset. 
    For other datsets, this function will return the filename without the extension.

    Args:
        dataset_path (str): path to the dataset
    
    Returns:
        Set[str]: a set of the record ID's
    """
    # dataset is a list of dictionaries with the audio path as the value of the 'audio' key.
    dataset = read_data_json(dataset_path)

    return set([path_to_id(xmpl['audio']) for xmpl in dataset])


def path_to_id(record_path:str)->str:
        #returns the basename of the path without the extension
        return os.path.basename(
            os.path.splitext(record_path)[0]
        )


def get_record_ids_map(metadata_path:str, id_names:list=None, has_url:bool=False)->dict:
    """This function returns a mapping from record_id to other ids like speaker, lesson,
    line, and target sentence. This function runs on recordings from the speak firestore database.

    Args:
        metadata_path (str): path to the tsv file that contains the various ids
        id_names (List[str]): names of the ids in the output dict 
            This is currented hard-coded to the list: ['lesson', 'target-sentence', 'speaker']
        has_url (bool): if true, the metadata file contains an extra column with an audio url

    Returns:
        Dict[str, Dict[str, str]]: a mapping from record_id to a dict
            where the value-dict's keys are the id_name and the values are the ids
    """
    assert os.path.splitext(metadata_path)[1] == '.tsv', \
        f"metadata file: {metadata_path} is not a tsv file"

    # check that input matches the expected values
    # TODO: hard-coding the id-names isn't flexible but is the best option for now
    expected_id_names = ['lesson', 'target_sentence', 'speaker']
    if id_names is None:
        id_names = expected_id_names
    assert id_names == expected_id_names, \
        f"input id_names: {id_names} do not match expected values: {expected_id_names}"

    # create a mapping from record_id to lesson, line, and speaker ids
    expected_row_len = 7 + int(has_url)     # add an extra column if audio_url exists
    with open(metadata_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        # this assert helps to ensure the row indexing below is correct
        assert len(header) == expected_row_len, \
            f"Expected metadata header length: {expected_row_len}, got: {len(header)}."
        # header: id, text, lessonId, lineId, uid(speaker_id), redWords_score, date
        print("header: ", header)

        # mapping from record_id to other ids like lesson, speaker, and line
        record_ids_map = dict()
        for row in tsv_reader:
            assert len(row) == expected_row_len, \
                f"row: {row} is len: {len(row)}. Expected len: {expected_row_len}"
            record_ids_map[row[0]] = {
                    "record": row[0],                   # adding record for disjoint_check
                    id_names[0]: row[2],                # lesson
                    id_names[1]: process_text(row[1]),  # using target_sentence instead of lineId
                    id_names[2]: row[4]                 # speaker
            }

    return record_ids_map

def total_duration(data: List[dict])->float:
    """Returns the total time (in hours) of the input data list. 
    The data list has the typical training format, which is a list of dictionaries
    the a "duration" key.

    Args:
        data (List[Dict[str, Any]]): list of dicts with key "duration"
    
    Returns:
        (float): total duration of the dataset in hours
    """

    total_duration_s = sum([xmpl['duration'] for xmpl in data])
    return round(total_duration_s/3600, 3)


def getsize(obj):
    """sum size of object & members.
    copied from: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    """
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

