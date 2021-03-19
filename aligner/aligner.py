# this file is a hodge-podge of scripts used to process lexicons and datasets using the 
# montreal forced aligner (MFA). the MFA is used provide potentially better phoneme labels
# Copyright: Speak Labs 2021
# Author: Dustin Zubke

import argparse
from collections import Counter, defaultdict
import copy
import csv
import glob
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import string
from typing import Dict, List, Tuple
import unicodedata
# third party libraries
import editdistance as ed
import numpy as np
import tqdm
# project libraries
from speech.utils import textgrid
from speech.utils.data_helpers import clean_phonemes, get_record_ids_map, path_to_id, process_text
from speech.utils.io import read_data_json, write_data_json
from speech.utils.wave import wav_duration

##########     LEXICON AUGMENTATION  FUNCTIONS   #############

def insert_mispronunciations(lex_path:str, spk_word_path:str, save_path:str):
    """This function adds additional pronunciations created by replacing certain phonemes for words
    in the speak dataset. The existing and additional pronunciations are saved to `save_path`.
    """
    # phoneme keys in the dict will be replaced with the phonmes in the values list
    phoneme_swaps = {
        'AE1':['EH1'],
        'B': ['P'],
        'D': ['D IY1', 'D AH1', 'T'],
        'DH' : ['D'], 
        'F': ['P'],
        'G': ['K']  ,
        'IH1': ['IY1'],
        'IY1': ['IH1'],
        'L' : ['R'],
        'P': ['F'],
        'R' : ['L'],
        'R AO1': ['AO1', 'R'],
        'S': ['SH'],
        'SH': ['S'],
        'TH': ['S'],
        'UH1': ['UW1'],
        'V': ['B'],
        'W UH1': ['UH1'],
        'W UW1': ['UW1'],
        'Z': ['JH', 'CH'],
    }

    lex_dict = load_lex_dict(lex_path, split_phones=True)
    spk_upper_words = load_spk_upper_words(spk_word_path)
    for word in spk_upper_words:
        # make all additions to a word in a separate list to prevent an infinite-loop
        new_pronunciations = list()
        # for each pronnciation of `word`
        for lex_phones in lex_dict[word]:
            # for each swap-phoneme as `src_phone`
            for src_phone, dst_phone_list in phoneme_swaps.items():
                if src_phone in lex_phones:
                    # for each possible `dst_phone`, swap `dst_phone` for `src_phone`
                    for dst_phone in dst_phone_list:
                        new_phones = copy.copy(lex_phones)
                        new_phones = [dst_phone if x==src_phone else x for x in lex_phones]
                        new_pronunciations.append(new_phones) 
                          
        lex_dict[word].extend(new_pronunciations) 

    save_lex_dict(lex_dict, save_path, split_phones=True)


def manual_entry(lex_path:str, save_path:str):
    """This function will manually add a few mispronunciations to the input lexicon
    """

    manual_entries = {
        "TRAVELING": "T R AE1 V ER0  L IH0 NG",        
    }


def remove_suffix(lex_path:str, spk_words_path:str, save_path:str):
    """A pronunciation mistake is to drop the 'ed' off the end of a word. 
    This function will add the pronunciation of an 'ed'-less word as a word's
    pronunciation. For example the pronunciation of 'dream' will be used for the
    word 'dreamed'.
    This addition will only be made for words in the `spk_words_path` file. 
    """

    lex_dict = load_lex_dict(lex_path)

    # character lengths of the phonemes for each suffix
    # allows for indexing as [-len:] to remove suffix phonemes
    suffix_lengths = {
        "ED": 2,
        "ING": 7
    }
    
    spk_upper_words = load_spk_upper_words(spk_words_path)

    for word in spk_upper_words:
        if word == 'GOING': # `G OW1 IH0 N` pronunciation is edge-case
            continue

        for suffix, suffix_len in suffix_lengths.items():
            if word.endswith(suffix):
                new_pronunciations = list()
                for phones in lex_dict[word]:
                    new_phones = copy.copy(phones)
                    new_pronunciations.append(new_phones[:-suffix_len])
                
                lex_dict[word].extend(new_pronunciations)
                        

    save_lex_dict(lex_dict, save_path)


def expand_contractions(lex_path:str, contractions_path:str, save_path:str):
    """This function adds entries to the output lexicon that expands contractions.
    For example the pronunciation of "i'll" will now have a new entry for the phonemes of
    the phrase "i will". The pronunciations of the expanded contraction will be added to the output
    lexicon.
    
    Args:
        lex_path (str): path to lexicon
        contractions_path (str): path to file with contraction-to-expansion mapping
        save_path (str): path to output lexicon
    """
    # create a lexicon dict where pronunciations are strings in a list
    # words with multiple pronunciations have len(list) > 1
    lex_dict = load_lex_dict(lex_path)
    
    # for each contraction-expansion pair, create a new entry in the lex_dict
    with open(contractions_path, 'r') as fid:
         for row in fid:
            row = row.strip().upper().split(' ')
            contraction, expansion = row[0], row[1:]
            
            # the combinations below only work for 2-word expansions
            assert len(expansion) == 2, f"expansion: {expansion} is not size 2"
            new_pronun = list()
            for phones_1 in lex_dict[expansion[0]]:
                for phones_2 in lex_dict[expansion[1]]:
                    lex_dict[contraction].append(phones_1 + " " + phones_2)

    save_lex_dict(lex_dict, save_path)      


############   LEXICON & CORPUS ASSESSMENT  FUNCTIONS   #############

def compute_lexicon_outliers(lexicon_path:str):
    """This function computes the outliers in the lexicon by length of pronunciation. It is meant
    to catch very short or long pronunciations.
    
    It does this by printing pronunciations that are more than 2 standard deviations away from the 
    mean ratio of the word length to pronunciation length, where word length is the number of 
    characters and pronunciation length is the number of phonemes. 
    
    Args:
        lexicon_path (str): path to lexicon
    """

    # returns lexicon with pronunciations as list of phoneme strings
    lex_dict = load_lex_dict(lexicon_path, split_phones=True)

    pronun_ratios  = [len(word)/len(phones) for word, phones in lex_dict.items()]
    mean_ratio = np.mean(pronun_ratios)
    stddev_ratio = np.std(pronun_ratios)

    outlier_factor = 4.0    # num std-deviations that define an outlier
    lower_bound = mean_ratio - outlier_factor * stddev_ratio
    upper_bound = mean_ratio + outlier_factor * stddev_ratio

    print(f"mean: {mean_ratio}, std-dev: {stddev_ratio}, # of stddev for outlier: {outlier_factor}")

    outliers = defaultdict(list)
    for word, phones in lex_dict.items():
        ratio = len(word)/ len(phones)
        if ratio < lower_bound or ratio > upper_bound:
            outliers[word].extend(phones)

    print(f"number of outliers outside bounds: {len(outliers)}")
    for word, phone_list in outliers.items():
        for phones in phone_list:
            print(f"{word} {phones}")


def phoneme_occurance(lex_path:str, save_path:str)->None:
    """This function computes an occurance count of the phonemes in the lexicon
    """

    lex_dict = load_lex_dict(lex_path)

    phone_counter = Counter()
    for word, phone_str_list in lex_dict.items():
        for phone_str in phone_str_list:
            phones = phone_str.strip().split(' ')
            phone_counter.update(phones)

    with open(save_path, 'w') as fid:
        for phone, count in phone_counter.most_common():
            fid.write(f"{phone} {count}\n")


def spk_word_count(metadata_file:str, out_path:str):
    """This funciton creates a count of all the words in the speak training set.
    Args:
        metadata_file (str): path to the metadata file with the word targets
        out_path (str): file where the count will be saved
    """

    word_counter = Counter()
    with open(metadata_file, 'r') as fid:
        reader = csv.reader(fid, delimiter='\t')
        header = next(reader)
        for row in tqdm.tqdm(reader, total=3.165e7):
            target = process_text(row[1])
            word_counter.update(target.split(' '))
            
    with open(out_path, 'w') as fid:
        for word, count in word_counter.most_common():
            fid.write(f"{word} {count}\n")


#############   IO  FUNCTIONS   #############

def load_spk_upper_words(spk_words_path:str)->List[str]:
    """This funciton returns a list of the uppecase words in the speak training set
    """
    spk_words = list()    
    with open(spk_words_path, 'r') as fid:
        for row in fid:
            word, _ = row.strip().split()
            spk_words.append(word.upper())

    return spk_words
                

def load_lex_dict(lex_path:str, split_phones=False)->Dict[str, List[str]]:
    """This function reads the lexicon path and returns a dictionary with 
    the uppercase words as keys and values as a list of one or more pronunciations as strings.
    
    Args:
        lex_path (str): path to lexicon
        split_phones (bool): if true, the phonemes will be split into a list of strings, default False    
    """
    lex_dict = defaultdict(list)
   
    # ensure all entries are space-separated, rather than tab-separated 
    with open(lex_path, 'r') as fid:
        for row in fid:
            row = row.replace('\t', ' ')
            row = row.strip().split(' ', maxsplit=1)
            # check of unexpected rows
            if len(row) != 2:
                print(f"short row: {row}")
                continue
            word, phones = row
            # remove (2) digit marker for alternative pronunciations
            word = re.sub("\(\d\)", "", word).upper()
            phones = phones.strip()
            if split_phones:
                phones = phones.split(' ')
            lex_dict[word].append(phones)

    return lex_dict    


def save_lex_dict(lex_dict:dict, save_path:str, split_phones=False):
    """Save the lex dict to the save_path

    Args:
        lex_dict (dict): dictionary of pronunications either as strings or list of strings
        save_path (str): output path for lexicon
        split_phnoes (bool): if True, values in lex_dict are a list of list of phoneme-strings,
            if False, values are list of single-strings of all phonemes
    """
    with open(save_path, 'w') as fid:
        for key in lex_dict:
            for phones in lex_dict[key]:
                if split_phones:
                    phones = " ".join(phones)
                fid.write(f"{key} {phones}\n")


def load_aligner_phones_lower(aligner_phone_path:str)->Dict[str,List[str]]:
    """Returns a dict mapping example_id to lowercase phonemes based on the input path to the
    aligner's phonemes.

    Args:
        aligner_phone_path (str): path to file output by `extract_phonemes` function
    
    Returns:
        (dict)
    """   
    phone_path = Path(aligner_phone_path)

    aligner_phones = dict()
    for row in phone_path.read_text().split('\n'):
        row = row.strip().split()
        if len(row) < 2:
            print(f"row: {row} as no phonemes")
            continue 
        
        file_id, phones = row[0], row[1:]
        # remove the digit and lower case
        phones = [phone.rstrip(string.digits).lower() for phone in phones]
        aligner_phones[file_id] = phones
    
    return aligner_phones




def combine_cmud_libsp_lexicons(cmu_path:str, libsp_path:str, save_path:str)->None:
    """This function combines the cmudict and librispeech lexicons and writes the combined
        file to the file in `save_path`. For cmudict, it removes the digits "(1)" demarking
        multiple pronunciations.
    
    Args:
        cmu_path (str): path to cmudict
        libsp_path (str): path to librispeech
        save_path (str): path to output file
    """

    # reads inputs and remove digit marker from cmu
    word_phone_list = list()
    with open(cmu_path, 'r') as cmuid:
        for row in cmuid:
            word_phones = row.strip().split(' ', maxsplit=1)
            if len(word_phones) != 2:
                # some entries are tab-delimited
                word_phones = row.strip().split('\t', maxsplit=1)
                if len(word_phones) != 2:
                    print(f"cmu: unexpected row size in row:  {word_phones}")
                    continue
            word, phones = word_phones
            # remove the "(1)" marker in alternate pronunciations
            word = re.sub("\(\d\)", '', word)
            word = word.upper()
            word_phone_list.append((word, phones))

    with open(libsp_path, 'r') as libid:
        for row in libid:
            word_phones = row.strip().split(' ', maxsplit=1)
            if len(word_phones) != 2:
                # some entries are tab-delimited
                word_phones = row.strip().split('\t', maxsplit=1)
                if len(word_phones) != 2:
                    print(f"libsp: unexpected row size in row:  {word_phones}")
                    continue
            word, phones = word_phones   
            word = word.upper()
            word_phone_list.append((word, phones))

    # pass list through set to  de-duplicate
    word_phone_list = sorted(set(word_phone_list))
    
    # write to file
    with open(save_path, 'w') as fid:
        for word, phones in word_phone_list:
            fid.write(f"{word} {phones}\n")




def update_train_json(old_json_path:str, aligner_phones_path:str, new_json_path:str):
    """Saves a new training json that replaces the phones in the existing training json at 
    `old_json_path` with the aligner phonemes in `aligner_phones_path`. The new training json
    is saved to `new_json_path`. 

    Args:
        old_json_path (str): path to existing training json
        aligner_phones_path (str): path to aligner phonemes
        new_json_path (str): path where new training json will be saved
    
    """
    # aligner_phones is a  mapping from example_id to aligner_phonemes
    aligner_phones = load_aligner_phones_lower(aligner_phones_path)

    # train_json is list of dicts with keys in ['audio', 'duration', 'text']
    train_json = read_data_json(old_json_path)
    
    # update train_json in-place with new phonemes
    examples_not_updated = 0
    for idx, xmpl in enumerate(train_json):
        audio_id = path_to_id(xmpl['audio'])
        if audio_id in aligner_phones:
            xmpl['text'] = aligner_phones[audio_id]
        # remove examples not in the aligner_phones. edit this design choice, if desired. 
        else:
            #train_json.pop(idx)
            examples_not_updated += 1

    print(f"num examples not updated: {examples_not_updated}")
    write_data_json(train_json, new_json_path) 


def compare_phonemes(aligner_phone_path:str, training_json_path:str, save_path:str)->None:
    """This function calculates the levenshtein distance between the new aligner phonemes
    and the existing phoneme labels and writes the sorted file-id and distances to `save_path`

    Args:
        aligner_phone_path (str): path to file with space separate file_id and aligner phonemes pairs
        training_json_path (str): patht to standard training json file
        save_path (str): path where sorted distances will be written
    """

    
    # load the lower-case aligner phones
    aligner_phones = load_aligner_phones_lower(aligner_phone_path)
    
    data = read_data_json(training_json_path)
    
    # records the distance between old and new phonemes
    distance_dict = dict()
    for xmpl in data:
        old_phones = xmpl['text']
        file_id = path_to_id(xmpl['audio'])
        
        try:
            new_phones = aligner_phones[file_id]
        except KeyError:
            print(f"file_id {file_id} not found in aligner phones. skipping")
            continue

        distance = ed.eval(old_phones, new_phones)
        distance_dict[xmpl['audio']] = (distance, old_phones, new_phones)

    # sort examples by distance 
    sorted_dist = sorted(distance_dict.items(), key=lambda x: x[1][0], reverse=True)
    
    # summary stats
    total_dist = 0
    total_old_phones = 0

    # write formatted output to `save_path`
    with open(save_path, 'w') as fid:
        for file_path, (dist, old_phones, new_phones) in sorted_dist:
            fid.write(f"file:       {file_path}\n")
            fid.write(f"dist:       {dist}\n") 
            fid.write(f"old_phones: {old_phones}\n")
            fid.write(f"new_phones: {new_phones}\n")
            fid.write("\n")

            total_dist += dist
            total_old_phones += len(old_phones)

    # print summary stats
    print(f"total distance: {total_dist} out of {total_old_phones} total phonemes")
    print(f"average distance: {round(total_dist/total_old_phones, 3)}")


def extract_aligner_phonemes(data_dir:str, save_path:str):
    """This function gathers all textgrid files whose parent dir is `data_dir`, extracts
        the phonemes from each textgrid file, and writes only the phonemes to `save_path`
        as a space-separate format prefixed by the filename.
    """
   
    file_phones = dict()
    remove_markers = ["sil", "sp", "spn"]
    data_dir = Path(data_dir)
    for tg_file in data_dir.rglob("*.TextGrid"):
        # create a TextGrid object
        tg_object = textgrid.TextGrid.load(str(tg_file))
        for tier in tg_object:
            if "phones" in tier.nameid:
                start_end_phone = tier.make_simple_transcript()
                phones = [elem[2] for elem in start_end_phone]
                # remove the 'sil', 'sp', and 'spn', markers
                phones = [phone for phone in phones if phone not in remove_markers] 
                file_phones[tg_file.stem] = " ".join(phones)

    with open(save_path, 'w') as fid:
        for filename, phones in file_phones.items():
            filename = filename.replace(".TextGrid", "")
            fid.write(f"{filename} {phones}\n")    
    

def extract_aligner_phonemes_mp(data_dir:str, save_path:str, num_processes:int=6):
    """This function gathers all textgrid files whose parent dir is `data_dir`, extracts
        the phonemes from each textgrid file, and writes only the phonemes to `save_path`
        as a space-separate format prefixed by the filename.
    """
 
    file_phones = dict()
    data_dir = Path(data_dir)
    files = [str(file_path) for file_path in data_dir.rglob("*.TextGrid")]
    with mp.Pool(processes=num_processes) as pool:
        phones_list = pool.map(extract_textgrid, files)
        pool.close()
        pool.join()
    
    with open(save_path, 'w') as fid:
        for filename, phones in phones_list:
            filename = filename.replace(".TextGrid", "")
            fid.write(f"{filename} {phones}\n")    
    
def extract_textgrid(tg_file:str)->str:
    remove_markers = ["sil", "sp", "spn"]
    # create a TextGrid object
    tg_object = textgrid.TextGrid.load(str(tg_file))
    tier = [tier for tier in tg_object if "phones" in tier.nameid][0]
    start_end_phone = tier.make_simple_transcript()
    phones = [elem[2] for elem in start_end_phone]
    # remove the 'sil', 'sp', and 'spn', markers
    phones = [phone for phone in phones if phone not in remove_markers] 
    filename = os.path.splitext(os.path.split(tg_file)[1])[0]
    return (filename, " ".join(phones))

################# TRANSCRIPT CREATION FUNCTIONS  ########################

##### GENERAL FUNCTIONS  #####


def write_textgrid_from_txt(data_dir:str):
    """This function creates textgrid files from existing .txt files in the data_dir.
    The newly created textgrid files will have the same filename as the original .txt files
    but will use the .TextGrid extension. The existing .txt files will be renamed .txt-ignore so 
    the MFA tool does not consider them in the alignment. 
    """

    data_dir = Path(data_dir)
    # remove the .trans.txt files
    txt_paths = [fn for fn in data_dir.rglob("*.txt") if "trans" not in str(fn)]   
 
    # a backup clause in case this function as already been run 
    # so that the .txt files are now .txt-ignore files
    if len(txt_paths) == 0: 
        print("no txt files found, using .txt-ignore extension")
        txt_paths = data_dir.rglob("*.txt-ignore")
    
    # spacing (in seconds) between beginning and end of recording
    tg_end_interval = 0.0  #0.5
    for txt_path in txt_paths:
        # taking the chapter name in librispeech, is this how the aligner seems to sort speaker
        speaker_name = txt_path.parents[0].name
        if "trans" in str(txt_path):
            continue
        audio_dur = wav_duration(txt_path.with_suffix(".wav"))
        tg_begin = 0.0  # 0.2
        tg_end = audio_dur - tg_end_interval
        transcript = txt_path.read_text()
        tg_str = gen_oo_textgrid_str(tg_begin, tg_end, audio_dur, transcript, speaker_name)    
        txt_path.with_suffix(".TextGrid").write_text(tg_str)
        txt_path.rename(txt_path.with_suffix(".txt-ignore"))


def gen_oo_textgrid_str(begin_t, end_t, duration, transcript, speaker_name)->str:
    """Creates a TextGrid file in oo-format with only one entry as the transcript
    with begins and ends at begin_t and end_t, respectively.

    Args:
        begin_t (float): beginning time
        end_t (float): end time
        transcript (str): uppercase transcript
    Returns:
        (str): string format of textgrid
    """
    oo_file = ""
    oo_file += "File type = \"ooTextFile\"\n"
    oo_file += "Object class = \"TextGrid\"\n\n"
    oo_file += "xmin = 0.0\n"
    oo_file += f"xmax = {duration}\n"
    oo_file += "tiers? <exists>\n"
    oo_file += "size = 1\n"
    oo_file += "item []:\n"
    
    oo_file += "%4s%s [%s]:\n" % ("", "item", 1)
    
    oo_file += "%8s%s = \"%s\"\n" % ("", "class", "IntervalTier")
    oo_file += "%8s%s = \"%s\"\n" % ("", "name", speaker_name)
    oo_file += "%8s%s = %s\n" % ("", "xmin", begin_t)
    oo_file += "%8s%s = %s\n" % ("", "xmax", end_t)
    oo_file += "%8s%s: %s\n" % ("", "interals", "size = 1")
    
    oo_file += "%12s%s\n" % ("", "intervals [1]:")
    
    oo_file += "%16s%s = %s\n" % ("", "xmin", begin_t)
    oo_file += "%16s%s = %s\n" % ("", "xmax", end_t)
    oo_file += "%16s%s = \"%s\"\n" % ("", "text", transcript)

    return oo_file

##### LIBRISPEECH  #####

def create_libsp_transcripts():
    libsp_glob = "/mnt/disks/data_disk/data/LibriSpeech/**/**/**/*trans.txt" 
    
    for trans_file in glob.glob(libsp_glob):
        dir_name = os.path.dirname(trans_file)
        with open(trans_file, 'r') as rfid:
            for row in rfid:
                filename, transcript = row.strip().split(' ', maxsplit=1)
                out_file = os.path.join(dir_name, filename)
                os.remove(out_file)
                out_file = out_file + ".txt"
                with open(out_file, 'w') as wfid:
                    wfid.write(transcript)


######   TEDLIUM   ######

def create_tedlium_transcripts(tedlium_dir:str, train_json_path:str):
    """This function creates separate txt transcript files for each utterance for the mfa aligner.
    The function uses the training_json file to check the duration of the utterance to ensure that
    the transcript is matched with the correct audio file. This double-check is necessary because the
    audio filenames do not reference the start and end times of each clip and this are not easy to match
    with the transcript stm files. 

    Args:
        tedlium_dir (str): path to directory that contains `stm` and `converted/wav` subdirectories
        train_json_path (str): path to training json file
    """
    # preprocessing constants
    MIN_DURATION = 1.0
    MAX_DURATION = 20.0
    
    tedlium_dir = Path(tedlium_dir)
    
    # create a dict mapping from audio_path to data example for later lookup
    tedlium_data = {
        xmpl['audio']: xmpl for xmpl in read_data_json(train_json_path)
    }

    excluded_utt_count = 0      # counts number of utterances not in training-json
    for stm_file in tedlium_dir.joinpath("stm").glob("*.stm"):
        utterance_dicts = get_utterances_from_stm(stm_file)
        # filter the utterances by min and max duration
        filtered_utterances = list()
        for utt in utterance_dicts:
            utt_duration = utt["end_time"] - utt["start_time"]
            if MIN_DURATION < utt_duration < MAX_DURATION:
                filtered_utterances.append(utt)
        
        for utt_idx, utt in enumerate(filtered_utterances):
            wav_file = f"{utt['filename']}_{str(utt_idx)}.wv"
            wav_file = tedlium_dir.joinpath(f"converted/wav/{wav_file}")
        for utt in utterance_dicts:
            utt_duration = utt["end_time"] - utt["start_time"]
            if MIN_DURATION < utt_duration < MAX_DURATION:
                filtered_utterances.append(utt)
        
        for utt_idx, utt in enumerate(filtered_utterances):
            wav_file = f"{utt['filename']}_{str(utt_idx)}.wv"
            wav_file = tedlium_dir.joinpath(f"converted/wav/{wav_file}")
            # unsure why this file didn't get converted, but it doesn't exist
            if 'ThomasGoetz_2010P_118.wv' in str(wav_file):
                continue
            assert wav_file.exists(), f"audio file: {wav_file} doesn't exist"
            # check that the training-json duration and utterance-duration match    
            if str(wav_file) not in tedlium_data:
                excluded_utt_count += 1
                continue
            train_duration = tedlium_data[str(wav_file)]['duration']
            utt_duration = utt["end_time"] - utt["start_time"]
            assert math.isclose(train_duration, utt_duration, rel_tol=1e-6), \
                f"durations: {train_duration}, {utt_duration} not close for {wav_file}"
            # write transcript in uppercase to new file
            transcript = utt['transcript'].upper()
            trans_path = wav_file.with_suffix(".txt")
            trans_path.open('w').write(transcript)
    
    print(f"number of excluded utterances: {excluded_utt_count}")


def get_utterances_from_stm(stm_file:str):
    """parses and stm file to extract the transcript. The unk_token is removed from transcript

    Note: below is a sample stm file:
        911Mothers_2010W 1 911Mothers_2010W 14.95 16.19 <NA> <unk> because of
        911Mothers_2010W 1 911Mothers_2010W 16.12 25.02 <NA> the fact that we have
    """
    unk_token = "<unk>"
    utterances = []

    with open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            stm_line = stm_line.replace("\t", " ").split()      # remove tabs
            start_time, end_time = float(stm_line[3]), float(stm_line[4])
            filename = stm_line[0]
            transcript = [word for word in stm_line[6:] if word != unk_token]
            transcript = unicodedata.normalize(
                "NFKD", " ".join(transcript).strip()
            ).encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                utterances.append({
                    "start_time": start_time, 
                    "end_time": end_time,
                    "filename": filename, 
                    "transcript": transcript
                })
                                                                                              
        return utterances

def tedlium_oov_words(transcript_dir:str, lex_path:str, oov_path: str):
    """This function determines the words in the transcripts in `transcript_dir` that are not
    included in the lexicon in `lex_path`.

    Args:
        transcript_dir (str): path to the directory that contains the transcripts
        lex_path (str): path to lexicon
    """

    transcript_dir = Path(transcript_dir)
    lexicon = load_lex_dict(lex_path)
    lex_words = set([word for word in lexicon])
    words_set = set()

    for trans_file in transcript_dir.glob("*.txt"):
        words = trans_file.read_text()
        words = words.replace("\t", " ")
        words = words.split()
        words_set.update(words_set)

    oov_words = words_set.difference(lex_words)

    with open(oov_path, 'w') as fid:
        for word in oov_words:
            fid.write(word+"\n")


######   SPEAK DATASET   ######

def create_spk_dir_tree(audio_dir:str, spk_metadata_path:str, spk_training_jsons:List[str])->None:
    """The mfa aligner uses features from the same speaker to perform alignments, which makes
    having directories with the same speak potentially useful. This function takes an existing
    directory structure where sequential subdirectories had at most 1000 audio samples and 
    creates new directories sorted by speaker. If working from a different directory structure, 
    change the pattern in the `rglob` function.
    
    The function also rewrites the `spk_training_jsons` with the paths to the new subdirectories.

    Note: both .wav and .txt files will be moved. The .txt files are used by the mfa aligner

    Args:
        spk_training_jsons (List[str]): List of paths to relevent speak training jsons
        audio_dir (str): path to directory that contains all speak training files. 
            These files will be moved into subdirectories under `audio_dir`
        spk_metadata_path (str): path to metadata
    """
    
    # create generator for '.wav' files in audio_dir
    audio_dir = Path(audio_dir)
    audio_files = audio_dir.rglob("*.wav")
    
    # dict maping record_id to file metadata
    metadata = get_record_ids_map(spk_metadata_path, has_url=True)

    # create a dict for each training_json whose values is a dict mapping audio paths to examples
    json_dict = dict()
    for json_path in spk_training_jsons:
        json_dict[json_path] = {
            xmpl['audio']: xmpl for xmpl in read_data_json(json_path)
        }
    
    # move the files to a new subdir and update the training jsons
    for audio_file in audio_files:
        # create a new sub-dir of the speaker-id if it doesn't already exist
        speaker_id = metadata[path_to_id(audio_file)]['speaker']
        speaker_sub_dir = audio_dir.joinpath(speaker_id)
        os.makedirs(speaker_sub_dir, exist_ok=True)

        # update the new file paths in each training json
        new_audio_file = speaker_sub_dir.joinpath(audio_file.name)
        for train_dict in json_dict.values():
            if audio_file in train_dict:
                train_dict[audio_file]['audio'] = str(new_audio_file)
        
        # move the .wav and .txt files to the new sub-dir
        shutil.move(audio_file, new_audio_file)
        if audio_file.with_suffix(".txt").exists():
            shutil.move(audio_file.with_suffix(".txt"), new_audio_file.with_suffix(".txt"))

    # save the updated training jsons to disk
    for json_path, xmpl_dict in json_dict.items():
        # TODO (drz): remove renaming of `json_path` once verifying the script works as expected
        json_path += "-new"
        write_data_json(xmpl_dict.values(), json_path)

    prune_empty_dirs(audio_dir)
    

def prune_empty_dirs(audio_dir:str)->None:
    """This function removes empty directories from the audio_dir.
    It should be called after `create_spk_dir_tree`

    Args:
        audio_dir (str): path to audio directory
    """
    # find the newly empty directories, record them, and remove them
    deleted_dirs = set()
    for dir_entry in os.scandir(audio_dir):
        if dir_entry.is_dir():
            # checks if directory is empty. scandir is supposedly faster than listdir
            if next(os.scandir(dir_entry), None) is None:
                os.rmdir(dir_entry) # will only delete dir if empty
                deleted_dirs.add(dir_entry.name)
 
    # previous dirs ranged from 1 to 1135
    expected_deleted_dirs = set(range(1, 1136))
    print(f"num expected dirs not deleted: {len(expected_deleted_dirs.difference(deleted_dirs))}")
    print(f"unexpected dirs: {sorted(list(expected_deleted_dirs.difference(deleted_dirs)))}")


def correct_train_jsons(audio_dir:str, spk_training_jsons:List[str]):
    """This functions corrects the training jsons files with the correct
    audio paths for the .wav files in audio_dir
    """
    # create generator for '.wav' files in audio_dir
    audio_dir = Path(audio_dir)
    audio_files = audio_dir.rglob("*.wav")
    # mapping from audio_id to actual audio_path
    id_path_map = {path_to_id(str(audio_path)): str(audio_path) for audio_path in audio_files}

    train_jsons = dict()
    for json_path in spk_training_jsons:                                                                                         
        train_jsons[json_path] = read_data_json(json_path)
    
    # move the files to a new subdir and update the training jsons
    for json_path, data in train_jsons.items():
        for xmpl in data:
            xmpl['audio'] = id_path_map[path_to_id(xmpl['audio'])]


    # save the updated training jsons to disk
    for json_path, data in train_jsons.items():
        # TODO (drz): remove renaming of `json_path` once verifying the script works as expected
        json_path += "-corrected"
        write_data_json(data, json_path)


def create_spk_transcripts(spk_metadata_path:str, spk_audio_dir:str, lex_path:str):
    """This function creates an uppercase transcript as an individual .txt file for each audio file
    in `audio_dir` using the `spk_metadata_path`. It does not create a transcript for files with words
    not included in the lexicon from `lex_path`. 
    """
    print(f"lex path: {lex_path}")
    record_ids_map = get_record_ids_map(spk_metadata_path, has_url=True)
    lex_dict = load_lex_dict(lex_path)
    lex_words = set([word for word in lex_dict])
    audio_files = Path(spk_audio_dir).rglob("*.wav") 
    examples = {
        "total": 0,
        "oov": 0
    }
    
    for audio_file in tqdm.tqdm(audio_files):
        file_id = audio_file.stem
        # trancript is already processed in `get_record_ids_map`
        transcript = record_ids_map[file_id]['target_sentence']
        transcript = transcript.upper().split()
        # checks if the transcript has an out-of-vocab word
        has_oov = any([(word not in lex_words) for word in transcript])
        if has_oov:
            examples['oov'] += 1
            continue
        examples['total'] += 1
        # write the transcript to a txt file
        txt_file = audio_file.with_suffix(".txt")
        with open(txt_file, 'w') as fid:
            fid.write(" ".join(transcript))
    print(f"num oov_examples: {examples['oov']} out to total: {examples['total']}")    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--action", help="determines what function to call"
    )
    parser.add_argument(
        "--data-paths", nargs="+",
        help="Overloaded arg with paths to relevant function data. A single path can be used."
    )
    parser.add_argument(
        "--audio-dir", help="Paths to directory that contains audio."
    )
    parser.add_argument(
        "--metadata-path", help="Path to speak metadata."
    )
    parser.add_argument(
        "--save-path", help="path to output file"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, help="number of jobs. used by applicable functions"
    )

    args = parser.parse_args()

    # sorted alphabetically, not by function or topic
    if args.action == "correct-train-jsons":
        correct_train_jsons(args.audio_dir, args.data_paths)
    elif args.action == "create-spk-transcripts":
        create_spk_transcripts(*args.data_paths)
    elif args.action == "create-subdirs":
        create_spk_dir_tree(args.audio_dir, args.metadata_path, args.data_paths)
    elif args.action == "create-tedlium-transcripts":
        create_tedlium_transcripts(*args.data_paths)
    elif args.action == "combine-cmu-libsp":
        combine_cmud_libsp_lexicons(*args.data_paths, save_path=args.save_path)
    elif args.action == "compare-phonemes":
        compare_phonemes(*args.data_paths, save_path=args.save_path)
    elif args.action == "compute-lex-outliers":
        compute_lexicon_outliers(*args.data_paths)
    elif args.action == "expand-contractions":
        expand_contractions(*args.data_paths, save_path=args.save_path)
    elif args.action == "extract-phonemes":
        extract_aligner_phonemes_mp(*args.data_paths, args.save_path, args.jobs) 
    elif args.action == "insert-mispronunciations":
        insert_mispronunciations(*args.data_paths, save_path=args.save_path)
    elif args.action ==  "phoneme-occurance":
        phoneme_occurance(*args.data_paths, save_path=args.save_path)
    elif args.action ==  "remove-suffix":
        remove_suffix(*args.data_paths, save_path=args.save_path)
    elif args.action == "update-train-json":
        update_train_json(*args.data_paths, args.save_path)
    elif args.action ==  "write-textgrid-from-txt":
        write_textgrid_from_txt(*args.data_paths)
