#!/usr/bin/env python
# standard libs
import codecs
import fnmatch
import glob
import os
import random
import re
import subprocess
import sys
import tarfile
from typing import Tuple
import unicodedata
import wave

# third party libs
import librosa
import pandas
import requests
import soundfile  # <= Has an external dependency on libsndfile

#from deepspeech_training.util.importers import validate_label_eng as validate_label

# ensure that you have downloaded the LDC dataset LDC97S62 and tar exists in a folder e.g.
# ./data/swb/swb1_LDC97S62.tgz
# from the deepspeech directory run with: ./bin/import_swb.py ./data/swb/


# ARCHIVE_NAME refers to ISIP alignments from 01/29/03
ARCHIVE_NAME = "switchboard_word_alignments.tar.gz"
ARCHIVE_URL = "http://www.openslr.org/resources/5/"
ARCHIVE_DIR_NAME = ''  #"LDC97S62"
LDC_DATASET = "swb1_LDC97S62.tgz"


def _download_and_preprocess_data(data_dir):
    new_data_dir = os.path.join(data_dir, ARCHIVE_DIR_NAME)
    target_dir = os.path.abspath(new_data_dir)
    archive_path = os.path.abspath(os.path.join(data_dir, LDC_DATASET))

    # Check swb1_LDC97S62.tgz then extract
    #assert os.path.isfile(archive_path)
    #_extract(target_dir, archive_path)

    # Transcripts
    #transcripts_path = maybe_download(ARCHIVE_URL, target_dir, ARCHIVE_NAME)
    #_extract(target_dir, transcripts_path)

    # Check swb1_d1/2/3/4/swb_ms98_transcriptions
    expected_folders = [
        "swb1_d1",
        "swb1_d2",
        "swb1_d3",
        "swb1_d4",
        "swb_ms98_transcriptions",
    ]
    assert all([os.path.isdir(os.path.join(target_dir, e)) for e in expected_folders])

    data_dirs = ["swb1_d1", "swb1_d2", "swb1_d3", "swb1_d4"] 

    # Conditionally convert swb sph data to wav
    for data_dir in data_dirs:
        wav_dir = data_dir + "-wav"
        _maybe_convert_wav(target_dir, data_dir, wav_dir)
        # number of wav files should be twice sph files because of 2 channels
        assert count_files(wav_dir, "*.wav") == 2 * count_files(data_dir, "*.sph"), \
            f"number of sph files in {data_dir} doesn't match wav files in {wav_dir}"

    print("Conditionally splitting data...")
    split_files = list()
    for data_dir in data_dirs:
        wav_dir = data_dir + "-wav"
        split_dir = data_dir + "-split-wav"
        split_files.extend(
            _maybe_split_wav_and_sentences(
                target_dir, "swb_ms98_transcriptions", wav_dir, split_dir
            )
        )

    train_files, dev_files, test_files = _split_sets(split_files)

    # Write sets to disk as CSV files
    files_names = [
        (train_files, "swb-train.csv"), 
        (dev_files, "swb-dev.csv"), 
        (test_files, "swb-test.csv")
    ]
    header = ["wav_filename", "wav_filesize", "transcript"]
    for data_files, set_name  in files_names:
        out_path = os.path.join(target_dir, set_name)
        with open(out_path, 'w') as fid:
            fid.write(','.join(header)+'\n')
            for row in data_files:
                fid.write(','.join(row)+'\n')


def _extract(target_dir, archive_path):
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)


def download_file(folder, url):
    # https://stackoverflow.com/a/16696317/738515
    local_filename = url.split("/")[-1]
    full_filename = os.path.join(folder, local_filename)
    r = requests.get(url, stream=True)
    with open(full_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_filename


def maybe_download(archive_url, target_dir, ldc_dataset):
    # If archive file does not exist, download it...
    archive_path = os.path.join(target_dir, ldc_dataset)
    ldc_path = archive_url + ldc_dataset
    if not os.path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        download_file(target_dir, ldc_path)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path


def _maybe_convert_wav(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert sph files to wav files
    if os.path.exists(target_dir):
        print("skipping maybe_convert_wav")
        return

    # Create target_dir
    os.makedirs(target_dir)

    # Loop over sph files in source_dir and convert each to 16-bit PCM wav
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.sph"):
            for channel in ["1", "2"]:
                sph_file = os.path.join(root, filename)
                wav_filename = (
                    os.path.splitext(os.path.basename(sph_file))[0]
                    + "-"
                    + channel
                    + ".wav"
                )
                wav_file = os.path.join(target_dir, wav_filename)
                temp_wav_filename = (
                    os.path.splitext(os.path.basename(sph_file))[0]
                    + "-"
                    + channel
                    + "-temp.wav"
                )
                temp_wav_file = os.path.join(target_dir, temp_wav_filename)
                print("converting {} to {}".format(sph_file, temp_wav_file))
                subprocess.check_call(
                    [
                        "sph2pipe",
                        "-c",
                        channel,
                        "-p",
                        "-f",
                        "rif",
                        sph_file,
                        temp_wav_file,
                    ]
                )
                print("upsampling {} to {}".format(temp_wav_file, wav_file))
                audioData, frameRate = librosa.load(temp_wav_file, sr=16000, mono=True)
                soundfile.write(wav_file, audioData, frameRate, "PCM_16")
                os.remove(temp_wav_file)


def count_files(data_dir:str, pattern:str)->int:
    """This function counts the number of files that match the input pattern
    contained in the input data_dir

    Args:
        data_dir (str): directory to search for matching files
        pattern (str): regex pattern for files
    Returns:
        (int): count of files that match patter in data_dir
    """
    return len(glob.glob(os.path.join(data_dir, "**", pattern)))


def _parse_transcriptions(trans_file):
    segments = []
    with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
            if line.startswith("#") or len(line) <= 1:
                continue

            tokens = line.split()
            start_time = float(tokens[1])
            stop_time = float(tokens[2])
            transcript = validate_label(" ".join(tokens[3:]))

            if transcript == None:
                continue

            # We need to do the encode-decode dance here because encode
            # returns a bytes() object on Python 3, and text_to_char_array
            # expects a string.
            transcript = (
                unicodedata.normalize("NFKD", transcript)
                .encode("ascii", "ignore")
                .decode("ascii", "ignore")
            )

            segments.append(
                {
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "transcript": transcript,
                }
            )
    return segments


def _maybe_split_wav_and_sentences(data_dir, trans_data, original_data, converted_data)->list:
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    #if os.path.exists(target_dir):
    #    print("skipping maybe_split_wav")
    #    return

    os.makedirs(target_dir, exist_ok=True)

    files = []

    # Loop over transcription files and split corresponding wav
    for trans_file in glob.glob(os.path.join(trans_dir, "**", "*trans.text"), recursive=True):
        segments = _parse_transcriptions(trans_file)

        # transcript_id use the format: 'swIIIIA' while audio_ids are formatted as: 'sw0IIII'
        # 
        trans_filename = os.path.basename(trans_file)
        trans_id = trans_filename.split('-')[0]
        assert len(trans_id) == 7
        trans_id = trans_id.replace("sw", "sw0")
        audio_id = trans_id[:-1]  # remove the A/B side character 
       
        # Open wav corresponding to transcription file
        channel = '1' if trans_id[-1] == "A" else '2'

        wav_filename = audio_id + '-' + channel + ".wav"

        wav_file = os.path.join(source_dir, wav_filename)
        print("splitting {} according to {}".format(wav_file, trans_file))

        # the connection between transcript and data_directory is unknown, so the constructed 
        # wav_file may be in another directory, so check its existence
        if not os.path.exists(wav_file):
            print("~~ wav_file not found: " + wav_file)
            continue

        origAudio = wave.open(wav_file, "r")

        # Loop over segments and split wav_file for each segment
        for segment in segments:
            # Create wav segment filename
            start_time = segment["start_time"]
            stop_time = segment["stop_time"]
            new_wav_filename = (
                audio_id
                + "-"
                + str(start_time)
                + "-"
                + str(stop_time)
                + ".wav"
            )
            if _is_wav_too_short(new_wav_filename):
                continue

            new_wav_file = os.path.join(target_dir, new_wav_filename)

            if not os.path.exists(new_wav_file):
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

            new_wav_filesize = os.path.getsize(new_wav_file)
            transcript = segment["transcript"]
            files.append(
                (os.path.abspath(new_wav_file), str(new_wav_filesize), transcript)
            )

        # Close origAudio
        origAudio.close()

    return files


def _is_wav_too_short(wav_filename):
    short_wav_filenames = [
        "sw2986A-ms98-a-trans-80.6385-83.358875.wav",
        "sw2663A-ms98-a-trans-161.12025-164.213375.wav",
    ]
    return wav_filename in short_wav_filenames


def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()


def _split_sets(filelist:list, split_factor:float=0.95)->Tuple[list, list, list]:
    """Will randomly split the input list of files into three sets based on the split_factor.
    A split_factor=0.95 will result in a 90/5/5 training/dev/test split
    A split_factor=0.9 will result in a 80/10/10 training/dev/test split 
    """

    random.shuffle(filelist)

    split_factor = 0.95
    train_beg = 0
    train_end = int(split_factor * len(filelist))

    dev_beg = int(split_factor * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return (
        filelist[train_beg: train_end],
        filelist[dev_beg: dev_end],
        filelist[test_beg: test_end],
    )


def _read_data_set(
    filelist,
    thread_count,
    batch_size,
    numcep,
    numcontext,
    stride=1,
    offset=0,
    next_index=lambda i: i + 1,
    limit=0,
):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    filelist = filelist[offset::stride]

    # Return DataSet
    return DataSet(
        txt_files, thread_count, batch_size, numcep, numcontext, next_index=next_index
    )


# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    """Validate and normalize transcriptions. Returns a cleaned version of the label 
    or None if it's invalid. The commented out sections preserve the original code. 
    The `remove_markers` has been added
    """
    
    # this only accepts [a-z '], and will remove all utterances with [noise] which isn't desired
    #if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
    #    return None

    # instead the noise, silence, and other tags will be removed
    remove_tags = ["[noise]", "[silence]", "<b_aside>", "<e_aside>", '[vocalized-noise]', "[laughter]"]
    for tag in remove_tags:
        label = label.replace(tag, "")


    # '-'. '_', '[' , ']' are used in the lexicon
    #label = label.replace("-", " ")
    #label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)  # this removes 2 or more consecutive spaces
    # these 7 punctuation are not in the lexicon, so removing them is fine
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.strip()
    # Capitalization matters in the lexicon
    #label = label.lower()

    return label if label else None




if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
