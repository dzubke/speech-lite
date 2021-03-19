"""
this file contains functions that allow for audio files to be fed to Google's
speech-to-text (STT) API. This will be used to see if the STT can identify misprounced words.
Copyright: Speak Labs 2021
Author: Dustin Zubke
"""
# standard libs
import argparse
from collections import OrderedDict
from functools import partial
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List
# third-party libs
import azure.cognitiveservices.speech as speechsdk
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from google.cloud import speech_v1p1beta1 as speech
from tqdm import tqdm
# project libs
from speech.utils.data_helpers import get_record_ids_map, path_to_id, process_text
from speech.utils.io import read_data_json, write_data_json

# not sure if there is a better way to do this.
# without this var-declaration, `set_global_client` throws NameError
client = None

def stt_on_datasets(
    data_paths:List[str], 
    save_path:str, 
    stt_provider:str="ibm", 
    resume_audio_path:str=None)->None:
    """This function calls Google's STT on all the audio files in the datasets in
    `data_paths` with the output formated and written in json format. 
    The combined datasets audio will be sorted and processed sequentially.
    If the process is disrupted, it will be restarted from `resume_audio_path` and the current
    audio path will be saved.

    Args:
        data_paths: list of dataset paths
        resume_audio_path: audio path where the api calls will be resumed from
        stt_provider: name of company providing stt service
        save_path: path to output json file
    """
    MULTI_PROCESS = False
    CHUNK_SIZE = 5   # number of elements to feed into multiprocess pool
    
    assert stt_provider in ["google", "ibm"], "stt_provider must be 'google' or 'ibm'."

    # combines all the datasets into a de-duplicated, sorted dict
    data_dict = combine_sort_datasets(data_paths)     

    # check to see if the save_path should be overwritten
    if os.path.exists(save_path):
        print(f"The save path: {save_path} already exists.")
        print("Are you sure you wanted to overwrite it? (y/n)")
        inp = input()
        if inp.lower() != "y":
            print("exiting")
            return None

    # start from the `resume_audio_path`
    resume_idx = -1
    if resume_audio_path is not None:
        for i, audio_path in enumerate(data_dict.keys()):
            if audio_path == resume_audio_path:
                resume_idx = i
    # only use files after the `resume_idx`
    audio_files = list(data_dict.keys())[resume_idx+1:]
    print
    
    print(f"number of de-duplicated examples: {len(data_dict.values())}")
    print(f"number of examples to process: {len(audio_files)}")   

    # call the STT API from `resume_audio_path` and write the formatted results to `save_path`
    
    count = {"api_calls": 0}
    # multi-process implementation
    if MULTI_PROCESS:

        manager = mp.Manager()
        queue = manager.Queue() 
        pool_fn = partial(
            call_api_write_json,
            #client = client,
            stt_provider = stt_provider,
        )
        with mp.Pool(processes=mp.cpu_count(), initializer=set_global_client) as pool:
            #out_dict_list = pool.map(pool_fn, audio_files, chunksize=CHUNK_SIZE)
            
            # start writer listener
            watcher = pool.apply_async(listener, (queue,))
            jobs = list()
            for audio_file in audio_files:
                job = pool.apply_async(pool_fn, (audio_file, queue))
                jobs.append(job)
            for job in jobs:
                job.get() 
            
            queue.put('kill')
            pool.close()
            pool.join()
                

    # single-process implementation
    else:
        with open(save_path, 'w') as fid:
            client = get_stt_client(stt_provider)
            for audio_path in tqdm(audio_files):
                count['api_calls'] += 1
                response = get_stt_response(audio_path, client, stt_provider)
                out_dict = format_response_dict(audio_path, response, stt_provider)
                json.dump(out_dict, fid)
                fid.write('\n')

        print(f"number of api calls: {count['api_calls']}")


def filter_datasets_by_stt(data_paths:List[str], metadata_path:str, stt_path:str, save_path:str)->None:
    """This function takes in a list of datasets paths `data_paths`, combines them, and
    filters out the examples where the transcript does not equal the transcript from Google's
    speech-to-text API saved at `stt_path`. The filtered dataset (which is a filtered superset
    of all of the datasets in `data_paths`) is written to `save_path`. 

    Args:
        data_paths: list of datasets to combine and fitler to output
        metadata_path: path to speak metadata tsv file
        stt_path: path to speech-to-text saved output from `stt_on_datasets` function
        save_path: path where filtered examples will be saved
    """
    data_dict = combine_sort_datasets(data_paths)
    metadata = get_record_ids_map(metadata_path, has_url=True)
    stt_data = read_data_json(stt_path)    
    filtered_data = list()

    count = {"total": 0, "filtered": 0}
    for datum in stt_data:        
        audio_id = path_to_id(datum['audio_path'])
        spk_trans = process_text(metadata[audio_id]['target_sentence'])
        ggl_trans = process_text(datum['transcript'])
        count['total'] += 1

        if spk_trans == ggl_trans:
            count['filtered'] += 1
            filtered_data.append(data_dict[datum['audio_path']])

    write_data_json(filtered_data, save_path)
    print(f"number of total de-duplicated examples: {count['total']}")
    print(f"number of filtered examples: {count['filtered']}")


def stt_on_sample(
    data_path:str, 
    metadata_path:str,
    save_path:str,
    stt_provider:str='ibm')->None:
    """Pulls a random sample of audio files from `data_path` and calls a speech-to-text API to 
    get transcript predictions. The STT output is formated and written to `save_path` along 
    with the files's transcript from `metadata_path`. 

    Args:
        data_path: path to training json 
        metadata_path: path to metadata tsv containing transcript
        save_path: path where output txt will be saved
        stt_provider: name of company providing STT model
    """
    random.seed(0)
    SAMPLE_SIZE = 100

    data = read_data_json(data_path)
    data_sample = random.choices(data, k=SAMPLE_SIZE)
    print(f"sampling {len(data_sample)} samples from {data_path}")

    # mapping from audio_id to transcript
    metadata = get_record_ids_map(metadata_path, has_url=True)

    client = get_stt_client(stt_provider)
    
    preds_with_two_trans = set()
    match_trans_entries = list()       # output list for matching transcripts
    diff_trans_entries = list()        # output list for non-matching transcripts
    for datum in data_sample:
        audio_path = datum['audio']
        audio_id = path_to_id(audio_path)
        id_plus_dir = os.path.join(*audio_path.split('/')[-2:])

        response = get_stt_response(audio_path, client, stt_provider)        
        resp_dict = format_response_dict(audio_path, response, stt_provider)

        ggl_trans = process_text(resp_dict['transcript'])
        apl_trans = process_text(metadata[audio_id]['target_sentence'])
        out_txt = format_txt_from_dict(resp_dict, apl_trans, id_plus_dir)
    
        if apl_trans == ggl_trans:
            match_trans_entries.append(out_txt)
        else:
            diff_trans_entries.append(out_txt)
    
    with open(save_path, 'w') as fid:
        for entries in [diff_trans_entries, match_trans_entries]:
            fid.write("-"*10+'\n')
            for entry in entries:
                fid.write(entry+'\n\n')


#######    HELPER FUNCTIONS    #######


def combine_sort_datasets(data_paths: List[str])->Dict[str, dict]:
    """combines all examples in the datasets in `data_paths` and creates a de-duplicated, sorted
    dict mapping audio_paths to examples.
    """
    data_dict = dict()
    total_xmpls = 0 
    for data_path in data_paths:
        dataset = read_data_json(data_path)
        total_xmpls += len(dataset)
        for xmpl in dataset:
            if xmpl['audio'] not in data_dict:
                data_dict[xmpl['audio']] = xmpl
            else:
                # checks that same entry in different datasest have same labels
                assert data_dict['audio']['text'] == xmpl['text'], \
                    "same entry in different dataset differ in phonemes labels"
    
    print(f"number of total examples: {total_xmpls}")
    return OrderedDict((audio, xmpl) for audio, xmpl in sorted(data_dict.items()))


def get_stt_client(stt_provider:str='ibm')->Any:
    """Returns the stt client based on the name of the `stt_provider`."""
    
    if stt_provider == "google":
        client = speech.SpeechClient()
    elif stt_provider == "ibm":
        assert "IBM_KEY" in os.environ, "environment is missing IBM API key"
        authenticator = IAMAuthenticator(os.getenv("IBM_KEY"))
        client = SpeechToTextV1(authenticator=authenticator)
        client.set_service_url(
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/970f9a72-c22f-4363-889f-8b538cc2a4a5"
        ) 
        client.set_default_headers({'x-watson-learning-opt-out': "true"})
    elif stt_provider == "azure":
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZR_KEY"), region="eastus")
    else:
        raise ValueError(f"stt provider: {stt_provider} is unacceptable. Use 'google' or 'ibm'.") 
    
    return client


def get_stt_response(audio_path:str, client:Any, stt_provider:str)->Any:
    """sends a call to the STT specified by the client for the input audio_path"""

    with open(audio_path, "rb") as fid:
        content = fid.read()

    if stt_provider == "google":
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_confidence=True,
            model="default",
        )
        response = client.recognize(config=config, audio=audio)

    elif stt_provider == "ibm":
        response = client.recognize(
            audio=content,
            content_type='audio/wav',
            model="en-US_BroadbandModel",
            word_confidence=True
        ).get_result()
    
    elif stt_provider == "azure":
        audio_input = speechsdk.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=client, audio_config=audio_input)
        result = speech_recognizer.recognize_once_async().get()

    else:
        raise ValueError(f"stt provider: {stt_provider} is unacceptable. Use 'google' or 'ibm'.")    

    return response


def format_response_dict(audio_path: str, response: Any, stt_provider:str)->dict:
    """Formats the STT response as a dict to be written to json"""

    transcript = list()
    conf = list()
    words_confs = list()
    
    if stt_provider == "google":
        for i in range(len(response.results)):
            alt = response.results[i].alternatives[0]
            transcript.append(alt.transcript.strip())
            conf.append(alt.confidence)
            # adds each word-confidence pair as a tuple to the `words_confs` list
            words_confs.extend(
                [(w_c.word, w_c.confidence) for w_c in list(alt.words)]
            )
    
    elif stt_provider == "ibm":
        for result in response['results']:
            alt = result['alternatives'][0]
            transcript.append(alt['transcript'].strip())
            conf.append(alt['confidence'])
            words_confs.extend(
                [(word, word_conf) for word, word_conf in alt['word_confidence']]
            )
    # filter out hesitation tag
    transcript = " ".join(transcript).replace("%HESITATION ", "")

    return {
        "audio_path": audio_path,
        "transcript": transcript,
        "confidence": conf,
        "words_confidence": words_confs
    }


def format_txt_from_dict(resp_dict, apl_trans:str, id_plus_dir:str)->str:
    """Formats the entry for each prediction"""

    lines = [
        id_plus_dir,
        apl_trans,
        resp_dict['transcript'], 
        " ".join([str(round(conf, 4)) for conf in resp_dict['confidence']]),
        ' '.join([f"({word}, {str(round(conf, 4))})" for word, conf in resp_dict['words_confidence']])
    ]
    
    return '\n'.join(lines)
   

def call_api_write_json(
    audio_path:str, 
    queue:mp.Queue,
    stt_provider:str, 
    #file_writer:io.TextIOWrapper
    )->dict: 
    """This function packages the STT API call, formatting of response, and writing to json."""

    response = get_stt_response(audio_path, client, stt_provider)
    out_dict = format_response_dict(audio_path, response, stt_provider)
    queue.put(out_dict)   
 
    return out_dict


def listener(queue:mp.Queue, save_file:str):
    '''listens for messages on the q, writes to file. '''

    with open(save_file, 'w') as fid:
        while True:
            out_dict = queue.get()
            if out_dict == 'kill':
                break
            json.dump(out_dict, fid)
            fid.write('\n')
            fid.flush()


def set_global_client(stt_provider:str='ibm'):
    """Creates a global variable client to initialize each process in multiprocess pool"""
    global client
    if not client:
        client = get_stt_client(stt_provider)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calls Google's speech-to-text api and writes results to json."
    )
    parser.add_argument(
        "--action", help="desired function to be called"
    )    
    parser.add_argument(
        "--data-paths", nargs="*", help="path to training json where audio examples will be samples from"
    )
    parser.add_argument(
        "--metadata-path", help="path to metadata file containing transcript"
    )
    parser.add_argument(
        "--optional-arg", help="overloaded arg for optional arguments", default=None,
    )
    parser.add_argument(
        "--save-path", help="path where output file will be saved"
    )
    parser.add_argument(
        "--stt-path", help="path to saved output speech-to-text file"
    )
    parser.add_argument(
        "--stt-provider", help="name of company providing speech-to-text service"
    )   

    args = parser.parse_args()
    
    start_time = time.time()

    if args.action == "filter-by-stt":
        filter_datasets_by_stt(args.data_paths, args.metadata_path, args.stt_path, args.save_path)
    elif args.action == "stt-on-datasets":
        stt_on_datasets(args.data_paths, args.save_path, args.stt_provider, args.optional_arg)
    elif args.action == "stt-on-sample":
        stt_on_sample(args.data_paths[0], args.metadata_path, args.save_path, args.stt_provider)

    print(f"processing time (min): {round((time.time() - start_time)/60, 2)}")
