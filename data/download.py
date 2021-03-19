# standard libraries
import argparse
import csv
import datetime
import functools
import glob
import json
from multiprocessing import Pool
import os
import random
import re
from shutil import copyfile
import tarfile
from tempfile import NamedTemporaryFile
import time
from typing import List, Set
import urllib
from zipfile import ZipFile
# third party libraries
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import tqdm
# project libraries
from speech.utils.convert import to_wave
from speech.utils.data_helpers import check_update_contraints, get_dataset_ids, get_record_ids_map
from speech.utils.data_helpers import path_to_id, process_text
from speech.utils.io import load_config, read_data_json


class Downloader(object):

    def __init__(self, output_dir, dataset_name):
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.download_dict = dict()
        self.ext = ".tar.gz"


    def download_dataset(self):
        save_dir = self.download_extract()
        print(f"Data saved and extracted to: {save_dir}")
        self.extract_samples(save_dir)

    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        for name, url in self.download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
    
            save_path = os.path.join(save_dir, name + self.ext)
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir
    
    def extract_samples(self, save_dir:str):
        """
        Most datasets won't need to extract samples
        """
        pass
    

class VoxforgeDownloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        super(VoxforgeDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data": "https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz",
            "lexicon": "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Lexicon/VoxForge.tgz"
        }
        self.data_dirname = "archive"


    def extract_samples(save_dir:str):
        """
        All samples are zipped in their own tar files.
        This function collects all the tar filenames and
        unzips themm.
        """
        pattern = "*.tgz"
        sample_dir = os.path.join(save_dir, self.data_dirname)
        tar_path = os.path.join(sample_dir, pattern)
        tar_files = glob.glob(tar_path)
        print("Extracting and removing sample files...")
        for tar_file in tqdm(tar_files):
            with tarfile.open(tar_file) as tf:
                tf.extractall(path=sample_dir)
            os.remove(tar_file)

class TatoebaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip"
        }
        self.data_dirname = "audio"

        # downloads
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2
        # https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2
        # https://downloads.tatoeba.org/exports/user_languages.tar.bz2
        # https://downloads.tatoeba.org/exports/users_sentences.csv


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


class TatoebaV2Downloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "",
            "data_csv": "https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2"
        }
        self.data_dirname = "audio"


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


class CommonvoiceDownloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        A previous version of common voice (v4) can be downloaded here:
        "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
        """
        super(CommonvoiceDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz"
        }
        self.data_dirname = "clips"


class WikipediaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        """
        super(WikipediaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_English.tar"
        }
        self.data_dirname = "not-used"
        self.ext = ".tar"
    

class SpeakTrainDownloader(Downloader):
    
    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        Downloading the Speak Train Data is significantly different than other datasets
        because the Speak data is not pre-packaged into a zip file. Therefore, the 
        `download_dataset` method will be overwritten. 

        Args:
            output_dir (str): directory where files will be downloaded
            dataset_name (str): name of the dataset 
                TODO: (only used as a subdirectory of `output_dir`, maybe unnecessary)
            config_path (dict): configuration file
        
        Properties:
            download_audio (bool): if True, audio filles will be downloaded. 
                If False, only metadata is downloaded.
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.download_audio = False         # if False, no audio will be downloaded
        self.last_id = None #'351999FC-1B14-4D9E-9D52-63D0EB66ABD1'
        self.metadata_fname = "metadata-with-url"

    def download_dataset(self):
        """
        This method loops through the firestore document database using paginated queries based on
        the document id. It filters out documents where `target != guess` and saves the audio file
        and target text into separate files. 

        The approach to index the queries based on the document `id` is based on the approach
        outlined here: 
        https://firebase.google.com/docs/firestore/query-data/query-cursors#paginate_a_query
            
        """

        PROJECT_ID = 'speak-v2-2a1f1'
        QUERY_LIMIT = 2500
        NUM_PROC = 50
        AUDIO_EXT = ".m4a"
     
        # verify and set the credientials
        CREDENTIAL_PATH = "/home/dzubke/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
        assert os.path.exists(CREDENTIAL_PATH), "Credential file does not exist or is in the wrong location."
        # set the enviroment variable that `firebase_admin.credentials` will use
        os.putenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIAL_PATH)
        
        # create the data-label path and initialize the tsv headers 
        audio_dir = os.path.join( self.output_dir,  "audio")
        today = datetime.date.today().isoformat()
        metadata_path = os.path.join(
            self.output_dir, f"{self.metadata_fname}_{today}.tsv"
        )

        # initialize the credentials and firebase db client
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
        db = firestore.client()

        # create the first query based on the constant QUERY_LIMIT
        rec_ref = db.collection(u'recordings')

        # if `last_id` is not set, start the query from the beginning
        if self.last_id is None:
            next_query = rec_ref.order_by(u'id').limit(QUERY_LIMIT)
            
            # write a header to a new file
            with open(metadata_path, 'w', newline='\n') as tsv_file:
                tsv_writer = csv.writer(tsv_file, delimiter='\t')
                header = [
                    "id", "target", "lessonId", "lineId", "uid", "redWords_score", "date"
                ]
                # add the audio url if not downloading audio
                if not self.download_audio: 
                    header.append("audio_url")

                tsv_writer.writerow(header)

        # or begin where a previous run left off at `self.last_id`
        else: 
            next_query = (rec_ref
                .order_by(u'id')
                .start_after({
                    u'id': self.last_id
                })
                .limit(QUERY_LIMIT))
        
        start_time = time.time()
        query_count = 0 
        
            
        # these two lines can be used for debugging by only looping a few times 
        #loop_iterations = 5
        #while loop_iterations > 0:
    
        # loops until break is called in try-except block
        while True:
            
            # converting generator to list to it can be referenced mutliple times
            # the documents are converted to_dict so they can be pickled in multiprocessing
            docs = list(map(lambda x: self._doc_trim_to_dict(x), next_query.stream()))
            
            try:
                # this `id` will be used to start the next query
                last_id = docs[-1][u'id']
            # if the docs list is empty, there are no new documents
            # and an IndexError will be raised and break the while loop
            except IndexError:
                break
            
            # fill in the keyword arguments of the multiprocess download function        
            mp_function = functools.partial(
                self.multiprocess_download, 
                audio_dir = audio_dir,
                metadata_path = metadata_path,
                audio_ext = AUDIO_EXT
            )

            #self.singleprocess_record(docs)
            pool = Pool(processes=NUM_PROC)
            results = pool.imap_unordered(mp_function, docs, chunksize=1)
            pool.close() 
            pool.join()
           
            # print the last_id so the script can pick up from the last_id if something breaks 
            query_count += QUERY_LIMIT
            print(f"last_id: {last_id} at count: {query_count}")
            print(f"script duration: {round((time.time() - start_time)/ 60, 2)} min")
            
            # create the next query starting after the last_id 
            next_query = (
                rec_ref
                .order_by(u'id')
                .start_after({
                    u'id': last_id
                })
                .limit(QUERY_LIMIT))
        
            # used for debugging with fixed number of loops
            #loop_iterations -= 1
        

    def multiprocess_download(self, doc:dict, audio_dir:str, metadata_path:str, audio_ext:str):
        """
        Takes in a single document and records and downloads the contents if the recording
        meets the criterion. Used by a multiprocessing pool.

        Args:
            doc (dict): dict of the record to bee processed. A dict is passed as it needs to be pickled.
            tsv_writer (csv.writer): tsv_writer object where the metadata will be written
            audio_dir (str): directory where audio is saved
            metadata_path (str): path where metadata tsv file will be saved
            audio_ext (str): extension of the saved audio file
        Returns:
            None - write to file
        """
        with open(metadata_path, 'a') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            # process the target and guess text to see if they are equal
            # some of the guess's don't include apostrophes so processing will remove apostrophes
            target = process_text(doc['info']['target'], remove_apost=True)
            guess = process_text(doc['result']['guess'], remove_apost=True)
            
            if target == guess:

                # if True, save the audio file from the link in the document
                if self.download_audio:
                    audio_url = doc['result']['audioDownloadUrl']
                    audio_save_path = os.path.join(audio_dir, doc['id'] + AUDIO_EXT)
                    try:
                        urllib.request.urlretrieve(audio_url, filename=audio_save_path)
                    except (ValueError, URLError) as e:
                        print(f"unable to download url: {audio_url} due to exception: {e}")

                # save the target and metadata in a tsv row
                # tsv header: "id", "target", "lessonId", "lineId", "uid", "redWords Score", "date"
                tsv_row =[
                    doc['id'],
                    doc['info']['target'],
                    doc['info']['lessonId'],
                    doc['info']['lineId'],
                    doc['user']['uid'],
                    doc['result']['score'],
                    doc['info']['date']
                ]
                # if not downloading the audio file, add the url to the metadata
                if not self.download_audio:
                    tsv_row.append(doc['result']['audioDownloadUrl'])
                tsv_writer.writerow(tsv_row)


    def singleprocess_download(self, docs_list:list):
        """
        Downloads the audio file associated with the record and records the transcript and various metadata
        Args:
            docs_list - List[firebase-document]: a list of document references
        """

        AUDIO_EXT = ".m4a"
        TXT_EXT = ".txt"
        audio_dir = os.path.join(self.output_dir, "audio")

        with open(self.data_label_path, 'a') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            # filter the documents where `target`==`guess`
            for doc in docs_list:
      
                doc_dict = doc.to_dict() 
                original_target = doc_dict['info']['target']
         
                # some of the guess's don't include apostrophes
                # so the filter criterion will not use apostrophes
                target = process_text(doc_dict['info']['target'])
                target_no_apostrophe = target.replace("'", "")
                
                guess = process_text(doc_dict['result']['guess'])
                guess_no_apostrophe = guess.replace("'", "")

                if target_no_apostrophe == guess_no_apostrophe:
                    # save the audio file from the link in the document
                    audio_url = doc_dict['result']['audioDownloadUrl']
                    audio_save_path = os.path.join(audio_dir, doc_dict['id'] + AUDIO_EXT)
                    try:
                        urllib.request.urlretrieve(audio_url, filename=audio_save_path)
                    except (ValueError, urllib.error.URLError) as e:
                        print(f"unable to download url: {audio_url} due to exception: {e}")
                        continue

                    # save the target in a tsv row
                    # tsv header: "id", "text", "lessonId", "lineId", "uid", "date"
                    tsv_row =[
                        doc_dict['id'], 
                        original_target, 
                        doc_dict['info']['lessonId'],
                        doc_dict['info']['lineId'],
                        doc_dict['user']['uid'],
                        doc_dict['result']['score'],
                        doc_dict['info']['date']
                    ]
                    tsv_writer.writerow(tsv_row)
                    

    @staticmethod
    def _doc_trim_to_dict(doc)->dict:
        """This function converts a document into a dict and removes certain keys if the keys
           exist in the dict. The removed keys have very large arrays for values that aren't necessary 
            and take up alot of memory. 

        Args:
            doc: firestore document
        Returns:
            dict: trimmed dictionary of the input document
        """
        doc = doc.to_dict()
        # remove large and unnecessary parts of the doc_dict
        if 'asrResults' in doc['result']:
            del doc['result']['asrResults']
        if 'processingResult' in doc['result']:
            del doc['result']['processingResult']
        if 'asrResultData' in doc['result']:
            del doc['result']['asrResultData']

        return doc



class SpeakEvalDownloader(SpeakTrainDownloader):
    """
    This class creates a small evaluation dataset from the Speak firestore database.


    ##### LIST OF PREVIOUS QUERIES ######

    ## querying only by day_range
    next_query = rec_ref.where(u'info.date', u'>', day_range).order_by(u'info.date').limit(QUERY_LIMIT)

    last_time = docs[-1]['info']['date']

    next_query = (
        rec_ref.where(
            u'info.date', u'>', day_range
        )
        .order_by(u'info.date')
        .start_after({
            u'info.date': last_time
        })
        .limit(QUERY_LIMIT)
    )
    """

    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        Properties:
            num_examples (int): number of examples to be downloaded
            target_eq_guess (bool): if True, the target == guess criterion will filter the downloaded files
        """
        super().__init__(output_dir, dataset_name)
        config = load_config(config_path)
        self.num_examples = config['num_examples']
        self.target_eq_guess = config['target_eq_guess']
        self.check_constraints = config['check_constraints']
        self.constraints = config['constraints']
        self.days_from_today = config['days_from_today']
        self.disjoint_metadata_tsv = config['disjoint_metadata_tsv']
        self.disjoint_id_names = config['disjoint_id_names']
        self.disjoint_datasets = config['disjoint_datasets']


    def download_dataset(self):
        """
        This method loops through the firestore document database using paginated queries based on
        the document id. It filters out documents where `target != guess` if `self.target_eq_guess` is True
        and saves the audio file and target text into separate files. 
        """

        PROJECT_ID = 'speak-v2-2a1f1'
        QUERY_LIMIT = 2000              # max size of query
        SAMPLES_PER_QUERY = 200         # number of random samples downloaded per query
        AUDIO_EXT = '.m4a'              # extension of downloaded audio
        audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # verify and set the credientials
        CREDENTIAL_PATH = "/home/dzubke/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
        assert os.path.exists(CREDENTIAL_PATH), "Credential file does not exist or is in the wrong location."
        # set the enviroment variable that `firebase_admin.credentials` will use
        os.putenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIAL_PATH)

        # initialize the credentials and firebase db client
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
        db = firestore.client()

        # create the data-label path and initialize the tsv headers 
        date = datetime.date.today().isoformat()
        self.data_label_path = os.path.join(self.output_dir, "eval2-v4_data_" + date + ".tsv")
        self.metadata_path = os.path.join(self.output_dir, "eval2-v4_metadata_" + date + ".json")

        # re-calculate the constraints in the `config` as integer counts based on the `dataset_size`
        self.constraints = {
            name: int(self.constraints[name] * self.num_examples) for name in self.constraints.keys()
        }
        # constraint_names will help to ensure the dict keys created later are consistent.
        constraint_names = list(self.constraints.keys())
        print("constraints: ", self.constraints)

        # id_counter keeps track of the counts for each speaker, lesson, and line ids
        id_counter = {name: dict() for name in constraint_names}

        # create a mapping from record_id to lesson, line, and speaker ids
        disjoint_ids_map = get_record_ids_map(metadata_path, constraint_names)

        # create a dict of sets of all the ids in the disjoint datasets that will not
        # be included in the filtered dataset
        disjoint_id_sets = {name: set() for name in self.disjoint_id_names}
        for disj_dataset_path in self.disjoint_datasets:
            disj_dataset = read_data_json(disj_dataset_path)
            # extracts the record_ids from the excluded datasets
            record_ids = [path_to_id(example['audio']) for example in disj_dataset]
            # loop through each record id
            for record_id in record_ids:
                # loop through each id_name and update the disjoint_id_sets
                for disjoint_id_name, disjoint_id_set in disjoint_id_sets.items():
                    disjoint_id_set.add(disjoint_ids_map[record_id][disjoint_id_name])

        # creating a data range from `self.days_from_today` in the correct format
        now = datetime.datetime.utcnow()
        day_delta = datetime.timedelta(days = self.days_from_today)
        day_range = now - day_delta
        day_range = day_range.isoformat("T") + "Z"

        with open(self.data_label_path, 'w', newline='\n') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            header = [
                "id", "target", "guess", "lessonId", "target_sentence", "lineId", "uid", "redWords_score", "date"
            ]
            tsv_writer.writerow(header)        

            # create the first query based on the constant QUERY_LIMIT
            rec_ref = db.collection(u'recordings')

            # this is the final record_id that was downloaded from the speak training set
            speak_train_last_id = 'SR9TIlF8bSWApZa1tqEBIHOQs5z1-1583920255'

            next_query = rec_ref\
                .order_by(u'id')\
                .start_after({u'id': speak_train_last_id})\
                .limit(QUERY_LIMIT)\

            # loop through the queries until the example_count is at least the num_examples
            example_count = 0
            # get the ids from the training and testsets to ensure the downloaded set is disjoint
            train_test_set = self.get_train_test_ids()

            while example_count < self.num_examples:
                print(f"another loop with {example_count} examples written")                
                # convert the generator to a list to retrieve the last doc_id
                docs = list(map(lambda x: self._doc_trim_to_dict(x), next_query.stream()))
                
                try:
                    # this time will be used to start the next query
                    last_id = docs[-1]['id']
                # if the docs list is empty, there are no new documents
                # and an IndexError will be raised and break the while loop
                except IndexError:
                    print("Exiting while loop")
                    break

                # selects a random sample of `SAMPLES_PER_QUERY` from the total queries
                #docs = random.sample(docs, SAMPLES_PER_QUERY)

                for doc in  docs:
                    # if num_examples is reached, break
                    if example_count >= self.num_examples:
                        break
                    
                    target = process_text(doc['info']['target'])

                    # check that the speaker, target-sentence, and record_Id are disjoint
                    if doc['user']['uid'] not in disjoint_id_sets['speaker']\
                    and target not in disjoint_id_sets['target_sentence']\
                    and doc['id'] not in train_test_set:
                        # set `self.target_eq_guess` to True in `init` if you want 
                        ## to filter by `target`==`guess`
                        if self.target_eq_guess:
                            # process the target and guess and remove apostrophe's for comparison
                            guess = process_text(doc['result']['guess'])
                            # removing apostrophes for comparison
                            target_no_apostrophe = target.replace("'", "")
                            guess_no_apostrophe = guess.replace("'", "")
                            # if targ != guess, skip the record
                            if target_no_apostrophe != guess_no_apostrophe:
                                continue
                        
                        # if `True` constraints on the records downloaded will be checked
                        if self.check_constraints:
                            # create a mapping to feed into `check_update_constraints`
                            record_ids_map = {
                                doc['id']: {
                                    'lesson': doc['info']['lessonId'],
                                    'target_sentence': target,         # using processed target as id 
                                    'speaker': doc['user']['uid']
                                }
                            }
                            pass_constraint = check_update_contraints(
                                doc['id'], 
                                record_ids_map, 
                                id_counter, 
                                self.constraints
                            )
                            # if the record doesn't pass the constraints, continue to the next record
                            if not pass_constraint:
                                continue

                        # save the audio file from the link in the document
                        audio_url = doc['result']['audioDownloadUrl']
                        audio_path = os.path.join(audio_dir, doc['id'] + AUDIO_EXT)


                        # convert the downloaded file to .wav format
                        # usually, this conversion done in the preprocessing step 
                        # but some eval sets don't need PER labels, and so this removes the need to 
                        # preprocess the evalset. 
                        base, raw_ext = os.path.splitext(audio_path)
                        # use the `.wv` extension if the original file is a `.wav`
                        wav_path = base + os.path.extsep + "wav"
                        # if the wave file doesn't exist, convert to wav
                        if not os.path.exists(wav_path):
                            try:
                                to_wave(audio_path, wav_path)
                            except subprocess.CalledProcessError:
                                # if the file can't be converted, skip the file by continuing
                                logging.info(f"Process Error converting file: {audio_path}")
                                continue

                        # save the target in a tsv row
                        # tsv header: "id", "target", "guess", "lessonId", "target_id", "lineId", "uid", "date"
                        tsv_row =[
                            doc['id'], 
                            doc['info']['target'],
                            doc['result']['guess'],
                            doc['info']['lessonId'],
                            target,     # using this to replace lineId
                            doc['info']['lineId'],
                            doc['user']['uid'],
                            doc['result']['score'],
                            doc['info']['date']
                        ]
                        tsv_writer.writerow(tsv_row)
                        # save all the metadata in a separate file
                        #with open(self.metadata_path, 'a') as jsonfile:
                        #    json.dump(doc, jsonfile)
                        #    jsonfile.write("\n")
                        
                        example_count += 1
                
                # create the next query starting after the last_id 
                next_query = (rec_ref\
                    .order_by(u'id')\
                    .start_after({u'id': last_id})\
                    .limit(QUERY_LIMIT)
                )
    

    @staticmethod
    def get_train_test_ids()->Set[str]:
        """
        This function returns a set of ids for records that are included in the speak training and
        test sets. The paths to the training and test sets are hardcoded to the paths on the cloud VM's. 
        Returns:
            Set[str]: a set of record_ids for the training and test recordings
        """
        # train_data_trim_2020-09-22.json is the entire 7M recordings in the full speak training set
        train_test_paths = [
            "/home/dzubke/awni_speech/data/speak_train/train_data_trim_2020-09-22.json",
            "/home/dzubke/awni_speech/data/speak_test_data/2020-05-27/speak-test_2020-05-27.json",
            "/home/dzubke/awni_speech/data/speak_test_data/2019-11-29/speak-test_2019-11-29.json"
        ]
        
        datasets = [read_data_json(path) for path in train_test_paths]
        
        train_test_ids = set()

        # loop throug the datasets and add the id's output from `path_to_id` to the set
        for dataset in datasets:
            train_test_ids.update([path_to_id(datum['audio']) for datum in dataset])

        return train_test_ids


class SpeakTestDownloader(SpeakTrainDownloader):
    """
    This class queries the firestore database and writes a tsv file with the metadata from the speak testsets. 
    This class does not download audio files like the `SpeakTrain` and `SpeakEval` Downloader classes. 
    """

    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        Properties:
            num_examples (int): number of examples to be downloaded
            target_eq_guess (bool): if True, the target == guess criterion will filter the downloaded files
        """
        super().__init__(output_dir, dataset_name)
        config = load_config(config_path)
        lists_of_ids = [
            get_dataset_ids(data_path) for data_path in config['datasets']
        ]
        self.record_ids = [ids for list_of_ids in lists_of_ids for ids in list_of_ids]

    def download_dataset(self):
        """
        This method doesn't download an audio dataset like in other Downloader classes. It writes the metadata
        of a set of recordings to a tsv file. 
        """

        PROJECT_ID = 'speak-v2-2a1f1'
        QUERY_LIMIT = 2000              # max size of query
        SAMPLES_PER_QUERY = 200         # number of random samples downloaded per query

        # verify and set the credientials
        CREDENTIAL_PATH = "/home/dzubke/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
        assert os.path.exists(CREDENTIAL_PATH), "Credential file does not exist or is in the wrong location."
        # set the enviroment variable that `firebase_admin.credentials` will use
        os.putenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIAL_PATH)

        # initialize the credentials and firebase db client
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
        db = firestore.client()
        # select the recordings collection
        rec_ref = db.collection(u'recordings')

        data_label_path = os.path.join(self.output_dir, "speak-test_metadata_2020-12-09.tsv")

        with open(data_label_path, 'w', newline='\n') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            header = [
                "id", "target", "guess", "lessonId", "lineId", "uid", "redWords_score", "date"
            ]
            tsv_writer.writerow(header) 

            # take the record_ids in batches of 10
            # the firestore `in` operater can take only a list of 10 elements
            for idx_10 in range(0, len(self.record_ids), 10):  

                batch_10_ids = self.record_ids[idx_10: idx_10 + 10]

                next_query = rec_ref.where(u'id', u'in', batch_10_ids) 

                for doc in next_query.stream():
                    doc = doc.to_dict()
                
                    target = process_text(doc['info']['target'])

                    tsv_writer.writerow([
                        doc['id'], 
                        doc['info']['target'],
                        doc['result']['guess'],
                        doc['info']['lessonId'],
                        doc['info']['lineId'],
                        doc['user']['uid'],
                        doc['result']['score'],
                        doc['info']['date']
                    ])

class Chime1Downloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        super(Chime1Downloader, self).__init__(output_dir, dataset_name)
        # original dataset is in 2-channel. 
        # in this case, I have downloaded and transfered the data to the VM myself. 


class DemandDownloader(Downloader):

    def __init__(self, output_dir, dataset_name, config_path=None):
        """
        Limitations: 
            - feed_model_dir, the directory where the noise is fed to the model, is hard-coded
        """
        super(DemandDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {}
        self.feed_model_dir = "/home/dzubke/awni_speech/data/noise/feed_to_model"
        self.load_download_dict()

    def load_download_dict(self):
        """
        Loads the download dictionary with download links and dataset names

        Assumptions: 
            Download links will still work in the future
        """

        noise_filenames = ["TMETRO_16k.zip", "TCAR_16k.zip", "SPSQUARE_16k.zip", "SCAFE_48k.zip",
                            "PSTATION_16k.zip", "PRESTO_16k.zip", "OMEETING_16k.zip", "OHALLWAY_16k.zip",
                            "NRIVER_16k.zip", "NPARK_16k.zip", "NFIELD_16k.zip", "DWASHING_16k.zip", 
                            "DLIVING_16k.zip", "DKITCHEN_16k.zip"]
        
        download_link = "https://zenodo.org/record/1227121/files/{}?download=1"
        
        for filename in noise_filenames:
            basename = os.path.splitext(filename)[0]
            self.download_dict.update({basename: download_link.format(filename)})


    def download_extract(self):
        """
        This method was overwritten because the download files were in zip format
        instead of tar format as in the base downloader class. 
        
        Limitations: 
            - this method is not idempotent as it will re-download and extract
                the files again if it is called again
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in self.download_dict.items():
            save_path = os.path.join(save_dir, name + ".zip")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with ZipFile(save_path) as zipfile:
                zipfile.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


    def extract_samples(self, save_dir:str):
        """
        Extracts the wav files from the directories and copies them into the noise_dir.
        The audio files in the "SCAFE_48k" data subset are in 48 kHz and should be converted
        to 16 kHz. The if-statement in the for-loop does this conversion.

        Assumptions: 
            - The directory structure of the zip files will not change
            - 
        """
        pattern = "*/*.wav"
        high_res_audio = {"SCAFE"}
        all_wav_paths = glob.glob(os.path.join(save_dir, pattern))

        print("Extracting and removing sample files...")
        for wav_path in tqdm.tqdm(all_wav_paths):
            filename = os.path.basename(wav_path)
            dirname = os.path.basename(os.path.dirname(wav_path))
            dst_filename = "{}_{}".format(dirname, filename)
            dst_wav_path = os.path.join(self.feed_model_dir, dst_filename)
            if os.path.exists(dst_wav_path):
                print(f"{dst_wav_path} exists. Skipping...")
                continue
            else:
                # if the wavs are high resolution, down-convert to 16kHz
                if dirname in high_res_audio:
                    to_wave(wav_path, dst_wav_path)
                # if not high-res, just copy
                else: 
                    copyfile(wav_path, dst_wav_path)


def download_sample_from_GCP():
    """This function downloads a random sample of audio from processed data in Google Cloud.
    """
    import subprocess

    sample_size = 100
    src_dataset_path = "/home/dzubke/awni_speech/data/speak_test_data/eval/eval2/eval2_data_2020-12-05.json"
    vm_string = "dzubke@phoneme-3:"
    full_dataset_string = vm_string + src_dataset_path
    dst_dir = "/Users/dustin/CS/work/speak/src/data/speak_eval/eval2/v1/"

    cmd_base = ["gcloud", "compute", "scp"]
    
    # ensure the dst directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # copy the datatset from the VM
    subprocess.run(cmd_base + [full_dataset_string, dst_dir])
    print(f"copied dataset {full_dataset_string} to {dst_dir}")

    dataset = read_data_json(os.path.join(dst_dir, os.path.basename(src_dataset_path)))

    data_subset = random.sample(dataset, sample_size)

    dst_audio_dir = os.path.join(dst_dir, "audio")
    os.makedirs(dst_audio_dir, exist_ok=True)
    print("destination audio dir: ", dst_audio_dir)
    for xmpl in data_subset:
        audio_path = xmpl['audio']
        full_audio_string = vm_string + audio_path
        dst_audio_path = os.path.join(dst_audio_dir, os.path.basename(audio_path))
        subprocess.run(cmd_base + [full_audio_string, dst_audio_path])







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download voxforge dataset.")

    parser.add_argument("--output-dir",
        help="The dataset is saved in <output-dir>/<dataset-name>.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with a capitalized first letter.")
    parser.add_argument("--config-path", type=str, default=None,
        help="Path to config file.")
    args = parser.parse_args()

    downloader = eval(args.dataset_name+"Downloader")(args.output_dir, args.dataset_name, args.config_path)
    print(f"type: {type(downloader)}")
    downloader.download_dataset()
