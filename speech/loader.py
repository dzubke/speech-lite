# compatibility libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import copy
import json
import math
import random
from typing import List, Tuple
# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud
from torch.utils.data.distributed import DistributedSampler
# project libraries
from speech.utils.wave import array_from_wave
from speech.utils.io import read_data_json
from speech.utils.signal_augment import (
    inject_noise, synthetic_gaussian_noise_inject, tempo_gain_pitch_perturb
)
from speech.utils.feature_augment import apply_spec_augment



class Preprocessor():

    END = "</s>"
    START = "<s>"

    def __init__(self, data_json:list, preproc_cfg:dict, logger=None, max_samples:int=1000,
                 start_and_end=False):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            preproc_json: A json file defining the preprocessing with attributes
                preprocessor: "log_spec" or "mfcc" to determine the type of preprocessing
                window_size: the size of the window in the spectrogram transform
                step_size: the size of the step in the spectrogram transform
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
        """

        # if true, data augmentation will be applied
        self.train_status = True
        
        assert preproc_cfg['preprocessor'] in ['log_spectrogram', 'log_mel', 'mfcc'], \
            f"preprocessor name: {preproc_cfg['preprocessor']} is unacceptable"
        self.preprocessor = preproc_cfg['preprocessor']
        self.window_size = preproc_cfg['window_size']
        self.step_size = preproc_cfg['step_size']
        self.use_feature_normalize =  preproc_cfg['use_feature_normalize']
        self.augment_from_normal = preproc_cfg.get('augment_from_normal', False)

        self.tempo_gain_pitch_perturb = preproc_cfg['tempo_gain_pitch_perturb']
        self.tempo_gain_pitch_prob = preproc_cfg.get('tempo_gain_pitch_prob', 1.0)
        self.tempo_range = preproc_cfg['tempo_range']
        self.gain_range = preproc_cfg['gain_range']
        self.pitch_range = preproc_cfg['pitch_range']
       
        self.synthetic_gaussian_noise = preproc_cfg.get('synthetic_gaussian_noise', False)
        self.gauss_noise_prob = preproc_cfg.get('gauss_noise_prob', 1.0)
        self.gauss_snr_db_range = preproc_cfg.get('gauss_snr_db_range',
                                        preproc_cfg.get('signal_to_noise_range_db'))

        self.background_noise = preproc_cfg.get('background_noise', preproc_cfg.get('inject_noise'))
        self.noise_dir = preproc_cfg.get('background_noise_dir', preproc_cfg.get('noise_directory'))
        self.background_noise_prob = preproc_cfg.get('background_noise_prob', preproc_cfg.get('noise_prob'))
        self.background_noise_range = preproc_cfg.get('background_noise_range', preproc_cfg.get('noise_levels'))     
        
        self.spec_augment = preproc_cfg.get('spec_augment', preproc_cfg.get('use_spec_augment'))
        self.spec_augment_prob = preproc_cfg.get('spec_augment_prob', 1.0)
        self.spec_augment_policy = preproc_cfg['spec_augment_policy']

        # Compute data mean, std from sample
        data = read_data_json(data_json)
        audio_files = [sample['audio'] for sample in data]
        random.shuffle(audio_files)
        self.mean, self.std = compute_mean_std(audio_files[:max_samples],
                                                self.preprocessor, 
                                                window_size = self.window_size, 
                                                step_size = self.step_size,
                                                use_feature_normalize = self.use_feature_normalize
        )
        self._input_dim = self.mean.shape[0]
        self.use_log = (logger is not None)
        self.logger = logger


        # Make char map
        chars = sorted(list(set(label for datum in data for label in datum['text'])))
        if start_and_end:
            # START must be last so it can easily be
            # excluded in the output classes of a model.
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end

        assert preproc_cfg['blank_idx'] in ['first', 'last'], \
            f"blank_idx: {preproc_cfg['blank_idx']} must be either 'first' or 'last'"  
        # if the blank_idx is 'first' then the int_to_char must start at 1 as 0 is already reserved
        ## for the blank
        if preproc_cfg['blank_idx'] == 'first':
            start_idx = 1
        else:   # if the blank_idx is 'last', then the int_to_char can start at 0
            start_idx = 0

        self.int_to_char = dict(enumerate(chars, start_idx))  # start at 1 so zero can be blank for native loss
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}
    

    def preprocess(self, wave_file:str, text:List[str])->Tuple[np.ndarray, List[int]]:
        """Performs the feature-processing pipeline on the input wave file and text transcript.
        Args: 
            wave_file (str): path to wav file
            text (List[str]): a list of labels 
        
        Returns:
            feature_data (np.ndarray): a feature array augmented and processed by a log-spec 
                or mfcc transformations
        targets (List[int]): a list of the integer-encoded phoneme labels
        """
        if self.use_log: self.logger.info(f"preproc: ======= Entering preprocess =====")
        if self.use_log: self.logger.info(f"preproc: wave_file: {wave_file}")
        if self.use_log: self.logger.info(f"preproc: text: {text}") 

        audio_data, samp_rate = self.signal_augmentations(wave_file)

        # apply audio processing function
        feature_data = process_audio(audio_data, 
                                    samp_rate, 
                                    self.window_size, 
                                    self.step_size, 
                                    self.preprocessor)
        
        # normalize
        feature_data = self.normalize(feature_data)
        if self.use_log: self.logger.info(f"preproc: normalized")
        
        # apply feature_augmentations
        feature_data = self.feature_augmentations(feature_data)

        # target encoding
        targets = self.encode(text)
        if self.use_log: self.logger.info(f"preproc: text encoded")
        if self.use_log: self.logger.info(f"preproc: ======= Exiting preprocess =====")

        return feature_data, targets


    def signal_augmentations(self, wave_file:str)-> tuple:
        """
        Performs all of the augmtations to the raw audio signal. The audio data is in pcm16 format.
        Arguments:
            wave_file - str: the path to the audio sample
        Returns:
            audio_data - np.ndarray: augmented np-array
            samp_rate - int: sample rate of the audio recording
        """
        if self.use_log: self.logger.info(f"preproc: audio_data read: {wave_file}")
        
        audio_data, samp_rate = array_from_wave(wave_file)

        # sox-based tempo, gain, pitch augmentations
        if self.tempo_gain_pitch_perturb and self.train_status:
            if np.random.binomial(1, self.tempo_gain_pitch_prob): 
                audio_data, samp_rate = tempo_gain_pitch_perturb(wave_file, 
                                                                samp_rate, 
                                                                self.tempo_range,
                                                                self.gain_range, 
                                                                self.pitch_range, 
                                                                self.augment_from_normal, 
                                                                logger=self.logger)
                if self.use_log: self.logger.info(f"preproc: tempo_gain_pitch applied")

        # synthetic gaussian noise
        if self.synthetic_gaussian_noise and self.train_status:
            if np.random.binomial(1, self.gauss_noise_prob): 
                audio_data = synthetic_gaussian_noise_inject(audio_data, 
                                                            self.gauss_snr_db_range,
                                                            self.augment_from_normal, 
                                                            logger=self.logger)
                if self.use_log: self.logger.info(f"preproc: synth_gauss_noise applied")

        # noise injection
        if self.background_noise and self.train_status:
            if np.random.binomial(1, self.background_noise_prob):
                audio_data =  inject_noise(audio_data, 
                                            samp_rate, 
                                            self.noise_dir, 
                                            self.background_noise_range, 
                                            self.augment_from_normal, 
                                            self.logger) 
                if self.use_log: self.logger.info(f"preproc: noise injected")
        
        return audio_data, samp_rate


    def feature_augmentations(self, feature_data:np.ndarray)->np.ndarray:
        """
        Performs feature augmentations to the 2d array of features
        """
        # spec-augment
        if self.spec_augment and self.train_status:
            if np.random.binomial(1, self.spec_augment_prob):
                feature_data = apply_spec_augment(feature_data, 
                                                  self.spec_augment_policy, 
                                                  self.logger)
                if self.use_log: self.logger.info(f"preproc: spec_aug applied")

        return feature_data


    def normalize(self, feature_array:np.ndarray)->np.ndarray:
        if self.use_feature_normalize:
            feature_array = feature_normalize(feature_array)
        feature_array = (feature_array - self.mean) / self.std
        assert feature_array.dtype == np.float32, "feature_array is not float32"
        return feature_array


    def encode(self, text):
        text = list(text)
        if self.start_and_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]


    def decode(self, seq):
        try:
            text = [self.int_to_char[s] for s in seq]
        except KeyError as e:
            raise KeyError(f"Key Error in {seq} as {e}")
        if not self.start_and_end:
            return text

        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]


    def update(self):
        """
        Updates an old, saved instance with new attributes.
        """
        if not hasattr(self, 'tempo_gain_pitch_perturb'):
            if hasattr(self, 'speed_vol_perturb'):
                self.tempo_gain_pitch_perturb = self.speed_vol_perturb
                self.pitch_range = [0,0]    # no pitch augmentation
            else:
                self.tempo_gain_pitch_perturb = False
        if not hasattr(self, 'train_status'):
            self.train_status = True
        if not hasattr(self, 'synthetic_gaussian_noise'):
            self.synthetic_gaussian_noise = False
        if not hasattr(self, "gauss_snr_db_range"):
            self.gauss_snr_db_range=(100, 100)
        if self.preprocessor == "log_spec":
            self.preprocessor = "log_spectrogram"
        if not hasattr(self, 'background_noise'):
            self.background_noise = False
        if not hasattr(self, 'use_feature_normalize'):
            self.use_feature_normalize = False
        # removing the old attritube to separate feature_normalize
        # self.normalize is now a method
        if type(self.normalize) == str:
            del self.normalize

    def set_eval(self):
        """
        turns off the data augmentation for evaluation
        """
        self.train_status = False
        self.use_log = False

    def set_train(self):
        """
        turns on data augmentation for training
        """
        self.train_status = True


    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)

    def __str__(self):
        string = str()
        for name, value in vars(self).items():
            string += f"\n{name}: {value}"
        return string


def feature_normalize(feature_array:np.ndarray, eps=1e-7)->np.ndarray:
    """
    Normalizes the features so that the entire 2d input array
    has zero mean and unit (1) std deviation
    The first assert checks std is not zero. If it is zero, will get NaN
    """
    assert feature_array.dtype == np.float32, "feature_array is not float32"

    mean = feature_array.mean(dtype='float32')
    std = feature_array.std(dtype='float32')
    # the eps factor will prevent from getting NaN value, but the assert is just to surface
    # the the std value is zero
    assert std != 0, "feature_normalize: std dev is zero, may get NaN"
    assert std == std, "NaN value in feature array!"
    feature_array -= mean
    feature_array /= (std + eps)
    assert feature_array.dtype == np.float32, "feature_array is not float32"
    return feature_array


def compute_mean_std(audio_files: List[str],
                     preprocessor: str, 
                     window_size: int, 
                     step_size: int, 
                     use_feature_normalize:bool)->Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and std deviation of all of the feature bins (frequency bins if log_spec
    preprocessor). Will first normalize the audio samples if use_feature_normalize is true.
    Args:
        audio_files - List[str]: a list of shuffled audio files. len = max_samples
        preprocessor (str): name of preprocessor
        window_size - int: window_size of preprocessor
        step_size - int: step_size of preprocessor
        use_feature_normalize - bool: whether or not the features themselves are normalized
    Returns:
        mean - np.ndarray: the mean of the feature bins - shape = (# feature bins,)
        std  - np.ndarray: the std deviation of the feature bins - shape = (# bins,)
    """
    assert len(audio_files) > 0, "input list of audio_files is empty"

    samples = []
    for audio_file in audio_files: 
        audio_data, samp_rate = array_from_wave(audio_file)
        feature_array = process_audio(audio_data, samp_rate, window_size, step_size, preprocessor)
        if use_feature_normalize:
            feature_array = feature_normalize(feature_array)   # normalize the feature
        samples.append(feature_array)
    
    # compute mean and std dev of the feature bins (along axis=0)
    # feature arrays aka samples are time x feature bin
    samples = np.vstack(samples) # stacks along time axis: shape = (all_time, feature bin)
    mean = np.mean(samples, axis=0, dtype='float32') # computes mean along time axis: shape = (feature bin,)
    std = np.std(samples, axis=0, dtype='float32')
    return mean, std


class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):
        """
        this code sorts the samples in data based on the length of the transcript lables and the audio
        sample duration. It does this by creating a number of buckets and sorting the samples
        into different buckets based on the length of the labels. It then sorts the buckets based 
        on the duration of the audio sample.
        """

        data = read_data_json(data_json)        #loads the data_json into a list
        self.preproc = preproc                  # assign the preproc object

        bucket_diff = 4                             # number of different buckets
        max_len = max(len(x['text']) for x in data) # max number of phoneme labels in data
        num_buckets = max_len // bucket_diff        # the number of buckets
        buckets = [[] for _ in range(num_buckets)]  # creating an empy list for the buckets
        
        for sample in data:                          
            bucket_id = min(len(sample['text']) // bucket_diff, num_buckets - 1)
            buckets[bucket_id].append(sample)

        sort_fn = lambda x: (round(x['duration'], 1), len(x['text']))

        for bucket in buckets:
            bucket.sort(key=sort_fn)
        
        # unpack the data in the buckets into a list
        data = [sample for bucket in buckets for sample in bucket]
        self.data = data
        print(f"in AudioDataset: length of data: {len(data)}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        datum = self.preproc.preprocess(datum["audio"],
                                        datum["text"])
        return datum


class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        
        if len(data_source) < batch_size:
            raise ValueError("batch_size is greater than data length")

        it_end = len(data_source) - batch_size + 1
        self.batches = [
            range(i, i + batch_size) for i in range(0, it_end, batch_size)
        ]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


class DistributedBatchRandomSampler(DistributedSampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement with distributed data parallel
    compatibility.

    Args: 
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        batch_size (int): number of samples in batch

    Instructive to review parent class: 
        https://pytorch.org/docs/0.4.1/_modules/torch/utils/data/distributed.html#DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        
        if len(dataset) < batch_size:
            raise ValueError("batch_size is greater than data length")
        
        self.batch_size = batch_size
        self.n_batch_per_replica = int(math.floor(len(self.dataset)//batch_size * 1.0 / self.num_replicas))
        self.total_size = self.n_batch_per_replica * self.num_replicas
        
        # leaves off the last unfilled batch. the last batch shouldn't be filled from the initial values 
        # because the audio lengths will be very different
        it_end = len(dataset) - batch_size + 1
        self.batches = [
            range(i, i + batch_size) for i in range(0, it_end, batch_size)
        ]
        print(f"in DistBatchSamp: rank: {self.rank} dataset size: {len(self.dataset)}")
        print(f"in DistBatchSamp: rank: {self.rank} batch size: {batch_size}")
        print(f"in DistBatchSamp: rank: {self.rank} num batches: {len(self.batches)}")
        print(f"in DistBatchSamp: rank: {self.rank} num_replicas: {self.num_replicas}")
        print(f"in DistBatchSamp: rank: {self.rank} batches per replica: {self.n_batch_per_replica}")
        print(f"in DistBatchSamp: rank: {self.rank} iterator_end: {it_end}")
        
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch_indices = list(torch.randperm(len(self.batches), generator=g))
        print(f"in DistBatchSamp: rank: {self.rank} len batch_indices: {len(batch_indices)}")

        # add extra batches to make the total num batches evenly divisible by num_replicas
        batch_indices = batch_indices[:self.total_size]  #+= batch_indices[:(self.total_size - len(batch_indices))]
        print(f"in DistBatchSamp: rank: {self.rank} new len batch_indices: {len(batch_indices)}")
        assert len(batch_indices) == self.total_size

        # subsample the batches for individual replica based on rank
        offset = self.n_batch_per_replica * self.rank
        batch_indices = batch_indices[offset:offset + self.n_batch_per_replica]
        assert len(batch_indices) == self.n_batch_per_replica
        
        print(f"in DistBatchSamp: rank: {self.rank} batches per replica: {len(batch_indices)}")
        print(f"in DistBatchSamp: rank: {self.rank} total_size: {self.total_size}")
        print(f"in DistBatchSamp: rank: {self.rank} offset_begin: {offset} offset_end: {offset + self.n_batch_per_replica}")
        
        #assert all([self.batch_size == len(batch) for batch in self.batches]),\
        #    f"at least one batch is not of size: {self.batch_size}"

        return  (idx for batch_idx in batch_indices for idx in self.batches[batch_idx])

    def __len__(self):
        return self.n_batch_per_replica * self.batch_size


def make_loader(dataset_json, preproc,
                batch_size, num_workers=4):
    dataset = AudioDataset(dataset_json, preproc, batch_size)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True)
    return loader

def make_ddp_loader(dataset_json, 
                    preproc,
                    batch_size, 
                    num_workers=4):
    """Creates a load compatibile with distributed data parallel (ddp).
    """
    
    dataset = AudioDataset(dataset_json, preproc, batch_size)
    sampler = DistributedBatchRandomSampler(dataset, batch_size=batch_size)
    loader = tud.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True,
                pin_memory=True
    )
    return loader

class CustomBatch:
    """
    This class is based on: https://pytorch.org/docs/stable/data.html#memory-pinning. 
    It was used to implemented pinned memory to speed up training. I don't think it is 
    currently in use. 
    """
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)
 
def collate_fn(batch):  
    """This needed to be named function instead of an anonymous lambda function so the loader can be
    serialized during distributed data parallel training.
    """
    return zip(*batch)


#######    DATA PREPROCESSING    ########

def process_audio(audio, samp_rate:int, window_size=32, step_size=16, processing='log_spectrogram'):
    """Processes audio through the provided processing function.

    Args:
        audio (str or np.ndarray): path to audio or audio array
        samp_rate (int): sample rate of audio
        window_size (int): size of window in processing function
        step_size (int): step in processing function
        processing (str): name of processing function. 
            'log_spectogram', 'mfcc', and 'log_mel' are acceptable.
    Returns: 
        np.ndarray: processed array of dimensions: time x processor_bins
    """
    assert isinstance(audio, (str, np.ndarray)), \
        f"audio must be type str or np.ndarray, not {type(audio)}"

    # process audio from audio path
    if isinstance(audio, str):
        audio, samp_rate = array_from_wave(audio_path)

    audio = average_channels(audio)

    if processing == 'log_spectrogram':
        output = log_spectrogram(audio, samp_rate, window_size, step_size)
    elif processing == 'mfcc':
        output = mfcc(audio, samp_rate, window_size, step_size)
    elif processing == 'log_mel':
        output = log_mel_filterbank(audio, samp_rate, window_size, step_size)
    else:
        raise ValueError(f"processing value: {processing} is unacceptable")

    return output


def mfcc(audio, sample_rate: int, window_size, step_size):
    """Returns the mfcc's as well as the first and second order deltas.
    Hanning window used in mfccs for parity with log_spectrogram function.

    Args:
        audio (np.ndarray): audio signal array
        sample_rate (int): sample_rate of signal
        window_size (int): window size
        step_size (int): step size

    Returns:
        np.ndarray: log mel filterbank, delta, and delta-deltas
    """
    delta_window = 1
    mfcc = python_speech_features.mfcc( audio, 
                                        sample_rate, 
                                        winlen=window_size/1000, 
                                        winstep=step_size/1000,
                                        winfunc=np.hanning
    )
    delta = python_speech_features.delta(mfcc, N=delta_window)
    delta_delta = python_speech_features.delta(delta, N=delta_window) 
    output = np.concatenate((mfcc, delta, delta_delta), axis=1)

    return output.astype(np.float32)


def log_spectrogram(audio, sample_rate, window_size, step_size, eps=1e-10):
    """
    Computes the log of the spectrogram for input audio. Hanning window is used.
    Dimensions are time x freq. The step size is converted into the overlap noverlap.
    
    Arguments:
        audio_data (np.ndarray)
    Returns:
        np.ndarray: log of the spectrogram as returned by log_specgram
            transposed so dimensions are time x frequency    
    """
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int( (window_size - step_size) * sample_rate / 1e3)
    f, t, spec = scipy.signal.spectrogram(  audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False
    )
    return np.log(spec.T.astype(np.float32) + eps)


def log_mel_filterbank(audio, sample_rate, window_size, step_size):
    """Returns the log of the mel filterbank energies as well as the first and second order deltas.
    Hanning window used for parity with log_spectrogram function.

    Args:
        audio (np.ndarray): audio signal array
        sample_rate (int): sample_rate of signal
        window_size (int): window size
        step_size (int): step size

    Returns:
        np.ndarray: log mel filterbank, delta, and delta-deltas
    """
    delta_window=1
    log_mel = python_speech_features.base.logfbank(  audio,
                                                    sample_rate,
                                                    winlen=window_size/1000,
                                                    winstep=step_size/1000,
                                                    winfunc=np.hanning
    )
    delta = python_speech_features.delta(log_mel, N=delta_window)
    delta_delta = python_speech_features.delta(delta, N=delta_window) 
    output = np.concatenate((log_mel, delta, delta_delta), axis=1)

    return output.astype(np.float32)


def average_channels(audio):
    """This function will return an audio file averaged across channels if multiple channels exist
    """
    
    if len(audio.shape)>1:     # there are multiple channels
        if audio.shape[1] == 1:
            audio = audio.squeeze()
        else:
            audio = audio.mean(axis=1, dtype='float32')  # multiple channels, average

    return audio


def compare_log_spec_from_file(audio_file_1: str, audio_file_2: str, plot=False):
    """
    This function takes in two audio paths and calculates the difference between the spectrograms 
        by subtracting them. 
    """
    audio_1, sr_1 = array_from_wave(audio_file_1)
    audio_2, sr_2 = array_from_wave(audio_file_2)

    if len(audio_1.shape)>1:
        audio_1 = audio_1[:,0]  # take the first channel
    if len(audio_2.shape)>1:
        audio_2 = audio_2[:,0]  # take the first channel
    
    window_size = 20
    step_size = 10

    nperseg_1 = int(window_size * sr_1 / 1e3)
    noverlap_1 = int(step_size * sr_1 / 1e3)
    nperseg_2 = int(window_size * sr_2 / 1e3)
    noverlap_2 = int(step_size * sr_2 / 1e3)

    freq_1, time_1, spec_1 = scipy.signal.spectrogram(audio_1,
                    fs=sr_1,
                    window='hann',
                    nperseg=nperseg_1,
                    noverlap=noverlap_1,
                    detrend=False)

    freq_2, time_2, spec_2 = scipy.signal.spectrogram(audio_2,
                    fs=sr_2,
                    window='hann',
                    nperseg=nperseg_2,
                    noverlap=noverlap_2,
                    detrend=False)
    
    spec_diff = spec_1 - spec_2 
    freq_diff = freq_1 - freq_2
    time_diff = time_1 - time_2

    if plot:
        plot_spectrogram(freq_diff, time_diff, spec_diff)
        #plot_spectrogram(freq_1, time_1, spec_2)
        #plot_spectrogram(freq_2, time_2, spec_2)
    
    return spec_diff


def plot_spectrogram(f, t, Sxx):
    """This function plots a spectrogram using matplotlib

    Arguments
    ----------
    f: the frequency output of the scipy.signal.spectrogram
    t: the time series output of the scipy.signal.spectrogram
    Sxx: the spectrogram output of scipy.signal.spectrogram

    Returns
    --------
    None

    Note: the function scipy.signal.spectrogram returns f, t, Sxx in that order
    """
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
