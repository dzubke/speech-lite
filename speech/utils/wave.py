# compatibility libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import multiprocessing as mp
from pathlib import Path
# third-party libraries   
import numpy as np
import soundfile

def array_from_wave(file_name:str):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    return audio, samp_rate

def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration
 
def array_to_wave(filename:str, audio_data:np.ndarray, samp_rate:int):
    """
    Writes an array to wave in the in the signed int-16 subtype (PCM_16)
    """
    soundfile.write(filename, audio_data, samp_rate, subtype='PCM_16', format='WAV')


def corpus_audio_duration(
    corpus_dir: str, 
    audio_ext:str='wav', 
    workers:int=3,
    chunk_size:int=5000)->int:
    """Calculates the total duration of all audio files (in seconds) with the input extension
    contained in the corpus directory.

    Args:
        corpus_dir (str): path to corpus directory
        audio_ext (str): audio extension, default 'wav'
        workers (int): number of processes to run the 
        chunk_size (int): size of chunks to feed into multiprocessing pool
    """

    corpus_dir = Path(corpus_dir)

    # remove the period if present
    if '.' in audio_ext:
        audio_ext = audio_ext.replace('.', '')

    audio_files = corpus_dir.rglob("*." + audio_ext)

    duration_list = list()
    with mp.Pool(processes=workers) as pool:
        while True:
            audio_chunk = list()
            try:
                for _ in range(chunk_size):
                    audio_chunk.append(next(audio_files))
            except StopIteration:
                out_list = pool.map(wav_duration, audio_chunk)
                duration_list.extend(out_list)
                break
            
            out_list = pool.map(wav_duration, audio_chunk)
            duration_list.extend(out_list)
        
        pool.close()
        pool.join() 

    return sum(duration_list)


def simple_audio_duration(corpus_dir: str, audio_ext:str='wav')->int:
    """Single process implementation of audio duration.
    Calculates the total duration of all audio files (in seconds) with the input extension
    contained in the corpus directory.

    Args:
        corpus_dir (str): path to corpus directory
        audio_ext (str): audio extension, default 'wav'
    """
    corpus_dir = Path(corpus_dir)
    audio_files = corpus_dir.rglob("*." + audio_ext)
    total_duration = 0.0
    for audio_file in audio_files:
        total_duration += wav_duration(audio_file)
    return total_duration
    
