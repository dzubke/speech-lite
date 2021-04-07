"""This script aids in review audio. It needs to be run on a local machine as I don't know how to 
listen to audio on a VM. Use gcsfuse to create a connection to the spk_dataset google cloud bucket.
"""

# standard libs
import argparse
import os
import time
import subprocess
# third-party libs
# project libs
from speech.utils.io import read_data_json


def review_audio(data_path:str, gcsfuse_dir:str):
    """

    Args:
        data_path: path to dataset
        gcsfuse_dir: path to directory connected to gsc bucket
    """
    dataset = read_data_json(data_path)

    for xmpl in dataset:

        next_recording = False
        while not next_recording:
            
            print(xmpl['text'])
            time.sleep(0.5)
            plan_fn(xmpl['audio'], gcsfuse_dir)
            
            print("(n) next rec, (a) play again, (p) print full entry")    
            action = input()
            if action == 'n':
                next_recording = True
            elif action == 'a':
                pass
            elif action == 'p':
                print(xmpl)
            else:
                print("invalid entry")


def play_fn(audio_path, gcsfuse_dir):
    split_path = audio_path.split('/')
    # there is a subdir that exists on the VM but not in the gcs bucket so that is removed
    audio_path = os.path.join('/'.join(split_path[-5:-2]), split_path[-1])
    audio_path = os.path.join(gcsfuse_dir, audio_path)
    cmd = ['play', audio_path]
    
    subprocess.run(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Function that makes it easier to review audio files."
    )
    parser.add_argument(
        "--dataset-path", help="path to dataset"
    )
    parser.add_argument(
        "--gcsfuse-dir", help="path to directory connected to google storage bucket"
    )

    args = parser.parse_args()       

    review_audio(args.dataset_path, args.gcsfuse_dir)
