# standard libraries
import argparse
import glob
import os
from typing import List
# third-party libraries
import editdistance
import numpy as np
import torch
# project libraries
from speech.models.ctc_decoder import decode as ctc_decode
from speech.utils.convert import to_numpy
from speech.utils.io import read_data_json, read_pickle
from onnx_to_coreml import onnx_to_coreml
from torch_to_onnx import torch_to_onnx


def compare_precision(model_name:str, 
                      num_frames:int, 
                      half_precision:bool, 
                      quarter_precision:bool,
                      audio_dir:str):
    """
    Arguments
    ---------
    model_name: str
        filename of the model
    num_frames: int
        number of feature frames that will fix the model's size
    half_precision: bool
        whether to convert the coreml model to half precision
    quarter_precision: bool
        whether to conver the coreml model to quarter precision
    audio_dir: str
        path to the directory containing the audio files to evaluate the models
    """

    BLANK_INDEX=39

    torch_model, _ = torch_to_onnx(model_name, 
                                    num_frames, 
                                    use_state_dict=True, 
                                    return_models=True)
    

    _, coreml_model_single_prec = onnx_to_coreml(model_name, 
                                                half_precision=False,
                                                quarter_precision=False,
                                                return_or_save='return')

    _, coreml_model_half_prec = onnx_to_coreml(model_name, 
                                                half_precision=True,
                                                quarter_precision=False,
                                                return_or_save='return')

    _, coreml_model_quarter_prec = onnx_to_coreml(model_name,
                                                half_precision=False,
                                                quarter_precision=True,
                                                return_or_save='return')
    
    models_to_test = {
        "single-precision": coreml_model_single_prec,
        "half-precison": coreml_model_half_prec,
        "quarter-precision": coreml_model_quarter_prec
    }

    # ready the models for evaluation
    torch_model.eval()

    # load and ready preproc
    preproc_path = os.path.join(os.getcwd(), "preproc", model_name+"_preproc.pyc")
    preproc = read_pickle(preproc_path)
    preproc.use_log = False
    preproc.set_eval()
    
    # TODO load data_json
    audio_paths = get_audio_paths(audio_dir)

    for model_name, coreml_model in models_to_test.items():

        total_dist = 0      # total distance between model predictions
        total_len = 0       # total number of labels 

        for audio_path in audio_paths:
            test_h = np.zeros((5, 1, 512)).astype(np.float32)
            test_c = np.zeros((5, 1, 512)).astype(np.float32)

            # empty label
            label = []
                
            feature_array, label = process_audio(preproc, audio_path, label, num_frames)
        
            # if the time dimension doesn't match num_frames (is too short), skip it
            if feature_array.shape[1] != num_frames:
                continue

            # running inference on torch model
            torch_output = torch_model(torch.from_numpy(feature_array),
                                        (torch.from_numpy(test_h), torch.from_numpy(test_c))
                                        )
            # unpacking the torch_ouput and converting to numpy
            torch_probs, torch_h, torch_c = to_numpy(torch_output[0]), to_numpy(torch_output[1][0]), to_numpy(torch_output[1][1])
            torch_preds_int = ctc_decode(torch_probs[0], beam_size=50, blank=BLANK_INDEX)
            # converting the integers to character labels
            torch_preds = preproc.decode(torch_preds_int[0])

            # get coreml predictions
            coreml_input = {'input': feature_array, 'hidden_prev': test_h, 'cell_prev': test_c}
            coreml_output = coreml_model.predict(coreml_input, useCPUOnly=True)
            coreml_probs = np.array(coreml_output['output'])
            coreml_h = np.array(coreml_output['hidden'])
            coreml_c = np.array(coreml_output['cell'])
            coreml_preds_int = ctc_decode(coreml_probs[0], beam_size=50,blank=BLANK_INDEX)
            coreml_preds = preproc.decode(coreml_preds_int[0])

            dist = editdistance.eval(torch_preds, coreml_preds)
            total_dist += dist
            total_len += len(torch_preds)

        print(f"\n ========== {model_name} ==========")
        print("total distance: ", total_dist)
        print("total number of labels: ", total_len)
        print("ratio of distance to total labels: ", round(total_dist/ total_len, 3))


def process_audio(preproc, audio_path, label, num_frames):

        feature_array, label = preproc.preprocess(audio_path, label)
        # segmenting to only include the first num_frames
        feature_array = feature_array[:num_frames,:]
        # adding batch dimension to features
        feature_array = np.expand_dims(feature_array, 0)

        return feature_array, label


def get_audio_paths(audio_dir:str)->List[str]:
    pattern = "*.w*v"
    return glob.glob(os.path.join(audio_dir, pattern))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares the performance of models at different precisions.")
    parser.add_argument("--model-name", help="name of the model.")
    parser.add_argument("--num-frames", type=int, help="number of input frames in time dimension hard-coded in onnx model")
    parser.add_argument("--audio-dir", type=str, help="path to the dataset.json file used for validation.")
    parser.add_argument("--half-precision", action='store_true', default=False,  help="converts the model to half precision.")
    parser.add_argument("--quarter-precision", action='store_true', default=False, help="converts the model to quarter precision.")
    args = parser.parse_args()

    compare_precision(args.model_name, args.num_frames, args.half_precision, 
                        args.quarter_precision, args.audio_dir)
