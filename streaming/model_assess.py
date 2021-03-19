# This code aims to assess the inference time for different model sizes and architectures
# The architectures to be tested include: 
#   - original models = CNN/LSTM
#   - CNN/GRU
#   - CNN/RNN
#   - CNN only
# 
# Different sizes to test include:
#   - regular model = (32-32-96 CNN/ 512-unit LSTM)
#   - large model = (64-64-128 CNN/ 1024-unit LSTM)
#   
#

# standard libs
import time
# third party libs
import torch
from prettytable import PrettyTable
# local libs
#from streaming.streaming_validation import full_audio_infer, list_chunk_infer_full_chunks
from speech.models import model, ctc_model


INPUT_DIM = 257
OUTPUT_DIM = 39


def main(no_cuda:bool=False):

    print('Initializing models...')
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    print(f"device: {device}")
    model_dict = initialize_models()

    audio_dur = 10 #seconds
    feature_step = 0.016 # seconds, 16 ms
    dummy_data = torch.randn((1, int(audio_dur/ feature_step), INPUT_DIM), dtype=torch.float32)
    table = PrettyTable(["Model name", "Time (s)", r"% realtime", "Parameters"])

    for model_name, model in model_dict.items():
        model.eval()
        model.to(device)
        dummy_data = dummy_data.to(device)

        infer_time = time.time()
        output = model(dummy_data)
        infer_time = time.time() - infer_time

        total_params = simple_parameter_count(model)

        table.add_row([
            model_name,
            round(infer_time, 3),
            round(infer_time/audio_dur*100, 2),
            f'{total_params:,}'
        ])

        #detailed_parameter_count(model)
    table.sortby = "Time (s)"
    print(table)

def initialize_models():

    # dict to contain all the models
    model_dict = dict()


    ###############  CREATE REGULAR MODEL ########################
    reg_cnn_lstm_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [32, 11, 41, 1, 2, 0, 20],
                [32, 11, 21, 1, 2, 0, 10],
                [96, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'LSTM',
                'dim': 512,
                'bidirectional': False,
                'layers': 5
            }
        }
    }

    reg_cnn_lstm_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        reg_cnn_lstm_cfg
    )

    model_dict.update({"reg_cnn_lstm": reg_cnn_lstm_model})


    ###############  CREATE LARGE MODEL ########################3
    large_cnn_lstm_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'LSTM',
                'dim': 1024,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    large_cnn_lstm_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_cnn_lstm_cfg
    )

    model_dict.update({"large_cnn_lstm": large_cnn_lstm_model})


    ###############  CREATE SMALL MODEL ########################3
    small_cnn_lstm_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [16, 11, 41, 1, 2, 0, 20],
                [16, 11, 21, 1, 2, 0, 10],
                [48, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'LSTM',
                'dim': 256,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    small_cnn_lstm_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        small_cnn_lstm_cfg
    )

    model_dict.update({"small_cnn_lstm": small_cnn_lstm_model})


    ###############  CREATE REG GRU MODEL ########################3
    reg_cnn_gru_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [32, 11, 41, 1, 2, 0, 20],
                [32, 11, 21, 1, 2, 0, 10],
                [96, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'GRU',
                'dim': 512,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    reg_cnn_gru_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        reg_cnn_gru_cfg
    )

    model_dict.update({"reg_cnn_gru": reg_cnn_gru_model})


    ###############  CREATE LARGE GRU MODEL ########################3
    large_cnn_gru_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'GRU',
                'dim': 1024,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    large_cnn_gru_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_cnn_gru_cfg
    )

    model_dict.update({"large_cnn_gru": large_cnn_gru_model})

    ###############  CREATE REGULAR RNN MODEL ########################3
    reg_cnn_rnn_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [32, 11, 41, 1, 2, 0, 20],
                [32, 11, 21, 1, 2, 0, 10],
                [96, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'RNN',
                'dim': 512,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    reg_cnn_rnn_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        reg_cnn_rnn_cfg
    )

    model_dict.update({"reg_cnn_rnn": reg_cnn_rnn_model})


    ###############  CREATE LARGE RNN MODEL ########################3
    large_cnn_rnn_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'type': 'RNN',
                'dim': 1024,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    large_cnn_rnn_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_cnn_rnn_cfg
    )

    model_dict.update({"large_cnn_rnn": large_cnn_rnn_model})


    ###############  CREATE CNN ONLY MODEL ########################3
    large_cnn_only_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'use_rnn': False,
                'type': 'RNN',
                'dim': 0,
                'bidirectional': False,
                'layers': 0
            }
    }   }

    large_cnn_only_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_cnn_only_cfg
    )

    model_dict.update({"large_cnn_only": large_cnn_only_model})


    ###############  CREATE LSTM ONLY MODEL ########################3
    large_lstm_only_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'use_conv': False,
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'use_rnn': True,
                'type': 'LSTM',
                'dim': 1024,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    large_lstm_only_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_lstm_only_cfg
    )

    model_dict.update({"large_lstm_only": large_lstm_only_model})


    ###############  CREATE XL LSTM ONLY MODEL ########################3
    xl_lstm_only_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'use_conv': False,
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
                [64, 11, 21, 1, 2, 0, 10],
                [128, 11, 21, 1, 1, 0, 10]
            ],
            'rnn': {
                'use_rnn': True,
                'type': 'LSTM',
                'dim': 2048,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    xl_lstm_only_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        xl_lstm_only_cfg
    )

    model_dict.update({"xl_lstm_only": xl_lstm_only_model})


     ###############  CREATE LARGE LSTM SMALL CNN MODEL ########################3
    large_lstm_small_cnn_cfg = {
        'dropout': 0.4,
        'blank_idx': 'last',
        'encoder': {
            'use_conv': True,
            'conv':[
                [64, 11, 41, 1, 2, 0, 20],
            ],
            'rnn': {
                'use_rnn': True,
                'type': 'LSTM',
                'dim': 1024,
                'bidirectional': False,
                'layers': 5
            }
    }   }

    large_lstm_small_cnn_model = ctc_model.CTC(
        INPUT_DIM,
        OUTPUT_DIM,
        large_lstm_small_cnn_cfg
    )

    model_dict.update({"large_lstm_small_cnn": large_lstm_small_cnn_model})

    return model_dict




def detailed_parameter_count(model):
    """This function is taken from:
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    # count the parameters by layer groups
    total_params = 0
    param_count_dict = dict()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: 
            name_base = name.split('.')[0]
            param_count = parameter.numel()
            param_count_dict[name_base] =  param_count_dict.get(name_base, 0) + param_count
            total_params += param_count

    # create and print the table
    table = PrettyTable(["Modules", "Parameters"])
    for name_base, count in param_count_dict.items():
        table.add_row([name_base, f'{count:,}'])
    table.add_row(['Total', f'{total_params:,}'])
    print(table)
    

def simple_parameter_count(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total Trainable Params: {total_params:,}")
    return total_params



class CNN_GRU(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super().__init__(freq_dim, config)

        self.fc = model.LinearND(self.encoder_dim, output_dim + 1)
    
    def forward(self, inputs):
        inputs = self.encode(inputs)
        outputs = self.fc(inputs)
        return outputs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Tests the timing for a variety of model architectures.")
    parser.add_argument(
         '--no-cuda', action='store_true', help="uses the cpu even if cuda is available"
    )
    ARGS = parser.parse_args()
    main(ARGS.no_cuda)
    
