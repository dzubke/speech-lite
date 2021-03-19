#!/bin/bash
# Run `source setup.sh` from this directory.
export PYTHONPATH=`pwd`:`pwd`/libs/warp-ctc-naren/pytorch_binding:`pwd`/libs:$PYTHONPATH

export PYTHONPATH=`pwd`/libs/warp-ctc/pytorch_binding:$PYTHONPATH

export PYTHONPATH=`pwd`/libs/apex:$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc/build

#export PYTHONPATH=$PYTHONPATH:/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc-naren/build
