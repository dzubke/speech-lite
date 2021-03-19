#!/bin/bash
# this command should be called from inside the model_convert/ directory
# command structure: bash cp_convert_zip.sh <base_path_to_model> <model_name> <num_frames> 

# assigning the arguments

display_help(){
    echo "$(basename "$0") [-mp MODEL_PATH] [-mn MODEL_NAME] [-nf NUM_FRAMES] 
            (options) [--quarter-precision] [--half-precision] [--best]" >& 2
    echo "call this function inside the <main>/model_convert directory"
    echo "where:
            -mp or --model-path:  is the path to the model directory
            -mn or --model-name: is the model name
            -nf or --num-frames: is the number of frames
            -hs or --hidden-size: size of RNN hidden state
            --quarter-precision: for quarter precision
            --half-precision:    for half precision
            -t or --tag: can be 'best' or 'ckpt'
    "
}


while [[ $# -gt 0 ]]
do
    key="$1"
    VALUE="$2"

    case $key in
        -mp|--model-path)
        MODEL_PATH="$VALUE"
        shift # past argument
        shift # past value
        ;;
        -mn|--model-name)
        MODEL_NAME="$VALUE"
        shift # past argument
        shift # past value
        ;;
        -nf|--num-frames)
        NUM_FRAMES="$VALUE"
        shift # past argument
        shift # past value
        ;;
        -t|--tag)
        TAG="$VALUE"_
        shift # past argument
        shift # past value
        ;;
        --half-precision)
        HALF_PRECISION="--half-precision"
        shift # past argument
        ;;
        --quarter-precision)
        QUARTER_PRECISION="--quarter-precision"
        shift # past argument
        ;;
        --help|-h)
        display_help
        exit 1
        ;;
        -*)
        echo "Error: Unknown option: $1" >&2
        display_help
        exit 1
        ;;
    esac
done


# creating the functions

copy_files(){

    MODEL_PATH=$1
    MODEL_NAME=$2
    TAG=$3
    echo "tag is ${TAG}"
    cp ${MODEL_PATH}/${TAG}model_state_dict.pth ./torch_models/${MODEL_NAME}_model.pth
    echo "copied ${MODEL_PATH}/${TAG}model_state_dict.pth to ./torch_models/${MODEL_NAME}_model.pth"
    cp ${MODEL_PATH}/${TAG}preproc.pyc ./preproc/${MODEL_NAME}_preproc.pyc
    cp ${MODEL_PATH}/*.yaml ./config/${MODEL_NAME}_config.yaml
}

convert_model(){
    
    MODEL_NAME=$1
    NUM_FRAMES=$2
    HALF_PRECISION=$3
    QUARTER_PRECISION=$4

    echo "precision used: " $HALF_PRECISION $QUARTER_PRECISION

    # the `sed` commands were used when the import was include in the model class. they are not needed anymore
    #sed -i '' 's/import functions\.ctc/#import functions\.ctc/g' ../speech/models/ctc_model_train.py
    python torch_to_onnx.py --model-name "$MODEL_NAME" --num-frames "$NUM_FRAMES"
    python onnx_to_coreml.py "$MODEL_NAME" $HALF_PRECISION $QUARTER_PRECISION

    # print a warning about `validation.py` if half or quarter precision is used
    if [ "$HALF_PRECISION" = "--half-precision" ] || [ "$QUARTER_PRECISION" = "--quarter-precision" ]
    then
    echo "
    ~~~ CoreML model converted to lower precision. Probabilities in validation.py may not fully agree. ~~~
    "
    fi
    # validate the model conversion
    python validation.py $MODEL_NAME --num-frames $NUM_FRAMES

    # the `sed` commands were used when the import was include in the model class. they are not needed anymore
    #sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ../speech/models/ctc_model_train.py
}

zip_files(){
    # $1 IS MODEL_NAME

    echo "Zipping $1.zip"
    zip -j ./zips/$1.zip ./coreml_models/$1_model.mlmodel ./preproc/$1_metadata.json ./config/$1_config.yaml ./output/$1_output.json
}

cleanup(){
    # is run if SIGINT is sent
    # ensure the modified ctc_model_train file is put back to original state
    sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ../speech/models/ctc_model_train.py
}


# function  execution

# runs the cleanup finction if SIGINT is entered
trap cleanup SIGINT

copy_files $MODEL_PATH $MODEL_NAME $TAG

convert_model $MODEL_NAME $NUM_FRAMES $HALF_PRECISION $QUARTER_PRECISION

zip_files $MODEL_NAME

