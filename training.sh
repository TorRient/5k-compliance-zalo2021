# Example: ./training.sh train_5k train_mask train_distancing train
DIR_MODEL_5K='/model/train_5k'
DIR_MODEL_MASK='/model/train_mask'
DIR_MODEL_DISTANCING='/model/train_distancing'
DIR_TRAIN='/model/train'

# DIR_MODEL_5K=$1
# DIR_MODEL_MASK=$2
# DIR_MODEL_DISTANCING=$3
# DIR_TRAIN=$4
python3 main_mask.py --dir_model_mask $DIR_MODEL_MASK --dir_train $DIR_TRAIN

python3 main_distancing.py --dir_model_distancing $DIR_MODEL_DISTANCING --dir_model_mask $DIR_MODEL_MASK --dir_train $DIR_TRAIN

python3 main_5k.py --dir_model_5k $DIR_MODEL_5K --dir_model_distancing $DIR_MODEL_DISTANCING --dir_train $DIR_TRAIN

