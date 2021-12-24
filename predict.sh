# Example: ./predict.sh train_5k/ train_mask/ public_test/ submition.csv
# DIR_MODEL_5K=$1
# DIR_MODEL_MASK=$2
# DIR_TEST=$3
# OUTPUT_SUBMIT=$4
# python3 predict.py --dir_model_5k $DIR_MODEL_5K --dir_model_mask $DIR_MODEL_MASK --dir_test $DIR_TEST -- output_submit $OUTPUT_SUBMIT
./predict.sh /model/train_5k/ /model/train_mask/ /model/public_test/ /result/submition.csv