export MODE=$1  # name
export MODEL=$2 # model to start with
export LANG=$3  # lang to finetune on
echo "DOING $MODE with $MODEL on $LANG"

mkdir -p logs


export MAX_LENGTH=200
# where is the data
MAIN_DATA_DIR=../../../data/masakhane-ner/data
TMP_DATA_DIR=$MAIN_DATA_DIR

# If you want to save, e.g. nfs storage performance, then you can copy the data over to /tmp and use it from there.
# # MAIN_DATA_DIR=../../../data/masakhane-ner/data/$LANG
# TMP_DATA_DIR=/tmp/username/mner/$LANG"_"$MODE"_"$NUM_EPOCHS
# mkdir -p $TMP_DATA_DIR
# rsync -r  $MAIN_DATA_DIR $TMP_DATA_DIR

# Some params
export BERT_MODEL=$MODEL
export NUM_EPOCHS=50
export OUTPUT_DIR=models/v20/$LANG"_"$MODE"_"$NUM_EPOCHS
export BATCH_SIZE=1 # you can make this larger if needed
export SAVE_STEPS=10000
export SEED=1 # just evaluation, no training.

mkdir -p $OUTPUT_DIR
python3 evaluate_ner.py --data_dir $TMP_DATA_DIR/$LANG \
--model_type xlmroberta \
--learning_rate 5e-5 \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--gradient_accumulation_steps 16 \
--do_predict 2>&1 | tee logs/$MODE"_"$NUM_EPOCHS"_"`date +%s`