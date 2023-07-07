TOT_CUDA="0"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="11451"

DATA_PATH="insturct.json" 
OUTPUT_PATH="checkpoints/"
MODEL_PATH="/remote-home/share/MOSS_7B_Base"
TEST_SIZE=1
use_zero_offload=1
if [ ${use_zero_offload} == "1" ]
then
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT finetune_fp16.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --eval_steps 200 \
    --save_steps 200 \
    --test_size $TEST_SIZE \
    --deepspeed
else
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_fp16.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --eval_steps 200 \
    --save_steps 200 \
    --test_size $TEST_SIZE
fi
