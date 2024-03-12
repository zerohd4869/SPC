#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/SPC"
MODEL_BASE="/SPC/ptms/roberta-base"

model_name="spc"
VV="a100-2401-st-${model_name}"
SEED="0 1 2 3 4"
DATA_PER="1.0"
LR="5e-5"
WARMUP_RATIO=0.1
BS=128
EPOCH_NUM=20
PASTIENT_NUM=5
MAX_LEN=128


# ==============================================================================
ALL_TASK_NAME="isear_v3"
task_type="cls"
VAR_W="10"
CLU_W="0.1"
L2="0"
DP="0.2"


for task_name in ${ALL_TASK_NAME[@]}
do
for l2 in ${L2[@]}
do
for dp in ${DP[@]}
do
for var_w in ${VAR_W[@]}
do
for clu_w in ${CLU_W[@]}
do
for seed in ${SEED[@]}
do
EXP_NO="${VV}_${task_name}_var${var_w}-clu${clu_w}_d${DATA_PER}_lr${LR}_l2${l2}_WA${WARMUP_RATIO}_bs${BS}_dp${dp}_len${MAX_LEN}_s${seed}"
OUT_DIR="${WORK_DIR}/outputs/${model_name}/${task_name}/${task_name}"
LOG_PATH="${WORK_DIR}/logs/${model_name}/${task_name}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi
echo "VV: ${VV}"
echo "TASK_NAME: ${task_name}"
echo "EXP_NO: ${EXP_NO}"
echo "OUT_DIR: ${OUT_DIR}"
echo "LOG_DIR: ${LOG_PATH}/${EXP_NO}.out"

#--normalize_flag \
python -u ${WORK_DIR}/main.py   \
--epochs    ${EPOCH_NUM} \
--patience  ${PASTIENT_NUM} \
--seed      ${seed} \
--fine_tune_task        ${task_name} \
--task_type             ${task_type} \
--dataset_percentage    ${DATA_PER} \
--lr        ${LR}   \
--weight_decay          ${l2}   \
--warmup_ratio  ${WARMUP_RATIO} \
--bs        ${BS}   \
--dropout   ${dp}   \
--max_length    ${MAX_LEN} \
--pretrained_model_path     ${MODEL_BASE} \
--output_dir    ${OUT_DIR} \
--var_weight    ${var_w} \
--clu_weight    ${clu_w} \
--batch_sampling_flag \
>> ${LOG_PATH}/${EXP_NO}.out
done
done
done
done
done
done
