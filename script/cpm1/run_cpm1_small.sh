#! /bin/bash

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/mnt/sfs_turbo/hx/ModelCenter"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/configs/cpm1/cpm1-small"
OPTS+=" --batch-size 128"
OPTS+=" --train-iters 500000"
OPTS+=" --save-iters 500"
OPTS+=" --max-length 512"
OPTS+=" --save ${BASE_PATH}/results/"
OPTS+=" --save-name small"
OPTS+=" --lr 0.25"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 500"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2097152"
OPTS+=" --start-step 73001"
OPTS+=" --load ${BASE_PATH}/results/small-73000.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/pretrain_cpm1.py ${OPTS}"
echo ${CMD}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} | tee ${BASE_PATH}/logs/small_noam.log
else
    ${CMD}
fi
