#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/mnt/sfs_turbo/hx/ModelCenter"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
# OPTS+=" --model-config ${BASE_PATH}/configs/cpm1/cpm1-large"
OPTS+=" --model-config ${BASE_PATH}/configs/cpm1/cpm1-small"
OPTS+=" --batch-size 52"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --max-length 512"
OPTS+=" --save ${BASE_PATH}/results"
# OPTS+=" --load ${BASE_PATH}/results/noam-0.25-0.01-checkpoint-22000.pt"
OPTS+=" --load ${BASE_PATH}/results/small-46500.pt"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/infer.py ${OPTS}"
echo ${CMD}

${CMD} | tee ${BASE_PATH}/logs/new_noam.log
