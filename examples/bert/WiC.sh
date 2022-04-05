#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/home/hx/lyq/BigModels"
VERSION="bert-large-cased"
DATASET="WiC"

OPTS=""
OPTS+=" --model-config ${VERSION}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 64"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr 0.00005"
OPTS+=" --max-decoder-length 512"
OPTS+=" --train-iters 400"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --loss-scale 128"

python3 -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS} 
