#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6002
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
DATASET="CB"

OPTS=""
OPTS+=" --model-config ${VERSION}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 16"
OPTS+=" --lr 0.00001"
OPTS+=" --max-decoder-length 512"
OPTS+=" --train-iters 400"
OPTS+=" --lr-decay-style constant"
OPTS+=" --warmup-iters 40"
OPTS+=" --weight-decay 2e-3"
OPTS+=" --loss-scale 128"

#CMD="python3 >${LOGFILE} -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS} ${CMDOPTS}"
#echo ${CMD}

python3 -u -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS} ${CMDOPTS} #>${LOGFILE}
