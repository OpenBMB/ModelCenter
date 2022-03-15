#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/home/hx/lyq/BigModels"
#VERSION="bert-base-uncased"
#VERSION="bert-large-uncased"
#VERSION="bert-base-cased"
#VERSION="bert-large-cased"
#VERSION="bert-base-multilingual-cased"
VERSION="bert-base-chinese"

OPTS=""
OPTS+=" --model-config ${BASE_PATH}/configs/bert/${VERSION}"
OPTS+=" --load ${BASE_PATH}/results/${VERSION}.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/run_bert.py ${OPTS}"
echo ${CMD}

${CMD}