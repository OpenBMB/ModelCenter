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

BASE_PATH="/home/hx/lyq/BigModels"
VERSION="multilingual"

OPTS=""
OPTS+=" --model-config ${BASE_PATH}/configs/bert/bert-multilingual"
OPTS+=" --load ${BASE_PATH}/results/BERT-${VERSION}.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/bert_test.py ${OPTS}"
echo ${CMD}

${CMD}
