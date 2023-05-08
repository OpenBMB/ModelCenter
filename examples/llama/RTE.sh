#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/data/ModelCenter"
VERSION="7b"
DATASET="RTE"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/results/llama-${VERSION}"
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 1400"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 512"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-llama-${DATASET}"
OPTS+=" --lr 0.00001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 140"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/llama/finetune_llama.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/llama_superglue/finetune-llama-${VERSION}-${DATASET}.log
