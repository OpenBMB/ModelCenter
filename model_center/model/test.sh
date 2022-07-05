export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=1   vit_base.py
