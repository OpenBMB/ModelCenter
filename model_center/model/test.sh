export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
python -m debugpy --listen 0.0.0.0:5678 -m torch.distributed.launch --master_port 2224 --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost test_vit.py
# torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --master_port 2228 --rdzv_backend=c10d --rdzv_endpoint=localhost test_vit.py
