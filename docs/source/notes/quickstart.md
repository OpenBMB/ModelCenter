# Quick Start

ModelCenter are examples of model implemented based on [BMTrain](https://bmtrain.readthedocs.io/en/latest/index.html).

```python
# init bmtrain backend
import bmtrain as bmt
bmt.init_distributed()

# get model and tokenizer
from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased")
t5 = Bert.from_pretrained("bert-base-uncased")

# get optimizer
optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())

# get learning rate scheduler
lr_scheduler = bmt.lr_scheduler.NoDecay(
    optimizer, 
    start_lr = 1e-5,
    warmup_iter = 100, 
    end_iter = -1,
    num_iter = 1000,
)

# get loss function
loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
```

For training:

TODO data processing

```python
optimizer.zero_grad()

# model forward
logits = model(input_ids, attention_mask, return_logits=True)

# calc loss
loss = loss_func(logits, labels)

# scale loss to avoid precision underflow of fp16
loss = optimizer.loss_scale(loss)

# model backward
loss.backward()

# clip gradient norm. with loss scale, the clip_grad_norm function is slightly different from torch.nn.utils.clip_grad_norm_
grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

# change optimizer.step() to bmt.optim_step(optimizer)
bmt.optim_step(optimizer, lr_scheduler)

# print information only on rank 0 of the distributed training
# bmt.sum_loss(loss) to gather all loss information from other processes
bmt.print_rank(
    "loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
        bmt.sum_loss(loss).item(),
        lr_scheduler.current_lr,
        int(optimizer.scale),
        grad_norm,
    )
)
```