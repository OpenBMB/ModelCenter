# Quick Start

ModelCenter are examples of model implemented based on `BMTrain <https://bmtrain.readthedocs.io/en/latest/index.html>`_ .

```python
# init bmtrain backend
import bmtrain as bmt
bmt.init_distributed()

# get model and tokenizer
from model_center.tokenizer import T5Tokenizer
from model_center.model import T5Config, T5

tokenizer = T5Tokenizer.from_pretrained("bert-base-uncased")
config = T5Config.from_pretrained("bert-base-uncased")
t5 = T5.from_pretrained("bert-base-uncased")

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

```python
optimizer.zero_grad()

logits = model(input_ids, attention_mask, return_logits=True)

loss = loss_func(logits, labels)

loss = optimizer.loss_scale(loss)

loss.backward()

grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

bmt.optim_step(optimizer, lr_scheduler)

bmt.print_rank(
    "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
        epoch,
        it,
        len(dataloader["train"]),
        global_loss,
        lr_scheduler.current_lr,
        int(optimizer.scale),
        grad_norm,
    )
)
```