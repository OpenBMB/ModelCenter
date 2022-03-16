# How to write a new model

A classic transformer is implemented in the following structure:

Parameters are wrapped by [bmtrain.DistributedParameter](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.DistributedParameter),
Modules basically inherit [bmtrain.DistributedModule](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.DistributedModule),
Transformer blocks are wrapped in [bmtrain.CheckpointBlock](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.CheckpointBlock),
Repeated Transformer blocks are in [bmtrain.TransformerBlockList](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.TransformerBlockList).

For more information, see [BMTrain's Quick Start](https://bmtrain.readthedocs.io/en/latest/notes/quickstart-zh.html)

```
T5(
  (input_embedding): Embedding()
  (position_bias_enc): RelativePositionEmbedding()
  (position_bias_dec): RelativePositionEmbedding()
  (encoder): Encoder(
    (layers): bmtrain.TransformerBlockList(
      (0): bmtrain.CheckpointBlock(
        TransformerBlock(
          (self_att): SelfAttentionBlock(
            (layernorm_before_attention): LayerNorm()
            (attention): Attention(
              (project_q): Linear()
              (project_k): Linear()
              (project_v): Linear()
              (attention_out): Linear()
            )
          )
          (ffn): FFNBlock(
            (layernorm_before_ffn): LayerNorm()
            (ffn): FeedForward(
              (w_in): DenseACT(
                (w): Linear()
                (act): ReLU()
              )
              (w_out): Linear()
            )
          )
        )
      )
      (1): bmtrain.CheckpointBlock()
      .
      .
      .
    )
    (output_layernorm): LayerNorm()
  )
  (decoder): Decoder(
    (layers): bmtrain.TransformerBlockList(
      (0): bmtrain.CheckpointBlock(
        (self_att): SelfAttentionBlock(
          (layernorm_before_attention): LayerNorm()
          (attention): Attention(
            (project_q): Linear()
            (project_k): Linear()
            (project_v): Linear()
            (attention_out): Linear()
          )
        )
        (cross_att): CrossAttentionBlock(
          (layernorm_before_attention): LayerNorm()
          (attention): Attention(
            (project_q): Linear()
            (project_k): Linear()
            (project_v): Linear()
            (attention_out): Linear()
          )
        )
        (ffn): FFNBlock(
          (layernorm_before_ffn): LayerNorm()
          (ffn): FeedForward(
            (w_in): DenseACT(
              (w): Linear()
              (act): ReLU()
            )
            (w_out): Linear()
          )
        )
      )
      (1): bmtrain.CheckpointBlock()
      .
      .
      .
    )
    (output_layernorm): LayerNorm()
  )
  (output_projection): Projection(
    (w): Linear(
      (weight): bmtrain.DistributedParameter()
      (bias): bmtrain.DistributedParameter()
    )
  )
)
```