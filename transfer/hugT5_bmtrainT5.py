from collections import OrderedDict
import torch
from tqdm import tqdm

def main():
    ver = "11b"
    layernum = 24
    inpath = f"../results/t5-{ver}-pytorch_model.bin"
    outpath = f"../results/T5-{ver}.pt"
    scale = 100
    inp = torch.load(inpath)
    out = OrderedDict()
    out["input_embedding.weight"] = inp["shared.weight"].contiguous()
    with torch.no_grad(): out["input_embedding.weight"] /= scale
    out["output_projection.w.weight"] = inp["lm_head.weight"].contiguous()
    out["encoder.output_layernorm.weight"] = inp["encoder.final_layer_norm.weight"].contiguous()
    out["position_bias_enc.relative_attention_bias"] = inp["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].transpose(0,1).contiguous()
    out["decoder.output_layernorm.weight"] = inp["decoder.final_layer_norm.weight"].contiguous()
    out["position_bias_dec.relative_attention_bias"] = inp["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].transpose(0,1).contiguous()
    for i in range(layernum):
        prefix = f"encoder.layers.{i}"
        old_prefix = f"encoder.block.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.0.layer_norm.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"].contiguous() #[2*attn_project_size:]
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.self_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layer.1.layer_norm.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.layer.1.DenseReluDense.wi.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_in.w.weight"] /= scale**0.5
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.layer.1.DenseReluDense.wo.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_out.weight"] /= scale**0.5

    for i in range(layernum):
        prefix = f"decoder.layers.{i}"
        old_prefix = f"decoder.block.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.0.layer_norm.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"].contiguous() #[2*attn_project_size:]
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.self_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.cross_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.1.layer_norm.weight"].contiguous()
        out[f"{prefix}.cross_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.q.weight"].contiguous()
        out[f"{prefix}.cross_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.k.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.cross_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.v.weight"].contiguous() #[attn_project_size:]
        out[f"{prefix}.cross_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.cross_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layer.2.layer_norm.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.layer.2.DenseReluDense.wi.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_in.w.weight"] /= scale**0.5
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.layer.2.DenseReluDense.wo.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_out.weight"] /= scale**0.5

    for k, v in out.items():
        out[k] = out[k].half()

    torch.save(out, outpath)

if __name__ == "__main__":
    main()
