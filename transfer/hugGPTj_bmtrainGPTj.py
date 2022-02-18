from collections import OrderedDict
import torch
from tqdm import tqdm

def main():
    ver = "6b"
    layernum = 28
    inpath = f"../results/gptj-{ver}-pytorch_model.bin"
    outpath = f"../results/GPTj-{ver}.pt"
    inp = torch.load(inpath)
    out = OrderedDict()
    out["input_embedding.weight"] = inp["transformer.wte.weight"].contiguous() # original vocab size is an odd number
    out["output_projection.w.weight"] = inp["lm_head.weight"].contiguous()
    out["output_projection.w.bias"] = inp["lm_head.bias"].contiguous()
    out["decoder.output_layernorm.weight"] = inp["transformer.ln_f.weight"].contiguous()
    out["decoder.output_layernorm.bias"] = inp["transformer.ln_f.bias"].contiguous()
    for i in range(layernum):
        prefix = f"decoder.layers.{i}"
        old_prefix = f"transformer.h.{i}"
        # parallel, share the same layernorm
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.ln_1.weight"].contiguous()
        out[f"{prefix}.self_att.layernorm_before_attention.bias"] = inp[f"{old_prefix}.ln_1.bias"].contiguous()
        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.ln_1.weight"].contiguous()
        out[f"{prefix}.ffn.layernorm_before_ffn.bias"] = inp[f"{old_prefix}.ln_1.bias"].contiguous()

        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.attn.q_proj.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.attn.k_proj.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.attn.v_proj.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.attn.out_proj.weight"].contiguous()

        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.mlp.fc_in.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.bias"] = inp[f"{old_prefix}.mlp.fc_in.bias"].contiguous()
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.mlp.fc_out.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_out.bias"] = inp[f"{old_prefix}.mlp.fc_out.bias"].contiguous()

    for k, v in out.items():
        out[k] = out[k].half()

    torch.save(out, outpath)

if __name__ == "__main__":
    main()
