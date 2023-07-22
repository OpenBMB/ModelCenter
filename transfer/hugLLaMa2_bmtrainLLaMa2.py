from transformers import LlamaConfig
import torch, os
import json
from collections import OrderedDict

ver_layernum = [
    # "7b",
    "13b",
]

ver = ver_layernum[0]

inpath = f"../results/llama-2-{ver}-hf"
outpath = f"../results/llama-2-{ver}"

hf_config = LlamaConfig.from_pretrained(inpath)
config = {
    'dim_model': hf_config.hidden_size,
    'dim_ff': hf_config.intermediate_size,
    'num_layers': hf_config.num_hidden_layers,
    'num_heads': hf_config.num_attention_heads,
    'num_heads_kv': hf_config.num_key_value_heads,
    'dim_head': hf_config.hidden_size // hf_config.num_attention_heads,
    'norm_eps': hf_config.rms_norm_eps,
}
with open(os.path.join(outpath, "config.json"), 'w') as f:
    json.dump(config, f)

layernum = config['num_layers']

model_hf = OrderedDict()
ckpt_num = None
for name in os.listdir(inpath):
    if name.startswith("pytorch_model-") and name.endswith(".bin"):
        ckpt_num = int(name[-9:-4])
for i in range(1, ckpt_num + 1):
    part = torch.load(os.path.join(inpath, f"pytorch_model-{i:05d}-of-{ckpt_num:05d}.bin"))
    model_hf.update(part)

out = OrderedDict()

out["input_embedding.weight"] = model_hf['model.embed_tokens.weight'].contiguous()
out["encoder.output_layernorm.weight"] = model_hf['model.norm.weight'].contiguous()
out['output_projection.weight'] = model_hf['lm_head.weight'].contiguous()
for lnum in range(layernum):
    hf_pfx = f"model.layers.{lnum}"
    bmt_pfx = f"encoder.layers.{lnum}"
    
    out[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"] = model_hf[f"{hf_pfx}.input_layernorm.weight"].contiguous()

    out[f"{bmt_pfx}.self_att.self_attention.project_q.weight"] = model_hf[f"{hf_pfx}.self_attn.q_proj.weight"].contiguous()
    out[f"{bmt_pfx}.self_att.self_attention.project_k.weight"] = model_hf[f"{hf_pfx}.self_attn.k_proj.weight"].contiguous()
    out[f"{bmt_pfx}.self_att.self_attention.project_v.weight"] = model_hf[f"{hf_pfx}.self_attn.v_proj.weight"].contiguous()
    out[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"] = model_hf[f"{hf_pfx}.self_attn.o_proj.weight"].contiguous()

    out[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"] = model_hf[f"{hf_pfx}.post_attention_layernorm.weight"].contiguous()

    out[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"] = model_hf[f"{hf_pfx}.mlp.gate_proj.weight"].contiguous()
    out[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"] = model_hf[f"{hf_pfx}.mlp.up_proj.weight"].contiguous()

    out[f"{bmt_pfx}.ffn.ffn.w_out.weight"] = model_hf[f"{hf_pfx}.mlp.down_proj.weight"].contiguous()
    
    
for key in out:
    out[key] = out[key].half()

if not os.path.exists(outpath):
    os.makedirs(outpath)
torch.save(out, os.path.join(outpath, "pytorch_model.pt"))
