from collections import OrderedDict
import torch
from tqdm import tqdm

def main():
    layernum = 48
    inpath = f"../results/glm-10b_origin.pt"
    outpath = f"../results/glm-10b.pt"
    inp = torch.load(inpath)['module']
    out = OrderedDict()
    out["input_embedding.weight"] = inp["word_embeddings.weight"].cpu().detach().clone().contiguous()
    out["position_embedding.weight"] = inp["transformer.position_embeddings.weight"].cpu().detach().clone().contiguous()
    out["block_position_embedding.weight"] = inp["transformer.block_position_embeddings.weight"].cpu().detach().clone().contiguous()
    out["encoder.output_layernorm.weight"] = inp["transformer.final_layernorm.weight"].cpu().detach().clone().contiguous()
    out["encoder.output_layernorm.bias"] = inp["transformer.final_layernorm.bias"].cpu().detach().clone().contiguous()
    for i in range(layernum):
        prefix = f"encoder.layers.{i}"
        old_prefix = f"transformer.layers.{i}"
        attn_size = inp[f"{old_prefix}.attention.query_key_value.weight"].shape[0]//3
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.input_layernorm.weight"].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.layernorm_before_attention.bias"] = inp[f"{old_prefix}.input_layernorm.bias"].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.attention.query_key_value.weight"][:attn_size, :].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.bias"] = inp[f"{old_prefix}.attention.query_key_value.bias"][:attn_size].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.attention.query_key_value.weight"][attn_size:2*attn_size, :].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_k.bias"] = inp[f"{old_prefix}.attention.query_key_value.bias"][attn_size:2*attn_size].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.attention.query_key_value.weight"][2*attn_size:, :].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.project_v.bias"] = inp[f"{old_prefix}.attention.query_key_value.bias"][2*attn_size:].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.attention.dense.weight"].cpu().detach().clone().contiguous()
        out[f"{prefix}.self_att.self_attention.attention_out.bias"] = inp[f"{old_prefix}.attention.dense.bias"].cpu().detach().clone().contiguous()

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.post_attention_layernorm.weight"].cpu().detach().clone().contiguous()
        out[f"{prefix}.ffn.layernorm_before_ffn.bias"] = inp[f"{old_prefix}.post_attention_layernorm.bias"].cpu().detach().clone().contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.mlp.dense_h_to_4h.weight"].cpu().detach().clone().contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.bias"] = inp[f"{old_prefix}.mlp.dense_h_to_4h.bias"].cpu().detach().clone().contiguous()
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.mlp.dense_4h_to_h.weight"].cpu().detach().clone().contiguous()
        out[f"{prefix}.ffn.ffn.w_out.bias"] = inp[f"{old_prefix}.mlp.dense_4h_to_h.bias"].cpu().detach().clone().contiguous()

        drop_keys = [key for key in inp.keys() if key.startswith(old_prefix+'.')]
        print(drop_keys)
        for key in drop_keys: inp.pop(key)

    torch.save(out, outpath)

if __name__ == "__main__":
    main()
