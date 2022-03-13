from collections import OrderedDict
import torch

def main():
    inp = torch.load("results/noam-xxlargelr-ckpt-110000.pt")
    out = OrderedDict()
    out["input_embedding.weight"] = inp["input_embedding.weight"]
    out["output_projection.weight"] = inp["output_projection.weight"]
    out["encoder.output_layernorm.weight"] = inp["layernorm_after_enc.weight"]
    out["position_bias_enc.relative_attention_bias"] = inp["position_bias_enc.weight"]
    out["decoder.output_layernorm.weight"] = inp["layernorm_after_dec.weight"]
    out["position_bias_dec.relative_attention_bias"] = inp["position_bias_dec.weight"]
    for i in range(24):
        prefix = f"encoder.layers.{i}"
        old_prefix = f"enc_layers.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layernorm_before_attention.weight"]
        out[f"{prefix}.self_att.self_attention.project_q"] = inp[f"{old_prefix}.self_attention.project_q"]
        out[f"{prefix}.self_att.self_attention.project_k"] = inp[f"{old_prefix}.self_attention.project_k"]
        out[f"{prefix}.self_att.self_attention.project_v"] = inp[f"{old_prefix}.self_attention.project_v"]
        out[f"{prefix}.self_att.self_attention.attention_out"] = inp[f"{old_prefix}.self_attention.attention_out"]
        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layernorm_before_ff.weight"]
        out[f"{prefix}.ffn.ffn.w_0.weight"] = inp[f"{old_prefix}.ff.w_0"]
        out[f"{prefix}.ffn.ffn.w_1.weight"] = inp[f"{old_prefix}.ff.w_1"]
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.ff.w_out"]

    for i in range(24):
        prefix = f"decoder.layers.{i}"
        old_prefix = f"dec_layers.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layernorm_before_self_attention.weight"]
        out[f"{prefix}.self_att.self_attention.project_q"] = inp[f"{old_prefix}.self_attention.project_q"]
        out[f"{prefix}.self_att.self_attention.project_k"] = inp[f"{old_prefix}.self_attention.project_k"]
        out[f"{prefix}.self_att.self_attention.project_v"] = inp[f"{old_prefix}.self_attention.project_v"]
        out[f"{prefix}.self_att.self_attention.attention_out"] = inp[f"{old_prefix}.self_attention.attention_out"]

        out[f"{prefix}.cross_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layernorm_before_cross_attention.weight"]
        out[f"{prefix}.cross_att.self_attention.project_q"] = inp[f"{old_prefix}.cross_attention.project_q"]
        out[f"{prefix}.cross_att.self_attention.project_k"] = inp[f"{old_prefix}.cross_attention.project_k"]
        out[f"{prefix}.cross_att.self_attention.project_v"] = inp[f"{old_prefix}.cross_attention.project_v"]
        out[f"{prefix}.cross_att.self_attention.attention_out"] = inp[f"{old_prefix}.cross_attention.attention_out"]
        
        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layernorm_before_ff.weight"]
        out[f"{prefix}.ffn.ffn.w_0.weight"] = inp[f"{old_prefix}.ff.w_0"]
        out[f"{prefix}.ffn.ffn.w_1.weight"] = inp[f"{old_prefix}.ff.w_1"]
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.ff.w_out"]

    torch.save(out, "another_migrate.pt")

if __name__ == "__main__":
    main()