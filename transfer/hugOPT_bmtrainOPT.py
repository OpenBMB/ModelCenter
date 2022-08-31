# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
import torch
from tqdm import tqdm
import json

def main():
    vers = ["125m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b"]
    for ver in vers:
        inpath = f"/yinxr/zwl/.cache/transformers/opt-{ver}/pytorch_model.bin"
        config_inpath = f"/yinxr/zwl/.cache/transformers/opt-{ver}/config.json"
        outpath = f"/yinxr/zwl/.cache/model_center/opt-{ver}/pytorch_model.pt"
        config_outpath = f"/yinxr/zwl/.cache/model_center/opt-{ver}/config.json"
        inp = torch.load(inpath)
        inp_config = json.load(open(config_inpath, "r"))
        out = OrderedDict()
        out_config = OrderedDict()

        out_config["vocab_size"] = inp_config["vocab_size"]
        out_config["dim_model"] = inp_config["hidden_size"]
        out_config["num_heads"] = inp_config["num_attention_heads"]
        out_config["dim_head"] = out_config["dim_model"] // out_config["num_heads"]
        out_config["dim_ff"] = inp_config["ffn_dim"]
        out_config["num_layers"] = inp_config["num_hidden_layers"]
        out_config["dropout_p"] = inp_config["dropout"]
        out_config["emb_init_mean"] = 0.0
        out_config["emb_init_std"] = inp_config["init_std"]
        out_config["pos_bias_type"] = "none"
        out_config["pad_token_id"] = inp_config["pad_token_id"]
        out_config["position_size"] = inp_config["max_position_embeddings"]
        out_config["att_init_mean"] = 0.0
        out_config["att_init_std"] = inp_config["init_std"]
        out_config["att_bias"] = True
        out_config["ffn_init_mean"] = 0.0
        out_config["ffn_init_std"] = inp_config["init_std"]
        out_config["ffn_bias"] = True
        out_config["ffn_activate_fn"] = inp_config["activation_function"]
        out_config["post_layer_norm"] = not inp_config["do_layer_norm_before"]

        out["input_embedding.weight"] = inp["model.decoder.embed_tokens.weight"].contiguous()
        out["position_embedding.weight"] = inp["model.decoder.embed_positions.weight"].contiguous()
        out["decoder.output_layernorm.weight"] = inp["model.decoder.final_layer_norm.weight"].contiguous()
        out["decoder.output_layernorm.bias"] = inp["model.decoder.final_layer_norm.bias"].contiguous()
        for i in range(inp_config["num_hidden_layers"]):
            prefix = f"decoder.layers.{i}"
            old_prefix = f"model.decoder.layers.{i}"

            out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.self_attn_layer_norm.weight"].contiguous()
            out[f"{prefix}.self_att.layernorm_before_attention.bias"] = inp[f"{old_prefix}.self_attn_layer_norm.bias"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.self_attn.q_proj.weight"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_q.bias"] = inp[f"{old_prefix}.self_attn.q_proj.bias"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.self_attn.k_proj.weight"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_k.bias"] = inp[f"{old_prefix}.self_attn.k_proj.bias"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.self_attn.v_proj.weight"].contiguous()
            out[f"{prefix}.self_att.self_attention.project_v.bias"] = inp[f"{old_prefix}.self_attn.v_proj.bias"].contiguous()
            out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.self_attn.out_proj.weight"].contiguous()
            out[f"{prefix}.self_att.self_attention.attention_out.bias"] = inp[f"{old_prefix}.self_attn.out_proj.bias"].contiguous()

            out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.final_layer_norm.weight"].contiguous()
            out[f"{prefix}.ffn.layernorm_before_ffn.bias"] = inp[f"{old_prefix}.final_layer_norm.bias"].contiguous()
            out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.fc1.weight"].contiguous()
            out[f"{prefix}.ffn.ffn.w_in.w.bias"] = inp[f"{old_prefix}.fc1.bias"].contiguous()
            out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.fc2.weight"].contiguous()
            out[f"{prefix}.ffn.ffn.w_out.bias"] = inp[f"{old_prefix}.fc2.bias"].contiguous()
        for k, v in out.items():
            out[k] = out[k].half()

        torch.save(out, outpath)
        json.dump(out_config, open(config_outpath, "w"), indent=4)

if __name__ == "__main__":
    main()
