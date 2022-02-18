from collections import OrderedDict
import torch

def main():
    inp = torch.load("../../CPM-1/results/noam-1e-3-0.01-checkpoint-74500.pt")
    print(inp.keys())
    for i in range(32):
        prefix = f"encoder.layers.{i}"
        for j in range(2):
            fr = f"{prefix}.ffn.ffn.w_{j}"
            to = f"{prefix}.ffn.ffn.w_in.w_{j}.weight"
            inp[to] = inp[fr]
            inp.pop(fr)


        fr = f"{prefix}.ffn.ffn.w_out"
        to = f"{prefix}.ffn.ffn.w_out.weight"
        inp[to] = inp[fr]
        inp.pop(fr)
        
        fr = f"{prefix}.self_att.self_attention.project_q"
        to = f"{prefix}.self_att.self_attention.project_q.weight"
        inp[to] = inp[fr]
        inp.pop(fr)

        fr = f"{prefix}.self_att.self_attention.project_k"
        to = f"{prefix}.self_att.self_attention.project_k.weight"
        inp[to] = inp[fr]
        inp.pop(fr)

        fr = f"{prefix}.self_att.self_attention.project_v"
        to = f"{prefix}.self_att.self_attention.project_v.weight"
        inp[to] = inp[fr]
        inp.pop(fr)

        fr = f"{prefix}.self_att.self_attention.attention_out"
        to = f"{prefix}.self_att.self_attention.attention_out.weight"
        inp[to] = inp[fr]
        inp.pop(fr)

    torch.save(inp, "../results/CPM1-new.pt")

if __name__ == "__main__":
    main()