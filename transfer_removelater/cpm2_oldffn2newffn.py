from collections import OrderedDict
import torch

def main():
    inp = torch.load("../../CPM-1/results/another_migrate.pt")
    for i in range(24):
        for j in range(2):
            prefix = f"encoder.layers.{i}"
            fr = f"{prefix}.ffn.ffn.w_{j}"
            to = f"{prefix}.ffn.ffn.w_in.w_{j}.weight"
            print(f"{fr} -> {to}")
            inp[to] = inp[fr]
            inp.pop(fr)
    for i in range(24):
        for j in range(2):
            prefix = f"decoder.layers.{i}"
            fr = f"{prefix}.ffn.ffn.w_{j}"
            to = f"{prefix}.ffn.ffn.w_in.w_{j}.weight"
            print(f"{fr} -> {to}")
            inp[to] = inp[fr]
            inp.pop(fr)

    torch.save(inp, "../results/CPM2-0.25-0.005-checkpoint-110000.pt")

if __name__ == "__main__":
    main()