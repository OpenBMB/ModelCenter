import json
import os
from tqdm import tqdm
import re

p = re.compile(r"\n+ *")

input_dir = "/home/hanwentao/dataset/novel_train.jsonl"
output_dir = "/home/hanwentao/dataset/preprocessed"

with open(input_dir, "r") as f:
    lines = f.readlines()

with open(os.path.join(output_dir, "novel_train.txt"), "w") as f:
    for line in tqdm(lines):
        text = json.loads(line)["text"]
        text = p.sub("<n>", text)
        if len(text) < 8:
            continue
        f.write(text.strip() + "\n")
