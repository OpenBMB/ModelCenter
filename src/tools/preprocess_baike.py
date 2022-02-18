import json
import os
from tqdm import tqdm
from pprint import pprint

input_dir = "/home/hanwentao/dataset/563w_baidubaike.json"
output_dir = "/home/hanwentao/dataset/preprocessed"

fw = open(os.path.join(output_dir, "baike_train.txt"), "w")

with open(input_dir, "r") as f:
    for line in tqdm(f):
        jobj = json.loads(line)
        if jobj["summary"] is not None:
            text = jobj["summary"].replace("\n", "<n>") + "\n"
            if len(text) < 8:
                pass
            else:
                fw.write(text)
        for s in jobj["sections"]:
            text = s["content"].replace("\n", "<n>") + "\n"
            if len(text) < 8:
                pass
            else:
                fw.write(text)
