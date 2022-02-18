import os
from tqdm import tqdm
import random
import json
import re

input_dir = "/home/hanwentao/dataset/zhihu/final"
output_dir = "/home/hanwentao/dataset/preprocessed"

fw = open(os.path.join(output_dir, "zhihu_train.txt"), "w")

p1 = re.compile(r"(<n>)+")

for _, _, files in os.walk(input_dir):
    for file in tqdm(files):
        if file[0] == ".":
            continue
        
        with open(os.path.join(input_dir, file), "r") as f:
            for line in f:
                jobj = json.loads(line)

                q_title = jobj["q_title"].strip()
                if jobj["q-content"] is not None:
                    q_content = jobj["q-content"].strip()
                else:
                    q_content = ""
                ans_content = jobj["ans-content"].strip()

                if random.random() < 0.5:
                    text = "问题：" + q_title + "<n>"+ "描述：" + q_content + "<n>" + "答案：" + ans_content
                else:
                    text = ans_content

                text = text.replace("[图片]", "")
                text = p1.sub("<n>", text)
                if len(text) < 8:
                    pass
                else:
                    fw.write(text + "\n")
        
fw.close()
