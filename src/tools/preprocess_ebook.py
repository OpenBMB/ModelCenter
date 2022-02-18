import os
from tqdm import tqdm

input_dir = "/home/hanwentao/dataset/e-book/txt_books"
output_dir = "/home/hanwentao/dataset/preprocessed"

fw = open(os.path.join(output_dir, "ebook_train.txt"), "w")

for _, _, files in os.walk(input_dir):
    for file in tqdm(files):
        if file[0] == ".":
            continue
        doc = []
        with open(os.path.join(input_dir, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    text = "<n>".join(doc) + "\n"
                    if len(text) < 8:
                        pass
                    else:
                        fw.write(text)
                    doc = []
                else:
                    doc.append(line)

        if len(doc) > 0:
            text = "<n>".join(doc) + "\n"
            if len(text) < 8:
                pass
            else:
                fw.write(text)
            doc = []

fw.close()
