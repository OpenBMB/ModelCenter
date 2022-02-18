with open("vocab_origin.txt", "r") as f:
    lines = f.readlines()

for i in range(190):
    lines.append("<s_{}>".format(i))

with open("vocab.txt", "w") as f:
    for line in lines:
        f.write(line.strip() + "\n")
