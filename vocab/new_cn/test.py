#coding:utf-8

with open("vocab.txt", "r") as f:
    content = f.readlines()
    hash = {}
    for i in content:
        i = i.strip().decode("utf-8")
        hash[i.strip()]=1

with open("gg.txt", "r") as f:
    content = f.readlines()
    for i in content:
        i = i.strip().decode("utf-8")
        if not i in hash:
            print (i)