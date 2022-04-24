import os, re

data_path = "./data/OpenBG-IMG/"

with open(os.path.join(data_path, 'ent_id'), 'r') as fp:
    entity2id = fp.readlines()
    entity2id = [re.findall(r'(.+?)\t(.+?)\n',i)[0] for i in entity2id]
    ent2id = {a[1]:a[0] for a in entity2id}

with open(os.path.join(data_path, 'rel_id'), 'r') as fp:
    relation2id = fp.readlines()
    relation2id = [re.findall(r'(.+?)\t(.+?)\n',i)[0] for i in relation2id]
    rel2id = {a[1]:a[0] for a in relation2id}

with open('./results/result.txt', 'r') as fp:
    res = fp.readlines()

new_res = []
for i in res:
    resline = i[:-1].split(' ')
    new_resline = []
    for idx, li in enumerate(resline):
        if idx == 1:
            new_resline.append(rel2id[li])
        else:
            new_resline.append(ent2id[li])
    new_resline = '\t'.join(new_resline) + '\n'
    new_res.append(new_resline)


with open('./results/result.tsv', 'w') as fp:
    fp.writelines(new_res)