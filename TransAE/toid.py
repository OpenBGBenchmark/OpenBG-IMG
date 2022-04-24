import os, re

src_path = './data/OpenBG-IMG/'
path = lambda x: os.path.join(src_path, x)
file_list = ['train','valid','test']
out = './data/OpenBG-IMG/'
path2 = lambda x: os.path.join(out, x)

all = []
allset = set()
for file in file_list:
    with open(path(file+'.tsv'), 'r') as fp:
        data = fp.readlines()
        all.append(data)
        allset.update(data)


def toid(data):
    ent2id = {}
    id2ent = {}
    ent_num = 0
    rel2id = {}
    id2rel = {}
    t_ids = set()
    rel_num = 0
    def updateDict(idto, toid, num, item):
        if item in toid.keys():
            return toid[item], num
        toid[item] = num
        idto[num] = item
        return num, num+1
    update_ent = lambda num, item: updateDict(id2ent, ent2id, num, item)
    update_rel = lambda num, item: updateDict(id2rel, rel2id, num, item)
    
    LL = []
    for ii in data:
        L = []
        for aa in ii:
            h,r,t = re.findall('(.+?)\t(.+?)\t(.+?)\n',aa)[0]
            h_id, ent_num = update_ent(ent_num, h)
            t_id, ent_num = update_ent(ent_num, t)
            r_id, rel_num = update_rel(rel_num, r)
            t_ids.add(f'{t_id}\n')
            L.append(f'{h_id} {t_id} {r_id}\n')
        L = [f'{len(L)}\n'] + L
        LL.append(L)
    assert len(ent2id) == len(id2ent)
    assert len(rel2id) == len(id2rel)
    Ent = [f'{len(ent2id)}\n']
    for i in range(len(ent2id)):
        Ent.append(f'{id2ent[i]}\t{i}\n')
    Rel = [f'{len(rel2id)}\n']
    for i in range(len(rel2id)):
        Rel.append(f'{id2rel[i]}\t{i}\n')

    return LL, Ent, Rel, list(t_ids)


all, ent2id, rel2id, t_ids = toid(all)

with open(path2('entity2id.txt'), 'w') as fp:
    fp.writelines(ent2id)

with open(path2('relation2id.txt'), 'w') as fp:
    fp.writelines(rel2id)

# with open(path2('t_ids.txt'), 'w') as fp:
#     fp.writelines(t_ids)

for idx, file in enumerate(file_list):
    with open(path2(file+'2id.txt'), 'w') as fp:
        fp.writelines(all[idx])
    
