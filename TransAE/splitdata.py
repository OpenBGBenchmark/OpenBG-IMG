import os
import random


if not os.path.exists('./data'):
    os.mkdir('./data/')

if not os.path.exists('./data/OpenBG-IMG'):
    os.mkdir('./data/OpenBG-IMG/')

src_path = '../data/OpenBG-IMG/'
tgt_path = './data/OpenBG-IMG/'

random.seed(123)

with open(os.path.join(src_path, 'train.tsv'), 'r') as fp:
    train_data = fp.readlines()

random.shuffle(train_data)

#split = len(train_data) - 1
split = len(train_data) - len(train_data) / 20.0
split = int(split)

new_train = train_data[:split]
new_valid = train_data[split:]

with open(os.path.join(tgt_path, 'train.tsv'), 'w') as fp:
    fp.writelines(new_train)

with open(os.path.join(tgt_path, 'valid.tsv'), 'w') as fp:
    fp.writelines(new_valid)

with open(os.path.join(src_path, 'test.tsv'), 'r') as fp:
    test_data = fp.readlines()
    new_test = []
    for line in test_data:
        data = line[:-1].split('\t')
        data.append('ent_000000')
        data = '\t'.join(data)
        new_test.append(data+'\n')

with open(os.path.join(tgt_path, 'test.tsv'), 'w') as fp:
    fp.writelines(new_test)
