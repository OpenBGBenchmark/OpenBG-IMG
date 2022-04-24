from pathlib import Path
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from models import KBCModel
import os


# DATA_PATH = Path(pkg_resources.resource_filename('k bc', 'data/'))
DATA_PATH=Path('data')

class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            # print(str(self.root / (f + '.pickle')))
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)

        inp_f.close()
        # inp_f = open(str(self.root / f'mul_hot.pickle'), 'rb')
        # self.mul_hot: [[]]= pickle.load(inp_f)
        # inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        # mul_hot = np.zeros((2*len(self.data['train']), self.n_entities))
        # for i in range(len(self.data['train'])):
        #     print(i)
        #     mul_hot[i,self.to_skip['rhs'][(self.data['train'][i,0],self.data['train'][i,1])]]=1
        #     mul_hot[i+len(self.data['train']), self.to_skip['lhs'][(self.data['train'][i, 2], self.data['train'][i, 1]+self.n_predicates //2 )]] = 1


        return np.vstack((self.data['train'], copy))
            # ,mul_hot

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'rhs',
            at: Tuple[int] = (1, 3, 10)
    ):
        if not os.path.exists('./results'):
            os.mkdir('./results/')
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            model.get_ranking(q, self.to_skip[m], batch_size=500)
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, mean_rank, hits_at

    def predict(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'rhs',
            at: Tuple[int] = (1, 3, 10)
    ):
        if not os.path.exists('./results'):
            os.mkdir('./results/')
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            model.get_ranking(q, self.to_skip[m], batch_size=500, predict = True)
            # ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            # mean_rank[m] = torch.mean(ranks).item()
            # mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            # hits_at[m] = torch.FloatTensor((list(map(
            #     lambda x: torch.mean((ranks <= x).float()).item(),
            #     at
            # ))))

        # return mean_reciprocal_rank, mean_rank, hits_at
        return

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities