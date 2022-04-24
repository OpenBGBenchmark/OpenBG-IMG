from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import pickle
import torch.nn.functional as F
import numpy as np
from config import alpha,beta,random_gate,forget_gate,remember_rate,constant


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1, predict = False
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if not constant:
                        r_embeddings, img_embeddings= self.get_rhs(c_begin, chunk_size)
                        h_r = self.get_queries(these_queries)
                        n = len(h_r)
                        scores_str = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

                        for i in range(n):
                            i_alpha = self.alpha[(these_queries[i, 1])]
                            single_score = h_r[[i], :] @ (
                                    (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0,
                                                                                                                      1)
                            scores_str = torch.cat((scores_str, single_score.detach()), 0)
                    else:
                        rhs = self.get_rhs(c_begin, chunk_size)
                        q = self.get_queries(these_queries)
                        scores_str = q @ rhs

                    lhs_img = F.normalize(self.img_vec[these_queries[:,0]], p=2, dim=1)
                    rhs_img = F.normalize(self.img_vec, p=2, dim=1).transpose(0, 1)
                    score_img=lhs_img@rhs_img
                    # beta=0.95
                    if forget_gate:
                        scores=beta*scores_str+(1-beta)*score_img*self.rel_pd[these_queries[:,1]]
                    else:
                        scores = beta * scores_str + (1 - beta) * score_img
                    if not predict:
                        targets = self.score(these_queries)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6

                    if not predict:
                        ranks[b_begin:b_begin + batch_size] += torch.sum(
                            (scores >=targets).float(), dim=1
                        ).cpu()
                    else:
                        for i in range(len(these_queries)):
                            score_one = scores[i]
                            res = [str(i.item()) for i in torch.topk(score_one, k = 10, largest = True).indices]
                            with open('./results/result.txt','a+') as fp:
                                fp.write(f"{these_queries[i][0]} {these_queries[i][1]} {' '.join(res)}\n")
                    b_begin += batch_size

                c_begin += chunk_size
        if not predict:
            return ranks
        else:
            return


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

def sc_wz_01(len,num_1):
    A=[1 for i in range(num_1)]
    B=[0 for i in range(len-num_1)]
    C=A+B
    np.random.shuffle(C)
    return np.array(C,dtype=np.float)


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            # img_info='img_vec_id_fb15k_20_vit.pickle',
            # sig_alpha='rel_MPR_SIG_vit_20.pickle',
            # rel_pd='rel_MPR_PD_vit_20_mrp{}.pickle'
            img_info='./data/OpenBG-IMG/img_vec_id_openbg_vit.pickle',
            sig_alpha='./data/OpenBG-IMG/rel_MPR_SIG_vit.pickle',
            rel_pd='./data/OpenBG-IMG/rel_MPR_PD_vit_mrp{}.pickle'
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank


        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        if not constant:
            self.alpha=torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha=torch.cat((self.alpha,self.alpha),dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd=torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate),'rb'))).cuda()
        else:
            tmp=pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd=torch.from_numpy(sc_wz_01(len(tmp),np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd=torch.cat((self.rel_pd,self.rel_pd),dim=0)
        # self.alpha[self.img_info['missed'], :] = 1

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)

    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img


        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight +  self.alpha* img_embeddings

            lhs = embedding[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = embedding[(x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img=self.img_vec[(x[:, 0])]
            rhs_img=self.img_vec[(x[:, 2])]

            # score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)
            if forget_gate:
                score_img=torch.cosine_similarity(lhs_img,rhs_img,1).unsqueeze(1)*rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img,rhs_img, 1).unsqueeze(1)


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str=torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta*score_str+(1-beta)*score_img

    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                            (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            lhs = embedding[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = embedding[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]

            to_score = embedding
            to_score = to_score[:, :self.rank], to_score[:, self.rank:]

            return (
                           (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                           (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
                   ), (
                       torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                       torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                       torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                   )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)



        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            lhs = embedding[(queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)

if __name__ == '__main__':
    # pickle.pickle.load(open(img_info, 'rb'))
    pass