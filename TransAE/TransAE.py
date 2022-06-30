import torch
import os
import re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils import data
from collections import Counter
import torch.optim as optim
from tqdm import tqdm
import ctypes
import pickle
from dataloader import TrainDataLoader, TestDataLoader

RES = 'result.txt'

class Trainer(object):

    def __init__(self, 
                 model = None,
                 data_loader = None,
                 train_times = 1000,
                 alpha = 0.5,
                 use_gpu = True,
                 opt_method = "sgd",
                 save_steps = None,
                 checkpoint_dir = None):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir = None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

class MarginLoss(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0):
        super(MarginLoss,self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
            
    
    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "./release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            # score = self.test_one_step(data_head)
            # self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            # self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        # self.lib.test_link_prediction(type_constrain)

        # mrr = self.lib.getTestLinkMRR(type_constrain)
        # mr = self.lib.getTestLinkMR(type_constrain)
        # hit10 = self.lib.getTestLinkHit10(type_constrain)
        # hit3 = self.lib.getTestLinkHit3(type_constrain)
        # hit1 = self.lib.getTestLinkHit1(type_constrain)
        # print (hit10)
        # return mrr, mr, hit10, hit3, hit1
        return

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod


class IMG_Encoder(nn.Module):
    def __init__(self, embedding_dim = 4096, dim = 200, margin = None, epsilon = None):
        super(IMG_Encoder, self).__init__()
        with open('./data/OpenBG-IMG/entity2id.txt') as fp:
            entity2id = fp.readlines()[1:]
            entity2id = [i.split('\t')[0] for i in entity2id]

        self.entity2id = entity2id
        self.activation = nn.ReLU()
        self.entity_count = len(entity2id)
        self.dim = dim
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.criterion = nn.MSELoss(reduction='mean') 
        self.raw_embedding = nn.Embedding(self.entity_count, self.dim)

        self.visual_embedding = self._init_embedding()

        self.encoder = nn.Sequential(
                torch.nn.Linear(embedding_dim, 1024),
                self.activation
            )
        
        self.encoder2 = nn.Sequential(
                torch.nn.Linear(1024, self.dim),
                self.activation
            )

        self.decoder2 = nn.Sequential(
                torch.nn.Linear(self.dim, 1024),
                self.activation
            )

        self.decoder = nn.Sequential(
                torch.nn.Linear(1024, embedding_dim),
                self.activation
            )

    def _init_embedding(self):
        self.ent_embeddings = nn.Embedding(self.entity_count, self.embedding_dim)
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        weights = torch.empty(self.entity_count, self.embedding_dim)
        embed = 0
        for index, entity in tqdm(enumerate(self.entity2id)):
            # print(index, entity)
            try:
                entity_ = entity.replace('/', '.')
                
                with open("./data/OpenBG-IMG/img_em/" + entity_ + "/avg_embedding.pkl", "rb") as visef:
                    embed = embed+1
                    em = pickle.load(visef)
                    weights[index] = em
                    
            except:
                # print(index, entity)
                weights[index] = self.ent_embeddings(torch.LongTensor([index])).clone().detach()
                continue
        print(embed)
        entities_emb = nn.Embedding.from_pretrained(weights)

        return entities_emb

    def forward(self, entity_id):
        v1 = self.visual_embedding(entity_id)
        v2 = self.encoder(v1)

        v2_ = self.encoder2(v2)
        v3_ = self.decoder2(v2_)

        v3 = self.decoder(v3_)
        loss = self.criterion(v1, v3)
        return v2_, loss


class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(TransE, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.tail_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_embeddings = IMG_Encoder(dim = self.dim, margin = self.margin, epsilon = self.epsilon)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.tail_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_embeddings.weight.data, 
                a = -self.embedding_range.item(), 
                b = self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embeddings.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )

        if not os.path.exists('./results'):
            os.mkdir('./results/')
        with open(f'./results/{RES}','w') as _:
            pass

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        #self.ent_embeddings.encoder[0].weight.data.div_(self.ent_embeddings.encoder[0].weight.data.norm(p=2, dim=1, keepdim=True))
        
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h, hloss = self.ent_embeddings(batch_h)
        #t, tloss = self.ent_embeddings(batch_t)
        #h = self.ent_embeddings(batch_h)
        t = self.tail_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h ,t, r, mode) + hloss
        #score = self._calc(h ,t, r, mode)# + hloss + tloss
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)

        if data['mode'] == 'tail_batch':
            res = [str(i.item()) for i in torch.topk(score, k = 10, largest = False).indices]
            with open(f'./results/{RES}','a+') as fp:
                fp.write(f"{data['batch_h'][0]} {data['batch_r'][0]} {' '.join(res)}\n")
        
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

class TransE_raw(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(TransE_raw, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        # self.ent_embeddings = IMG_Encoder(dim = self.dim, margin = self.margin, epsilon = self.epsilon)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_embeddings.weight.data, 
                a = -self.embedding_range.item(), 
                b = self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embeddings.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        #self.ent_embeddings.encoder[0].weight.data.div_(self.ent_embeddings.encoder[0].weight.data.norm(p=2, dim=1, keepdim=True))
        
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        # h, hloss = self.ent_embeddings(batch_h)
        #t, tloss = self.ent_embeddings(batch_t)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h ,t, r, mode)# + hloss + tloss
        #score = self._calc(h ,t, r, mode)# + hloss + tloss
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)

        if data['mode'] == 'tail_batch':
            res = [str(i.item()) for i in torch.topk(score, k = 10, largest = False).indices]
            with open('./results/{RES}','a+') as fp:
                fp.write(f"{data['batch_h'][0]} {data['batch_r'][0]} {' '.join(res)}\n")

        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

class NegativeSampling(nn.Module):

    def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
        super(NegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def forward(self, data):
        score = self.model(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        return loss_res

data_path = "./data/OpenBG-IMG/"

train_dataloader = TrainDataLoader(
    in_path = data_path, 
    nbatches = 100,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 25)

# dataloader for test
test_dataloader = TestDataLoader(data_path, "link")

with open(os.path.join(data_path, 'entity2id.txt'), 'r') as fp:
    ents_count = int(fp.readline()[:-1])

with open(os.path.join(data_path, 'relation2id.txt'), 'r') as fp:
    rels_count = int(fp.readline()[:-1])

# define the model
transe = TransE(
    ent_tot = ents_count,
    rel_tot = rels_count,
    dim = 200, 
    p_norm = 1, 
    norm_flag = True)

model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints/')

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoints/OpenBG-IMG.ckpt')

# test the model
transe.load_checkpoint('./checkpoints/OpenBG-IMG.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)