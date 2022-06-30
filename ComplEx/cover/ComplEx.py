import torch
import torch.nn as nn
import os
import json
import numpy as np

RES = 'result.txt'

class BaseModule(nn.Module):

	def __init__(self):
		super(BaseModule, self).__init__()
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

	def load_checkpoint(self, path):
		self.load_state_dict(torch.load(os.path.join(path)))
		self.eval()

	def save_checkpoint(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		f = open(path, "r")
		parameters = json.loads(f.read())
		f.close()
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

	def save_parameters(self, path):
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def get_parameters(self, mode = "numpy", param_dict = None):
		all_param_dict = self.state_dict()
		if param_dict == None:
			param_dict = all_param_dict.keys()
		res = {}
		for param in param_dict:
			if mode == "numpy":
				res[param] = all_param_dict[param].cpu().numpy()
			elif mode == "list":
				res[param] = all_param_dict[param].cpu().numpy().tolist()
			else:
				res[param] = all_param_dict[param]
		return res

	def set_parameters(self, parameters):
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

class ComplEx(BaseModule):
	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(ComplEx, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot
		self.dim = dim
		self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)

		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

		if not os.path.exists('./results'):
			os.mkdir('./results/')
		with open(f'./results/{RES}','w') as _:
			pass

	def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
		return torch.sum(
			h_re * t_re * r_re
			+ h_im * t_im * r_re
			+ h_re * t_im * r_im
			- h_im * t_re * r_im,
			-1
		)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		regul = (torch.mean(h_re ** 2) + 
				 torch.mean(h_im ** 2) + 
				 torch.mean(t_re ** 2) +
				 torch.mean(t_im ** 2) +
				 torch.mean(r_re ** 2) +
				 torch.mean(r_im ** 2)) / 6
		return regul

	def predict(self, data):
		score = -self.forward(data)

		if data['mode'] == 'tail_batch':
			res = [str(i.item()) for i in torch.topk(score, k = 10, largest = False).indices]
			with open(f'./results/{RES}','a+') as fp:
				fp.write(f"{data['batch_h'][0]} {data['batch_r'][0]} {' '.join(res)}\n")

		return score.cpu().data.numpy()