#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-27 11:14:53
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
import math
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")
class RGCLayer(Module):
	def __init__(self,input_dim,h_dim,supprot,num_base,featureless,drop_prob):
		super(RGCLayer,self).__init__()
		self.num_base=num_base
		self.input_dim=input_dim
		self.supprot=supprot
		self.h_dim=h_dim
		self.featureless=featureless
		self.drop_prob=drop_prob
		if num_base>0:
			self.W=Parameter(torch.empty(input_dim*self.num_base,h_dim,dtype=torch.float32,device=device))
			self.W_comp=Parameter(torch.empty(supprot,num_base,dtype=torch.float32,device=device))
		else:
			self.W=Parameter(torch.empty(input_dim*self.supprot,h_dim,dtype=torch.float32,device=device))
		self.B=Parameter(torch.FloatTensor(h_dim))
		self.reset_parameters()
	def reset_parameters(self):
		nn.init.xavier_uniform_(self.W)
		if self.num_base>0:
			nn.init.xavier_uniform_(self.W_comp)
		self.B.data.fill_(0)
	def forward(self,vertex,A):
		supports=[]
		nodes_num=A[0].shape[0]
		for i,adj in enumerate(A):
			if not self.featureless:
				supports.append(torch.spmm(adj,vertex))
			else:
				supports.append(adj)
		supports=torch.cat(supports,dim=1)
		if self.num_base>0:
			V=torch.matmul(self.W_comp,torch.reshape(self.W,(self.num_base,self.input_dim,self.h_dim)).permute(1,0,2))
			V=torch.reshape(V,(self.input_dim*self.supprot,self.h_dim))
			output=torch.spmm(supports,V)
		else:
			output=torch.spmm(supports,self.W)
		if self.featureless:
			temp=torch.ones(nodes_num).to(device)
			temp_drop=F.dropout(temp,self.drop_prob)
			output=(output.transpose(1,0)*temp_drop).transpose(1,0)
		output+=self.B
		return output

