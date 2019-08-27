#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-27 11:32:45
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch.nn import Module,Sequential,Linear,ReLU,Dropout,LogSoftmax
from layers import RGCLayer
import torch.nn.functional as F
class RGCN(Module):
	def __init__(self,i_dim,h_dim,drop_prob,supprot,num_bases,featureless=True):
		super(RGCN,self).__init__()
		self.drop_prob=drop_prob
		self.gc1=RGCLayer(i_dim,h_dim,supprot,num_bases,featureless,drop_prob)
		self.gc2=RGCLayer(h_dim,h_dim,supprot,num_bases,False,drop_prob)

		self.fc1=Sequential(
			Linear(h_dim,h_dim),
			ReLU(),
			Dropout(drop_prob))
		self.fc2=Sequential(
			Linear(h_dim,4),
			LogSoftmax())
	def forward(self,vertex,A,idxs):
		gc1=F.dropout(F.relu(self.gc1(vertex,A)),self.drop_prob)
		gc2=F.dropout(F.relu(self.gc2(gc1,A)),self.drop_prob)

		embs=gc2[idxs]

		fc1=self.fc1(embs)
		fc2=self.fc2(fc1)
		return fc2