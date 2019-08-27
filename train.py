#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-26 20:53:30
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
from utils import load_data_pkl,compute_accuracy,normalize
from models import RGCN
from torch import nn
from torch import optim
import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def adjust_learning_rate(optimizer,lr):
	for param in optimizer.param_groups:
		param["lr"]=lr
def to_sparse_tensor(sparse_array):
	if len(sp.find(sparse_array)[-1])>0:
		v=torch.FloatTensor(sp.find(sparse_array)[-1])
		i=torch.LongTensor(sparse_array.nonzero())
		shape=sparse_array.shape
		sparse_tensor=torch.sparse.FloatTensor(i,v,torch.Size(shape))
	else:
		sparse_tensor=torch.sparse.FloatTensor(sparse_array.shape[0],sparse_array.shape[1])
	return sparse_tensor
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

pkl_path="E:/relationalGCNDatasets/aifb/aifb.pickle"
A,y,train_idx,test_idx=load_data_pkl(pkl_path)
y=y.todense()

saved_models="E:/relationalGCNDatasets/saved_models_aifb";mkdirs(saved_models)
epochs=5000
hidden_dim=50
drop_prob=0.
lr=0.01
weight_decay=0
best_loss=9999
vertex_features_dim=A[0].shape[0]

tensor_A=[]
for a in A:
	nor_a=normalize(a)
	if len(nor_a.nonzero()[0])>0:
		tensor_a=to_sparse_tensor(nor_a)
		tensor_A.append(tensor_a.to(device))


tensor_x=None

model=RGCN(vertex_features_dim,hidden_dim,drop_prob,len(tensor_A),40)
model.to(device)

# optimizer=optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay)
optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion=nn.CrossEntropyLoss()

train_losses=[]
val_losses=[]
val_accs=[]
no_update_cnter=0
for e in range(epochs):
	if no_update_cnter>=10:
		lr*=0.1
		# adjust_learning_rate(optimizer,lr)
		no_update_cnter=0

	labels=y[train_idx]
	labels=np.argmax(labels,axis=1)
	labels=torch.LongTensor(labels).to(device).squeeze()
	optimizer.zero_grad()
	batch_preds=model(tensor_x,tensor_A,train_idx)
	batch_loss=criterion(batch_preds,labels)
	e_loss=batch_loss.item()
	batch_loss.backward()
	optimizer.step()
	train_losses.append(e_loss)

	labels=y[test_idx]
	labels=np.argmax(labels,axis=1)
	labels=torch.LongTensor(labels).to(device).squeeze()
	batch_preds=model(tensor_x,tensor_A,test_idx)
	batch_loss=criterion(batch_preds,labels)
	e_loss=batch_loss.item()
	val_acc=compute_accuracy(batch_preds.detach().cpu().numpy(),labels.cpu().numpy())
	val_losses.append(e_loss)
	val_accs.append(val_acc)

	if best_loss>val_losses[-1]+0.01:
		torch.save({
			"model":model.state_dict(),
			"optimizer":optimizer.state_dict()
			},os.path.join(saved_models,"model_{}.tar".format(e)))
		best_loss=val_losses[-1]
		print("UPDATE\tEpoch {}: train loss {}\tval loss {}\tval acc{}".format(e,train_losses[-1],val_losses[-1],val_accs[-1]))
		no_update_cnter=0
	else:
		no_update_cnter+=1
	if e%10==0:
		print("Epoch {}: train loss {}\tval loss {}\tval acc{}".format(e,train_losses[-1],val_losses[-1],val_accs[-1]))


			









