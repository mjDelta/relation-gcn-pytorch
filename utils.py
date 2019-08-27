#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-27 11:41:09
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import numpy as np
import pickle as pkl 
import scipy.sparse as sp
def compute_accuracy(preds,trues):

	preds_=np.argmax(preds,axis=1)
	trues_=trues
	cnts=np.sum(preds_==trues_)
	return cnts/len(preds)
def load_data_pkl(pkl_path):
	with open(pkl_path,"rb") as f:
		data=pkl.load(f)
	A=data["A"]
	y=data["y"]
	train_idx=data["train_idx"]
	test_idx=data["test_idx"]

	return A,y,train_idx,test_idx

def normalize(matrix):
	row_sum=np.array(matrix.sum(axis=1)).flatten()
	row_sum_inv=1./row_sum
	row_sum_inv[np.isinf(row_sum_inv)]=0
	D_inv=sp.diags(row_sum_inv)
	out=D_inv.dot(matrix).tocsr()
	return out