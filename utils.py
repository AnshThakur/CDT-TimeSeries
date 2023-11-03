# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:24:02 2021

@author: anshul
"""
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

def get_validation_score(model,Val_T,Val_L):
    model.eval()
    tensor_x = torch.Tensor(Val_T).to(model.device)
    preds=model(tensor_x)[:,0]
    return roc_auc_score(Val_L, preds.cpu().detach().numpy())  


def get_validation_score_AE(model,Val_T,Val_L):
    model.eval()
    tensor_x = torch.Tensor(Val_T).to(model.device)
    recon,preds=model(tensor_x)
    LOSS=nn.BCELoss().to(model.device)
    val_loss=LOSS(preds[:,0],torch.Tensor(Val_L).type(torch.FloatTensor).to(model.device))
    return roc_auc_score(Val_L, preds[:,0].cpu().detach().numpy()),val_loss  


def get_validation_score_VAE(model,Val_T,Val_L):
    model.eval()
    tensor_x = torch.Tensor(Val_T).to(model.device)
    recon,preds,mu,std=model(tensor_x)
    LOSS=nn.BCELoss().to(model.device)
    val_loss=LOSS(preds[:,0],torch.Tensor(Val_L).type(torch.FloatTensor).to(model.device))
    return roc_auc_score(Val_L, preds[:,0].cpu().detach().numpy()),val_loss 

 

def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res


def sample_standard_gaussian(mu, sigma,device):
	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()
