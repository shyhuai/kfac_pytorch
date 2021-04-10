# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import numpy as np

import tcmm
import reader
import torchsso

def add_value_to_diagonal(X, value):
    return X.add_(torch.diag(X.new(X.shape[0]).fill_(value)))

def precondition_full_G(g):
    G = g @ g.t()
    add_value_to_diagonal(G, 0.002)
    inverse_G = torchsso.utils.inv(G); 
    g = inverse_G @ g
    return g

def precondition_sparse_G(g, ratio=0.01):
    g = torch.clone(g)
    flatten_g = g.view(-1)
    abs_g = torch.abs(flatten_g)
    k = int(g.numel() * ratio)
    tmpvalues, tmpindexes = torch.topk(abs_g, k=k)
    tmpvalues = flatten_g[tmpindexes]
    sg = tmpvalues.view(-1, 1)
    G =  sg @ sg.t()
    add_value_to_diagonal(G, 0.002)
    inverse_G = torchsso.utils.inv(G); 
    sg = inverse_G @ sg
    g.view(-1)[tmpindexes] = sg.view(-1)
    return g

def sparse(g, ratio=0.01):
    shape = g.size()
    g = g.view(-1)
    d = g.numel()
    k = int(d * ratio)
    tmpvalues, tmpindexes = torch.topk(g, k=d-k)
    g[tmpindexes] = 0.0
    return g.view(shape)


def bench():
    n = 1024
    g = torch.rand(n, 1).float().cuda()
    g = sparse(g)
    full_pg = precondition_full_G(g)
    sparse_pg = precondition_sparse_G(g)
    diff = torch.norm(full_pg-sparse_pg)
    print('norm full_g: ', full_pg.norm())
    print('norm sparse_pg: ', sparse_pg.norm())
    print('diff: ', diff)


if __name__ == '__main__':
    bench()
