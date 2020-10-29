# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time


def compute_eigen(matrix):
    A = matrix
    d, Q = torch.symeig(A, eigenvectors=True)
    eps = 1e-10  # for numerical stability
    d = torch.mul(d, (d > eps).float())
    return d, Q

def bench_ops(n, num_iters, warmup=5):
    a = torch.rand(n).float().cuda()
    a = a.view(-1, a.size(-1))
    print('a shape: ', a.shape)
    A = a.t() @ (a)
    print('A shape: ', A.shape)
    for i in range(warmup):
        compute_eigen(A)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        compute_eigen(A)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used


def bench():
    ns = range(2**10, 2**20, 1024) 
    #ns = ns+range(2**20, 2**29, 2**20) 
    #ns = range(2**20, 2**29, 2**20) 
    for n in ns:
        num_iters = 50
        if n > 2**19:
            num_iters = 10
        t = bench_ops(n, num_iters)
        print('%d,%f'%(n,t))



if __name__ == '__main__':
    bench()
