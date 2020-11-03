# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time

import tcmm


def compute_eigen(matrix):
    A = matrix
    d, Q = tcmm.f_symeig(A)
    Q = Q.transpose(-2, -1)
    #d, Q = torch.symeig(A, eigenvectors=True)
    #eps = 1e-10  # for numerical stability
    #d = torch.mul(d, (d > eps).float())
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
    ns = range(64, 2048, 64) 
    #ns = range(512, 512+64, 64) 
    #ns = ns+range(2**20, 2**29, 2**20) 
    #ns = range(2**20, 2**29, 2**20) 
    for n in ns:
        num_iters = 50
        if n > 2**19:
            num_iters = 10
        t = bench_ops(n, num_iters)
        print('%d,%f'%(n,t))

def check():
    n = 4
    a = torch.rand(n).float().cuda()
    a = a.view(-1, a.size(-1))
    A = a.t() @ (a)
    d, Q = torch.symeig(A, eigenvectors=True)
    print('GT shape: ', d.shape, Q.shape)
    d1, Q1 = tcmm.f_syseig(A)
    Q1 = Q1.transpose(-2, -1)
    print('Customize shape: ', d1.shape, Q1.shape)
    print('eigenvalues norm: ', (d-d1).norm(), d.norm(), d1.norm())
    print('eigenvectors norm: ', (Q-Q1).norm(), Q.norm(), Q1.norm())
    def _goback(d, Q):
        back = torch.matmul(Q, torch.matmul(d.diag_embed(), Q.transpose(-2, -1)))
        return back
    print('A: ', A)
    print('d: ', d)
    print('Q: ', Q)
    print('bA: ', _goback(d, Q))
    print('d1: ', d1)
    print('Q1: ', Q1)
    print('bA1: ', _goback(d1, Q1))

if __name__ == '__main__':
    bench()
    #check()
