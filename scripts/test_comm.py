# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import tcmm
import time
import mpi4py
import horovod.torch as hvd
from kfac import comm
torch.random.manual_seed(10)
hvd.init()


def allreduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    tensor = torch.rand(2).cuda()
    print('before rank: %d' % rank, tensor)
    communicator.allReduce(tensor)
    print('after rank: %d' % rank, tensor)

def multi_bcast():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    ntensors = 2
    tensors = []
    for i in range(ntensors):
        t = torch.rand(2).cuda()
        tensors.append(t)
    def _op(tensor):
        tensor.mul_(2)
        return None
    print('before rank: %d' % rank, tensors)
    communicator.multiBcast(tensors, _op)
    print('after rank: %d' % rank, tensors)

def reduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    nstreams = 1
    #communicator = tcmm.Communicator(rank, size, nstreams)
    communicator  = comm.MultiTensorReduce(symmetric=False)
    communicator2  = comm.MultiTensorReduce(symmetric=True)
    n_elements = 4
    tensor = torch.rand(n_elements, 1).cuda()
    tensor2 = torch.rand(n_elements, 1).cuda()
    tensor = tensor @ tensor.t()
    tensor2 = tensor2 @ tensor2.t()
    tensor_copy = tensor.clone()
    tensor2_copy = tensor2.clone()
    communicator.reduce_async_(['t'], [tensor], 1)
    communicator.reduce_async_(['t2'], [tensor2], 1)
    communicator2.reduce_async_(['tcopy'], [tensor_copy], 1)
    communicator2.reduce_async_(['tcopy2'], [tensor2_copy], 1)
    communicator.synchronize()
    communicator2.synchronize()
    #print('after rank: %d' % rank, tensor)
    print('after rank copy: %d' % rank, tensor_copy)
    print('diff copy: %d' % rank, torch.norm(tensor_copy-tensor))
    print('diff copy 2: %d' % rank, torch.norm(tensor2_copy-tensor2))
    

if __name__ == '__main__':
    #allreduce()
    #multi_bcast()
    reduce()
