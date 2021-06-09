# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import tcmm
import time
import mpi4py
import horovod.torch as hvd
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
    communicator = tcmm.Communicator(rank, size, nstreams)
    n_elements = 32* 1024
    iterations = 100
    tensor = torch.rand(n_elements).cuda()
    if rank == 0:
        print('before rank: %d' % rank, time.time())
    for i in range(nstreams):
        communicator.reduce(tensor, 0)
    #communicator.allReduce(tensor)
    #hvd.allreduce(tensor)
    communicator.synchronize()
    start = time.time()
    previous = start
    for i in range(iterations):
        communicator.reduce(tensor, 0)
        #communicator.allReduce(tensor)
        #hvd.allreduce(tensor)
        current = time.time()
        if rank ==0:
            print('i: ', i, current-previous)
        previous = current
    communicator.synchronize()
    end = time.time()
    if rank == 0:
        print('after rank: %d' % rank, time.time(), (end-start)/iterations)
        print('throughput: ', n_elements * 4 *1e-9/ ((end-start)/iterations), 'GB/s')

if __name__ == '__main__':
    #allreduce()
    #multi_bcast()
    reduce()
