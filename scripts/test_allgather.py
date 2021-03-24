import torch
import horovod.torch as hvd
from mpi4py import MPI
comm = MPI.COMM_WORLD

def test_allgather():
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    tensor = torch.rand(10).float().cuda()
    print('rank: ', rank, ', tensor: ', tensor)
    #handle = hvd.allgather_async(tensor)
    #tensor = hvd.synchronize(handle)
    handle = hvd.broadcast_async(tensor, 0)
    hvd.synchronize(handle)
    comm.Barrier()
    print('---------')
    print('rank: ', rank, ', tensor: ', tensor)

if __name__ == '__main__':
    hvd.init()
    test_allgather()
