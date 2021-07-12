import torch
import horovod.torch as hvd
import tcmm

from kfac import comm
from profiling import CommunicationProfiler

hvd.init()

def benchmark_comm():
    torch.cuda.set_device(hvd.local_rank())
    comm_op = hvd.broadcast_async_
    #comm_op = hvd.allreduce_async_
    sync_op = hvd.synchronize
    #sizes = [2**i for i in range(10, 30)]
    sizes = [] #[1024*i for i in range(1, 1024)] 
    large_sizes = [1024*1024*i for i in range(1, 513)] # 1M to 512M
    sizes += large_sizes
    profiler = CommunicationProfiler(comm_op, sync_op, sizes)
    sizes, times = profiler.benchmark(num_iters=20)
    if hvd.rank() == 0:
        for s, t in zip(sizes, times):
            print(s, t)


def benchmark_custom_comm():
    torch.cuda.set_device(hvd.local_rank())
    merged_comm = tcmm.Communicator(hvd.rank(), hvd.size(), 1)
    comm_op = merged_comm.reduce 
    sync_op = merged_comm.synchronize
    sizes = [2**i for i in range(10, 11)]
    #sizes = [] #[1024*i for i in range(1, 1024)] 
    large_sizes = [] #[1024*1024*i for i in range(1, 513)] # 1M to 512M
    sizes += large_sizes
    profiler = CommunicationProfiler(comm_op, sync_op, sizes)
    for root in range(hvd.size()):
        sizes, times = profiler.benchmark(root, num_iters=50)
        if hvd.rank() == 0:
            print('root: %d' % root)
            for s, t in zip(sizes, times):
                print(s, t, str(s*4/t*1e-6)+' MB/s')
            print()



if __name__ == '__main__':
    #benchmark_comm()
    benchmark_custom_comm()
