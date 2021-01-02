from profiling import CommunicationProfiler
import torch
import horovod.torch as hvd
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
    sizes, times = profiler.benchmark()
    if hvd.rank() == 0:
        for s, t in zip(sizes, times):
            print(s, t)

if __name__ == '__main__':
    benchmark_comm()
