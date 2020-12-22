from __future__ import print_function
import numpy as np
import reader

def read_bcast_log():
    #fn='logs/nccl-bcast-n16IB.log'
    fn='logs/nccl-bcast-n64.log'
    sizes, comms, errors = reader.read_times_from_nccl_log(fn, original=True)
    print('sizes: ', sizes)
    print('comms: ', comms)
    print('errors: ', errors)


if __name__ == '__main__':
    read_bcast_log()

