# -*- coding: utf-8 -*-
from __future__ import print_function

def read_conv_shapes(logfile):
    def _extract_args(string):
        args = {}
        items = string.split(',')
        input_c = int(items[0].split('Conv2d(')[-1])
        output_c = int(items[1])
        args['input_c'] = input_c
        args['output_c'] = output_c
        keys = ['kernel_size', 'stride', 'padding']
        for k in keys:
            if string.find(k) >= 0:
                args[k] = tuple([int(i) for i in string.split(k)[1].split('),')[0][2:].split(',')])
            else:
                args[k] = (0, 0)
        return args
    workloads = []
    with open(logfile) as f:
        for line in f.readlines():
            if line.find('Conv2d') >= 0:
                args = _extract_args(line)
                workloads.append(args)
    #print(workloads)
    return workloads

def read_tensor_sizes(logfile):
    def _extract_args(string):
        args = {}
        items = string.split('torch.Size([')
        str_tz = items[-1][:-3]
        tensor_size = tuple([int(i) for i in str_tz.split(',')])
        return tensor_size
    workloads = []
    with open(logfile) as f:
        for line in f.readlines():
            if line.find('torch.Size') >= 0:
                args = _extract_args(line)
                workloads.append(args)
    print(workloads)
    return workloads

