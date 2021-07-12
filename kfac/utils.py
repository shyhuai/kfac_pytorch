from __future__ import print_function
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import tcmm
import random
import math
import numpy as np

TENSOR_CORE=False
TENSOR_CORE_THRES=1024 #2048*1024

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x

class cycle:
    def __init__(self, iterable):
        """Iterator that produces tuples indefinitely.

        Example:
          iterator = tuple_cycle([1,2,3], 2)
          assert iterator.next(2) == (1, 2)
          assert iterator.next(1) == (3,)
          assert iterator.next(4) == (1, 2, 3, 1)

        Args:
          iterable: Any iterable to iterate over indefinitely
        """
        self.iterable = iterable
        self.reset()

    def reset(self):
        """Reset iterable to start"""
        self.iterator = itertools.cycle(self.iterable)

    def next(self, size):
        """Get next tuple of size in rotation.

        Returns:
          iterator that returns a tuple of size each time next
          is called.
        """
        return tuple([next(self.iterator) for x in range(size)])

def get_block_boundary(index, block_count, shape):
    """Computes start and end indicies when block diagonalizing a matrix"""
    if index >= block_count:
        raise ValueError("Index ({}) greater than number of requested blocks "
                         "({})".format(index, block_count))
    if block_count > min(shape):
        raise ValueError("Requested blocks ({}) greater than minimum possible "
                         "blocks for shape {}".format(block_count, shape))
    block_shape = [x // block_count for x in shape]
    block_start = [x * index for x in block_shape]
    block_end = [x * (index+1) if (index+1) < block_count 
                           else shape[i] 
                 for i, x in enumerate(block_shape)]
    return block_start, block_end

def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_avg(new, current, alpha):
    """Compute running average of matrix in-place

    current = alpha*new + (1-alpha)*current
    """
    current *= alpha / (1 - alpha)
    current += new
    current *= (1 - alpha)

def use_tensor_core(tensor):
    #if TENSOR_CORE and tensor.shape[0] > TENSOR_CORE_THRES and tensor.shape[1] > TENSOR_CORE_THRES:
    if TENSOR_CORE and tensor.shape[1] > TENSOR_CORE_THRES:
        return True
    return False


residualsA = {}
residualsG = {}
def sparsification(tensor, layer, ratio=0.01, residuals=None):
    t = tensor.view(-1)
    k = int(ratio * t.numel())
    if residuals is not None:
        if layer not in residuals:
            residuals[layer] = torch.zeros_like(t)
        t.add_(residuals[layer])

    abs_t = torch.abs(t)
    tmpvalues, tmpindexes = torch.topk(abs_t, k=k)
    values = t[tmpindexes]

    if residuals is not None:
        residuals[layer].data = t + 0.0 
        residuals[layer].data[tmpindexes] = 0. 
        t.sub_(residuals[layer])
    #tensor = t.view(tensor.shape)
    return values, tmpindexes

def fake_sparsification(tensor, layer, ratio=0.01, residuals=None):
    t = tensor.view(-1)
    indexes = (t.abs() > 1e-4).nonzero().data.squeeze().view(-1)
    values = t[indexes]
    return values, indexes 

def sparsification_randk(tensor, layer, ratio=0.01, residuals=None):
    t = tensor.view(-1)
    d = t.numel()
    k = int(ratio * d)
    if residuals is not None:
        if layer not in residuals:
            residuals[layer] = torch.zeros_like(t)
        t.data.add_(residuals[layer])

    indice = random.sample(range(d), k)
    indexes = torch.tensor(indice, device=t.device)
    values = t[indexes]

    if residuals is not None:
        residuals[layer].data = t.data + 0.0 
        residuals[layer].data[indexes] = 0.0
        t.data.sub_(residuals[layer])
    return values, indexes 

class ComputeA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        #cov_a = sparsification(cov_a, layer, residuals=residualsA)
        return cov_a

    @classmethod
    def get_data(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear_data(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv_data(a, layer)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_a

    @classmethod
    def get_dimension(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.conv2d_data(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.linear_data(a, layer)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_a.shape[-1]

    @staticmethod
    def conv_data(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a.div_(spatial_size)
        batch_size = a.size(0)
        return a

    @staticmethod
    def linear_data(a, layer):
        batch_size = a.size(0)
        #if len(a.shape) > 2:
        #    a = a.view(-1, a.shape[-1])
        #    a = torch.mean(a, list(range(len(a.shape)))[1:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a


    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a.div_(spatial_size)
        # FIXME(CW): do we need to divide the output feature map's size?
        #return torch.einsum('ki,kj->ij', a, a/batch_size) 
        #return a.t() @ (a / batch_size) 
        #print('a dimension conv: ', a.shape)
        if use_tensor_core(a):
            return tcmm.f_gemm_ex(a.t(), a/batch_size)
        else:
            return a.t() @ (a / batch_size) 

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        #if len(a.shape) > 2:
        #    a = a.view(-1, a.shape[-1])
        #    a = torch.mean(a, list(range(len(a.shape)))[1:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        #return torch.einsum('ki,kj->ij', a, a/batch_size) 
        #return a.t() @ (a / batch_size)
        #print('a dimension linear: ', a.shape)
        if use_tensor_core(a):
            return tcmm.f_gemm_ex(a.t(), a/batch_size)
        else:
            return a.t() @ (a / batch_size)


class ComputeG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        #cov_g = sparsification(cov_g, layer, residuals=residualsG)
        return cov_g

    @classmethod
    def get_dimension(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Linear):
            cov_g = cls.conv2d_data(g, layer, batch_averaged)
        elif isinstance(layer, nn.Conv2d):
            cov_g = cls.linear_data(g, layer, batch_averaged)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_g.shape[-1]

    @classmethod
    def get_data(cls, a, layer, batch_averaged):
        if isinstance(layer, nn.Linear):
            cov_g = cls.linear_data(a, layer, batch_averaged)
        elif isinstance(layer, nn.Conv2d):
            cov_g = cls.conv_data(a, layer, batch_averaged)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_g

    @staticmethod
    def conv_data(g, layer, batch_averaged):
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        return g

    @staticmethod
    def linear_data(g, layer, batch_averaged):
        batch_size = g.size(0)
        if batch_averaged:
            return g * batch_size
        else:
            return g / batch_size

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        #cov_g = torch.einsum('ki,kj->ij', g, g/g.size(0)) 
        #cov_g = g.t() @ (g / g.size(0))
        if use_tensor_core(g): 
            cov_g = tcmm.f_gemm_ex(g.t(), g/g.size(0))
        else:
            cov_g = g.t() @ (g / g.size(0))
        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)
        #if len(g.shape) > 2:
        #    #g = g.view(-1, g.shape[-1])
        #    g = torch.mean(g, list(range(len(g.shape)))[1:-1])
        if batch_averaged:
            #cov_g = torch.einsum('ki,kj->ij', g, g*batch_size) 
            #cov_g = g.t() @ (g * batch_size)
            if use_tensor_core(g):
                cov_g = tcmm.f_gemm_ex(g.t(), g*batch_size)
            else:
                cov_g = g.t() @ (g * batch_size)
        else:
            #cov_g = torch.einsum('ki,kj->ij', g, g/batch_size) 
            #cov_g = g.t() @ (g / batch_size)
            if use_tensor_core(g): 
                cov_g = tcmm.f_gemm_ex(g.t(), g/batch_size)
            else:
                cov_g = g.t() @ (g / batch_size)
        return cov_g

def estimate_bcast_time(n, nworkers):
    if nworkers == 16:
        return 2*5*4*1.2636384924990847e-05+1.0847816780156976e-10*n*4
    return 2*5*4*6.374037687522862e-06 + 1.840345743984339e-10* n*4

#alpha_allreduce = 0.0122
#beta_allreduce = 1.45e-9
#alpha_allreduce, beta_allreduce = 0.000929,1.4335e-9 

p_alpha_beta_100Gbps = {
        64: (0.00080632079996292579, 1.8*3.2713239529771973e-10),
        32: (0.00040632079996292579, 1.5*3.2713239529771973e-10),
        16: (0.00023583677659915685*3, 4.0594787739537565e-10),
        8: (9.75367204301171e-05, 3.0568230536676206e-10),
        4: (4.204298980348825e-05, 2.0589360830118177e-10),
        2: (2.554691138304671e-06, 9.837548167872609e-11)
        }

def get_alpha_beta(nworkers):
    alpha_allreduce, beta_allreduce = 0.000929,1.4335e-9
    return alpha_allreduce, beta_allreduce

def estimate_allreduce_time(n, nworkers):
    alpha, beta = get_alpha_beta(nworkers)
    return alpha + beta * n


inverse_times = None 
def estimate_inverse_time(dimension, dnn='resnet'):
    global inverse_times
    if inverse_times is None:
        inverse_times = {}
        with open('logs/inverse-%s.log'%dnn) as f:
            for line in f.readlines():
                items = line.split(',')
                if len(items) == 2:
                    size = int(items[0])
                    t = float(items[1][:-1])
                    inverse_times[size] = t
    return inverse_times[dimension]

def estimate_inverse_time_with_model(dimension):
    alpha = 3.64e-3
    beta = 4.77e-4
    return alpha * math.exp(beta * d) 

def get_per_worker_load(N, P, weights, placement):
    load = np.zeros(P)
    for i in range(N):
        root = placement[i]
        load[root] += weights[i]
    return load

def get_optimal_block_partition(N, P, weights):
    """
    input: 
        #tasks (N), #workers (P), weights of tasks
    output: 
        placement: task assignment results, i.e., a list of worker IDs
    """
    if N <= P:
        return np.arange(N)

    # compute the optimal bottleneck by dynamic programming
    # , where B[i, j] is the optimal bottleneck given the first (j+1) tasks and (i+1) workers

    W = np.cumsum(weights)
    B = np.zeros((P, N)) 
    B[0, :] = W

    # # original version
    # for p in range(1, P): 
    #     for i in range(p, N - P + p + 1):
    #         B[p, i] = min([max(B[p-1, j], W[i] - W[j]) for j in range(p-1,i)])

    # improved version
    for p in range(1, P):
        j = p - 1
        for i in range(p, N - P + p + 1):
            if W[i] - W[j] > B[p-1, j]:
                while W[i] - W[j] > B[p-1, j]:
                    j += 1
                if j == i or W[i] - W[j-1] < B[p-1, j]: # Important: deal with the special case of j == i
                    j = j - 1
                    B[p, i] = W[i] - W[j]
                else:
                    B[p, i] = B[p-1, j]
            else:
                B[p, i] = B[p-1, j]
            j = p - 1

    bottleneck = B[P-1, N-1]
    # print(bottleneck)

    # continuous placement until the bottleneck is reached
    placement = np.zeros(N, dtype=int)
    root = 0
    load = 0
    for i in range(N):
        if load + weights[i] <= bottleneck + 1e-06: 
            placement[i] = root
            load += weights[i]
        else:
            root += 1
            placement[i] = root
            load = weights[i]

    assert root < P

    # fill in the empty workers or not ?
    if root < P - 1:
        for i in range(1, N-1):
            if placement[i-1] == placement[i] and placement[i-1] != placement[i+1]:
                root += 1
                placement[i] = root
                if root == P-1:
                    break

    return placement


