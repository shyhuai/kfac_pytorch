import torch
import horovod.torch as hvd
import numpy as np
from kfac.utils  import estimate_allreduce_time, get_alpha_beta
import logging
import tcmm
logger = logging.getLogger()

sync_tensor = torch.zeros(1)

class TensorGroup:
    def __init__(self, tensor_names, single_layer, tensors=None):
        self._tensor_names = tensor_names
        self._single_layer = single_layer
        self._groups, self._group_indices_by_name = self._generate_groups()
        self._group_buffers = {}
        self.reset_merge()

    def reset_merge(self):
        if self._group_buffers is not None:
            for k in self._group_buffers:
                buf = self._group_buffers[k]
                del buf
        self._group_flags = [[0]*len(g) for g in self._groups]
        self._group_keys = [':'.join(g) for g in self._groups]
        self._group_storages = [[None] * len(g) for g in self._groups]
        self._group_buffers = {}

    def _generate_groups(self):
        groups = []
        group_indices_by_name = {}
        current_group = []
        group_idx = 0
        for i, t in enumerate(self._tensor_names):
            group_indices_by_name[t] = (group_idx, len(current_group))
            current_group.append(t)
            #if i % len(self._tensor_names) == 0 and i > 0:
            if not self._single_layer and i % 3 == 0 and i > 0:
                groups.append(current_group)
                current_group = []
                group_idx += 1
        if len(current_group) > 0:
            groups.append(current_group)
        return groups, group_indices_by_name

    def is_merged(self):
        return len(self._tensor_names) != len(self._groups)

    def get_group_index_by_name(self, name):
        group_idx, sub_idx = self._group_indices_by_name[name]
        return group_idx, sub_idx

    def clear_group_flags(self):
        self._group_flags = [[0]*len(g) for g in self._groups]

    def check_group_full(self, name):
        group_idx, sub_idx = self.get_group_index_by_name(name) 
        if np.sum(self._group_flags[group_idx]) < len(self._group_flags[group_idx]):
            return False
        return True

    def push_tensor(self, name, tensor):
        group_idx, sub_idx = self.get_group_index_by_name(name) 
        group_key = self._group_keys[group_idx]
        numel = tensor.numel()
        self._group_flags[group_idx][sub_idx] = 1
        self._group_storages[group_idx][sub_idx] = tensor
        if self.check_group_full(name):
            if group_key not in self._group_buffers:
                total_size = 0
                for t in self._group_storages[group_idx]:
                    total_size += t.numel()
                self._group_buffers[group_key] = tensor.new_zeros(total_size)
            buf = self._group_buffers[group_key]
            offset = 0
            for t in self._group_storages[group_idx]:
                numel = t.numel()
                buf.data[offset:offset+numel].copy_(t.view(numel))
                offset += numel
            return group_key, buf
        return name, None

    def pull_alltensors(self):
        for group_key in self._group_buffers:
            names = group_key.split(':')
            group_idx, sub_idx = self.get_group_index_by_name(names[0]) 
            buf = self._group_buffers[group_key]

            offset = 0
            for t in self._group_storages[group_idx]:
                numel = t.numel()
                t.copy_(buf.data[offset:offset+numel].view(t.shape))
                offset += numel 

    def update_groups(self, sizes, times, symmetric=False, reverse=False):
        if self._single_layer:
            return
        self._groups, self._group_indices_by_name = self._generate_groups_spd(sizes, times, symmetric, reverse)
        self.reset_merge()
        torch.cuda.empty_cache()

    def _generate_groups_spd(self, sizes, times, symmetric, reverse=False):
        num_of_workers = hvd.size()
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            #taoc[L-1] = taob[L-1] + tb[L-1]
            taoc[0] = taob[0] + tb[0]
            for l in range(1, L):
                taoc[l] = max(taoc[l-1] + tc[l-1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l+1] = p[l+1]+p[l]
            p[l] = 0
            tc[l+1] = estimate_allreduce_time(p[l+1], num_of_workers)
        if reverse:
            seq_layernames = self._tensor_names[::-1]
        else:
            seq_layernames = self._tensor_names
        p = sizes[:]

        if symmetric:
            p = [np.sqrt(s) * (np.sqrt(s)+1) /2 for s in sizes]

        L = len(sizes)

        tc = [estimate_allreduce_time(s, num_of_workers) for s in sizes]

        tb = list(times)
        taob = [0]*L
        for l in range(1,L):
            taob[l] = taob[l-1] + tb[l-1]

        taoc = __calculate_comm_start(tc, tb, taob, L)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = 0
        key = seq_layernames[l] 
        group_indices_by_name = {}
        key_groupidx_maps[key] = idx
        alpha, beta = get_alpha_beta(num_of_workers)
        for l in range(0, L-1):
            key = seq_layernames[l]
            group_indices_by_name[key] = (idx, len(group))
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l+1] + tb[l+1]
            merged=False
            if current_taob < taoc[l]+tc[l]:
                if taoc[l] > current_taob:
                    __merge(taob, tc, p, l)
                    taoc = __calculate_comm_start(tc, tb, taob, L)
                    merged=True
                else:
                    t_wait = current_taob - taoc[l]
                    t_saved = alpha
                    if t_wait < t_saved:
                        __merge(taob, tc, p, l)
                        taoc = __calculate_comm_start(tc, tb, taob, L)
                        merged=True
            if not merged:
                idx += 1
                groups.append(group)
                group = []
        l = L-1
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group_indices_by_name[key] = (idx, len(group))
        group.append(key)
        groups.append(group)

        if hvd.rank() == 0:
            logger.info('Merged sizes: %s', p)
            logger.info('# of parameters: %f', np.sum(p))
        return groups, group_indices_by_name


class MergedCommAllReduce:
    def __init__(self, tensor_names, prefix='flag', merge=False, single_layer=False, symmetric=False, fp16=False, residual=False):
        self._tensor_names = tensor_names
        self.merge = merge
        self.fp16 = fp16
        self.residual = residual
        self.prefix = prefix
        self.op = hvd.Sum
        if merge:
            self._tensor_group = TensorGroup(tensor_names, single_layer=single_layer) 
            self.merge = self._tensor_group.is_merged()
        else:
            self._tensor_group = None
        self._name_tensors = {}
        self._residuals = {}
        self.handles = []
        self.symmetric = symmetric

    def allreduce_async_(self, name, tensor, op=hvd.Average):
        self.op = op
        if self.merge:
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor
            if self.fp16:
                if self.residual:
                    if name not in self._residuals:
                        self._residuals[name] = comm_tensor.new_zeros(comm_tensor.shape)
                    comm_tensor.add_(self._residuals[name])
                half_tensor  = comm_tensor.half() 
                if self.residual:
                    self._residuals[name] = comm_tensor - half_tensor
                comm_tensor = half_tensor
            self._name_tensors[name] = (tensor, comm_tensor)
            new_name, new_tensor = self._tensor_group.push_tensor(name, comm_tensor)
            if new_tensor is not None:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()

                handle = hvd.allreduce_async_(new_tensor, op=hvd.Sum, name=self.prefix+new_name)
                self.handles.append(handle)
        else:
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor
            if self.fp16:
                if self.residual:
                    if name not in self._residuals:
                        self._residuals[name] = comm_tensor.new_zeros(comm_tensor.shape)
                    comm_tensor.add_(self._residuals[name])
                half_tensor  = comm_tensor.half() 
                if self.residual:
                    self._residuals[name] = comm_tensor - half_tensor
                comm_tensor = half_tensor #comm_tensor.half()
                #comm_tensor = comm_tensor.bfloat16() 
            self._name_tensors[name] = (tensor, comm_tensor)
            handle = hvd.allreduce_async_(comm_tensor, op=hvd.Sum)
            self.handles.append(handle)

    def update_groups(self, sizes, times, reverse=False):
        if self.merge and self._tensor_group:
            self._tensor_group.update_groups(sizes, times, self.symmetric, reverse=reverse)
            self.merge = self._tensor_group.is_merged()

    def synchronize(self):
        for h in self.handles:
            hvd.synchronize(h)
        if self.merge:
            self._tensor_group.pull_alltensors()
            self._tensor_group.clear_group_flags()
        for name in self._name_tensors:
            tensor, comm_tensor = self._name_tensors[name]
            if self.symmetric:
                if self.fp16:
                    comm_tensor = comm_tensor.float()
                lower_indices = torch.tril_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                tensor[upper_indices[0], upper_indices[1]] = comm_tensor
                tensor[lower_indices[0], lower_indices[1]] = tensor.t()[lower_indices[0], lower_indices[1]]
            else:
                if self.fp16:
                    comm_tensor = comm_tensor.float()
                    tensor.copy_(comm_tensor)
            if self.op == hvd.Average:
                tensor.div_(hvd.size())
        self._name_tensors.clear()
        self.handles.clear()

class MergedCommBcast:
    def __init__(self, tensor_names, prefix='flag', fp16=False):
        self._tensor_names = tensor_names
        self.merge = False
        self.prefix = prefix
        self.fp16 = fp16
        if self.merge:
            self._tensor_group = TensorGroup(tensor_names, single_layer=False) 
        else:
            self._tensor_group = None
        self._name_tensors = {}
        self.handles = []

    def bcast_async_(self, name, tensor, rank):
        if self.merge:
            new_name, new_tensor = self._tensor_group.push_tensor(name, tensor)
            self._name_tensors[name] = tensor
            if new_tensor is not None:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()

                handle = hvd.broadcast_async_(new_tensor, rank, name=self.prefix+new_name)
                self.handles.append(handle)
        else:
            handle = hvd.broadcast_async_(tensor, rank)
            self.handles.append(handle)

    def allgather_sync(self, tensors, ranks):
        nworkers = hvd.size()
        rank = hvd.rank()
        start = 0
        sub_ranks = ranks[start:start+nworkers]
        sub_tensors = tensors[start:start+nworkers]
        while len(sub_ranks) > 0:
            #print('len(sub_ranks): ', len(sub_ranks))
            #print('len(sub_tensors): ', len(sub_tensors))
            try:
                idx = sub_ranks.index(rank)
            except:
                idx = -1
            if idx < 0:
                tensor = sub_tensors[0].new(0) 
            else:
                tensor = sub_tensors[idx]
            handle = hvd.allgather_async(tensor.view(-1))
            sync_tensors = hvd.synchronize(handle)
            offset = 0
            for i, r in enumerate(sub_ranks):
                if idx < 0:
                    continue
                original_t = sub_tensors[r]
                numel = original_t.numel()
                t = sync_tensors[offset:offset+numel]
                original_t.copy_(t.view(original_t.shape))
                offset += numel

            start += nworkers
            sub_ranks = ranks[start:start+nworkers]
            sub_tensors = tensors[start:start+nworkers]

    def synchronize(self):
        for h in self.handles:
            hvd.synchronize(h)
        self.handles.clear()
        if self.merge:
            self._tensor_group.pull_alltensors()
            self._tensor_group.clear_group_flags()


class MultiTensorComm:
    def __init__(self, symmetric=False, fp16=False):
        self.handles = []
        self.symmetric = symmetric
        self.fp16 = fp16
        self.merged_tensors = {}


    def bcast_async_(self, names, tensors, rank):
        #name = 'merged_tensor_comm_'+','.join(names)
        if self.fp16:
            comm_tensors = [t.half() for t in tensors]
        else:
            comm_tensors = tensors

        if self.symmetric:
            sym_comm_tensors = []
            for tensor in comm_tensors:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
                sym_comm_tensors.append(comm_tensor)
            comm_tensors = sym_comm_tensors

        name = ','.join(names)
        if name not in self.merged_tensors:
            size = 0
            if len(comm_tensors) > 1:
                for t in comm_tensors:
                    size += t.numel()
                buf = comm_tensors[0].new_zeros(size)
                self.merged_tensors[name] = buf
            else:
                self.merged_tensors[name] = comm_tensors[0]
        buf = self.merged_tensors[name]
        if len(comm_tensors) > 1:
            offset = 0
            for t in comm_tensors:
                numel = t.numel()
                buf.data[offset:offset+numel].copy_(t.view(numel))
                offset += numel
        #handle = hvd.broadcast_async_(buf, rank, name=name)
        handle = hvd.broadcast_async_(buf, rank)
        self.handles.append((handle, names, tensors, comm_tensors))

    def synchronize(self):
        for h in self.handles:
            handle, names, tensors, comm_tensors = h
            hvd.synchronize(handle)
            #name = 'merged_tensor_comm_'+','.join(names)
            name = ','.join(names)

            offset = 0
            buf = self.merged_tensors[name]
            if self.fp16:
                buf = buf.float()
            for i, t in enumerate(tensors):
                numel = comm_tensors[i].numel()
                comm_tensor = buf.data[offset:offset+numel]

                if self.symmetric:
                    lower_indices = torch.tril_indices(t.shape[0], t.shape[1], device=t.device)
                    upper_indices = torch.triu_indices(t.shape[0], t.shape[1], device=t.device)
                    t[upper_indices[0], upper_indices[1]] = comm_tensor.view(comm_tensors[i].shape)
                    t[lower_indices[0], lower_indices[1]] = t.t()[lower_indices[0], lower_indices[1]]
                else:
                    t.copy_(comm_tensor.view(t.shape))
                offset += numel 
        self.handles.clear()


class MultiTensorReduce:
    def __init__(self, symmetric=False, fp16=False):
        self.handles = []
        self.symmetric = symmetric
        self.fp16 = fp16 # dosen't support fp16 at the current stage
        self.merged_tensors = {}
        nstreams = 1
        self.merged_comm = tcmm.Communicator(hvd.rank(), hvd.size(), nstreams)

    def reduce_async_(self, names, tensors, rank):
        if self.fp16:
            comm_tensors = [t.half() for t in tensors]
        else:
            comm_tensors = tensors

        if self.symmetric:
            sym_comm_tensors = []
            for tensor in comm_tensors:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
                sym_comm_tensors.append(comm_tensor)
            comm_tensors = sym_comm_tensors

        name = ','.join(names)

        if name not in self.merged_tensors:
            size = 0
            if len(comm_tensors) > 1:
                for t in comm_tensors:
                    size += t.numel()
                buf = comm_tensors[0].new_zeros(size)
                self.merged_tensors[name] = buf
            else:
                self.merged_tensors[name] = comm_tensors[0]
        buf = self.merged_tensors[name]
        if len(comm_tensors) > 1:
            offset = 0
            for t in comm_tensors:
                numel = t.numel()
                buf.data[offset:offset+numel].copy_(t.view(numel))
                offset += numel
        handle = self.merged_comm.reduce(buf, rank)
        self.handles.append((handle, names, tensors, comm_tensors))

    def synchronize(self):
        self.merged_comm.synchronize()
        for h in self.handles:
            handle, names, tensors, comm_tensors = h

            name = ','.join(names)
            offset = 0
            buf = self.merged_tensors[name]

            if self.fp16:
                buf = buf.float()
            for i, t in enumerate(tensors):
                numel = comm_tensors[i].numel()
                comm_tensor = buf.data[offset:offset+numel]

                if self.symmetric:
                    lower_indices = torch.tril_indices(t.shape[0], t.shape[1], device=t.device)
                    upper_indices = torch.triu_indices(t.shape[0], t.shape[1], device=t.device)
                    t[upper_indices[0], upper_indices[1]] = comm_tensor
                    t[lower_indices[0], lower_indices[1]] = t.t()[lower_indices[0], lower_indices[1]]
                else:
                    t.copy_(comm_tensor.view(t.shape))
                offset += numel 
        self.handles.clear()

def barrier():
    torch.cuda.synchronize()
    handle = hvd.broadcast_async_(sync_tensor, root_rank=0)
    hvd.synchronize(handle)
