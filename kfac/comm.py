import torch
import horovod.torch as hvd
import numpy as np


class TensorGroup:
    def __init__(self, tensor_names, single_layer):
        self._tensor_names = tensor_names
        self._single_layer = single_layer
        self._groups, self._group_indices_by_name = self._generate_groups()
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
        new_name_tensors = {}
        for group_key in self._group_buffers:
            names = group_key.split(':')
            group_idx, sub_idx = self.get_group_index_by_name(names[0]) 
            buf = self._group_buffers[group_key]

            offset = 0
            for t in self._group_storages[group_idx]:
                numel = t.numel()
                t.copy_(buf.data[offset:offset+numel].view(t.shape))
                offset += numel 


class MergedComm:
    def __init__(self, tensor_names, prefix='flag', merge=False, single_layer=False):
        self._tensor_names = tensor_names
        self.merge = merge
        self.prefix = prefix
        if merge:
            self._tensor_group = TensorGroup(tensor_names, single_layer=single_layer) 
        else:
            self._tensor_group = None
        self._name_tensors = {}
        self.handles = []

    def allreduce_async_(self, name, tensor, op=hvd.Average):
        if self.merge:
            new_name, new_tensor = self._tensor_group.push_tensor(name, tensor)
            self._name_tensors[name] = tensor
            if new_tensor is not None:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()

                handle = hvd.allreduce_async_(new_tensor, op=op, name=self.prefix+new_name)
                self.handles.append(handle)
        else:
            handle = hvd.allreduce_async_(tensor, op=hvd.Average)
            self.handles.append(handle)

    def synchronize(self):
        for h in self.handles:
            hvd.synchronize(h)
        if self.merge:
            self._tensor_group.pull_alltensors()
            self.handles.clear()
            self._tensor_group.clear_group_flags()


class MergedCommBcast:
    def __init__(self, tensor_names, prefix='flag'):
        self._tensor_names = tensor_names
        self.merge = False
        self.prefix = prefix
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

    def synchronize(self):
        for h in self.handles:
            hvd.synchronize(h)
        if self.merge:
            self._tensor_group.pull_alltensors()
            self.handles.clear()
            self._tensor_group.clear_group_flags()

