# 定义一些定制化操作
# 修改BatchSampler、Collate_fn等

import random
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Mapping

class DynamicBatchSampler(Sampler[List[int]]):

    def __init__(self, dataset, sampler, max_token=500, shuffle=True) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.sampler = sampler
        self.max_token = max_token
        self.batch_indices = dataset.dynamic_batch_sample(max_token)
        bs = [len(b) for b in self.batch_indices]
        print(f'MAX batchsize: {max(bs)}  MIN batchsize: {min(bs)}')
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.shuffle:
            random.shuffle(self.batch_indices)
        
        for batch in self.batch_indices:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.batch_indices)
    

def my_collate_fn(data_batch:Sequence, needPad=False):
        data_item = data_batch[0]
        data_item_type = type(data_item)

        if isinstance(data_item, (str, bytes ,bool)):
            return data_batch
        elif isinstance(data_item,Sequence):
            if isinstance(data_item[0],(int,float)):
                return data_batch
            else:
                return data_item_type([my_collate_fn(samples) for samples in list(zip(*tuple(data_batch)))])
        elif isinstance(data_item, Mapping):
            return data_item_type({
                key: my_collate_fn([d[key] for d in data_batch], needPad=True)
                if ('target' in key or 'gt' in key) else
                my_collate_fn([d[key] for d in data_batch]) for key in data_item 
            })
        elif isinstance(data_item, torch.Tensor):
            shapes = [data.shape for data in data_batch]
            if not needPad:
                try:
                    return torch.concat([data.unsqueeze(0) for data in data_batch])
                except:
                    return data_batch
            else:
                # pad poly points
                ls = [shape[0] for shape in shapes]
                max_len = max(ls)
                if data_item.ndim == 2:
                    polys = [ F.pad(data_batch[i],(0,0,0,max_len-length),mode='constant', value=-1) for i,length in enumerate(ls)]
                elif data_item.ndim == 1:
                    polys = [ F.pad(data_batch[i],(0,max_len-length),mode='constant', value=-1) for i,length in enumerate(ls)]
                else:
                    raise NotImplementedError
                polys = torch.concat([data.unsqueeze(0) for data in polys])
                pad_mask = torch.zeros_like(polys)
                pad_mask[polys==-1] = 1
                return {'data':polys,'pad_mask':pad_mask}
        else:
            raise NotImplementedError
            

