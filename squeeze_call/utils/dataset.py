
import collections
import os.path
from collections.abc import Callable
import copy
import sys
import tarfile
import logging
import h5py
from typing import List
from glob import glob

import numpy as np
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes
from torch.utils.data.datapipes.iter import Mapper
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)

    feats_lengths = torch.tensor([ x["feat"].size(0) for x in sample],
                                 dtype=torch.int32)

    label_lengths = torch.tensor([x['label'].shape[0] for x in sample],
                                 dtype=torch.int32)

    padding_labels = pad_sequence([torch.tensor(x['label'], dtype=torch.int64) if not torch.is_tensor(x['label']) else x['label'].to(torch.int64) for x in sample],
                                  batch_first=True,
                                  padding_value=-1)
    feats = [ x['feat']  if x["feat"].size(0) % 5 == 0 else  torch.nn.functional.pad(x['feat'], (0, 0, 0, 5-(x["feat"].size(0) % 5)), 'constant', 0) for x in sample]

    padding_feats = pad_sequence(feats,
                                  batch_first=True,
                                  padding_value=0)

    batch = {
        "keys": [x["key"] for x in sample],
        # "feats": torch.stack([x["feat"] for x in sample]),
        "feats": padding_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
        "target_lengths": label_lengths,
    }
    if "read_lengths" in sample[0]:
        batch["read_lengths"] = [x["read_lengths"] for x in sample]
    return batch


@functional_datapipe("map_ignore_error")
class MapperIgnoreErrorDataPipe(Mapper):

    def __init__(self,
                 dataset: IterDataPipe,
                 fn: Callable,
                 input_col=None,
                 output_col=None,
                 log_error: bool = True) -> None:
        super().__init__(dataset, fn, input_col, output_col)
        self._iter = None
        self.log_error = log_error

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self.datapipe)

        while True:
            try:
                elem = next(self._iter)
                yield self._apply_fn(elem)
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if self.log_error:
                    logging.warning(str(ex))


@functional_datapipe('bucket_by_sequence_length')
class BucketBySequenceLengthDataPipe(IterDataPipe):

    def __init__(
        self,
        dataset: IterDataPipe,
        elem_length_func,
        bucket_boundaries: List[int],
        bucket_batch_sizes: List[int],
        wrapper_class=None,
    ) -> None:
        super().__init__()
        _check_unpickable_fn(elem_length_func)
        assert len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        self.bucket_batch_sizes = bucket_batch_sizes
        self.bucket_boundaries = bucket_boundaries + [sys.maxsize]
        self.elem_length_func = elem_length_func

        self._group_dp = GroupByWindowDataPipe(dataset,
                                               self._element_to_bucket_id,
                                               self._window_size_func,
                                               wrapper_class=wrapper_class)

    def __iter__(self):
        yield from self._group_dp

    def _element_to_bucket_id(self, elem):
        seq_len = self.elem_length_func(elem)
        bucket_id = 0
        for (i, b) in enumerate(self.bucket_boundaries):
            if seq_len < b:
                bucket_id = i
                break
        return bucket_id

    def _window_size_func(self, bucket_id):
        return self.bucket_batch_sizes[bucket_id]


@functional_datapipe("group_by_window")
class GroupByWindowDataPipe(datapipes.iter.Grouper):

    def __init__(
        self,
        dataset: IterDataPipe,
        key_func,
        window_size_func,
        wrapper_class=None,
    ):
        super().__init__(dataset,
                         key_func,
                         keep_key=False,
                         group_size=None,
                         drop_remaining=False)
        _check_unpickable_fn(window_size_func)
        self.dp = dataset
        self.window_size_func = window_size_func
        if wrapper_class is not None:
            _check_unpickable_fn(wrapper_class)
            del self.wrapper_class
            self.wrapper_class = wrapper_class

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            group_size = self.window_size_func(key)
            if group_size == len(self.buffer_elements[key]):
                result = self.wrapper_class(self.buffer_elements[key])
                yield result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield result


@functional_datapipe("sort")
class SortDataPipe(IterDataPipe):

    def __init__(self,
                 dataset: IterDataPipe,
                 buffer_size: int = 500,
                 key_func=None,
                 reverse=False) -> None:
        if key_func is not None:
            _check_unpickable_fn(key_func)
        self.buffer_size = buffer_size
        super().__init__()
        self.dp = dataset
        self._buffer = []
        self.key_func = key_func
        self.reverse = reverse

    def __iter__(self):
        for elem in self.dp:
            self._buffer.append(elem)
            if len(self._buffer) >= self.buffer_size:
                self._buffer.sort(key=self.key_func, reverse=self.reverse)
                for x in self._buffer:
                    yield x
                del self._buffer
                self._buffer = []
        # The sample left over
        self._buffer.sort(key=self.key_func, reverse=self.reverse)
        for x in self._buffer:
            yield x
        del self._buffer
        self._buffer = []


@functional_datapipe("dynamic_batch")
class DynamicBatchDataPipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, window_class,
                 wrapper_class) -> None:
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        assert window_class is not None
        assert wrapper_class is not None
        self.window_class = window_class
        self._buffer = []
        self._wrappr_class = wrapper_class

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        del self._buffer
        self._buffer = []


@functional_datapipe("prefetch")
class PrefetchDataPipe(IterDataPipe):
    """Performs prefetching"""

    def __init__(
        self,
        dataset: IterDataPipe,
        buffer_size: int = 500,
    ):
        # TODO(Mddct): support multiprocessing pool with shared-memory to
        #   prefetch
        super().__init__()
        self.dp = dataset
        self._iter = None
        self._prefetch_buffer_size = buffer_size
        self._buffer = None
        if self._prefetch_buffer_size > 0:
            self._buffer = collections.deque(maxlen=self._prefetch_buffer_size)

    def __iter__(self):
        if self._prefetch_buffer_size > 0:
            if self._iter is None:
                self._iter = iter(self.dp)
            assert self._buffer is not None

            while True:
                if len(self._buffer) <= self._prefetch_buffer_size // 2:
                    while len(self._buffer) < self._prefetch_buffer_size:
                        try:
                            self._buffer.append(next(self._iter))
                        except StopIteration:
                            if len(self._buffer) != 0:
                                while len(self._buffer) > 0:
                                    yield self._buffer.popleft()
                            self._iter = None
                            return
                while len(self._buffer) > self._prefetch_buffer_size // 2:
                    elem = self._buffer.popleft()
                    yield elem

        else:
            yield from self.dp


@functional_datapipe("repeat")
class RepeatDatapipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, count: int = -1):
        super().__init__()
        self.dp = dataset
        self.count = count

    def __iter__(self):
        if self.count == 1:
            yield from self.dp
            return
        i = 0
        while self.count < 0 or i < self.count:
            for elem in self.dp:
                new_elem = copy.copy(elem)
                yield new_elem
            i += 1


@functional_datapipe("shard")
class ShardDataPipe(ShardingFilterIterDataPipe):

    def __init__(self, dataset: IterDataPipe, partition: bool = False):
        super().__init__(dataset, None)
        self.partition = partition
        self.dp = dataset

    def apply_sharding(self, num_of_instances: int, instance_id: int,
                       sharding_group: SHARDING_PRIORITIES):
        if self.partition:
            return super().apply_sharding(num_of_instances, instance_id,
                                          sharding_group)
        else:
            # We can not handle uneven data for CV on DDP, so we don't
            # sample data by rank, that means every GPU gets the same
            # and all the CV data
            info = torch.utils.data.get_worker_info()
            if info is None:
                self.num_of_instances = 1
                self.instance_id = 0
            else:
                n_workers_per_device = info.num_workers
                self.num_of_instances = n_workers_per_device
                self.instance_id = info.id


class NpzDatasetSource(IterDataPipe):
    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition: bool = True,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 cycle: int = 1) -> None:
        super().__init__()
            
        _dp = datapipes.iter.FileLister(filenames, recursive=True)
        _dp = datapipes.iter.FileOpener(_dp, mode="b")
        self.dp = _dp
        self.partition = partition
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.repeat(cycle).prefetch(prefetch)
        self.dp = self.dp.shard(partition)

        self.total_size = 0
        for filename, stream in self.dp:
            arr = np.load(filename, mmap_mode='r')
            self.total_size += arr['y_length'].shape[0]

    def __len__(self):
        return self.total_size

    def __iter__(self):
        for filename, stream in self.dp:
            print(f"__iter__ {filename}")
            arr = np.load(filename, mmap_mode='r')
            x = arr['x']
            y = arr['y']
            y_length = arr['y_length']
            size = y_length.shape[0]
            for i in range(size):
                # print(f"{filename} {i=}")
                chunk = torch.tensor(x[i])
                target = torch.tensor(y[i])
                length = y_length[i]
                target = torch.tensor(target[:length])
                yield {"key": "ont",
                       "chunk_size": len(chunk),
                       "feat": chunk.unsqueeze(1),
                       "label": target
                       }
                


def Dataset_from_npz(npz_files, cycle=1, list_shuffle=False, partition=True, batch_size=16):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    # stage1 shuffle: source
    list_shuffle_size = sys.maxsize

    dataset = NpzDatasetSource(npz_files,
                                partition=partition,
                                shuffle=list_shuffle,
                                shuffle_size=list_shuffle_size,
                                cycle=cycle)
    
    dataset = dataset.batch(batch_size, wrapper_class=padding)
    
    return dataset    


def get_npz_dataloader(npz_files, 
                        cycle=1, 
                        batch_size=16,
                        num_workers=1,
                        shuffle=False, 
                        partition=True):
    dataset = NpzDatasetSource(npz_files,
                                partition=partition,
                                shuffle=shuffle,
                                cycle=cycle)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10_000)

    dataset = dataset.batch(batch_size, wrapper_class=padding)


    data_loader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=True,
                            num_workers=num_workers,
                            persistent_workers=True,
                            generator=torch.Generator(),
                            prefetch_factor=500)
    return data_loader