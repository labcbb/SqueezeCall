import numpy as np
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call')
from squeeze_call.utils.dataset import get_npz_dataloader

data_dir = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/data"


def demo0():
    npz_file = "/home/share/huadjyin/home/cyclone_ops/basecall_data/nanopore_benchmark/jain/data_split_paper/chunk3600_overlap0_npz_demo/train_15.npz"
    arr = np.load(npz_file, mmap_mode='r')
    x = arr['x']
    y = arr['y']
    y_length = arr['y_length']
    print(f"x:{x.shape}, y:{y.shape}, y_length:{y_length.shape}")

    size_per_file = 1000
    output_dir = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/data"

    for i,index in enumerate(range(0, y_length.shape[0], size_per_file)):
        np.savez(os.path.join(output_dir, f"train_{i}.npz"), 
                 x=arr['x'][index: index+size_per_file].astype(np.float32), 
                 y=arr['y'][index: index+size_per_file].astype(np.int64), 
                 y_length=arr['y_length'][index: index+size_per_file].astype(np.int64)
                 )    

def demo1():
    train_npz_files = glob(f"{data_dir}/train*.npz")
    train_loader = get_npz_dataloader(train_npz_files, batch_size=4, cycle=1)
    for i,batch in enumerate(train_loader):
        print(f"{i} {batch}")
        break
    

def demo2():
    dataset = BaseNanoporeDataset(data_dir, None, None, 0.9, shuffle=False)
    print(f"train-size: {dataset.train_size}")

    dataloader_train = DataLoader(
        dataset, 
        batch_size = 512, 
        sampler = dataset.train_sampler, 
        num_workers = 1,
        drop_last=True
    )

    for train_batch_num, train_batch in enumerate(dataloader_train):
        if train_batch_num%100 == 0:
            print(f"{train_batch_num=}")



demo2()