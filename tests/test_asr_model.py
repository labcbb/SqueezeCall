import numpy as np
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import yaml
import sys
sys.path.append('/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call')
from squeeze_call.asr_model import ASRModel
from squeeze_call.utils.dataset import get_npz_dataloader
from squeeze_call.encoder import SqueezeformerEncoder
from squeeze_call.decoder import BiTransformerDecoder
from squeeze_call.ctc import CRF
from squeeze_call.utils.init_model import init_model


data_dir = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/data"
config_path = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/config/base.yaml"


def asr_model_demo():
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config)
    print(model)



def demo0():
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config)
    device = 'cuda:0'
    model = model.to(device)
    
    train_npz_files = glob(f"{data_dir}/train*.npz")
    batch_size = config['dataset_conf']['batch_conf']['batch_size']
    train_loader = get_npz_dataloader(train_npz_files, batch_size=batch_size, cycle=1)
    #print(len(train_loader))
    for i, batch in enumerate(train_loader):

        batch['feats'] = batch['feats'].to(device)
        batch['feats_lengths'] = batch['feats_lengths'].to(device)
        batch['target'] = batch['target'].to(device)
        batch['target_lengths'] = batch['target_lengths'].to(device)
        model(batch)
        
    
    

demo0()