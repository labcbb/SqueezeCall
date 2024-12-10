import yaml
import sys
import torch
sys.path.append("/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call")

from squeeze_call.encoder import SqueezeformerEncoder
from squeeze_call.decoder import BiTransformerDecoder


def encoder_demo0():
    config_path = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/config/block8_hs512_kmer_crf_att_conv1d5_layer3_ln.yaml"
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    input_dim = configs['input_dim']  # 1
    vocab_size = configs['output_dim']  # 7    
    encoder = SqueezeformerEncoder(input_dim, **configs['encoder_conf'])

    xs = torch.rand((4, 3600, 1))
    xs_len = torch.tensor([3600, 3600, 3600, 3600])
     # xs(bsz, 1200, 512), masks(bsz,1,1200),  intermediate_outputs[0]:(bsz, 1200, 512)
    xs1, masks, intermediate_outputs = encoder(xs, xs_len)
    print(f"xs1:{xs1.shape}, masks:{masks.shape}, intermediate_outputs:{intermediate_outputs[0].shape}")


def decoder_demo():
    config_path = "/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call/config/block8_hs512_kmer_crf_att_conv1d5_layer3_ln.yaml"
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    decode_conf = configs['decoder_conf']
    # print(decode_conf)
    decoder = BiTransformerDecoder(vocab_size=7, encoder_output_size=512, **decode_conf)
    print(decoder)




decoder_demo()