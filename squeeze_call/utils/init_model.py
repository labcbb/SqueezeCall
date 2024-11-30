import numpy as np
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import yaml
import sys
# sys.path.append('/home/share/huadjyin/home/cyclone_ops/users/ryl/code_repo/basenet/squeeze_call')
from squeeze_call.asr_model import ASRModel
from squeeze_call.encoder import SqueezeformerEncoder
from squeeze_call.decoder import BiTransformerDecoder
from squeeze_call.ctc import CRF


def init_model(config):
    input_dim = config['input_dim']  # 1
    vocab_size = config['output_dim']  # 7    
    encoder = SqueezeformerEncoder(input_dim, **config['encoder_conf'])
    
    decoder = BiTransformerDecoder(vocab_size=vocab_size, 
                                   encoder_output_size=encoder.output_size(), 
                                   **config['decoder_conf'])
    ctc = CRF(vocab_size,
                encoder.output_size(),
                reduce=config['ctc_conf']['reduce'] if 'ctc_conf' in config else True,
                use_focal_loss=config['ctc_conf']['use_focal_loss'] if 'ctc_conf' in config else False,
                blank_id=config['ctc_conf']['ctc_blank_id'] if 'ctc_conf' in config else 0)
    
    gate_intermediate = config['encoder_conf'].get("gate_intermediate", False)

    model = ASRModel(
                vocab_size=vocab_size,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                gate_intermediate=gate_intermediate,
                special_tokens=config.get('tokenizer_conf',
                                            {}).get('special_tokens', None),
                **config['model_conf'])
    
    return model
