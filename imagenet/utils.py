import os
import yaml


def name_model(config):
    formatted_injection_info = ''
    for info in config['injection_info']:
        formatted_injection_info += str(tuple(info))
   
    model_name = 'CSAM_Approach{}_BN_PosEmb{}_AfterLayer{}_Temp{}_StochStride{}_Stride{}'.format(
        config['name'], 
        config['pos_emb_dim'], 
        formatted_injection_info, 
        config['softmax_temp'], 
        config['apply_stochastic_stride'], 
        config['stride']
    )
    return model_name

def read_yaml(path):
    return yaml.safe_load(
        open(
            os.path.join(
                path
            )
        )
    )

def save_yaml(path, data, verbose=True):
    if verbose:
        print('Saving yaml to: {}'.format(path))
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)