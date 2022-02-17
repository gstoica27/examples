import os
import yaml
import sys

def name_model(config):
    formatted_injection_info = ''
    for info in config['injection_info']:
        formatted_injection_info += str(tuple(info))
   
    model_name = 'CSAM_Approach{}_BN_PosEmb{}_AfterConv{}_Temp{}_StochStride{}_Stride{}_Residual{}'.format(
        config['approach_name'], 
        config['pos_emb_dim'], 
        formatted_injection_info, 
        config['softmax_temp'], 
        config['apply_stochastic_stride'], 
        config['stride'],
        config['use_residual_connection']
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

def get_network(name):
    """ return given network
    """

    # if name == 'vgg16':
    #     from models.vgg import vgg16_bn
    #     net = vgg16_bn()
    # elif name == 'vgg13':
    #     from models.vgg import vgg13_bn
    #     net = vgg13_bn()
    # elif name == 'vgg11':
    #     from models.vgg import vgg11_bn
    #     net = vgg11_bn()
    # elif name == 'vgg19':
    #     from models.vgg import vgg19_bn
    #     net = vgg19_bn()
    # elif name == 'densenet121':
    #     from models.densenet import densenet121
    #     net = densenet121()
    # elif name == 'densenet161':
    #     from models.densenet import densenet161
    #     net = densenet161()
    # elif name == 'densenet169':
    #     from models.densenet import densenet169
    #     net = densenet169()
    # elif name == 'densenet201':
    #     from models.densenet import densenet201
    #     net = densenet201()
    # elif name == 'googlenet':
    #     from models.googlenet import googlenet
    #     net = googlenet()
    # elif name == 'inceptionv3':
    #     from models.inceptionv3 import inceptionv3
    #     net = inceptionv3()
    # elif name == 'inceptionv4':
    #     from models.inceptionv4 import inceptionv4
    #     net = inceptionv4()
    # elif name == 'inceptionresnetv2':
    #     from models.inceptionv4 import inception_resnet_v2
    #     net = inception_resnet_v2()
    # elif name == 'xception':
    #     from models.xception import xception
    #     net = xception()
    if name == 'resnet18':
        from resnet import resnet18
        net = resnet18()
    elif name == 'resnet34':
        from resnet import resnet34
        net = resnet34()
    elif name == 'resnet50':
        from resnet import resnet50
        net = resnet50()
    elif name == 'resnet101':
        from resnet import resnet101
        net = resnet101()
    elif name == 'resnet152':
        from resnet import resnet152
        net = resnet152()
    # elif name == 'preactresnet18':
    #     from models.preactresnet import preactresnet18
    #     net = preactresnet18()
    # elif name == 'preactresnet34':
    #     from models.preactresnet import preactresnet34
    #     net = preactresnet34()
    # elif name == 'preactresnet50':
    #     from models.preactresnet import preactresnet50
    #     net = preactresnet50()
    # elif name == 'preactresnet101':
    #     from models.preactresnet import preactresnet101
    #     net = preactresnet101()
    # elif name == 'preactresnet152':
    #     from models.preactresnet import preactresnet152
    #     net = preactresnet152()
    # elif name == 'resnext50':
    #     from models.resnext import resnext50
    #     net = resnext50()
    # elif name == 'resnext101':
    #     from models.resnext import resnext101
    #     net = resnext101()
    # elif name == 'resnext152':
    #     from models.resnext import resnext152
    #     net = resnext152()
    # elif name == 'shufflenet':
    #     from models.shufflenet import shufflenet
    #     net = shufflenet()
    # elif name == 'shufflenetv2':
    #     from models.shufflenetv2 import shufflenetv2
    #     net = shufflenetv2()
    # elif name == 'squeezenet':
    #     from models.squeezenet import squeezenet
    #     net = squeezenet()
    # elif name == 'mobilenet':
    #     from models.mobilenet import mobilenet
    #     net = mobilenet()
    # elif name == 'mobilenetv2':
    #     from models.mobilenetv2 import mobilenetv2
    #     net = mobilenetv2()
    # elif name == 'nasnet':
    #     from models.nasnet import nasnet
    #     net = nasnet()
    # elif name == 'attention56':
    #     from models.attention import attention56
    #     net = attention56()
    # elif name == 'attention92':
    #     from models.attention import attention92
    #     net = attention92()
    # elif name == 'seresnet18':
    #     from models.senet import seresnet18
    #     net = seresnet18()
    # elif name == 'seresnet34':
    #     from models.senet import seresnet34
    #     net = seresnet34()
    # elif name == 'seresnet50':
    #     from models.senet import seresnet50
    #     net = seresnet50()
    # elif name == 'seresnet101':
    #     from models.senet import seresnet101
    #     net = seresnet101()
    # elif name == 'seresnet152':
    #     from models.senet import seresnet152
    #     net = seresnet152()
    # elif name == 'wideresnet':
    #     from models.wideresidual import wideresnet
    #     net = wideresnet()
    # elif name == 'stochasticdepth18':
    #     from models.stochasticdepth import stochastic_depth_resnet18
    #     net = stochastic_depth_resnet18()
    # elif name == 'stochasticdepth34':
    #     from models.stochasticdepth import stochastic_depth_resnet34
    #     net = stochastic_depth_resnet34()
    # elif name == 'stochasticdepth50':
    #     from models.stochasticdepth import stochastic_depth_resnet50
    #     net = stochastic_depth_resnet50()
    # elif name == 'stochasticdepth101':
    #     from models.stochasticdepth import stochastic_depth_resnet101
    #     net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
