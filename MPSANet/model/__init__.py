import numpy as np

np.random.seed(0)

from KFwsi.model.resnet import (resnet18, resnet34, resnet50, resnet101,
                                resnet152)
from KFwsi.model.vgg import (vgg11, vgg13, vgg16, vgg19)

from  KFwsi.model.pre_resnet import (resnet18, resnet34, resnet50, resnet101,
                                     resnet152)

from KFwsi.model.Shuff import shufflenet_v2_x0_5
from KFwsi.model.Mobile import mobilenet_v2
from KFwsi.model.Goole import googlenet
# from KFwsi.model.pre_efficient import  EfficientNet
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_name('efficientnet-b0')


'''
预训练之后的resnet比之前没有使用预训练的好，
并且预训练的resnet比VGG好用

'''


MODELS = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152}


MODELS_vgg = {'vgg11': vgg11,
          'vgg13': vgg13,
          'vgg16': vgg16,
          'vgg19': vgg19}


MODELS_resnet = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152}
#这几个都是新加的没有跑的

MODELS_shuff = {'shufflenet_v2': shufflenet_v2_x0_5}

MODELS_Mobile = {'mobilenet_v2': mobilenet_v2}

MODELS_Goole = {'googlenet': googlenet}

# MODELS_Efficient ={
#          'efficientnet-b0': EfficientNet.from_name('efficientnet-b0'),
#           'efficientnet-b3': EfficientNet.from_name('efficientnet-b3'),
#           'efficientnet-b5': EfficientNet.from_name('efficientnet-b5'),
#           'efficientnet-b7': EfficientNet.from_name('efficientnet-b7'),
#                   }

