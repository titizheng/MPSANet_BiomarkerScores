import numpy as np

np.random.seed(0)

from model.vgg import (vgg11, vgg13, vgg16, vgg19)

from model.resnets import (resnet18, resnet34, resnet50, resnet101,
                           resnet152)

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


