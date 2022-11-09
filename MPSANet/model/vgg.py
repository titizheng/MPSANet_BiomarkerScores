import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from KFwsi.model.layers import CRF

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1, num_nodes=1,use_crf=True,init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.crf = CRF(num_nodes) if use_crf else None
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #######-----------------------------------------------------------------
        batch_size, grid_size, _, crop_size = x.shape[0:4]
        # flatten grid_size dimension and combine it into batch dimension
        x = x.view(-1, 3, crop_size, crop_size)
        #######-----------------------------------------------------------------

        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        feats = x.view(x.size(0), -1)
        logits = self.classifier(feats)
        # restore grid_size dimension for CRF
        feats = feats.view((batch_size, grid_size, -1))
        logits = logits.view((batch_size, grid_size, -1))

        if self.crf:
            logits = self.crf(feats, logits)

        logits = torch.squeeze(logits)

        return logits
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

from collections import OrderedDict

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs) ##这个就是实例化对象
    if pretrained: ##这一部分才是加载预训练的参数
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress) ##下载预训练好的参数
        # print('输出原本的',state_dict)
        new_state_dict=state_dict##这个就相当于，加载的load，已经训练好的模型参数，字典形式
        dd=model.state_dict()##这个是模型初始化的参数。字典形式，这个形式是完全使用新模型的
        # print('输出第一个dd', dd) ##dd和state_dict结果是一样的，但是，内容不意义昂
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and not k.startswith('classifier'):  # 去掉classifier层中的参数，startswith是查找含有某个字符的数值
                print('yes')
                dd[k] = new_state_dict[k] ##用已经训练好的模型参数，替换掉，dd中初始化的部分参数。
        # print('输出dd',dd)
        model.load_state_dict(dd)##最后模型加载的还是dd参数。


        # dd = model.state_dict() ##这个和state_dict输出的格式是一样的
        # model.load_state_dict(state_dict)
        # new_state_dict=state_dict
        # for k in list(new_state_dict.keys()):
        #     print(k)
        #     if k.startswith('classifier'):
        #         new_state_dict.pop(k)
        # model.load_state_dict(new_state_dict)

        # stat_dict = torch.load(cfg['dict_path'])['state_dict']
        # model_dict = state_dict
        # stat_dict = {k: v for k, v in stat_dict.items() if
        #              k in model_dict.keys() and not k != 'module.fc.weight' and k != 'module.fc.bias'}
        # model_dict.update(stat_dict)
        # model.load_state_dict(model_dict)


        ##----------------------------下面是自己修改的加载模型的参数的方法------------------------------
        # new_state_dict = model.state_dict()  ##把权重字典化
        # dd = model.state_dict()  # net是自己定义的含有resnet backbone的模型
        # for k in new_state_dict.keys():
        #     print(k)
        #     if k in dd.keys() and not k.startswith('classifier'):  # 不使用全连接的参数
        #         print('yes')
        #         dd[k] = new_state_dict[k]
        # model.load_state_dict(dd)
        #
        # model_dict = model.state_dict()
        # stat_dict = {k: v for k, v in stat_dict.items() if
        #              k in model_dict.keys() and k != 'module.fc.weight' and k != 'module.fc.bias'}
        # model_dict.update(stat_dict)
        # model.load_state_dict(model_dict)

    return model


def vgg11(pretrained=True, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=True, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=True, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=True, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
