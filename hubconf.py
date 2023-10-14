from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.linear import linear
from cifar10_models.lenet5 import lenet5
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

import torch.nn as nn
if __name__ == "__main__":
    model = vgg19_bn(pretrained=False)
    print(model)
