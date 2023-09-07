import os
from collections import namedtuple

import torch
import torch.nn as nn

__all__ = ["linear"]

cifar10_pretrained_weight_urls = {
    'linear-cifar10': 'https://github.com/u7122029/PyTorch_CIFAR10/releases/download/pretrained_addons/linear.pt',
    'linear-svhn': "https://github.com/u7122029/PyTorch_CIFAR10/releases/download/pretrained_svhn/linear.pt"
}


def linear(pretrained=False, progress=True, device="cpu", dataset="cifar10", **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    model = Linear()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(cifar10_pretrained_weight_urls[f"linear-{dataset}"], map_location=device)
        model.load_state_dict(state_dict)
    return model


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(3*32*32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x