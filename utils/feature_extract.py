import torch
import torch.nn as nn

from torchvision.models.feature_extraction import create_feature_extractor


class FeatXception(nn.Module):
    def __init__(self, net):
        super().__init__()
        return_nodes = {'view': 'feat'}
        self.body = create_feature_extractor(
            model=net, return_nodes=return_nodes)

    def forward(self, x):
        x = self.body(x)
        return x
