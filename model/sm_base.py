import torch.nn as nn
import torch


class SaimeseNet(nn.Module):
    """
    TODO: Documentation
    """

    def __init__(self, backbone, encoder):
        super(SaimeseNet, self).__init__()

        self.backbone = backbone
        self.encoder = encoder

    def _forward(self, x):
        x = self.backbone(x)
        x = x.view(-1)
        x = self.encoder(x)
        return x

    def forward(self, input1, input2):
        y_hat1 = _forward(input1)
        y_hat2 = _forward(input2)
        return y_hat1, y_hat2
