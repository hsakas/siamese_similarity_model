from torchvision.models import *


def get_model(name='resnet18', pretrained=True, **kwargs):
    if name == 'resnet18':
        return resnet18(pretrained=pretrained)
    if name == 'vgg16':
        return vgg16(pretrained=pretrained)

