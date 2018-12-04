from model.encoder import FCN
from model.backbone import *
from utils.model_utils import compute_out_size

vgg1, vgg2 = get_model(name='vgg16', pretrained=False), get_model(
    name='resnet18', pretrained=False)

computed1 = compute_out_size(vgg1, 3, img_size, True, 'cpu')
computed2 = compute_out_size(vgg2, 3, img_size, True, 'cpu')

encoder1, encoder2 = FCN(10, computed1, [4048, 2024]), FCN(
    10, computed2, [4048, 2024])

vgg1.fc, vgg2.fc = encoder1, encoder2
