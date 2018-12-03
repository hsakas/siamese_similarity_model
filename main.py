from model.encoder import FCN
from model.backbone import *

res1, res2 = get_model(name='resnet18', pretrained=False), get_model(
    name='resnet18', pretrained=False)

encoder1, encoder2 = FCN(10, 1000, [100]), FCN(10, 1000, [100])

res1.fc, res2.fc = encoder1, encoder2

print(res1)
