from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.models import mobilenetv3, ResNet50_Weights
import torch.nn as nn
from loguru import logger


class Model4Band(nn.Module):
    def __init__(self, model_type, num_inp_feats=4) -> None:
        super(Model4Band, self).__init__()
        self.first = nn.Conv2d(num_inp_feats, 3, 1) # changes from 4 channels to 3 channels
        if model_type == "deeplab_mobilenet":
            backbone = deeplabv3_mobilenet_v3_large(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            )
            trans = mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
        elif model_type == "deeplab_resnet":
            backbone = deeplabv3_resnet50(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            )
            trans = ResNet50_Weights.IMAGENET1K_V1.transforms()
        elif model_type == "lraspp":
            backbone = deeplabv3_resnet50(
                weights_backbone="IMAGENET1K_V1",
                num_classes=1,
            )
            trans = mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
        else:
            raise NotImplementedError
        self.crop_size = trans.__dict__["crop_size"][0]
        self.backbone = backbone

    def forward(self, x):
        x = self.first(x)
        out = self.backbone(x)
        return out