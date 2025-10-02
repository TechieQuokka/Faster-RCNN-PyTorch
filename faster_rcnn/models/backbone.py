"""
Backbone Network 구현
- ResNet-50 기반 특징 추출기
- 사전학습된 가중치 로드
- Feature Map 출력
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    ResNet-50 Backbone

    Args:
        pretrained: 사전학습된 가중치 사용 여부
        freeze_bn: Batch Normalization 레이어 고정 여부
    """

    def __init__(self, pretrained=True, freeze_bn=True):
        super().__init__()

        # ResNet-50 로드
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        # ResNet의 레이어 추출 (conv5_x까지 사용)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

        # Feature Map 출력 채널 수
        self.out_channels = 2048  # ResNet-50의 layer4 출력 채널

        # Batch Normalization 레이어 고정
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, H, W] 입력 이미지

        Returns:
            features: Tensor [B, 2048, H/16, W/16] Feature Map
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def freeze_bn(self):
        """Batch Normalization 레이어 고정"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


class VGG16Backbone(nn.Module):
    """
    VGG16 Backbone (원논문 구현)

    Args:
        pretrained: 사전학습된 가중치 사용 여부
    """

    def __init__(self, pretrained=True):
        super().__init__()

        from torchvision.models import vgg16, VGG16_Weights

        # VGG16 로드
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            vgg = vgg16(weights=weights)
        else:
            vgg = vgg16(weights=None)

        # VGG16의 features 추출 (conv5_3까지)
        # VGG16 features는 30개의 레이어로 구성
        # conv5_3은 인덱스 29까지
        self.features = vgg.features[:30]

        # Feature Map 출력 채널 수
        self.out_channels = 512  # VGG16의 conv5_3 출력 채널

    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, H, W] 입력 이미지

        Returns:
            features: Tensor [B, 512, H/16, W/16] Feature Map
        """
        return self.features(x)


def build_backbone(name='resnet50', pretrained=True, **kwargs):
    """
    Backbone 생성 팩토리 함수

    Args:
        name: 'resnet50' 또는 'vgg16'
        pretrained: 사전학습된 가중치 사용 여부
        **kwargs: 추가 인자

    Returns:
        backbone: Backbone 모델
    """
    if name == 'resnet50':
        return ResNetBackbone(pretrained=pretrained, **kwargs)
    elif name == 'vgg16':
        return VGG16Backbone(pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
