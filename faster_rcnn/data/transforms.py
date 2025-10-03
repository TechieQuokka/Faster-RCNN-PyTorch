"""
데이터 변환 및 증강
- 리사이즈
- 정규화
- 랜덤 수평 뒤집기
- ToTensor 변환
"""

import torch
import torchvision.transforms.functional as F
import random
from PIL import Image


class Compose:
    """여러 변환을 순차적으로 적용"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """PIL Image를 Tensor로 변환"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Resize:
    """
    이미지 리사이즈 (shortest side 기준)

    Args:
        min_size: 최소 크기
        max_size: 최대 크기
    """

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # 원본 크기
        w, h = image.size

        # 스케일 계산
        size = self.min_size
        if self.max_size is not None:
            min_original_size = float(min(w, h))
            max_original_size = float(max(w, h))

            if max_original_size / min_original_size * size > self.max_size:
                size = int(round(self.max_size * min_original_size / max_original_size))

        # 새 크기 계산
        if w < h:
            new_w = size
            new_h = int(size * h / w)
        else:
            new_h = size
            new_w = int(size * w / h)

        # 리사이즈
        image = F.resize(image, (new_h, new_w))

        # 박스 스케일 조정 및 메타데이터 저장
        if target is not None:
            scale_x = new_w / w
            scale_y = new_h / h

            # 스케일 팩터 저장 (추론 시 박스 복원용)
            target['scale_factors'] = torch.tensor([scale_x, scale_y, scale_x, scale_y])
            target['original_size'] = (h, w)
            target['resized_size'] = (new_h, new_w)

            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    """
    랜덤 수평 뒤집기

    Args:
        prob: 뒤집기 확률
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 이미지 뒤집기
            image = F.hflip(image)

            # 박스 뒤집기
            if target is not None and 'boxes' in target:
                width = image.width if isinstance(image, Image.Image) else image.shape[-1]
                boxes = target['boxes']
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes

        return image, target


class Normalize:
    """
    이미지 정규화

    Args:
        mean: 평균 (RGB)
        std: 표준편차 (RGB)
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ColorJitter:
    """
    색상 증강

    Args:
        brightness: 밝기 조정 범위
        contrast: 대비 조정 범위
        saturation: 채도 조정 범위
        hue: 색조 조정 범위
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        # 밝기
        if self.brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - self.brightness), 1 + self.brightness
            )
            image = F.adjust_brightness(image, brightness_factor)

        # 대비
        if self.contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - self.contrast), 1 + self.contrast
            )
            image = F.adjust_contrast(image, contrast_factor)

        # 채도
        if self.saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - self.saturation), 1 + self.saturation
            )
            image = F.adjust_saturation(image, saturation_factor)

        # 색조
        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            image = F.adjust_hue(image, hue_factor)

        return image, target


def get_transform(train=True, config=None, min_size=600, max_size=1000):
    """
    데이터 변환 파이프라인 생성

    Args:
        train: 학습용 변환 여부
        config: 설정 객체 (우선순위 높음)
        min_size: 최소 이미지 크기 (config 없을 때)
        max_size: 최대 이미지 크기 (config 없을 때)

    Returns:
        transforms: Compose 객체
    """
    transforms = []

    # Config에서 파라미터 가져오기
    if config is not None:
        min_size = config.transforms.min_size
        max_size = config.transforms.max_size

    # 리사이즈 (PIL Image에서 수행)
    transforms.append(Resize(min_size, max_size))

    # 학습용 증강 (PIL Image에서 수행)
    if train:
        if config is not None and config.transforms.train_augmentation:
            # Color Jitter
            if config.transforms.get('color_jitter', False):
                transforms.append(ColorJitter(
                    brightness=config.transforms.get('brightness', 0.2),
                    contrast=config.transforms.get('contrast', 0.2),
                    saturation=config.transforms.get('saturation', 0.2),
                    hue=config.transforms.get('hue', 0.1)
                ))

            # Horizontal Flip
            flip_prob = config.transforms.get('horizontal_flip_prob', 0.5)
            if flip_prob > 0:
                transforms.append(RandomHorizontalFlip(flip_prob))
        else:
            # 기본 증강 (config 없을 때)
            transforms.append(RandomHorizontalFlip(0.5))

    # Tensor 변환
    transforms.append(ToTensor())

    # 정규화 (Tensor에서 수행)
    transforms.append(Normalize())

    return Compose(transforms)
