"""
배치 처리를 위한 Collate 함수
- 가변 크기 이미지 및 박스 처리
"""

import torch


def collate_fn(batch):
    """
    커스텀 Collate 함수

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: List[Tensor] 각 이미지
        targets: List[Dict] 각 타겟
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return images, targets


def collate_fn_batched(batch):
    """
    배치 텐서로 변환하는 Collate 함수
    (모든 이미지 크기가 같을 때만 사용 가능)

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: Tensor [B, 3, H, W]
        targets: List[Dict] 각 타겟
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # 이미지를 배치 텐서로 변환
    images = torch.stack(images, dim=0)

    return images, targets
