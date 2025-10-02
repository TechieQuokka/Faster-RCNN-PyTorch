"""
Anchor 생성기
- 다양한 스케일과 종횡비를 가진 Anchor 박스 생성
- Feature Map의 각 위치에 대해 Anchor 생성
"""

import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    """
    Anchor 박스 생성기

    Args:
        sizes: Anchor 크기 리스트 (예: [128, 256, 512])
        aspect_ratios: 종횡비 리스트 (예: [0.5, 1.0, 2.0])
        stride: Feature Map의 stride (예: 16)
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0), stride=16):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.num_anchors = len(sizes) * len(aspect_ratios)

    def generate_base_anchors(self):
        """
        기본 Anchor 템플릿 생성 (0, 0 위치 기준)

        Returns:
            anchors: Tensor [num_anchors, 4] (x1, y1, x2, y2)
        """
        anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                # 종횡비를 고려한 너비와 높이 계산
                h = size / torch.sqrt(torch.tensor(ratio))
                w = size * torch.sqrt(torch.tensor(ratio))

                # 중심이 (0, 0)인 박스 생성
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2

                anchors.append([x1, y1, x2, y2])

        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, feature_map_size, image_size, device):
        """
        Feature Map 전체에 대한 Anchor 생성

        Args:
            feature_map_size: (height, width) Feature Map 크기
            image_size: (height, width) 원본 이미지 크기
            device: 텐서가 생성될 디바이스

        Returns:
            anchors: Tensor [num_anchors_total, 4] 모든 Anchor 박스
        """
        feat_h, feat_w = feature_map_size
        image_h, image_w = image_size

        # 기본 Anchor 템플릿 생성
        base_anchors = self.generate_base_anchors().to(device)

        # Feature Map의 각 위치에 대한 중심점 계산
        shift_x = torch.arange(0, feat_w, device=device) * self.stride
        shift_y = torch.arange(0, feat_h, device=device) * self.stride

        # Grid 생성
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        # Shift 텐서 생성 [num_locations, 4]
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

        # 모든 위치에 대해 Anchor 생성
        # shifts: [num_locations, 4]
        # base_anchors: [num_anchors, 4]
        # anchors: [num_locations, num_anchors, 4] -> [num_locations * num_anchors, 4]
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)

        # 이미지 경계를 벗어난 Anchor는 클리핑
        anchors[:, 0].clamp_(min=0, max=image_w)
        anchors[:, 1].clamp_(min=0, max=image_h)
        anchors[:, 2].clamp_(min=0, max=image_w)
        anchors[:, 3].clamp_(min=0, max=image_h)

        return anchors

    def num_anchors_per_location(self):
        """위치당 Anchor 개수 반환"""
        return self.num_anchors
