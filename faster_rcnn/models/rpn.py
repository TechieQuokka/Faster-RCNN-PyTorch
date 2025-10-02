"""
Region Proposal Network (RPN) 구현
- 객체 존재 여부 예측 (objectness)
- Bounding Box 회귀
- Proposal 생성 및 샘플링
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.anchor_generator import AnchorGenerator
from ..utils.bbox_tools import encode_boxes, decode_boxes, clip_boxes_to_image, remove_small_boxes, bbox_iou
from ..utils.nms import nms


class RPNHead(nn.Module):
    """
    RPN Head 네트워크

    Args:
        in_channels: 입력 채널 수 (Backbone 출력)
        num_anchors: 위치당 Anchor 개수
    """

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        # 3x3 Convolution (sliding window)
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)

        # Objectness 예측 (객체/배경)
        self.cls_logits = nn.Conv2d(512, num_anchors, kernel_size=1, stride=1)

        # Bounding Box 회귀 (4개 좌표)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, C, H, W] Feature Map

        Returns:
            objectness: Tensor [B, num_anchors * H * W] Objectness scores
            bbox_deltas: Tensor [B, num_anchors * H * W, 4] Box deltas
        """
        # Sliding window convolution
        t = F.relu(self.conv(x))

        # Objectness 예측
        objectness = self.cls_logits(t)  # [B, num_anchors, H, W]
        objectness = objectness.permute(0, 2, 3, 1).flatten(1)  # [B, H*W*num_anchors]

        # Bounding Box 델타 예측
        bbox_deltas = self.bbox_pred(t)  # [B, num_anchors*4, H, W]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)  # [B, H*W*num_anchors, 4]

        return objectness, bbox_deltas


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network

    Args:
        in_channels: Backbone 출력 채널 수
        anchor_sizes: Anchor 크기
        anchor_ratios: Anchor 종횡비
        stride: Feature Map stride
        nms_thresh: NMS IoU 임계값
        min_size: 최소 박스 크기
        pre_nms_top_n_train: NMS 전 제안 개수 (학습)
        pre_nms_top_n_test: NMS 전 제안 개수 (테스트)
        post_nms_top_n_train: NMS 후 제안 개수 (학습)
        post_nms_top_n_test: NMS 후 제안 개수 (테스트)
    """

    def __init__(
        self,
        in_channels,
        anchor_sizes=(128, 256, 512),
        anchor_ratios=(0.5, 1.0, 2.0),
        stride=16,
        nms_thresh=0.7,
        min_size=16,
        pre_nms_top_n_train=12000,
        pre_nms_top_n_test=6000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=300,
    ):
        super().__init__()

        # Anchor 생성기
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
            stride=stride
        )

        # RPN Head
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = RPNHead(in_channels, num_anchors)

        # 파라미터
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.pre_nms_top_n = {
            'training': pre_nms_top_n_train,
            'testing': pre_nms_top_n_test
        }
        self.post_nms_top_n = {
            'training': post_nms_top_n_train,
            'testing': post_nms_top_n_test
        }

    def forward(self, features, image_size):
        """
        Args:
            features: Tensor [B, C, H, W] Feature Map
            image_size: (height, width) 원본 이미지 크기

        Returns:
            proposals: List[Tensor] 각 이미지별 제안 박스 [num_proposals, 4]
            objectness: Tensor [B, num_anchors] Objectness scores
            bbox_deltas: Tensor [B, num_anchors, 4] Box deltas
        """
        # RPN Head 통과
        objectness, bbox_deltas = self.head(features)

        # Anchor 생성
        feature_map_size = features.shape[-2:]
        anchors = self.anchor_generator(
            feature_map_size,
            image_size,
            features.device
        )  # [num_anchors, 4]

        # Proposal 생성
        proposals = self.generate_proposals(
            anchors,
            objectness,
            bbox_deltas,
            image_size
        )

        return proposals, objectness, bbox_deltas

    def generate_proposals(self, anchors, objectness, bbox_deltas, image_size):
        """
        Proposal 생성 및 필터링

        Args:
            anchors: Tensor [num_anchors, 4] Anchor 박스
            objectness: Tensor [B, num_anchors] Objectness scores
            bbox_deltas: Tensor [B, num_anchors, 4] Box deltas
            image_size: (height, width)

        Returns:
            proposals: List[Tensor] 각 이미지별 제안 박스
        """
        batch_size = objectness.size(0)
        mode = 'training' if self.training else 'testing'

        proposals_list = []

        for i in range(batch_size):
            # Objectness와 Box deltas
            scores = objectness[i]  # [num_anchors]
            deltas = bbox_deltas[i]  # [num_anchors, 4]

            # Box 디코딩
            proposals = decode_boxes(deltas, anchors)  # [num_anchors, 4]

            # 이미지 경계 내로 클리핑
            proposals = clip_boxes_to_image(proposals, image_size)

            # 작은 박스 제거
            keep = remove_small_boxes(proposals, self.min_size)
            proposals = proposals[keep]
            scores = scores[keep]

            # Top-N 선택 (NMS 전)
            pre_nms_top_n = min(self.pre_nms_top_n[mode], len(scores))
            top_n_idx = scores.topk(pre_nms_top_n)[1]
            proposals = proposals[top_n_idx]
            scores = scores[top_n_idx]

            # NMS
            keep = nms(proposals, scores, self.nms_thresh)
            keep = keep[:self.post_nms_top_n[mode]]
            proposals = proposals[keep]

            proposals_list.append(proposals)

        return proposals_list

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """
        Anchor에 GT 할당 (학습용)

        Args:
            anchors: Tensor [num_anchors, 4] Anchor 박스
            gt_boxes: List[Tensor] 각 이미지별 GT 박스

        Returns:
            labels: List[Tensor] 각 이미지별 레이블 (1: positive, 0: negative, -1: ignore)
            matched_gt_boxes: List[Tensor] 각 이미지별 매칭된 GT 박스
        """
        labels_list = []
        matched_gt_boxes_list = []

        for gt_boxes_per_image in gt_boxes:
            if gt_boxes_per_image.numel() == 0:
                # GT가 없는 경우
                device = anchors.device
                labels = torch.zeros(len(anchors), dtype=torch.float32, device=device)
                matched_gt_boxes = torch.zeros_like(anchors)
                labels_list.append(labels)
                matched_gt_boxes_list.append(matched_gt_boxes)
                continue

            # IoU 계산
            match_quality_matrix = bbox_iou(gt_boxes_per_image, anchors)  # [num_gt, num_anchors]

            # 각 Anchor에 대해 가장 높은 IoU를 가진 GT 찾기
            matched_vals, matches = match_quality_matrix.max(dim=0)  # [num_anchors]

            # 레이블 초기화 (-1: ignore)
            labels = torch.full((len(anchors),), -1, dtype=torch.float32, device=anchors.device)

            # Negative 샘플 (IoU < 0.3)
            labels[matched_vals < 0.3] = 0

            # Positive 샘플 (IoU > 0.7)
            labels[matched_vals >= 0.7] = 1

            # 각 GT에 대해 가장 높은 IoU를 가진 Anchor도 Positive로 설정
            _, best_anchor_for_gt = match_quality_matrix.max(dim=1)
            labels[best_anchor_for_gt] = 1

            # 매칭된 GT 박스
            matched_gt_boxes = gt_boxes_per_image[matches]

            labels_list.append(labels)
            matched_gt_boxes_list.append(matched_gt_boxes)

        return labels_list, matched_gt_boxes_list
