"""
RoI Head 구현
- RoI Pooling/Align
- Detection Head (분류 + 박스 회귀)
- Proposal 샘플링 및 타겟 할당
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool, RoIAlign
from ..utils.bbox_tools import bbox_iou, encode_boxes, decode_boxes, clip_boxes_to_image
from ..utils.nms import batched_nms


class TwoMLPHead(nn.Module):
    """
    두 개의 FC 레이어로 구성된 Head

    Args:
        in_channels: 입력 채널 수
        representation_size: 중간 레이어 크기 (기본값: 4096)
    """

    def __init__(self, in_channels, representation_size=4096):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    Fast R-CNN 예측 Head

    Args:
        in_channels: 입력 채널 수
        num_classes: 클래스 개수 (배경 포함)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # 분류 레이어
        self.cls_score = nn.Linear(in_channels, num_classes)

        # 박스 회귀 레이어 (클래스별로 4개 좌표)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        """
        Args:
            x: Tensor [N, in_channels]

        Returns:
            class_logits: Tensor [N, num_classes]
            box_regression: Tensor [N, num_classes * 4]
        """
        class_logits = self.cls_score(x)
        box_regression = self.bbox_pred(x)
        return class_logits, box_regression


class RoIHeads(nn.Module):
    """
    RoI Heads (RoI Pooling + Detection Head)

    Args:
        in_channels: Feature Map 채널 수
        num_classes: 클래스 개수 (배경 포함)
        roi_output_size: RoI Pooling 출력 크기 (기본값: 7)
        sampling_ratio: RoIAlign 샘플링 비율 (기본값: 2)
        fg_iou_thresh: Foreground IoU 임계값 (기본값: 0.5)
        bg_iou_thresh: Background IoU 임계값 (기본값: 0.5)
        batch_size_per_image: 이미지당 RoI 샘플 개수 (기본값: 128)
        positive_fraction: Positive 샘플 비율 (기본값: 0.25)
        score_thresh: 추론시 점수 임계값 (기본값: 0.05)
        nms_thresh: NMS IoU 임계값 (기본값: 0.3)
        detection_per_img: 이미지당 최대 검출 개수 (기본값: 100)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        roi_output_size=7,
        sampling_ratio=2,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=128,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.3,
        detection_per_img=100,
    ):
        super().__init__()

        # RoI Align (더 정확한 특징 추출)
        self.roi_align = RoIAlign(
            output_size=roi_output_size,
            spatial_scale=1.0 / 16,  # Feature Map stride
            sampling_ratio=sampling_ratio
        )

        # Detection Head
        resolution = roi_output_size
        representation_size = 4096
        self.head = TwoMLPHead(
            in_channels * resolution * resolution,
            representation_size
        )

        # Predictor
        self.predictor = FastRCNNPredictor(representation_size, num_classes)

        # 파라미터
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img
        self.num_classes = num_classes

    def forward(self, features, proposals, image_sizes, targets=None):
        """
        Args:
            features: Tensor [B, C, H, W] Feature Map
            proposals: List[Tensor] RPN 제안 박스
            image_sizes: List[(height, width)] 이미지 크기
            targets: List[Dict] GT 타겟 (학습시)

        Returns:
            detections: List[Dict] 검출 결과 (추론시)
            losses: Dict 손실 (학습시)
        """
        if self.training:
            # 학습 모드: Proposal 샘플링 및 타겟 할당
            proposals, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )

        # RoI Pooling/Align
        box_features = self.roi_align(features, proposals)

        # Detection Head
        box_features = self.head(box_features)
        class_logits, box_regression = self.predictor(box_features)

        if self.training:
            # 학습 모드
            return {
                'class_logits': class_logits,
                'box_regression': box_regression
            }, labels, regression_targets
        else:
            # 추론 모드
            detections = self.postprocess_detections(
                class_logits, box_regression, proposals, image_sizes
            )
            return detections

    def select_training_samples(self, proposals, targets):
        """
        학습용 Proposal 샘플링

        Args:
            proposals: List[Tensor] RPN 제안 박스
            targets: List[Dict] GT 타겟

        Returns:
            proposals: List[Tensor] 샘플링된 Proposal
            labels: Tensor [total_sampled] 클래스 레이블
            regression_targets: Tensor [total_sampled, 4] 회귀 타겟
        """
        sampled_proposals = []
        labels_list = []
        regression_targets_list = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            gt_boxes = targets_per_image['boxes']
            gt_labels = targets_per_image['labels']

            # IoU 계산
            match_quality_matrix = bbox_iou(gt_boxes, proposals_per_image)
            matched_vals, matches = match_quality_matrix.max(dim=0)

            # Foreground/Background 분류
            labels_per_image = gt_labels[matches]
            labels_per_image[matched_vals < self.bg_iou_thresh] = 0  # Background

            # Positive/Negative 샘플링
            positive = torch.where(matched_vals >= self.fg_iou_thresh)[0]
            negative = torch.where(matched_vals < self.bg_iou_thresh)[0]

            # 샘플 개수 결정
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            # 랜덤 샘플링
            perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx = positive[perm_pos]
            neg_idx = negative[perm_neg]

            sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)

            # 샘플링된 Proposal 및 레이블
            sampled_proposals.append(proposals_per_image[sampled_idx])
            labels_list.append(labels_per_image[sampled_idx])

            # 회귀 타겟 (Positive 샘플에 대해서만)
            matched_gt_boxes = gt_boxes[matches[sampled_idx]]
            regression_targets = encode_boxes(
                matched_gt_boxes, proposals_per_image[sampled_idx]
            )
            regression_targets_list.append(regression_targets)

        # Batch로 합치기
        labels = torch.cat(labels_list, dim=0)
        regression_targets = torch.cat(regression_targets_list, dim=0)

        return sampled_proposals, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_sizes):
        """
        추론 결과 후처리

        Args:
            class_logits: Tensor [N, num_classes]
            box_regression: Tensor [N, num_classes * 4]
            proposals: List[Tensor] Proposal 박스
            image_sizes: List[(height, width)]

        Returns:
            detections: List[Dict] 검출 결과
        """
        # 클래스별 확률
        pred_scores = F.softmax(class_logits, dim=-1)

        # Box regression reshape
        box_regression = box_regression.view(-1, self.num_classes, 4)

        # 각 이미지별로 분리
        detections = []
        boxes_per_image = [len(p) for p in proposals]
        pred_scores_split = pred_scores.split(boxes_per_image, dim=0)
        box_regression_split = box_regression.split(boxes_per_image, dim=0)

        for scores, boxes_delta, proposals_per_image, image_size in zip(
            pred_scores_split, box_regression_split, proposals, image_sizes
        ):
            # 배경 클래스 제외
            scores = scores[:, 1:]  # [num_proposals, num_classes - 1]

            # 각 클래스별로 박스 디코딩
            boxes = []
            for class_idx in range(self.num_classes - 1):
                boxes_per_class = decode_boxes(
                    boxes_delta[:, class_idx + 1], proposals_per_image
                )
                boxes_per_class = clip_boxes_to_image(boxes_per_class, image_size)
                boxes.append(boxes_per_class)

            boxes = torch.stack(boxes, dim=1)  # [num_proposals, num_classes - 1, 4]

            # 점수 임계값 적용
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = torch.arange(1, self.num_classes, device=scores.device)
            labels = labels.repeat_interleave(len(proposals_per_image))

            # 낮은 점수 제거
            keep = torch.where(scores > self.score_thresh)[0]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # NMS
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detection_per_img]

            detections.append({
                'boxes': boxes[keep],
                'labels': labels[keep],
                'scores': scores[keep]
            })

        return detections
