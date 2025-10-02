"""
Faster R-CNN 통합 모델
- Backbone, RPN, RoI Heads 통합
- 학습 및 추론 파이프라인
"""

import torch
import torch.nn as nn
from .backbone import build_backbone
from .rpn import RegionProposalNetwork
from .roi_head import RoIHeads
from ..utils.bbox_tools import encode_boxes


class FasterRCNN(nn.Module):
    """
    Faster R-CNN 모델

    Args:
        num_classes: 클래스 개수 (배경 포함)
        backbone_name: Backbone 이름 ('resnet50' or 'vgg16')
        pretrained_backbone: 사전학습된 Backbone 사용 여부
        min_size: 입력 이미지 최소 크기
        max_size: 입력 이미지 최대 크기
        **kwargs: RPN 및 RoI Heads 파라미터
    """

    def __init__(
        self,
        num_classes,
        backbone_name='resnet50',
        pretrained_backbone=True,
        min_size=600,
        max_size=1000,
        # RPN 파라미터
        rpn_anchor_sizes=(128, 256, 512),
        rpn_anchor_ratios=(0.5, 1.0, 2.0),
        rpn_nms_thresh=0.7,
        rpn_pre_nms_top_n_train=12000,
        rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=300,
        # RoI Heads 파라미터
        roi_output_size=7,
        roi_fg_iou_thresh=0.5,
        roi_bg_iou_thresh=0.5,
        roi_batch_size_per_image=128,
        roi_positive_fraction=0.25,
        roi_score_thresh=0.05,
        roi_nms_thresh=0.3,
        roi_detection_per_img=100,
    ):
        super().__init__()

        # Backbone
        self.backbone = build_backbone(backbone_name, pretrained=pretrained_backbone)

        # RPN
        self.rpn = RegionProposalNetwork(
            in_channels=self.backbone.out_channels,
            anchor_sizes=rpn_anchor_sizes,
            anchor_ratios=rpn_anchor_ratios,
            nms_thresh=rpn_nms_thresh,
            pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            post_nms_top_n_train=rpn_post_nms_top_n_train,
            post_nms_top_n_test=rpn_post_nms_top_n_test,
        )

        # RoI Heads
        self.roi_heads = RoIHeads(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            roi_output_size=roi_output_size,
            fg_iou_thresh=roi_fg_iou_thresh,
            bg_iou_thresh=roi_bg_iou_thresh,
            batch_size_per_image=roi_batch_size_per_image,
            positive_fraction=roi_positive_fraction,
            score_thresh=roi_score_thresh,
            nms_thresh=roi_nms_thresh,
            detection_per_img=roi_detection_per_img,
        )

        # 이미지 크기 제한
        self.min_size = min_size
        self.max_size = max_size

        # 이미지 정규화 (ImageNet 통계)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def forward(self, images, targets=None):
        """
        Args:
            images: List[Tensor] 또는 Tensor [B, 3, H, W]
            targets: List[Dict] GT 타겟 (학습시)
                각 Dict는 'boxes'와 'labels' 포함

        Returns:
            학습 모드: losses (Dict)
            추론 모드: detections (List[Dict])
        """
        # 입력 검증
        if self.training and targets is None:
            raise ValueError("학습 모드에서는 targets가 필요합니다")

        # 이미지 전처리
        original_image_sizes = []
        if isinstance(images, list):
            for img in images:
                original_image_sizes.append((img.shape[-2], img.shape[-1]))
        else:
            for i in range(images.shape[0]):
                original_image_sizes.append((images.shape[-2], images.shape[-1]))

        # 이미지 정규화
        images, targets = self.transform(images, targets)

        # Backbone을 통한 특징 추출
        features = self.backbone(images)

        # 이미지 크기
        image_sizes = [images.shape[-2:]] * images.shape[0]

        # RPN
        proposals, rpn_objectness, rpn_bbox_deltas = self.rpn(features, image_sizes[0])

        if self.training:
            # 학습 모드
            # RPN 타겟 준비
            rpn_targets = self.prepare_rpn_targets(features, targets, image_sizes[0])

            # RoI Heads
            det_outputs, det_labels, det_regression_targets = self.roi_heads(
                features, proposals, image_sizes, targets
            )

            # 손실 계산
            from ..utils.loss import FasterRCNNLoss
            criterion = FasterRCNNLoss()

            # RPN 출력 준비
            rpn_outputs = {
                'objectness': rpn_objectness,
                'bbox_deltas': rpn_bbox_deltas,
            }

            # Detection 출력 준비
            det_outputs_dict = {
                'class_logits': det_outputs['class_logits'],
                'box_regression': det_outputs['box_regression'],
            }

            # 타겟 준비
            targets_dict = {
                'rpn_labels': rpn_targets['labels'],
                'rpn_bbox_targets': rpn_targets['bbox_targets'],
                'det_labels': det_labels,
                'det_bbox_targets': det_regression_targets,
            }

            losses = criterion(rpn_outputs, det_outputs_dict, targets_dict)
            return losses

        else:
            # 추론 모드
            detections = self.roi_heads(features, proposals, image_sizes)

            # 원본 이미지 크기로 박스 스케일 복원
            detections = self.transform_detections(detections, image_sizes, original_image_sizes)

            return detections

    def transform(self, images, targets=None):
        """
        이미지 정규화 및 배치 처리

        Args:
            images: List[Tensor] 또는 Tensor
            targets: List[Dict] (옵션)

        Returns:
            images: Tensor [B, 3, H, W]
            targets: List[Dict] (옵션)
        """
        if isinstance(images, list):
            images = torch.stack(images)

        # 정규화
        device = images.device
        mean = torch.tensor(self.image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images, targets

    def prepare_rpn_targets(self, features, targets, image_size):
        """
        RPN 학습 타겟 준비

        Args:
            features: Feature Map
            targets: List[Dict] GT 타겟
            image_size: (height, width)

        Returns:
            rpn_targets: Dict with 'labels' and 'bbox_targets'
        """
        # Anchor 생성
        feature_map_size = features.shape[-2:]
        anchors = self.rpn.anchor_generator(
            feature_map_size, image_size, features.device
        )

        # GT 박스 추출
        gt_boxes_list = [t['boxes'] for t in targets]

        # Anchor 매칭
        labels_list, matched_gt_boxes_list = self.rpn.assign_targets_to_anchors(
            anchors, gt_boxes_list
        )

        # 회귀 타겟 계산
        bbox_targets_list = []
        for matched_gt_boxes in matched_gt_boxes_list:
            bbox_targets = encode_boxes(matched_gt_boxes, anchors)
            bbox_targets_list.append(bbox_targets)

        # Batch로 합치기
        labels = torch.stack(labels_list, dim=0)  # [B, num_anchors]
        bbox_targets = torch.stack(bbox_targets_list, dim=0)  # [B, num_anchors, 4]

        return {
            'labels': labels,
            'bbox_targets': bbox_targets
        }

    def transform_detections(self, detections, image_sizes, original_image_sizes):
        """
        검출 박스를 원본 이미지 크기로 스케일 복원

        Args:
            detections: List[Dict]
            image_sizes: List[(height, width)] 정규화된 크기
            original_image_sizes: List[(height, width)] 원본 크기

        Returns:
            detections: List[Dict] 스케일 복원된 검출 결과
        """
        for i, (detection, original_size) in enumerate(zip(detections, original_image_sizes)):
            boxes = detection['boxes']
            # 여기서는 크기가 같다고 가정 (실제로는 리사이즈 로직 필요)
            detections[i]['boxes'] = boxes

        return detections


def build_faster_rcnn(num_classes, **kwargs):
    """
    Faster R-CNN 모델 생성 팩토리 함수

    Args:
        num_classes: 클래스 개수 (배경 포함)
        **kwargs: 모델 파라미터

    Returns:
        model: Faster R-CNN 모델
    """
    return FasterRCNN(num_classes=num_classes, **kwargs)
