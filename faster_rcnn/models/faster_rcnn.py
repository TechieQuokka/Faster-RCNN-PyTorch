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

        # 이미지 크기 제한 (정보용, 실제 리사이즈는 Dataset에서 수행)
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, images, targets=None):
        """
        Args:
            images: List[Tensor] 또는 Tensor [B, 3, H, W]
                (이미 Dataset에서 정규화 및 리사이즈 완료)
            targets: List[Dict] GT 타겟 (학습시)
                각 Dict는 'boxes', 'labels', 'original_size', 'scale_factors' 포함

        Returns:
            학습 모드: losses (Dict)
            추론 모드: detections (List[Dict])
        """
        # 입력 검증
        if self.training and targets is None:
            raise ValueError("학습 모드에서는 targets가 필요합니다")

        # 이미지 리스트 확인
        if not isinstance(images, list):
            images = [images]

        # 원본 이미지 크기 저장 (추론 시 박스 복원용)
        original_image_sizes = []
        image_sizes = []

        for i, img in enumerate(images):
            image_sizes.append(img.shape[-2:])

            if not self.training and targets is not None and i < len(targets):
                if 'original_size' in targets[i]:
                    original_image_sizes.append(tuple(targets[i]['original_size']))
                else:
                    original_image_sizes.append(img.shape[-2:])
            else:
                original_image_sizes.append(img.shape[-2:])

        # 배치 크기
        batch_size = len(images)

        # 이미지를 하나의 배치로 처리하기 위해 패딩
        # 최대 크기 찾기
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)

        # 패딩된 이미지 생성
        padded_images = []
        for img in images:
            h, w = img.shape[-2:]
            padded_img = torch.zeros((3, max_h, max_w), dtype=img.dtype, device=img.device)
            padded_img[:, :h, :w] = img
            padded_images.append(padded_img)

        # 배치로 스택
        images_batched = torch.stack(padded_images, dim=0)

        # Backbone을 통한 특징 추출
        features = self.backbone(images_batched)

        # RPN (첫 번째 이미지 크기 사용, 모든 이미지가 패딩되어 같은 크기)
        proposals, rpn_objectness, rpn_bbox_deltas = self.rpn(features, (max_h, max_w))

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
            image_sizes: List[(height, width)] 리사이즈된 크기
            original_image_sizes: List[(height, width)] 원본 크기

        Returns:
            detections: List[Dict] 스케일 복원된 검출 결과
        """
        for i, (detection, resized_size, original_size) in enumerate(
            zip(detections, image_sizes, original_image_sizes)
        ):
            boxes = detection['boxes']

            if len(boxes) > 0:
                # 스케일 팩터 계산
                resized_h, resized_w = resized_size
                original_h, original_w = original_size

                scale_x = original_w / resized_w
                scale_y = original_h / resized_h

                # 박스 좌표 스케일 복원
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                # 원본 이미지 경계 내로 클리핑
                boxes[:, 0].clamp_(min=0, max=original_w)
                boxes[:, 1].clamp_(min=0, max=original_h)
                boxes[:, 2].clamp_(min=0, max=original_w)
                boxes[:, 3].clamp_(min=0, max=original_h)

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
