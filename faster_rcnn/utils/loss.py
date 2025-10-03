"""
Faster R-CNN 손실 함수
- RPN 손실 (분류 + 회귀)
- Detection 손실 (분류 + 회귀)
- Smooth L1 Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(pred, target, beta=1.0):
    """
    Smooth L1 Loss (Huber Loss)

    Args:
        pred: 예측값 Tensor [N, ...]
        target: 타겟값 Tensor [N, ...]
        beta: 전환점 (기본값: 1.0)

    Returns:
        loss: Smooth L1 Loss
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss


class RPNLoss(nn.Module):
    """
    RPN 손실 함수

    Args:
        lambda_reg: 회귀 손실 가중치 (기본값: 10.0)
    """

    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, objectness, pred_bbox_deltas, labels, bbox_targets):
        """
        Args:
            objectness: Tensor [N, num_anchors] 객체 존재 확률
            pred_bbox_deltas: Tensor [N, num_anchors, 4] 예측 박스 델타
            labels: Tensor [N, num_anchors] GT 레이블 (1: positive, 0: negative, -1: ignore)
            bbox_targets: Tensor [N, num_anchors, 4] GT 박스 델타

        Returns:
            total_loss: 전체 RPN 손실
            cls_loss: 분류 손실
            reg_loss: 회귀 손실
        """
        # 유효한 샘플 마스크 (ignore가 아닌 것들)
        valid_mask = labels >= 0
        num_valid = valid_mask.sum()

        # 분류 손실 (Binary Cross Entropy)
        if num_valid > 0:
            objectness_valid = objectness[valid_mask]
            labels_valid = labels[valid_mask].float()
            cls_loss = F.binary_cross_entropy_with_logits(
                objectness_valid, labels_valid, reduction='sum'
            ) / num_valid
        else:
            # 유효한 샘플이 없는 경우 (매우 드물지만 안전장치)
            cls_loss = torch.tensor(0.0, device=objectness.device, requires_grad=True)

        # 회귀 손실 (Smooth L1, positive 샘플에 대해서만)
        pos_mask = labels == 1
        num_pos = pos_mask.sum()

        if num_pos > 0:
            pred_bbox_deltas_pos = pred_bbox_deltas[pos_mask]
            bbox_targets_pos = bbox_targets[pos_mask]
            reg_loss = smooth_l1_loss(pred_bbox_deltas_pos, bbox_targets_pos).sum() / num_pos
        else:
            # Positive 샘플이 없는 경우
            reg_loss = torch.tensor(0.0, device=objectness.device, requires_grad=True)

        # 전체 손실
        total_loss = cls_loss + self.lambda_reg * reg_loss

        return total_loss, cls_loss, reg_loss


class DetectionLoss(nn.Module):
    """
    Detection Head 손실 함수

    Args:
        lambda_reg: 회귀 손실 가중치 (기본값: 10.0)
    """

    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, class_logits, box_regression, labels, regression_targets):
        """
        Args:
            class_logits: Tensor [N, num_classes] 클래스 로짓
            box_regression: Tensor [N, num_classes * 4] 예측 박스 델타
            labels: Tensor [N] GT 클래스 레이블 (0은 배경)
            regression_targets: Tensor [N, 4] GT 박스 델타

        Returns:
            total_loss: 전체 Detection 손실
            cls_loss: 분류 손실
            reg_loss: 회귀 손실
        """
        # 분류 손실 (Cross Entropy)
        cls_loss = F.cross_entropy(class_logits, labels)

        # 회귀 손실 (Smooth L1, positive 샘플에 대해서만)
        # 배경 클래스 (0)를 제외한 샘플
        pos_mask = labels > 0
        num_pos = pos_mask.sum()

        if num_pos > 0:
            # 해당 클래스의 박스만 선택
            labels_pos = labels[pos_mask]
            box_regression = box_regression.view(-1, box_regression.size(1) // 4, 4)

            # 각 샘플의 GT 클래스에 해당하는 박스 선택
            box_regression_pos = box_regression[pos_mask, labels_pos]
            regression_targets_pos = regression_targets[pos_mask]

            reg_loss = smooth_l1_loss(box_regression_pos, regression_targets_pos).sum() / num_pos
        else:
            # Positive 샘플이 없는 경우 (모두 배경)
            reg_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)

        # 전체 손실
        total_loss = cls_loss + self.lambda_reg * reg_loss

        return total_loss, cls_loss, reg_loss


class FasterRCNNLoss(nn.Module):
    """
    Faster R-CNN 전체 손실 함수
    """

    def __init__(self, lambda_rpn_reg=10.0, lambda_det_reg=10.0):
        super().__init__()
        self.rpn_loss = RPNLoss(lambda_reg=lambda_rpn_reg)
        self.det_loss = DetectionLoss(lambda_reg=lambda_det_reg)

    def forward(self, rpn_outputs, det_outputs, targets):
        """
        Args:
            rpn_outputs: RPN 출력 딕셔너리
            det_outputs: Detection Head 출력 딕셔너리
            targets: GT 타겟 딕셔너리

        Returns:
            losses: 모든 손실을 포함하는 딕셔너리
        """
        # RPN 손실
        rpn_total, rpn_cls, rpn_reg = self.rpn_loss(
            rpn_outputs['objectness'],
            rpn_outputs['bbox_deltas'],
            targets['rpn_labels'],
            targets['rpn_bbox_targets']
        )

        # Detection 손실
        det_total, det_cls, det_reg = self.det_loss(
            det_outputs['class_logits'],
            det_outputs['box_regression'],
            targets['det_labels'],
            targets['det_bbox_targets']
        )

        # 전체 손실
        total_loss = rpn_total + det_total

        losses = {
            'loss_total': total_loss,
            'loss_rpn': rpn_total,
            'loss_rpn_cls': rpn_cls,
            'loss_rpn_reg': rpn_reg,
            'loss_det': det_total,
            'loss_det_cls': det_cls,
            'loss_det_reg': det_reg,
        }

        return losses
