"""
Bounding Box 관련 유틸리티 함수
- IoU 계산
- Bounding Box 인코딩/디코딩
- Bounding Box 변환 (중심점 ↔ 좌표)
"""

import torch


def bbox_iou(boxes1, boxes2):
    """
    두 박스 집합 간의 IoU 계산

    Args:
        boxes1: Tensor [N, 4] (x1, y1, x2, y2)
        boxes2: Tensor [M, 4] (x1, y1, x2, y2)

    Returns:
        iou: Tensor [N, M] IoU 값
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 교집합 영역 계산
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # IoU 계산
    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou


def encode_boxes(gt_boxes, anchor_boxes):
    """
    Ground Truth 박스를 Anchor 박스 기준으로 인코딩

    Args:
        gt_boxes: Tensor [N, 4] (x1, y1, x2, y2)
        anchor_boxes: Tensor [N, 4] (x1, y1, x2, y2)

    Returns:
        encoded: Tensor [N, 4] (tx, ty, tw, th)
    """
    # 중심점 및 크기로 변환
    gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gt_ctr_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    anchor_ctr_x = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) * 0.5
    anchor_ctr_y = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) * 0.5
    anchor_w = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_h = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    # 인코딩
    tx = (gt_ctr_x - anchor_ctr_x) / anchor_w
    ty = (gt_ctr_y - anchor_ctr_y) / anchor_h
    tw = torch.log(gt_w / anchor_w)
    th = torch.log(gt_h / anchor_h)

    encoded = torch.stack([tx, ty, tw, th], dim=1)
    return encoded


def decode_boxes(deltas, anchor_boxes):
    """
    예측된 deltas를 실제 박스 좌표로 디코딩

    Args:
        deltas: Tensor [N, 4] (tx, ty, tw, th)
        anchor_boxes: Tensor [N, 4] (x1, y1, x2, y2)

    Returns:
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
    """
    # Anchor를 중심점 및 크기로 변환
    anchor_ctr_x = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) * 0.5
    anchor_ctr_y = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) * 0.5
    anchor_w = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_h = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    # 디코딩
    pred_ctr_x = deltas[:, 0] * anchor_w + anchor_ctr_x
    pred_ctr_y = deltas[:, 1] * anchor_h + anchor_ctr_y
    pred_w = torch.exp(deltas[:, 2]) * anchor_w
    pred_h = torch.exp(deltas[:, 3]) * anchor_h

    # 좌표 형식으로 변환
    pred_x1 = pred_ctr_x - pred_w * 0.5
    pred_y1 = pred_ctr_y - pred_h * 0.5
    pred_x2 = pred_ctr_x + pred_w * 0.5
    pred_y2 = pred_ctr_y + pred_h * 0.5

    boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    return boxes


def clip_boxes_to_image(boxes, image_shape):
    """
    박스 좌표를 이미지 경계 내로 제한

    Args:
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
        image_shape: (height, width)

    Returns:
        clipped_boxes: Tensor [N, 4]
    """
    height, width = image_shape
    boxes[:, 0].clamp_(min=0, max=width)
    boxes[:, 1].clamp_(min=0, max=height)
    boxes[:, 2].clamp_(min=0, max=width)
    boxes[:, 3].clamp_(min=0, max=height)
    return boxes


def remove_small_boxes(boxes, min_size):
    """
    너무 작은 박스 제거

    Args:
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
        min_size: 최소 크기 (픽셀)

    Returns:
        keep: Tensor [M] 유지할 박스의 인덱스
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep
