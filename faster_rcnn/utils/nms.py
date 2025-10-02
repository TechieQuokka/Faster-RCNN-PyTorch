"""
Non-Maximum Suppression (NMS) 구현
- 중복된 박스 제거
- IoU 기반 억제
"""

import torch
from .bbox_tools import bbox_iou


def nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression

    Args:
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
        scores: Tensor [N] 신뢰도 점수
        iou_threshold: IoU 임계값

    Returns:
        keep: Tensor [M] 유지할 박스의 인덱스
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 점수 기준 내림차순 정렬
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        # 가장 높은 점수의 박스 선택
        i = order[0]
        keep.append(i.item())

        # 나머지 박스들과 IoU 계산
        ious = bbox_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]

        # IoU가 임계값보다 낮은 박스만 유지
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    클래스별로 분리하여 NMS 수행

    Args:
        boxes: Tensor [N, 4] (x1, y1, x2, y2)
        scores: Tensor [N] 신뢰도 점수
        idxs: Tensor [N] 클래스 인덱스
        iou_threshold: IoU 임계값

    Returns:
        keep: Tensor [M] 유지할 박스의 인덱스
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 클래스별로 박스를 분리하기 위한 오프셋 추가
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
