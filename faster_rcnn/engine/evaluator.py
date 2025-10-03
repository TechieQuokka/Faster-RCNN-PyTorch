"""
평가 및 메트릭 계산
- mAP 계산
- Precision-Recall 곡선
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ..utils.bbox_tools import bbox_iou


class COCOEvaluator:
    """
    COCO 스타일 평가기 (mAP@[0.5:0.95])

    Args:
        iou_thresholds: IoU 임계값 리스트
    """

    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            # COCO 기본값: 0.5부터 0.95까지 0.05 간격
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = iou_thresholds

    @torch.no_grad()
    def evaluate(self, model, data_loader, device='cuda'):
        """
        모델 평가

        Args:
            model: Faster R-CNN 모델
            data_loader: 평가 데이터 로더
            device: 디바이스

        Returns:
            metrics: 평가 메트릭 딕셔너리
        """
        model.eval()
        model.to(device)

        # 예측 및 GT 수집
        all_predictions = []
        all_targets = []

        pbar = tqdm(data_loader, desc='Evaluation')

        for images, targets in pbar:
            images = [img.to(device) for img in images]

            # 추론
            predictions = model(images)

            # CPU로 이동
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            # Move only tensor values to CPU (skip tuples like original_size, resized_size)
            targets = [{k: v.cpu() if torch.is_tensor(v) else v
                        for k, v in t.items()} for t in targets]

            # **FIX**: predictions에 image_id 추가 (targets에서 복사)
            for pred, target in zip(predictions, targets):
                pred['image_id'] = target['image_id']

            all_predictions.extend(predictions)
            all_targets.extend(targets)

        # mAP 계산
        metrics = self.compute_map(all_predictions, all_targets)

        return metrics

    def compute_map(self, predictions, targets):
        """
        mAP 계산

        Args:
            predictions: List[Dict] 예측 결과
            targets: List[Dict] GT

        Returns:
            metrics: mAP 및 세부 메트릭
        """
        # 클래스별로 분리
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

            gt_boxes = target['boxes']
            gt_labels = target['labels']
            image_id = target['image_id'].item()

            # 예측 결과 저장 (pred image_id도 정수로 변환)
            pred_image_id = pred['image_id'].item() if torch.is_tensor(pred['image_id']) else (
                pred['image_id'][0] if isinstance(pred['image_id'], (list, tuple)) else pred['image_id']
            )
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                pred_by_class[label.item()].append({
                    'box': box,
                    'score': score.item(),
                    'image_id': pred_image_id  # 정수로 변환된 image_id 사용
                })

            # GT 저장
            for box, label in zip(gt_boxes, gt_labels):
                gt_by_class[label.item()].append({
                    'box': box,
                    'image_id': image_id
                })

        # 클래스별 AP 계산
        aps = []
        all_classes = set(list(pred_by_class.keys()) + list(gt_by_class.keys()))

        for cls in all_classes:
            preds = pred_by_class.get(cls, [])
            gts = gt_by_class.get(cls, [])

            if len(gts) == 0:
                continue

            # IoU 임계값별 AP 계산
            aps_per_iou = []
            for iou_thresh in self.iou_thresholds:
                ap = self.compute_ap(preds, gts, iou_thresh)
                aps_per_iou.append(ap)

            # 평균 AP (클래스별)
            aps.append(np.mean(aps_per_iou))

        # mAP
        if len(aps) == 0:
            mAP = 0.0
        else:
            mAP = np.mean(aps)

        # mAP@0.5 별도 계산
        aps_50 = []
        for cls in all_classes:
            preds = pred_by_class.get(cls, [])
            gts = gt_by_class.get(cls, [])

            if len(gts) == 0:
                continue

            ap = self.compute_ap(preds, gts, iou_thresh=0.5)
            aps_50.append(ap)

        mAP_50 = np.mean(aps_50) if len(aps_50) > 0 else 0.0

        metrics = {
            'mAP': mAP,
            'mAP@0.5': mAP_50,
            'num_classes': len(all_classes)
        }

        return metrics

    def compute_ap(self, predictions, ground_truths, iou_thresh=0.5):
        """
        특정 IoU 임계값에 대한 AP 계산

        Args:
            predictions: 예측 결과 리스트
            ground_truths: GT 리스트
            iou_thresh: IoU 임계값

        Returns:
            ap: Average Precision
        """
        if len(predictions) == 0 or len(ground_truths) == 0:
            return 0.0

        # 점수 기준 내림차순 정렬
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

        # GT 매칭 여부 추적
        gt_matched = defaultdict(lambda: defaultdict(bool))

        tp = []
        fp = []

        for pred in predictions:
            pred_box = pred['box']
            pred_image_id = pred['image_id']

            # 같은 이미지의 GT 찾기
            gts_in_image = [gt for gt in ground_truths if gt['image_id'] == pred_image_id]

            if len(gts_in_image) == 0:
                fp.append(1)
                tp.append(0)
                continue

            # IoU 계산
            gt_boxes = torch.stack([gt['box'] for gt in gts_in_image])
            ious = bbox_iou(pred_box.unsqueeze(0), gt_boxes)[0]

            # 최대 IoU GT 찾기
            max_iou, max_idx = ious.max(dim=0)

            if max_iou >= iou_thresh:
                # 해당 GT가 이미 매칭되었는지 확인
                gt_idx = max_idx.item()
                if not gt_matched[pred_image_id][gt_idx]:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[pred_image_id][gt_idx] = True
                else:
                    # 이미 매칭된 GT
                    fp.append(1)
                    tp.append(0)
            else:
                fp.append(1)
                tp.append(0)

        # Precision-Recall 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # AP 계산 (11-point interpolation)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return ap


class VOCEvaluator:
    """
    PASCAL VOC 스타일 평가기 (mAP@0.5)

    Args:
        iou_threshold: IoU 임계값 (기본값: 0.5)
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def evaluate(self, model, data_loader, device='cuda'):
        """
        모델 평가

        Args:
            model: Faster R-CNN 모델
            data_loader: 평가 데이터 로더
            device: 디바이스

        Returns:
            metrics: mAP@0.5
        """
        coco_evaluator = COCOEvaluator(iou_thresholds=[self.iou_threshold])
        metrics = coco_evaluator.evaluate(model, data_loader, device)

        return {
            'mAP@0.5': metrics['mAP'],
            'num_classes': metrics['num_classes']
        }
