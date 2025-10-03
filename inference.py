"""
Faster R-CNN 추론 스크립트
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import yaml

from faster_rcnn.models.faster_rcnn import build_faster_rcnn
from faster_rcnn.data.transforms import get_transform
from faster_rcnn.utils.config import load_config


def load_model(config_path, checkpoint_path, device='cuda'):
    """모델 로드"""
    # 설정 로드
    config = load_config(config_path)

    # 모델 생성
    model = build_faster_rcnn(
        num_classes=config.model.num_classes,
        backbone_name=config.model.backbone,
        pretrained_backbone=False,  # 체크포인트에서 로드
        # RPN 파라미터
        rpn_anchor_sizes=tuple(config.rpn.anchor_sizes),
        rpn_anchor_ratios=tuple(config.rpn.anchor_ratios),
        rpn_nms_thresh=config.rpn.nms_thresh,
        rpn_pre_nms_top_n_test=config.rpn.pre_nms_top_n_test,
        rpn_post_nms_top_n_test=config.rpn.post_nms_top_n_test,
        # RoI Head 파라미터
        roi_output_size=config.roi_head.roi_output_size,
        roi_fg_iou_thresh=config.roi_head.fg_iou_thresh,
        roi_bg_iou_thresh=config.roi_head.bg_iou_thresh,
        roi_score_thresh=config.roi_head.score_thresh,
        roi_nms_thresh=config.roi_head.nms_thresh,
        roi_detection_per_img=config.roi_head.detection_per_img,
    )

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def predict(model, image_path, transform, device='cuda'):
    """이미지에 대한 예측"""
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()

    # 변환 적용
    image_tensor, _ = transform(image, None)
    image_tensor = image_tensor.to(device)

    # 추론
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # CPU로 이동
    predictions = {k: v.cpu() for k, v in predictions.items()}

    return original_image, predictions


def visualize_predictions(image, predictions, class_names=None, score_threshold=0.5):
    """예측 결과 시각화"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # 박스 그리기
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # 레이블 표시
        class_name = class_names.get(label.item(), str(label.item())) if class_names else str(label.item())
        text = f'{class_name}: {score:.2f}'

        ax.text(
            x1, y1 - 5,
            text,
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10, color='white'
        )

    ax.axis('off')
    plt.tight_layout()
    return fig


def main(args):
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')

    # 모델 로드
    print('모델 로딩 중...')
    model, config = load_model(args.config, args.checkpoint, device)

    # 변환 생성
    transform = get_transform(train=False, config=config)

    # 클래스 이름 (PASCAL VOC 기본값)
    class_names = {
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
        6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
        11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
        16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }

    # 예측
    print('추론 중...')
    image, predictions = predict(model, args.image, transform, device)

    print(f'검출된 객체: {len(predictions["boxes"])}개')

    # 결과 시각화
    fig = visualize_predictions(
        image, predictions,
        class_names=class_names,
        score_threshold=args.score_threshold
    )

    # 저장 또는 표시
    if args.output:
        fig.savefig(args.output, bbox_inches='tight', dpi=150)
        print(f'결과 저장: {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN 추론')
    parser.add_argument('--config', type=str, default='faster_rcnn/configs/default.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--image', type=str, required=True,
                        help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 이미지 경로 (옵션)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='검출 점수 임계값')

    args = parser.parse_args()
    main(args)
