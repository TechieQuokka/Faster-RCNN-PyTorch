"""
Faster R-CNN 학습 스크립트
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import random
import numpy as np

from faster_rcnn.models.faster_rcnn import build_faster_rcnn
from faster_rcnn.data.dataset import VOCDataset, COCODataset
from faster_rcnn.data.transforms import get_transform
from faster_rcnn.data.collate import collate_fn
from faster_rcnn.engine.trainer import Trainer
from faster_rcnn.engine.evaluator import VOCEvaluator, COCOEvaluator


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_dataset(config, train=True):
    """데이터셋 생성"""
    dataset_name = config['dataset']['name']
    transforms = get_transform(
        train=train,
        min_size=config['transforms']['min_size'],
        max_size=config['transforms']['max_size']
    )

    if dataset_name == 'voc':
        dataset = VOCDataset(
            root=config['dataset']['root'],
            year=config['dataset']['year'],
            image_set=config['dataset']['train_set'] if train else config['dataset']['val_set'],
            transforms=transforms
        )
    elif dataset_name == 'coco':
        split = 'train' if train else 'val'
        dataset = COCODataset(
            root=config['dataset']['root'],
            annFile=config['dataset'][f'{split}_ann'],
            transforms=transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def build_optimizer(model, config):
    """옵티마이저 생성"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    return optimizer


def build_scheduler(optimizer, config):
    """학습률 스케줄러 생성"""
    scheduler_config = config['training']['lr_scheduler']

    if scheduler_config['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config['type']}")

    return scheduler


def main(args):
    # 설정 로드
    config = load_config(args.config)

    # 시드 설정
    set_seed(config['seed'])

    # 디바이스 설정
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')

    # 데이터셋 생성
    print('데이터셋 로딩 중...')
    train_dataset = build_dataset(config, train=True)
    val_dataset = build_dataset(config, train=False)

    print(f'학습 데이터: {len(train_dataset)} 이미지')
    print(f'검증 데이터: {len(val_dataset)} 이미지')

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    # 모델 생성
    print('모델 생성 중...')
    model = build_faster_rcnn(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone'],
        pretrained_backbone=config['model']['pretrained_backbone'],
        # RPN 파라미터
        rpn_anchor_sizes=tuple(config['rpn']['anchor_sizes']),
        rpn_anchor_ratios=tuple(config['rpn']['anchor_ratios']),
        rpn_nms_thresh=config['rpn']['nms_thresh'],
        rpn_pre_nms_top_n_train=config['rpn']['pre_nms_top_n_train'],
        rpn_pre_nms_top_n_test=config['rpn']['pre_nms_top_n_test'],
        rpn_post_nms_top_n_train=config['rpn']['post_nms_top_n_train'],
        rpn_post_nms_top_n_test=config['rpn']['post_nms_top_n_test'],
        # RoI Head 파라미터
        roi_output_size=config['roi_head']['roi_output_size'],
        roi_fg_iou_thresh=config['roi_head']['fg_iou_thresh'],
        roi_bg_iou_thresh=config['roi_head']['bg_iou_thresh'],
        roi_batch_size_per_image=config['roi_head']['batch_size_per_image'],
        roi_positive_fraction=config['roi_head']['positive_fraction'],
        roi_score_thresh=config['roi_head']['score_thresh'],
        roi_nms_thresh=config['roi_head']['nms_thresh'],
        roi_detection_per_img=config['roi_head']['detection_per_img'],
    )

    # 옵티마이저 및 스케줄러 생성
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Trainer 생성
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir'],
        print_freq=10
    )

    # 체크포인트 로드 (재개 시)
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # 학습
    print('학습 시작...')
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=config['training']['num_epochs'],
        val_loader=val_loader,
        scheduler=scheduler,
        save_freq=config['training']['save_freq']
    )

    print('학습 완료!')

    # 최종 평가
    print('최종 평가 중...')
    if config['evaluation']['metric'] == 'voc':
        evaluator = VOCEvaluator(iou_threshold=config['evaluation']['iou_threshold'])
    else:
        evaluator = COCOEvaluator()

    metrics = evaluator.evaluate(model, val_loader, device)

    print('평가 결과:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN 학습')
    parser.add_argument('--config', type=str, default='faster_rcnn/configs/default.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--resume', type=str, default=None,
                        help='재개할 체크포인트 경로')

    args = parser.parse_args()
    main(args)
