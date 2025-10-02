"""
더미 데이터셋으로 Faster R-CNN 학습 테스트
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import random
import numpy as np

from faster_rcnn.models.faster_rcnn import build_faster_rcnn
from faster_rcnn.data.dataset import VOCDataset
from faster_rcnn.data.transforms import get_transform
from faster_rcnn.data.collate import collate_fn
from faster_rcnn.engine.trainer import Trainer


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 시드 설정
    set_seed(42)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')

    # 데이터셋 생성
    print('데이터셋 로딩 중...')
    transforms_train = get_transform(train=True, min_size=400, max_size=600)
    transforms_val = get_transform(train=False, min_size=400, max_size=600)

    train_dataset = VOCDataset(
        root='./data',
        year='2012',
        image_set='train',
        transforms=transforms_train
    )

    val_dataset = VOCDataset(
        root='./data',
        year='2012',
        image_set='val',
        transforms=transforms_val
    )

    print(f'학습 데이터: {len(train_dataset)} 이미지')
    print(f'검증 데이터: {len(val_dataset)} 이미지')

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 작은 배치
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 모델 생성 (클래스 수는 더미 데이터에 맞게 21로 설정)
    print('모델 생성 중...')
    model = build_faster_rcnn(
        num_classes=21,  # PASCAL VOC 20 classes + background
        backbone_name='resnet50',
        pretrained_backbone=True,
        # 더 작은 설정으로 빠른 학습
        rpn_pre_nms_top_n_train=6000,
        rpn_post_nms_top_n_train=1000,
        roi_batch_size_per_image=64,
    )

    # 옵티마이저
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='./checkpoints',
        print_freq=5
    )

    # 학습
    print('학습 시작...')
    print('='*50)

    history = trainer.train(
        train_loader=train_loader,
        num_epochs=3,  # 짧은 테스트
        val_loader=val_loader,
        scheduler=scheduler,
        save_freq=1
    )

    print('\n학습 완료!')
    print('='*50)


if __name__ == '__main__':
    main()
