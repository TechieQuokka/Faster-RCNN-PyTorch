"""
torchvision을 사용하여 PASCAL VOC 2012 데이터셋 다운로드
"""

import torchvision
from torchvision.datasets import VOCDetection
import os

def download_voc_dataset(root='./data', year='2012'):
    """
    PASCAL VOC 데이터셋 다운로드

    Args:
        root: 데이터 저장 경로
        year: VOC 년도 ('2007', '2012')
    """
    print(f'PASCAL VOC {year} 데이터셋 다운로드 중...')
    print(f'저장 위치: {root}')

    # Train 데이터셋 다운로드
    print('\n학습 데이터셋 다운로드 중...')
    train_dataset = VOCDetection(
        root=root,
        year=year,
        image_set='train',
        download=True
    )
    print(f'학습 데이터: {len(train_dataset)} 이미지')

    # Val 데이터셋 다운로드
    print('\n검증 데이터셋 다운로드 중...')
    val_dataset = VOCDetection(
        root=root,
        year=year,
        image_set='val',
        download=True
    )
    print(f'검증 데이터: {len(val_dataset)} 이미지')

    print('\n다운로드 완료!')
    print(f'데이터 경로: {os.path.join(root, "VOCdevkit", f"VOC{year}")}')


if __name__ == '__main__':
    download_voc_dataset()
