# Faster R-CNN PyTorch 구현

Faster R-CNN (Faster Region-based Convolutional Neural Networks)의 PyTorch 구현입니다.

## 특징

- **End-to-End 학습**: RPN과 Detection Head가 통합된 구조
- **유연한 Backbone**: ResNet-50, VGG16 지원
- **다양한 데이터셋**: PASCAL VOC, MS COCO 지원
- **완전한 학습 파이프라인**: 학습, 검증, 평가, 추론
- **COCO 스타일 평가**: mAP@[0.5:0.95] 메트릭

## 프로젝트 구조

```
faster_rcnn/
├── models/
│   ├── backbone.py          # ResNet, VGG Backbone
│   ├── rpn.py               # Region Proposal Network
│   ├── roi_head.py          # RoI Pooling + Detection Head
│   └── faster_rcnn.py       # 전체 모델 통합
├── utils/
│   ├── anchor_generator.py  # Anchor 생성
│   ├── bbox_tools.py        # 박스 변환, IoU 계산
│   ├── nms.py               # Non-Maximum Suppression
│   └── loss.py              # 손실 함수
├── data/
│   ├── dataset.py           # 데이터셋 클래스
│   ├── transforms.py        # 데이터 증강
│   └── collate.py           # 배치 처리
├── engine/
│   ├── trainer.py           # 학습 루프
│   └── evaluator.py         # 평가 (mAP 계산)
└── configs/
    └── default.yaml         # 기본 설정
```

## 설치

### 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU 사용 시)

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd Faster_R-CNN

# 의존성 설치
pip install -r requirements.txt
```

## 데이터셋 준비

### PASCAL VOC

```bash
# VOC 2012 다운로드
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

# 디렉토리 구조
data/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── ImageSets/
        ├── JPEGImages/
        └── ...
```

### MS COCO

```bash
# COCO 2017 다운로드
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 압축 해제
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# 디렉토리 구조
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
```

## 학습

### 기본 학습

```bash
python train.py --config faster_rcnn/configs/default.yaml
```

### 학습 재개

```bash
python train.py --config faster_rcnn/configs/default.yaml --resume checkpoints/checkpoint_epoch_8.pth
```

### 설정 파일 수정

`faster_rcnn/configs/default.yaml` 파일을 편집하여 하이퍼파라미터 조정:

```yaml
# 모델 설정
model:
  num_classes: 21  # PASCAL VOC
  backbone: resnet50  # resnet50 또는 vgg16

# 학습 설정
training:
  batch_size: 2
  num_epochs: 12
  learning_rate: 0.001
```

## 추론

### 단일 이미지 추론

```bash
python inference.py \
  --config faster_rcnn/configs/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg \
  --output result.jpg \
  --score-threshold 0.7
```

### 예시

```bash
# 고양이 이미지 검출
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image examples/cat.jpg \
  --output results/cat_detected.jpg
```

## 평가

학습 스크립트에 포함되어 있으며, 학습 종료 후 자동으로 평가가 수행됩니다.

```python
# VOC 스타일 (mAP@0.5)
evaluator = VOCEvaluator(iou_threshold=0.5)
metrics = evaluator.evaluate(model, val_loader, device)

# COCO 스타일 (mAP@[0.5:0.95])
evaluator = COCOEvaluator()
metrics = evaluator.evaluate(model, val_loader, device)
```

## 성능 벤치마크

### PASCAL VOC 2007 Test

| Backbone | mAP@0.5 | FPS (GPU) |
|----------|---------|-----------|
| VGG16    | 69.9%   | 5         |
| ResNet-50| 76.4%   | 7         |

### MS COCO val2017

| Backbone | mAP@[0.5:0.95] | mAP@0.5 |
|----------|----------------|---------|
| ResNet-50| 37.4%          | 58.1%   |

## 아키텍처 세부사항

### 전체 파이프라인

```
입력 이미지
    ↓
Backbone (ResNet-50)
    ↓
Feature Maps
    ↓
RPN (Region Proposal Network)
    ↓
RoI Align
    ↓
Detection Head (분류 + 박스 회귀)
    ↓
NMS
    ↓
최종 검출 결과
```

### 주요 컴포넌트

1. **Backbone**: ResNet-50/VGG16 특징 추출기
2. **RPN**: 9개 Anchor (3 스케일 × 3 종횡비)
3. **RoI Align**: 7×7 고정 크기 특징
4. **Detection Head**: 2개 FC 레이어 (4096 dim)

## 학습 팁

### 메모리 부족 시

```yaml
training:
  batch_size: 1  # 배치 크기 감소
```

### 학습 속도 향상

```yaml
rpn:
  pre_nms_top_n_train: 6000  # Proposal 개수 감소
  post_nms_top_n_train: 1000
```

### 정확도 향상

```yaml
transforms:
  train_augmentation: true  # 데이터 증강 활성화

training:
  num_epochs: 20  # Epoch 증가
```

## 문제 해결

### Q: CUDA out of memory 오류
A: `batch_size`를 1로 줄이거나, `roi_batch_size_per_image`를 64로 감소

### Q: 학습이 안 됨 (loss가 감소하지 않음)
A:
- Anchor 크기가 데이터셋의 객체 크기와 맞는지 확인
- Learning rate를 0.0001로 감소
- Backbone의 사전학습 가중치가 로드되었는지 확인

### Q: mAP가 낮게 나옴
A:
- 더 많은 Epoch 학습 (20+ epochs)
- 데이터 증강 활성화
- NMS threshold 조정 (0.3 → 0.5)

## 참고 자료

- [원논문](https://arxiv.org/abs/1506.01497): Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (NIPS 2015)
- [PyTorch 공식 구현](https://github.com/pytorch/vision/tree/main/torchvision/models/detection)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## 라이센스

MIT License

## 기여

Pull Request와 Issue는 언제나 환영합니다!

## 연락처

문의사항이 있으시면 Issue를 등록해주세요.
