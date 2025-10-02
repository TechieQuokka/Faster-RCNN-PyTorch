# Faster R-CNN 아키텍처 설계

## 1. 시스템 개요

Faster R-CNN은 객체 탐지(Object Detection)를 위한 딥러닝 모델로, Region Proposal Network (RPN)과 Fast R-CNN을 결합한 end-to-end 학습 가능한 구조입니다.

## 2. 전체 아키텍처

```
입력 이미지
    ↓
[Backbone CNN (Feature Extractor)]
    ↓
Feature Maps
    ↓
┌─────────────────────────────┐
│  Region Proposal Network    │ → Region Proposals
│  (RPN)                      │
└─────────────────────────────┘
    ↓
[RoI Pooling/RoI Align]
    ↓
┌─────────────────────────────┐
│  Detection Head             │
│  ├─ Classification Branch   │ → 클래스 확률
│  └─ Bounding Box Regression │ → 박스 좌표
└─────────────────────────────┘
```

## 3. 주요 컴포넌트 설계

### 3.1 Backbone Network (특징 추출기)
**목적**: 입력 이미지로부터 고수준 특징 맵 추출

**구조 옵션**:
- VGG16 (원논문)
- ResNet-50/101 (더 나은 성능)
- MobileNet (경량화)

**출력**: Feature Maps (예: 1/16 해상도)

**구현 고려사항**:
- 사전학습된 가중치 사용 (ImageNet)
- 마지막 분류 레이어 제거
- 특정 레이어까지만 사용 (예: conv5_3)

### 3.2 Region Proposal Network (RPN)
**목적**: 객체가 있을 가능성이 높은 영역 제안

**입력**: Backbone의 Feature Maps
**출력**:
- 객체 존재 확률 (objectness score)
- 바운딩 박스 좌표 조정값 (bbox regression)

**구조**:
```
Feature Maps
    ↓
[3×3 Conv, 512 channels] (sliding window)
    ↓
    ├─ [1×1 Conv] → Objectness (2k scores)
    └─ [1×1 Conv] → Box Regression (4k coordinates)

k = Anchor 개수 (보통 9개)
```

**Anchor 설계**:
- 스케일: 3가지 (128², 256², 512² 픽셀)
- 종횡비: 3가지 (1:1, 1:2, 2:1)
- 총 9개 앵커 박스 (3 × 3)

**학습 전략**:
- Positive: IoU > 0.7 또는 가장 높은 IoU
- Negative: IoU < 0.3
- 무시: 0.3 ≤ IoU ≤ 0.7

### 3.3 RoI Pooling/Align
**목적**: 다양한 크기의 RoI를 고정 크기 특징으로 변환

**입력**:
- Feature Maps
- RPN이 제안한 Region Proposals

**출력**: 고정 크기 특징 (예: 7×7×512)

**옵션**:
- **RoI Pooling**: 원본 Faster R-CNN (양자화 오류 존재)
- **RoI Align**: Mask R-CNN에서 제안 (더 정확)

### 3.4 Detection Head
**목적**: 최종 객체 분류 및 바운딩 박스 정밀화

**입력**: RoI Pooling/Align 출력
**출력**:
- 클래스 확률 (C+1 클래스, +1은 배경)
- 박스 좌표 조정값 (4 × C)

**구조**:
```
RoI Features (7×7×512)
    ↓
[Flatten]
    ↓
[FC 4096] → ReLU → Dropout
    ↓
[FC 4096] → ReLU → Dropout
    ↓
    ├─ [FC C+1] → Softmax (Classification)
    └─ [FC 4×C] → (Bounding Box Regression)
```

## 4. 손실 함수 설계

### 4.1 RPN 손실
```
L_RPN = L_cls(RPN) + λ × L_reg(RPN)

L_cls: Binary Cross-Entropy (객체/배경)
L_reg: Smooth L1 Loss (박스 회귀)
λ = 10 (균형 가중치)
```

### 4.2 Detection 손실
```
L_Detection = L_cls(Detection) + λ × L_reg(Detection)

L_cls: Cross-Entropy (다중 클래스 분류)
L_reg: Smooth L1 Loss (박스 회귀)
```

### 4.3 전체 손실
```
L_total = L_RPN + L_Detection
```

## 5. 학습 전략

### 5.1 4-Step Alternating Training (원논문)
1. RPN 학습 (Backbone 고정)
2. Fast R-CNN 학습 (RPN 제안 사용)
3. RPN 미세조정 (Backbone 공유)
4. Fast R-CNN 미세조정

### 5.2 Approximate Joint Training (권장)
- RPN과 Detection Head 동시 학습
- 더 빠르고 구현이 간단
- 성능 차이 미미

### 5.3 하이퍼파라미터
```yaml
학습:
  batch_size: 1 (이미지당)
  learning_rate: 0.001
  optimizer: SGD (momentum=0.9)
  weight_decay: 0.0005
  epochs: 12-20

RPN:
  nms_threshold: 0.7
  train_proposals: 2000 → 256 샘플링
  test_proposals: 300
  positive_ratio: 0.5

Detection:
  roi_batch_size: 128
  positive_ratio: 0.25
  nms_threshold: 0.3
  score_threshold: 0.05
```

## 6. 추론 파이프라인

```
입력 이미지
    ↓
1. Backbone으로 특징 추출
    ↓
2. RPN으로 Region Proposals 생성 (~2000개)
    ↓
3. NMS로 중복 제거 (~300개)
    ↓
4. RoI Pooling/Align
    ↓
5. Detection Head로 분류 및 박스 정밀화
    ↓
6. NMS로 최종 탐지 결과 생성
    ↓
출력: [(class, score, bbox), ...]
```

## 7. 모듈 구조 설계

```
faster_rcnn/
├── models/
│   ├── backbone.py          # ResNet, VGG 등
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

## 8. 성능 최적화 고려사항

### 8.1 메모리 최적화
- Gradient Checkpointing
- Mixed Precision Training (FP16)
- 작은 배치 사이즈 + Gradient Accumulation

### 8.2 속도 최적화
- RoI 개수 제한
- Lightweight Backbone (MobileNet, EfficientNet)
- Feature Pyramid Network (FPN) 추가

### 8.3 정확도 향상
- Multi-scale Training/Testing
- Data Augmentation (Mixup, Mosaic)
- Focal Loss (불균형 데이터)
- Soft-NMS (겹친 객체 처리)

## 9. 평가 메트릭

```
주요 지표:
- mAP@0.5 (PASCAL VOC 스타일)
- mAP@[0.5:0.95] (COCO 스타일)
- Precision-Recall Curve
- FPS (초당 프레임)
```

## 10. 구현 순서 권장사항

1. **데이터 파이프라인** (dataset, transforms)
2. **Backbone 구현** (사전학습 모델 로드)
3. **Anchor 생성기** (다양한 스케일/비율)
4. **RPN 구현** (objectness + bbox regression)
5. **RoI Pooling/Align** (고정 크기 특징 추출)
6. **Detection Head** (분류 + 박스 정밀화)
7. **손실 함수** (RPN + Detection)
8. **학습 루프** (Approximate Joint Training)
9. **추론 파이프라인** (NMS 포함)
10. **평가 시스템** (mAP 계산)

## 11. 핵심 알고리즘 상세

### 11.1 Bounding Box 인코딩/디코딩

**인코딩 (Ground Truth → 학습 타겟)**:
```
tx = (x - xa) / wa
ty = (y - ya) / ha
tw = log(w / wa)
th = log(h / ha)

x, y, w, h: Ground Truth 박스 중심 및 크기
xa, ya, wa, ha: Anchor 박스 중심 및 크기
```

**디코딩 (예측값 → 실제 박스)**:
```
x = tx × wa + xa
y = ty × ha + ya
w = exp(tw) × wa
h = exp(th) × ha
```

### 11.2 Anchor 매칭 전략

**RPN 학습시**:
1. 각 Ground Truth와 가장 높은 IoU를 가진 Anchor → Positive
2. IoU > 0.7인 Anchor → Positive
3. IoU < 0.3인 Anchor → Negative
4. 나머지 → 무시 (학습에 사용 안함)

**Detection Head 학습시**:
1. RPN 제안 중 IoU > 0.5 → Positive
2. IoU < 0.5 → Negative (배경 클래스)

### 11.3 Non-Maximum Suppression (NMS)

```python
# 의사 코드
def nms(boxes, scores, threshold):
    sorted_indices = argsort(scores, descending=True)
    keep = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)

        # 현재 박스와 나머지 박스들의 IoU 계산
        ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])

        # IoU가 threshold보다 낮은 박스만 유지
        sorted_indices = sorted_indices[1:][ious < threshold]

    return keep
```

## 12. 데이터셋 형식

### 12.1 입력 데이터 구조
```python
{
    'image': Tensor [3, H, W],  # RGB 이미지
    'boxes': Tensor [N, 4],      # [x1, y1, x2, y2] 형식
    'labels': Tensor [N],        # 클래스 인덱스 (0: 배경)
    'image_id': int,             # 이미지 고유 ID
    'area': Tensor [N],          # 박스 면적
    'iscrowd': Tensor [N]        # 군집 객체 여부
}
```

### 12.2 지원 데이터셋
- PASCAL VOC (2007, 2012)
- MS COCO
- Custom 데이터셋 (COCO 형식)

## 13. 학습 파이프라인 상세

### 13.1 학습 단계
```
Epoch 루프:
  ├─ 데이터 로딩 및 전처리
  │   ├─ 이미지 리사이즈 (shortest side = 600px)
  │   ├─ 정규화 (ImageNet 통계)
  │   └─ 데이터 증강 (수평 뒤집기 등)
  │
  ├─ Forward Pass
  │   ├─ Backbone → Feature Maps
  │   ├─ RPN → Proposals + RPN Loss
  │   ├─ RoI Pooling → Fixed-size Features
  │   └─ Detection Head → Predictions + Detection Loss
  │
  ├─ Backward Pass
  │   ├─ Total Loss 계산
  │   ├─ Gradient 계산
  │   └─ Parameter 업데이트
  │
  └─ Logging 및 검증
      ├─ Loss 기록
      └─ 주기적 Validation (mAP 계산)
```

### 13.2 학습률 스케줄링
- Warm-up: 처음 500 iteration (lr × 1/3 → lr)
- Step Decay: 8 epoch, 11 epoch에서 lr × 0.1
- 또는 ReduceLROnPlateau 사용

## 14. 추론 최적화

### 14.1 속도 향상 기법
- Test-time Augmentation 제거
- RPN Proposal 개수 제한 (300개)
- Batch Inference (가능한 경우)
- TorchScript/ONNX 변환
- GPU 최적화 (CUDA, cuDNN)

### 14.2 정확도-속도 트레이드오프
```
고정확도 모드:
- Multi-scale Testing
- Soft-NMS
- Test-time Augmentation
- 더 많은 RPN Proposals

고속도 모드:
- Single-scale
- 표준 NMS
- 적은 RPN Proposals
- Lightweight Backbone
```

## 15. 디버깅 체크리스트

### 15.1 학습 안될 때
- [ ] Anchor 스케일이 객체 크기와 맞는지 확인
- [ ] Learning Rate가 너무 크거나 작지 않은지
- [ ] Positive/Negative 샘플 비율 확인
- [ ] Backbone 가중치가 제대로 로드되었는지
- [ ] 손실 함수 각 항의 밸런스 확인

### 15.2 성능 안나올 때
- [ ] 데이터 증강 적절성 검토
- [ ] NMS Threshold 조정
- [ ] Anchor 설계 재검토
- [ ] 더 많은 Epoch 학습
- [ ] Validation 데이터 오버피팅 확인

## 16. 참고 자료

- 원논문: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (NIPS 2015)
- PyTorch 공식 구현: torchvision.models.detection.fasterrcnn_resnet50_fpn
- Detectron2: Facebook AI Research의 객체 탐지 프레임워크
