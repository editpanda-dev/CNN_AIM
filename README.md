# CNN_AIM — AI광고 vs 인간광고 분류 모델

EfficientNet-B0 기반 Transfer Learning으로 AI 생성 광고와 인간 제작 광고를 분류하는 CNN 모델입니다.

---

## 성능

| 평가 방식 | 정확도 |
|---|---|
| 전체 데이터셋 TTA 정확도 | **95.0%** |
| 5-Fold CV 평균 (Fold 4) | 83.3% |
| 5-Fold CV 평균 (Fold 5) | 80.8% |
| 5-Fold CV 전체 평균 | 73.3% |

```
              precision  recall  f1-score  support
    인간광고       0.96    0.95      0.95      115
      AI광고       0.95    0.96      0.96      124
    accuracy                         0.95      239
```

---

## 데이터셋

- 총 239장 (AI광고 124장, 인간광고 115장)
- 경로: `광고_최종/AI광고/`, `광고_최종/인간광고/`

---

## 모델 구조

- **Backbone**: EfficientNet-B0 (ImageNet 사전학습)
- **Classifier**: Dropout(0.5) → Linear(1280→256) → SiLU → Dropout(0.25) → Linear(256→2)
- **학습 전략**: 3단계 학습
  1. **Phase 1** (10 epoch): Backbone 고정, Head만 학습 (LR=1e-3)
  2. **Phase 2** (30 epoch): 마지막 4블록 + Head fine-tune, Mixup (LR=5e-4)
  3. **Phase 3** (20 epoch): 전체 모델 fine-tune, Mixup (LR=5e-5)
- **추론**: TTA 5종 앙상블 (원본, 좌우반전, CenterCrop, ColorJitter, 회전)

---

## 파일 구조

```
CNN_AIM/
├── 광고_최종/
│   ├── AI광고/        # AI 생성 광고 이미지 (124장)
│   └── 인간광고/      # 인간 제작 광고 이미지 (115장)
├── CNN_AIM_Colab.ipynb  # 학습 노트북 (Colab용)
├── train.py             # 로컬 학습 스크립트
├── predict.py           # 단일 이미지 추론 스크립트
├── best_model.pth       # 학습된 모델 가중치 (95% acc)
├── requirements.txt     # 패키지 목록
└── README.md
```

---

## 실행 방법

### Google Colab (권장)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/editpanda-dev/CNN_AIM/blob/main/CNN_AIM_Colab.ipynb)

1. 위 버튼 클릭
2. 런타임 → 런타임 유형 변경 → **T4 GPU** 설정
3. 셀 순서대로 실행 (약 15~20분 소요)

### 로컬 실행

```bash
pip install -r requirements.txt

# 학습
python train.py

# 단일 이미지 추론
python predict.py 광고_최종/AI광고/AI_01_가방.png
```

---

## 의존성

```
torch >= 2.0.0
torchvision >= 0.15.0
Pillow >= 9.0.0
scikit-learn >= 1.2.0
numpy >= 1.23.0
```
