# VisDrone 항공 영상 객체 탐지

> 영상처리 수업 프로젝트 · **개인 작업**
> 과제 조건은 *"주어진 이미지를 인식하는 모델을 만들 것"* 뿐이었고,
> **데이터셋 선정 · 클래스 구성 · 모델 선택 · 학습 전략은 모두 직접 결정**했습니다.

드론 항공 영상에서 객체를 탐지하는 YOLOv8 모델입니다.
작은 객체가 많고 위에서 내려다보는 시점이라 일반 객체 탐지보다 난이도가 높습니다.

> ⚠️ **먼저 읽어 주세요 — 이 모델은 "사람 탐지" 모델이 아닙니다.**
> 목표는 사람 인식이었지만, 실제 학습 라벨은 사람뿐 아니라 **차량을 포함한 모든 객체를 하나의 클래스(`item`)로 합친 것**이었습니다.
> 아래 성능 수치는 모두 **클래스를 구분하지 않은 객체 탐지** 기준입니다. 자세한 내용은 [사후 점검](#사후-점검--라벨이-지표의-뜻을-바꾼다) 참고.

---

## 문제 정의

교수님이 제시한 것은 인식 대상 이미지 예시뿐이었습니다.
"이걸 인식하는 모델을 만들어라"에서 출발해, 다음을 직접 정해야 했습니다.

1. 어떤 공개 데이터셋을 쓸 것인가
2. 어떤 클래스 체계로 학습할 것인가
3. 어떤 모델과 입력 해상도를 쓸 것인가
4. 어디까지 학습시킬 것인가

---

## 설계 결정

### 1. 데이터셋 — VisDrone2019-DET

항공 시점 · 소형 객체 · 밀집 장면이라는 조건이 과제 이미지와 가장 가까워 선택했습니다.
원본은 10개 클래스(pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)입니다.

### 2. 단일 클래스 학습

10개 클래스를 구분하는 대신 하나의 클래스로 학습했습니다.
의도는 사람(`pedestrian`, `people`)만 남기는 것이었지만, 실제로는 **10개 클래스 전체가 한 클래스로 합쳐졌습니다.**
(아래 [사후 점검](#사후-점검--라벨이-지표의-뜻을-바꾼다) 참고)

### 3. 입력 해상도 — 960

항공 영상의 객체는 화면에서 매우 작게 나타납니다.
기본값 640으로는 다운샘플링 과정에서 특징이 사라진다고 보고 **960으로 올려** 학습했습니다.
대신 학습 시간을 감수했습니다.

### 4. 학습 환경

Ubuntu 환경에서 GPU 2장(`device: 0,1`)으로 100 에폭, batch 16을 학습했습니다.
최종 학습에 약 **4.9시간**이 소요되었습니다.

---

## 결과

### 실험별 성능

| 실험 | 클래스 구성 | 에폭 | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| train | VisDrone 10클래스 | 1 | 0.238 | 0.136 | — | — |
| train2 | 단일 클래스 (전체 객체) | 27 | 0.731 | 0.445 | 0.808 | 0.631 |
| **train3** | 단일 클래스 (전체 객체) | **100** | **0.755** | **0.464** | **0.826** | **0.655** |

- **train → train2** 사이에는 클래스 구성(10개 → 1개)과 에폭 수(1 → 27)가 **동시에 바뀌었습니다.**
  그래서 0.238 → 0.731 상승을 어느 한 요인으로 설명할 수 없습니다.
  게다가 10클래스 mAP는 클래스별 평균이고 단일 클래스 mAP는 분류 없는 탐지라, 두 수치는 애초에 같은 과제의 점수가 아닙니다.
- **train2 → train3**은 조건이 같고 에폭만 27 → 100으로 늘었습니다. 상승폭이 mAP50 기준 +0.024에 그쳐, 이 설정에서는 거의 수렴한 상태로 봅니다.

### 과제 이미지에 적용 — 사전학습 vs 파인튜닝

동일한 과제 이미지 3장(`s418`, `s863`, `s891`)에 COCO 사전학습 모델과 파인튜닝 모델을 적용해 비교했습니다 → `runs/compare/`

`s418`에서 COCO 사전학습 모델은 사람 두 명을 `suitcase`로 잘못 잡았지만, 파인튜닝 모델에서는 그런 오인이 사라졌습니다.

### 산출물

| 파일 | 내용 |
|---|---|
| `runs/train3/results.csv` | 에폭별 손실·정밀도·재현율·mAP 전체 기록 |
| `runs/train3/results.png` | 학습 곡선 |
| `runs/train3/BoxPR_curve.png` | Precision-Recall 곡선 |
| `runs/train3/confusion_matrix.png` | 혼동 행렬 |
| `runs/train3/labels.jpg` | 학습 라벨 분포 — **클래스 `item` 하나에 343,201개** |
| `runs/train3/val_batch0_labels.jpg` | 검증셋 정답 라벨 — 차량도 `item`으로 표시됨 |
| `runs/train3/val_batch0_pred.jpg` | 검증셋 예측 결과 |
| `runs/train3/weights/best.pt` | 학습된 가중치 |

---

## 사후 점검 — 라벨이 지표의 뜻을 바꾼다

목표는 사람 탐지였지만, 학습 라벨을 다시 확인해 보니 **사람·승용차·승합차·트럭 등 모든 객체가 하나의 클래스(`item`)로 합쳐져 있었습니다.**

근거는 두 가지입니다.

1. `runs/train3/labels.jpg` — 학습 인스턴스가 `item` 한 클래스에 **343,201개**입니다. 이는 VisDrone2019-DET 학습셋 10개 클래스 **전체의 객체 수와 거의 같습니다.** 사람(`pedestrian` + `people`)만 남겼다면 이보다 훨씬 적어야 합니다.
2. `runs/train3/val_batch0_labels.jpg` — 검증 데이터의 **정답** 라벨에 승용차·승합차 박스가 `item`으로 표시되어 있습니다.

따라서 mAP50 0.755는 '사람'이 아니라 **'클래스를 구분하지 않은 객체 탐지'** 성능입니다.
라벨 재매핑 단계에서 `pedestrian`(0)·`people`(1)만 남기고 나머지 객체 줄은 지웠어야 했습니다.

> **배운 점** — 같은 mAP라도 무엇을 한 클래스로 묶었느냐에 따라 뜻이 완전히 달라집니다.
> 숫자를 보기 전에 라벨부터 봐야 한다는 것을 이 프로젝트에서 가장 크게 배웠습니다.

`configs/vis_person.yaml`은 사람 단일 클래스를 **의도한** 설정 파일입니다(`names: ['person']`).
실제 학습에 쓰인 라벨·클래스명(`item`)과는 맞지 않으니, 이 파일을 학습 결과의 설명으로 읽지 마세요.

---

## 그 밖의 한계

- **낮은 재현율** — Recall 0.655로 Precision 0.826보다 낮아, 밀집 장면에서 놓치는 객체가 있습니다.
- **객체 크기별 분석 부재** — VisDrone의 핵심 난점인 작은 객체의 성능을 따로 분해하지 않았습니다.
- **해상도 효과 미검증** — 960의 효과를 640과 정량 비교하지 않았습니다.
- **중단된 아키텍처 실험** — YOLOv8n의 C2f 블록을 YOLO11의 C3k2로 바꾸는 실험을 COCO · 600 에폭 · batch 128 설정으로 시도했지만,
  **GPU 전원 문제로 학습이 중단되어 결과 지표를 얻지 못했습니다.** 실행 설정만 `experiments/`에 남깁니다.
  C3k2는 C2f 구조를 유지하면서 내부 bottleneck을 커널 크기 조절이 가능한 C3k로 바꾼 블록입니다.
  재실행한다면 baseline(C2f)과 C3k2 변형을 동일 조건에서 학습해 mAP50-95, 파라미터 수, 추론 속도를 함께 비교해야 합니다.

---

## 다시 한다면 — 진짜 사람 탐지로 학습하기

VisDrone을 YOLO 형식으로 변환한 라벨에서 사람 클래스만 남기면 됩니다.
(`configs/VisDrone.yaml` 기준 `pedestrian`=0, `people`=1)

```python
from pathlib import Path

PERSON = {"0", "1"}  # pedestrian, people

for txt in Path("VisDrone/labels").rglob("*.txt"):
    kept = [
        "0 " + line.split(maxsplit=1)[1]
        for line in txt.read_text().splitlines()
        if line.split() and line.split()[0] in PERSON
    ]
    txt.write_text("\n".join(kept) + ("\n" if kept else ""))
```

재매핑한 뒤에는 학습 전에 `labels.jpg`의 인스턴스 수가 사람 객체 수와 맞는지 먼저 확인합니다.

---

## 실행

```bash
pip install -r requirements.txt
```

VisDrone 데이터셋은 [공식 저장소](https://github.com/VisDrone/VisDrone-Dataset)에서 받아
`configs/VisDrone.yaml`의 `path`에 맞춰 배치하세요.

```bash
yolo train data=configs/vis_person.yaml model=yolov8n.pt epochs=100 imgsz=960
```

---

## 저장소 구조

```
├── notebooks/
│   └── 01_visdrone_training.ipynb   학습 파이프라인
├── configs/
│   ├── VisDrone.yaml                원본 10클래스 설정
│   └── vis_person.yaml              사람 단일 클래스로 '의도한' 설정 (실제 학습 라벨과 다름)
├── runs/
│   ├── train3/                      최종 학습 결과 (100에폭, 단일 클래스 · 전체 객체)
│   └── compare/                     사전학습 vs 파인튜닝 시각 비교
└── experiments/                     중단된 C3k2 아키텍처 실험 설정
```

포트폴리오: https://pr-log.github.io/portfolio/#p-visdrone
