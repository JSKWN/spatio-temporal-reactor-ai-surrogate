# Tesla FSD — 자율주행 세계모델 역공학

> **출처**: Tesla AI Day 2021/2022, Ashok Elluswamy 발표, 특허 (2021-2025)
> **공개 수준**: 역공학추론 (공식 논문 미공개. AI Day 발표, 특허, 인터뷰 기반 추론)

---

## 1. 시스템 개요

### 직관

Tesla FSD는 8대 카메라 영상만으로 (LiDAR/HD맵 없이) 자율주행을 수행한다. 핵심은 카메라 영상 → 3D 공간 이해 → 미래 예측 → 경로 계획의 end-to-end 파이프라인이다.

### 추론된 아키텍처 (FSD v12+, 2023-2025)

```
8 Cameras (1280×960, 36fps)
    ↓
[Per-camera CNN backbone (RegNet/EfficientNet + BiFPN)]
    ↓
[Spatial Transformer → BEV Feature Map]    ← 2D 이미지 → 3D 공간 변환
    ↓
[Temporal Fusion Module]                   ← 과거 BEV 프레임과 융합
    ↓
[3D Occupancy Network]                     ← 복셀 점유/의미/속도 예측
    ↓
[Planning Transformer]                      ← 궤적 생성
    ↓
[Steering, Acceleration commands]
```

**규모**: ~1-1.5B params (HW3/HW4 칩에서 추론)

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 — Temporal Fusion

**확인된 정보** (확신도: 중간):
- BEV feature map의 시간 융합은 **ego-motion 보상 + 어텐션/RNN 게이팅**으로 수행
- 과거 프레임의 BEV feature를 자차 이동량으로 워핑(warping) 후, 현재 프레임과 결합
- Deformable attention 또는 GRU 기반 융합 (정확한 방식 미공개)
- **SSM(Mamba) 사용은 미확인** (2025년 5월 기준)

**Factorized attention** (확신도: 높음):
- 공간 attention (프레임 내) + 시간 attention (프레임 간) 분리
- 인과적 시간 attention: 미래 프레임에 attend 불가

**우리 문제 대응**: 우리의 "인코더(공간) → Mamba(시간)" 분리는 Tesla의 factorized attention과 동일한 철학. 공간과 시간을 분리 처리하는 것이 산업 수준에서도 채택된 전략.

### 2.2 상태 표현 — Occupancy Network + BEV

**Occupancy Network** (확신도: 높음, AI Day 2022 발표):
- 차량 주변 100m × 100m × 8m 3D 복셀 그리드
- 복셀당 예측: 점유 여부, 의미 클래스, 속도 벡터
- Triplanar feature: 3D feature를 XY/XZ/YZ 3개 평면으로 분해 → 메모리 절약
- NeRF 유사 자기지도학습: 예측된 복셀에서 렌더링한 영상과 실제 영상 비교

**우리 문제 대응**:
| Tesla Occupancy | 우리 노심 상태 |
|---|---|
| 3D 복셀 (100×100×8m) | 3D cell (20×6×6 halo) |
| 복셀당: 점유/의미/속도 | cell당: 10ch 물리량 (power, Xe, flux, ...) |
| 복셀 간 관계: 장애물 연속성 | cell 간 관계: 중성자 확산 결합 |
| BEV 2D 압축 | Quarter symmetry (5×5 → 6×6 halo) |

### 2.3 제어/조건 주입 — 행동 조건부 세계모델

**Neural World Simulator** (확신도: 중간):
- 현재 BEV 상태 + 후보 궤적 → 미래 BEV 상태 예측
- 후보 궤적이 **제어 입력**에 해당
- VQ-VAE 또는 유사한 이산 토큰화로 장면 표현을 압축
- 자기회귀적으로 미래 상태 생성

**우리 문제 대응**: Tesla의 "후보 궤적 → 미래 예측"은 우리의 "rod offset → Branch 예측"과 구조적으로 동일:
| Tesla | 우리 |
|---|---|
| 후보 궤적 N개 → 미래 장면 N개 예측 | rod offset 29개 → Branch 29개 예측 |
| 충돌/불편 기준으로 궤적 선택 | 물리 제약/안전 기준으로 rod 위치 선택 |
| MCTS로 최적 궤적 탐색 | 29개 직접 비교 (GT 존재) |

### 2.4 시공간 결합

**4D 시공간 표현** (확신도: 중간):
- Occupancy flow: 현재 점유뿐 아니라 단기 미래 점유도 예측
- 시공간 복셀 볼륨 (3D 공간 + 시간) 처리
- Factorized: 공간 내 attention → 시간 간 attention (분리)

### 2.5 롤아웃/추론 안정화

- **모방학습(Imitation Learning)**: 수십억 마일의 실제 주행 데이터로 학습
- **DAgger 유사 온라인 학습**: 모델 실수를 교정하여 재학습
- **RL 미세조정**: 학습된 보상 모델로 플래너 개선

---

## 3. 역공학 추론

| 추론 항목 | 공개 근거 | 확신도 |
|----------|----------|--------|
| RegNet/EfficientNet backbone | AI Day 2021/2022 발표 | 높음 |
| BEV spatial transformer | AI Day 2021 Karpathy 발표 | 높음 |
| 3D Occupancy Network | AI Day 2022 직접 발표 | 높음 |
| GRU/Attention 시간 융합 | AI Day 발표 + 특허 US11,727,710 | 중간 |
| VQ-VAE 장면 토큰화 | 세계모델 특허 US2023/0415777 | 중간 |
| End-to-end planning transformer | 특허 US2024/0025449 + Elluswamy 인터뷰 | 중간 |
| ~1.5B params | Elluswamy X(Twitter) 게시 | 중간 |
| SSM/Mamba 사용 | 미확인 | 낮음 |

---

## 4. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| **Factorized 시공간 처리** | O (이미 채택) | 인코더(공간) + Mamba(시간) 분리. Tesla와 동일 철학 | - |
| 3D 복셀 상태 표현 | O (이미 유사) | 우리 (20,6,6) 격자 = 3D 복셀 구조 | - |
| **잔여(residual) 예측** | △ | Tesla는 delta 기반 예측 사용 가능성. GraphCast와 동일. 실험 후 판단 | 하 |
| MCTS 기반 궤적 탐색 | X | 우리는 29 Branch GT 직접 비교. 탐색 불필요 | - |
| VQ-VAE 이산 토큰화 | X | 연속 잠재 벡터(128D)가 물리량 표현에 더 적합 | - |
| NeRF 자기지도학습 | X | 우리는 GT 있음. 자기지도 불필요 | - |
| 모방학습 + DAgger | △ | 개념적으로는 teacher forcing → scheduled sampling과 유사 | - |

---

## 5. 핵심 차용 후보

### 이미 반영된 설계 패턴
- **Factorized 시공간 처리**: 공간(인코더 FullAttn) + 시간(Mamba) 분리. Tesla 규모에서도 이 분리가 작동함을 확인
- **3D 격자 상태 표현**: 복셀(Tesla) / cell(우리)의 3D 구조화된 상태 표현

### 수정 후 적용 가능
- **잔차 예측**: `state(t+1) = state(t) + model_output`. 학습 안정성 향상 가능. Tesla와 GraphCast 모두 이 패턴 사용 추정

### 부적합
- **MCTS/RL planning**: Branch GT가 있으므로 불필요
- **VQ-VAE 이산화**: 물리량의 연속성 보존이 중요
