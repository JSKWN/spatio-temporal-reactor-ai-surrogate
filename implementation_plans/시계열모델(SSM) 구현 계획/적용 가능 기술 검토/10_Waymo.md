# Waymo UniSim — Neural Closed-Loop Sensor Simulation

> **출처**: Yang et al. (2023). *UniSim: A Neural Closed-Loop Sensor Simulator.* CVPR 2023. [arXiv:2308.01898](https://arxiv.org/abs/2308.01898)
> **공개 수준**: 부분공개 (논문 공개, 코드 미공개)
> **우리 아키텍처 맥락**: 구성적(compositional) 장면 표현과 Neural Field 기반 재구성. 디코더 설계 관점에서 참고

---

## 1. 시스템 개요

### 직관

UniSim은 실제 주행 로그에서 **신경 장면 재구성**을 학습하고, 자율주행 정책을 평가하기 위한 **닫힌 루프(closed-loop) 시뮬레이터**를 구축한다. 세계모델이 아닌 **센서 시뮬레이터**이며, 관측을 생성하는 역할.

### 아키텍처

```
[주행 로그] → [Static Background: Neural Radiance Field (Instant-NGP)]
           → [Dynamic Actors: HyperNet → 개별 Neural Field]
           → [Volume Rendering → 저해상도 Feature Map]
           → [CNN Decoder → 고해상도 이미지/LiDAR 생성]
```

---

## 2. 기능별 상세 분석

### 2.1 상태 표현 — 구성적 장면 분해

UniSim의 핵심 기여는 장면을 **3가지 독립 요소로 분해**하는 것:
1. **정적 배경**: 도로, 건물 등. 단일 Neural Field로 표현
2. **동적 객체**: 차량, 보행자. 각각 **독립 Neural Field** (HyperNet이 생성)
3. **원거리 영역**: 하늘, 먼 배경. 별도 모델

**우리 문제에 대한 시사점**: 원자로 노심도 구성적으로 분해 가능:
| UniSim 요소 | 우리 대응 |
|---|---|
| 정적 배경 (도로/건물) | **연료 조성 (xs_fuel)**: LP별 고정, 시간 불변 |
| 동적 객체 (차량/보행자) | **시변 물리량 (Xe, flux, 온도)**: 매 시점 변화 |
| 제어 가능 요소 (자차 궤적) | **제어봉 (rod_map), 출력 (p_load)**: 외부 제어 입력 |

이 분해가 시사하는 것: xs_fuel(고정)과 시변 물리량(변화)을 **분리 처리**하는 것이 구조적으로 자연스럽다. 현재 설계에서 xs_fuel은 인코더 입력 21ch의 일부로 매 시점 제공되지만, 실제로는 시간에 따라 변하지 않으므로 별도 처리(예: FiLM conditioning)도 검토 가능.

### 2.2 HyperNet — 공유 사전지식에서 개별 표현 생성

**직관**: HyperNet은 "하나의 네트워크가 다른 네트워크의 가중치를 생성"하는 구조. UniSim에서는 하나의 HyperNet이 각 차량/보행자의 Neural Field 파라미터를 생성한다.

**우리 문제와의 관련**: 100개 LP의 각 cell은 서로 다른 연료 조성을 가지지만, 동일한 물리 법칙을 따른다. HyperNet 개념을 적용하면 "LP의 xs_fuel에서 cell별 Mamba 초기 상태 또는 바이어스를 생성"하는 방안이 가능. 단, 현재는 xs_fuel을 입력에 concat하는 것으로 충분할 수 있음.

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| 구성적 장면 분해 | △ | xs_fuel(고정) vs 시변 물리량 분리 처리 검토 가치. 현재는 concat으로 충분 |
| HyperNet | △ | LP별 Mamba 바이어스 생성. 복잡도 대비 이점 불확실 |
| Neural Field 표현 | X | 우리 격자는 이산적 (720 cell). 연속 표현 불필요 |
| CNN 디코더 업샘플링 | X | 우리 출력 해상도 = 입력 해상도 (5×5). 업샘플링 불필요 |

### 핵심 차용
- **구성적 분해 관점**: 고정 데이터(xs_fuel)와 시변 데이터의 역할을 명확히 분리하는 설계 원칙 참고

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| UniSim | Yang et al. CVPR 2023. [arXiv:2308.01898](https://arxiv.org/abs/2308.01898) |
