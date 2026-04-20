# V-JEPA 2 — 동결 인코더 위에 행동 예측기 올리기

> **출처**: Meta AI. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning.* [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
> **공개 수준**: 완전공개 (논문 + 모델 공개)

---

## 1. 시스템 개요

### 직관

V-JEPA 2는 두 가지를 분리한다:
1. **"세상을 이해하는 능력"** — 대규모 비디오로 사전학습한 ViT 인코더 (ViT-g, ~1B params)
2. **"행동이 세상을 어떻게 바꾸는지 배우는 능력"** — 소량의 로봇 데이터(62시간!)로 학습한 Action-Conditioned Predictor

핵심 발견: 좋은 표현(인코더)이 있으면, 시간 동역학 모델(predictor)은 **매우 적은 데이터**로도 학습할 수 있다.

**우리 문제에서의 중요성**: 이것은 우리 2-Phase 학습 설계를 직접적으로 정당화한다.
- Phase 1: 인코더/디코더를 Branch 데이터(26.68M 샘플)로 사전학습 → "좋은 표현" 구축
- Phase 2: 인코더 동결/미세조정 + Mamba 학습 → CRS 데이터(920K 샘플)로 시간 동역학 학습

### 아키텍처 다이어그램

```
[V-JEPA 2 기본 (자기지도 사전학습)]

비디오 프레임들
     ↓
[마스크 적용] → 일부 패치 은닉
     ↓
[ViT-g 인코더 (frozen after pretraining)]
     ↓
보이는 패치의 표현 z_visible
     ↓
[Predictor] → 은닉된 패치의 표현 z_hat_masked
     ↓
L = ||z_hat_masked - sg(z_masked)||  ← 잠재 공간에서 예측
```

```
[V-JEPA 2-AC (Action-Conditioned)]

z_t (인코더 출력, 16×16×1408)  +  a_t (7-dim 행동)  +  s_t (end-effector 상태)
     ↓
[Transformer Predictor: 24L, 16H, d=1024, block-causal attention]
     ↓
z_hat_{t+1} (다음 시점 예측)

L = L_teacher_forcing (1-step) + L_rollout (multi-step, T=2)
Loss: L1 distance in latent space
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 (Temporal Dynamics)

#### Transformer Predictor — Block-Causal Attention

**직관**: Predictor는 "과거 관측 + 현재 행동 → 다음 관측의 잠재 표현" 을 예측한다. Transformer 기반이며, **block-causal attention**으로 미래 정보 유출을 방지한다.

**Block-causal attention**:
```
시점 1: [z_1, a_1, s_1]  → 자기 시점 내에서만 attend
시점 2: [z_2, a_2, s_2]  → 시점 1, 2에 attend (과거 + 현재)
시점 3: [z_3, a_3, s_3]  → 시점 1, 2, 3에 attend
```
각 시점의 토큰들은 **과거 시점에만 attend** 가능. 이것은 GPT의 causal mask를 "블록 단위"로 적용한 것.

**우리 문제와의 대응**: 우리 Mamba는 자연스럽게 causal (h_t는 과거 정보만 포함). Transformer 기반 predictor는 Mamba의 대안으로 검토 가능하지만, 575 step에서 O(T²) 복잡도가 문제.

#### 1-step + Multi-step Loss

```
L = L_teacher_forcing + L_rollout

L_teacher_forcing: 매 시점 1-step 예측 loss (안정적)
L_rollout: 2-step 이상 자기회귀 예측 loss (누적 오차 학습)
```

**우리 문제에서의 시사점**: 우리도 teacher forcing → scheduled sampling 전환을 계획하고 있다. V-JEPA 2의 1-step + multi-step 혼합 loss는 이 전환을 부드럽게 만드는 방법.

### 2.2 상태 표현 (State Representation)

#### 동결 인코더 위의 행동 예측

**가장 중요한 설계 결정: 인코더 완전 동결**

```
ViT-g 인코더 (사전학습, ~1B params): 완전 동결 (gradient 없음)
Action-Conditioned Predictor: 62시간 로봇 데이터로만 학습
```

- 인코더는 비디오 자기지도학습으로 이미 "좋은 공간 표현"을 학습함
- Predictor만 학습하면 되므로 **데이터 효율이 극히 높음** (62시간!)
- 인코더 동결로 표현이 바뀌지 않아 학습 안정성 확보

**우리 2-Phase 설계와의 직접 대응**:

| V-JEPA 2 | 우리 Phase 1 → Phase 2 |
|---|---|
| 비디오 자기지도학습 → ViT 동결 | Branch 지도학습 → 인코더 동결 또는 미세조정 |
| 62시간 로봇 데이터 → Predictor 학습 | 920K CRS 데이터 → Mamba 학습 |
| 인코더가 공간 이해 담당 | 인코더가 노심 공간 이해 담당 |
| Predictor가 시간 동역학 학습 | Mamba가 시간 동역학 학습 |

**V-JEPA 2가 우리 2-Phase를 정당화하는 이유**:
1. 좋은 표현 위에 시간 모델을 올리면 수렴이 빠르다 (62시간만에 zero-shot planning 성공)
2. 전체를 end-to-end로 처음부터 학습하면 데이터와 계산이 훨씬 많이 필요
3. 인코더 동결/미세조정으로 Phase 1에서 학습한 표현을 보존

#### 3D RoPE

V-JEPA 2는 **3D Rotary Position Embedding**을 사용한다:
- Temporal + Height + Width 3축에 대해 독립적인 회전 인코딩
- STRING과 동일 계열 (RoPE의 다차원 확장)
- 우리 인코더가 이미 STRING(3D RoPE)을 사용 중 — 동일한 접근

### 2.3 제어/조건 주입 (Action/Condition Injection)

#### Action Vector Concat + Transformer 입력

```
입력 시퀀스 = [z_t (16×16 패치 토큰), a_t (7-dim), s_t (end-effector)]
→ 각각을 embedding하여 Transformer 입력으로 결합
→ Block-causal attention으로 처리
```

- 행동(a_t)과 상태(s_t)를 **토큰으로 변환**하여 관측 토큰과 함께 Transformer에 입력
- FiLM이나 AdaLN 없이, **동일한 attention 메커니즘**으로 처리
- 행동 토큰이 관측 토큰에 attend하고, 관측 토큰도 행동 토큰에 attend

**우리 문제와의 비교**: V-JEPA 2는 행동을 "추가 토큰"으로 넣는 반면, 우리는 p_load를 Mamba에 concat한다. 원리는 유사하지만 구현이 다름.

### 2.4 시공간 결합 (Spatiotemporal Coupling)

V-JEPA 2의 Predictor Transformer는 **시공간 통합 처리**:
- 입력이 (시점 × 패치) 시퀀스이므로 Attention이 시간과 공간을 동시에 처리
- Block-causal로 시간적 인과성은 보존
- 공간 내에서는 양방향 attention

**우리 설계와의 차이**: 우리는 공간(인코더 attention)과 시간(Mamba)을 분리. V-JEPA 2는 통합. 통합 방식이 우수할 수 있으나, 720 token × 575 step의 규모에서는 분리가 현실적.

### 2.5 롤아웃/추론 안정화

#### MPC (Model Predictive Control) 기반 Planning

V-JEPA 2는 잠재 공간에서 **CEM(Cross-Entropy Method)**으로 최적 행동을 탐색한다:
```
E(a_{1:T}; z_k, z_goal) = ||Predictor(a_{1:T}; z_k) - z_goal||_1
→ CEM으로 행동 시퀀스 최적화
→ MPC: 첫 행동만 실행, 다음 시점에서 재계획
```

**우리 문제에서의 관련성**: 행동 최적화 자체는 불필요 (Branch GT로 직접 평가). 그러나 "잠재 공간에서 미래를 예측하고, 그 예측의 품질로 행동을 평가한다"는 패러다임은 우리 Branch 평가의 개념적 배경.

### 2.6 분기/다중경로 평가

CEM이 여러 행동 후보를 병렬로 평가하는 구조. 우리 29개 Branch 평가와 유사한 구조이나, 구현 세부사항은 다름.

---

## 3. 역공학 추론

해당 없음 (완전공개).

---

## 4. 우리 문제 적용성 평가

| 기능 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| **동결 인코더 + Predictor 학습** | **O** | 우리 2-Phase 설계의 직접적 정당화 | - |
| 3D RoPE | **O** | 이미 STRING으로 채택. 동일 계열 확인 | - |
| Block-causal attention predictor | △ | Mamba의 대안으로 검토 가능. O(T²) 비용이 문제 | 중 |
| 1-step + Multi-step 혼합 loss | **O** | teacher forcing → scheduled sampling 전환의 부드러운 방법 | 하 |
| L1 잠재 예측 loss | **O** | CDP의 cosine 대안. 크기 정보도 보존 | 하 |
| MPC/CEM planning | X | Branch GT로 직접 평가. 행동 최적화 불필요 | - |

---

## 5. 핵심 차용 후보

### 즉시 적용 가능 (이미 채택 또는 설계에 반영)
- **2-Phase 설계 정당화**: V-JEPA 2의 "동결 인코더 + 소량 데이터 Predictor 학습" 패턴이 우리 Phase 1 (Branch pretrain) → Phase 2 (CRS Mamba 학습)의 타당성을 뒷받침
- **3D RoPE**: STRING과 동일 계열. 우리 인코더에서 이미 사용 중

### 수정 후 적용 가능
- **1-step + Multi-step 혼합 loss**: Phase 2에서 teacher forcing 비율을 줄이면서 동시에 2~4 step unroll loss를 혼합하는 스케줄. pushforward trick과 유사하지만 loss 수준에서 혼합하는 점이 다름
- **L1 잠재 예측 loss**: Dreamer-CDP의 cosine 대신 L1 사용. V-JEPA 2에서 L1이 잠재 예측에 효과적임을 확인. 우리 Mamba 잠재 예측 loss에도 L1을 시험 가치

### 부적합
- **ViT-g 규모 인코더**: 우리 인코더는 ~780K params (ViT-g의 1/1000). 규모가 다르므로 동결 전략의 효과가 동일하지 않을 수 있음. Phase 2에서 인코더를 완전 동결 대신 **낮은 lr로 미세조정**이 더 적합할 수 있음
- **Transformer Predictor**: 575 step × 720 token 규모에서 O(T²) 비용이 과도. Mamba의 O(T) 복잡도가 적합
