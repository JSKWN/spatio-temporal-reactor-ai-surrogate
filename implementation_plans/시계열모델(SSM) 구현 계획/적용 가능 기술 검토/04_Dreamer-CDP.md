# Dreamer-CDP — 재구성 없이 세계를 이해하기

> **출처**: Mohammadi, Halvagal, & Zenke. (2026). *Reconstruction-Free World Models.* [arXiv:2603.07083](https://arxiv.org/abs/2603.07083)
> **공개 수준**: 완전공개

---

## 1. 시스템 개요

### 직관

DreamerV3는 세계모델을 학습할 때 **디코더로 관측을 재구성**한다 (이미지 → 잠재 상태 → 이미지 복원). 이 재구성 loss가 인코더에 "유의미한 잠재 표현을 만들라"는 학습 신호를 제공한다.

Dreamer-CDP의 핵심 질문: **디코더 없이도 좋은 잠재 표현을 학습할 수 있는가?**

답: **그렇다.** 디코더를 제거하고, 대신 잠재 공간에서의 **예측 일관성 손실(Contrastive Dynamics Prediction, CDP)**로 대체하면 동등 이상의 성능을 얻는다. 핵심 아이디어는 "다음 시점의 잠재 표현을 정확히 예측하라"는 목표를 잠재 공간 안에서 직접 부여하는 것이다.

**우리 문제에서의 중요성**: 우리는 디코더를 유지할 것이므로 디코더 제거 자체는 적용하지 않는다. 그러나 **잠재 예측 loss를 추가 학습 신호로 사용하는 아이디어**는 Mamba에 직접적인 gradient를 제공하여 학습을 가속할 수 있다.

### 아키텍처 다이어그램

```
[DreamerV3 기본 구조 유지]
  RSSM (GRU h_t + Cat z_t) — 변경 없음
  Actor-Critic — 변경 없음

[변경 사항]
  디코더 삭제
  + Predictor MLP 추가:
      h_t → [MLP: 8192 → 4096 → 4096] → u_hat_t  (예측된 표현)
  + Feature Extractor:
      x_t → [Feature Extractor] → u_t              (목표 표현, stop-gradient)

  L_CDP = -cos(sg(u_t), u_hat_t)    ← cosine similarity, target에 stop-gradient
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 (Temporal Dynamics)

DreamerV3와 동일 (Block GRU). 시간 모델링 자체에 변경 없음.

### 2.2 상태 표현 (State Representation)

DreamerV3와 동일 (h_t + Cat(32×32) z_t). 상태 표현 자체에 변경 없음.

**단, KL 항(L_dyn, L_rep)은 반드시 유지해야 한다**:
- CDP에서 KL 항을 제거하면 **표현 붕괴(representation collapse)** 발생
- 인코더가 모든 입력에 대해 같은 표현을 출력하여 CDP loss를 0으로 만들 수 있음
- KL 항이 Prior와 Posterior 사이의 정보 전달을 강제하여 붕괴 방지

**우리 문제에서의 시사점**: 우리는 확률적 잠재를 쓰지 않으므로 KL 항이 없다. 만약 잠재 예측 loss를 도입한다면, 표현 붕괴를 방지하는 별도 메커니즘이 필요:
- **방법 1**: L_data(재구성 loss)가 있으므로 인코더가 의미 있는 표현을 유지 → 붕괴 위험 낮음
- **방법 2**: EMA(Exponential Moving Average) target encoder 사용 (V-JEPA 스타일)
- **결론**: 우리는 디코더를 유지하므로 L_data 자체가 붕괴 방지 역할. 추가 메커니즘 불필요할 가능성 높음

### 2.3 제어/조건 주입 (Action/Condition Injection)

DreamerV3와 동일. 변경 없음.

### 2.4 시공간 결합 (Spatiotemporal Coupling)

DreamerV3와 동일. 변경 없음.

### 2.5 롤아웃/추론 안정화

#### CDP 손실 — Mamba에 직접적인 학습 신호 제공

**이것이 Dreamer-CDP에서 가장 중요한 기여이다.**

**현재 우리 설계의 문제**:
```
Mamba에 대한 gradient 경로:
  Mamba 출력 → Decoder → L_data (재구성) → 역전파 → Decoder → Mamba
  Mamba 출력 → Decoder → L_physics (물리 제약) → 역전파 → Decoder → Mamba
```
모든 gradient가 **디코더를 경유**한다. Mamba가 "다음 잠재 상태를 정확히 예측하라"는 직접적인 신호를 받지 못한다.

**CDP 추가 시**:
```
기존 경로 유지 +
추가: Mamba 출력 d_t → Predictor MLP → u_hat_t
      Encoder(state_{t+1}_GT) → sg(z_{t+1}) = u_t
      L_latent = -cos(sg(u_t), u_hat_t)     ← Mamba에 직접 gradient!
```

**물리적 비유**: 기존에는 Mamba가 "디코더가 만든 결과물의 품질로 평가"받았다면, CDP를 추가하면 "다음 시점의 잠재 상태를 얼마나 정확히 예측했는지"로 **직접 평가**받는다.

**구현 세부사항** (Dreamer-CDP 논문 기준):
- Predictor MLP: 8192 → 4096 → 4096 (상당히 큰 네트워크)
- β_CDP = 500 (매우 높은 가중치)
- 인코더 learning rate = 6e-6 (매우 낮게 설정하여 인코더 과적합 방지)
- EMA target network **없음** — stop-gradient만 사용

**우리 문제에 맞는 구현**:
```python
# 의사코드
z_t = Encoder(state_t)              # 현재 시점 인코더 출력
d_t = Mamba(z_t, p_load_t)          # Mamba 출력
z_next_gt = Encoder(state_{t+1}_GT)  # 다음 시점 인코더 출력 (GT 사용)

# 잠재 예측 loss (Mamba에 직접 gradient)
u_hat = Predictor_MLP(d_t)          # Mamba 출력에서 예측
u_gt = tf.stop_gradient(z_next_gt)  # target에 stop-gradient

L_latent = -tf.reduce_mean(
    tf.reduce_sum(u_hat * u_gt, axis=-1) /
    (tf.norm(u_hat, axis=-1) * tf.norm(u_gt, axis=-1))
)  # cosine similarity loss

# 전체 loss
L = L_data + L_physics + λ_latent * L_latent
```

**λ_latent 튜닝**: Dreamer-CDP에서 β_CDP=500은 디코더가 없기 때문이다. 우리는 디코더가 있으므로 훨씬 낮은 가중치(0.1~1.0)에서 시작하여 조절.

### 2.6 분기/다중경로 평가

해당 없음. DreamerV3와 동일한 imagination 패턴.

---

## 3. 역공학 추론

해당 없음 (완전공개).

---

## 4. 우리 문제 적용성 평가

| 기능 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| 디코더 제거 | X | 우리는 공간 재구성(10ch 물리량)이 필수 출력 | - |
| **잠재 예측 loss (CDP)** | **O** | Mamba에 직접 gradient 제공. 학습 가속 기대 | 중 |
| Predictor MLP 추가 | O | d_t → MLP → 다음 z_{t+1} 예측. 간단한 추가 모듈 | 하 |
| Stop-gradient target | O | sg(z_{t+1})로 target 고정. 구현 단순 | 하 |
| KL 항 유지 (붕괴 방지) | △ | 우리는 L_data가 있으므로 KL 불필요할 가능성. 모니터링 필요 | - |
| Cosine similarity loss | O | L1, L2 대비 방향 일관성에 초점. 물리량 스케일 불변 | 하 |

---

## 5. 핵심 차용 후보

### 즉시 적용 가능
- **잠재 예측 loss**: Mamba 출력 d_t에서 Predictor MLP를 통해 다음 시점 인코더 출력 sg(z_{t+1})을 예측. Mamba에 디코더를 거치지 않는 직접적 학습 신호 제공
- **Cosine similarity loss**: 방향 일관성에 초점을 맞추어, 잠재 공간에서의 예측 품질을 평가. 물리량의 절대 스케일에 무관

### 수정 후 적용 가능
- **가중치 조절**: 원본의 β_CDP=500은 디코더 없는 환경의 값. 우리는 L_data + L_physics가 있으므로 λ_latent=0.1~1.0에서 시작. 학습 초기에 높게 설정하여 Mamba가 빠르게 수렴하도록 하고, 후기에 줄이는 스케줄도 검토
- **L1/L2 vs Cosine**: Dreamer-CDP는 cosine을 사용하지만, 우리 문제에서 잠재 벡터의 크기(magnitude)도 중요할 수 있음. L1 loss를 대안으로 시험 가치 있음 (V-JEPA 2는 L1 사용)

### 부적합
- **디코더 제거**: 우리 최종 출력은 10채널 3D 물리량이므로 디코더(공간 복원)가 필수
- **EMA target encoder**: 우리는 L_data가 인코더를 안정적으로 유지하므로 EMA 없이도 표현 붕괴 위험이 낮음. 단, 모니터링 필요
