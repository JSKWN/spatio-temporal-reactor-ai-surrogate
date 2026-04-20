# DreamerV3 — 범용 세계모델의 기준선

> **출처**: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* [arXiv:2301.04104](https://arxiv.org/abs/2301.04104). ICLR 2024, Nature 2025.
> **공개 수준**: 완전공개 (논문 + 코드 + 하이퍼파라미터)

---

## 1. 시스템 개요

### 직관

DreamerV3는 "관측만으로 세상이 어떻게 돌아가는지 배우고, 그 상상 속에서 행동을 연습한다"는 모델 기반 강화학습(MBRL)의 대표 시스템이다. 실제 환경과 상호작용하는 횟수를 최소화하면서도, 내부 모델이 만들어낸 가상 궤적(imagination)으로 정책을 개선한다.

우리 문제와의 핵심 연결점: DreamerV3의 "세계모델"은 본질적으로 **상태 전이 예측기**이다. 현재 상태 + 행동(제어 입력) → 다음 상태를 예측한다. 이것은 우리의 "현재 노심 상태 + 제어봉/출력 → 다음 시점 노심 상태" 문제와 구조적으로 동일하다.

### 아키텍처 다이어그램

```
[관측 x_t]                                [행동 a_{t-1}]
     ↓                                         ↓
[CNN Encoder]                          [z_{t-1} concat a_{t-1}]
     ↓                                         ↓
   enc_t                              [Linear → Block GRU]
     ↓                                         ↓
[Posterior MLP(h_t, enc_t)]              h_t (deterministic)
     ↓                                         ↓
z_t ~ Cat(32×32)                      [Prior MLP(h_t)]
     ↓                                         ↓
[CNN Decoder] → x_hat_t              z_hat_t ~ Cat(32×32)
[Reward MLP]  → r_hat_t
[Continue MLP] → c_hat_t
```

**RSSM (Recurrent State Space Model)**: 상태를 두 부분으로 나눈다.
- **h_t** (deterministic): Block GRU로 전파되는 "확실한 기억". 과거 관측과 행동의 요약.
- **z_t** (stochastic): 32개 카테고리 분포 × 32 클래스 = 1024차원 이산 잠재 상태. "불확실한 현재 인식".

### 입출력

| | 형태 | 설명 |
|---|---|---|
| 입력 | 이미지 (64×64×3) 또는 벡터 | 환경 관측 |
| 행동 | 이산 또는 연속 벡터 | 에이전트의 행동 |
| 출력 | 다음 관측 예측 + 보상 예측 | 세계모델 예측 |

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 (Temporal Dynamics)

#### Block GRU — 시간 전이의 핵심

**직관**: GRU(Gated Recurrent Unit)는 LSTM보다 단순한 순환 신경망으로, "어떤 정보를 유지하고 어떤 정보를 갱신할지" 게이트로 결정한다. DreamerV3는 이를 **Block GRU**로 변형했다.

**Block GRU가 일반 GRU와 다른 점**:
- 은닉 상태 h_t를 8개 독립 블록으로 분할
- 각 블록이 독립적으로 업데이트 (블록 간 간섭 없음)
- 각 블록 후 RMSNorm + SiLU 활성화
- 효과: 그래디언트 흐름 개선, 안정적 장기 학습

**우리 문제와의 대응**:

| DreamerV3 | 우리 |
|---|---|
| Block GRU h_t | Mamba hidden state h_t |
| 8개 독립 블록 | 720개 독립 cell (더 극단적인 분리) |
| a_{t-1} concat → GRU 입력 | p_load concat → Mamba 입력 |
| 학습: TBPTT (짧은 윈도우) | 학습: parallel scan (전 시퀀스 한번에) |
| 추론: 1-step GRU | 추론: 1-step Mamba recurrence |

**핵심 차이**: GRU의 게이트(reset, update)는 고정된 sigmoid 구조인 반면, Mamba의 A/B/C/Delta는 입력마다 달라지는 data-dependent 파라미터다. Mamba가 더 유연하게 "이번 시점에서 무엇을 기억하고 무엇을 잊을지"를 결정할 수 있다.

#### 시간 전이 흐름

```
1. 이전 잠재 상태 z_{t-1}과 행동 a_{t-1}을 concat
2. Linear projection으로 GRU 입력 크기에 맞춤
3. Block GRU 업데이트 → h_t 생성
4. h_t에서 Prior MLP로 z_hat_t 예측 (관측 없이 상상하는 다음 상태)
5. 실제 관측이 있으면 Posterior MLP로 z_t 생성 (관측을 반영한 교정)
```

### 2.2 상태 표현 (State Representation)

#### RSSM의 이중 상태 구조

**직관**: 왜 상태를 deterministic(h_t)과 stochastic(z_t) 두 개로 나눌까?

물리적 비유로 설명하면:
- **h_t**: 노심의 장기 추세 (Xe 축적 이력, 온도 변화 방향). 과거를 "확실하게" 요약한 기억.
- **z_t**: 현재 시점의 세밀한 상태 (정확한 출력 분포, 온도 분포). 관측으로 교정해야 알 수 있는 "불확실한" 부분.

이 분리가 중요한 이유: imagination(상상) 시에는 관측이 없으므로 z_hat_t(Prior에서 샘플링)만 사용한다. 실제 관측이 들어오면 z_t(Posterior)로 교정한다. 이 구조가 "닫힌 루프(실제) vs 열린 루프(상상)"를 자연스럽게 전환한다.

#### Cat(32×32) 이산 잠재 상태

- 32개 카테고리 분포, 각각 32개 클래스
- Straight-through gradient로 이산 샘플링의 역전파 문제 해결
- 연속 잠재 변수보다 후방 붕괴(posterior collapse) 방지에 효과적
- 총 1024차원의 이산 코드 → h_t와 합쳐 전체 상태 구성

**우리 문제와의 대응**: 우리는 확률적 잠재 상태를 사용하지 않는다 (결정적 128차원 벡터). 이는 우리 문제의 특성 때문:
- MASTER는 결정론적 시뮬레이터 → 동일 입력에 동일 출력
- 불확실성 모델링보다 정확한 예측이 목표
- KL balancing 등 복잡한 학습 안정화가 불필요

### 2.3 제어/조건 주입 (Action/Condition Injection)

#### 행동 주입 방식: Concat → Linear → GRU

DreamerV3의 행동(action) 주입은 매우 단순하다:

```python
# 의사코드
gru_input = Linear(concat(z_{t-1}, a_{t-1}))  # 잠재 상태 + 행동 결합
h_t = BlockGRU(gru_input, h_{t-1})             # GRU 업데이트
```

- 행동을 별도 경로로 넣지 않고, 이전 잠재 상태와 **직접 concat**
- 단순하지만 효과적: 시간 전이 동역학에 행동이 직접 영향
- 디코더에는 행동이 주입되지 않음 (디코더는 h_t + z_t만 사용)

**우리 문제와의 대응**:

| DreamerV3 행동 주입 | 우리 제어 주입 |
|---|---|
| a_{t-1} → GRU 입력 (concat) | p_load → Mamba 입력 (concat) |
| 디코더에 행동 없음 | 디코더에 p_load AdaLN-Zero |
| 단일 행동 벡터 | rod_map(공간) + p_load(스칼라) 이원화 |

**DreamerV3와의 차이점**: DreamerV3의 행동은 단순 벡터(조이스틱 방향 등)이므로 concat 하나로 충분하다. 우리는 rod_map이 3D 공간 분포를 가지므로 인코더 입력(21ch)에 넣고, p_load(스칼라)만 Mamba에 concat한다. 디코더에도 p_load를 AdaLN으로 넣는 것은 DreamerV3에는 없는 설계이며, 이는 출력이 p_load 수준에 강하게 의존하기 때문이다.

### 2.4 시공간 결합 (Spatiotemporal Coupling)

#### DreamerV3의 접근: 완전 분리

```
공간: CNN Encoder/Decoder (관측 ↔ 잠재 벡터 변환)
시간: GRU (잠재 벡터 시퀀스의 시간 전이)
```

- CNN은 개별 시점의 관측을 처리 (시간 축 없음)
- GRU는 잠재 벡터의 시퀀스를 처리 (공간 구조 없음)
- **완전한 공간-시간 분리**: 공간 정보는 인코더가 압축하고, 시간 정보는 GRU가 관리

**우리와의 유사성**: 우리도 동일한 분리 전략을 채택했다.
- 공간: Spatial Encoder (FullAttention3D) → 720 cell token 생성
- 시간: Mamba → cell-wise 시간 전이

**차이**: DreamerV3의 CNN은 공간을 **하나의 벡터**로 압축하지만, 우리 인코더는 **720개 토큰**을 유지한다. 이는 TokenWM의 접근과 더 유사하며, 공간 구조를 시간 모델에 전달할 수 있다 (현재는 cell-wise 독립이므로 실제로는 전달하지 않지만).

### 2.5 롤아웃/추론 안정화

#### Imagination Rollout

DreamerV3의 추론은 **열린 루프 상상(open-loop imagination)**이다:

```
h_t에서 시작 (실제 관측으로 구축된 상태)
    → Prior에서 z_hat_{t+1} 샘플
    → Actor가 a_{t+1} 생성
    → GRU step: h_{t+2} = GRU(concat(z_hat_{t+1}, a_{t+1}), h_{t+1})
    → Prior에서 z_hat_{t+2} 샘플
    → ... 반복 (보통 15 step)
```

- 상상 중에는 관측이 없으므로 Prior만 사용
- 오차가 누적될 수 있으나, 15 step 정도는 충분히 정확
- KL balancing이 Prior와 Posterior를 가깝게 유지하여 상상 품질 보장

**우리 문제에서의 시사점**: 우리의 575-step 자기회귀 롤아웃은 DreamerV3의 15-step imagination보다 **38배 길다**. 이 차이가 의미하는 것:
- DreamerV3 수준의 단순한 open-loop은 575 step에서 오차가 폭발할 가능성 높음
- pushforward trick, scheduled sampling 등 추가 안정화 기법이 필수
- 물리 제약 Loss (L_Bateman, L_diff_rel)가 물리법칙 이탈을 제한하는 "가드레일" 역할

#### 손실 함수

```
L = L_pred + L_dyn + 0.1 × L_rep

L_pred = L_image + L_reward + L_continue    (재구성 + 보상 + 종료 예측)
L_dyn  = KL(sg(posterior) || prior)         (Prior를 Posterior에 맞추라)
L_rep  = KL(posterior || sg(prior))         (Posterior를 Prior에 맞추라)
```

- **KL balancing**: L_dyn 가중치 > L_rep 가중치 (기본 1:0.1)
- sg = stop-gradient. Prior와 Posterior가 서로를 "끌어당기되", Prior가 더 많이 움직이도록 유도
- Free bits = 1 nat: KL이 1 미만이면 gradient를 차단하여 과도한 압축 방지

**우리와의 대응**: 확률적 잠재 상태를 쓰지 않으므로 KL balancing은 불필요. 대신 Dreamer-CDP의 잠재 예측 loss가 유사한 역할 가능 (04_Dreamer-CDP.md 참조).

### 2.6 분기/다중경로 평가

#### Imagination as Branch Evaluation

DreamerV3에서 Actor-Critic은 잠재 공간에서 여러 미래 궤적을 상상하여 최적 행동을 학습한다:

```
h_t에서 시작
    → Actor가 a_t 생성 → 궤적 1 상상
    → Actor가 a_t' 생성 → 궤적 2 상상
    → Lambda-return으로 각 궤적의 가치 평가
    → Actor를 가치가 높은 방향으로 업데이트
```

**우리와의 핵심 차이**: DreamerV3의 imagination은 **모델이 생성한 불완전한 미래**이지만, 우리 Branch GT는 **MASTER가 계산한 정밀한 미래**이다.

| | DreamerV3 Imagination | 우리 Branch GT |
|---|---|---|
| 정확도 | 모델 오차 누적 | MASTER 정밀도 (0.1% 이내) |
| 다양성 | Actor가 생성 (탐색 제한) | 29개 rod offset 체계적 탐색 |
| 학습 신호 | 보상 예측 (간접) | 직접 지도학습 (GT 존재) |
| Actor-Critic | 필수 | 불필요 |

**결론**: Actor-Critic 구조 자체는 불필요하지만, imagination의 "h_t에서 분기하여 여러 미래를 평가한다"는 사고방식은 Branch-in-CRS 문제의 원형이다. h_t snapshot → Branch 평가 → 원래 h_t 복귀의 패턴은 DreamerV3에서 시작된 아이디어.

---

## 3. 역공학 추론

해당 없음 (완전공개).

---

## 4. 우리 문제 적용성 평가

| 기능 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| Block GRU 시간 전이 | △ | 구조는 참고하되, Mamba가 data-dependent 선택성에서 우위 | - |
| RSSM 이중 상태 (h_t + z_t) | X | 결정론적 시뮬레이터이므로 확률적 잠재 불필요 | - |
| Cat(32×32) 이산 잠재 | X | 연속 벡터(128D)로 충분, KL balancing 복잡성 회피 | - |
| Action concat 주입 | O | p_load concat 방식의 직접적 근거 (이미 채택) | 하 |
| 공간-시간 분리 구조 | O | 인코더(공간) + Mamba(시간) 분리의 근거 (이미 채택) | - |
| KL balancing 손실 | X | 확률적 잠재 미사용 → 해당 없음 | - |
| Imagination 분기 패턴 | △ | Branch-in-CRS의 원형. h_t snapshot 아이디어는 적용 가능 | 중 |
| Symlog 예측 | △ | 출력 범위가 큰 물리량(Q_abs 등)에 적용 검토 가능 | 하 |

### Symlog 예측 (추가 검토 항목)

DreamerV3는 보상 예측에 **symlog 변환**을 사용한다:
```
symlog(x) = sign(x) · log(|x| + 1)
```
이는 극단적으로 큰 보상 값을 안정적으로 예측하기 위함이다. 우리 Q_abs(절대출력)도 범위가 넓으므로 (0 ~ 수 MW/node) log_zscore와 유사한 효과를 기대할 수 있다. 단, 우리는 이미 log_zscore 정규화를 사용하므로 추가 효과는 제한적.

---

## 5. 핵심 차용 후보

### 즉시 적용 가능 (이미 채택)
- **Action concat 주입**: p_load → Mamba concat. DreamerV3 패턴과 동일
- **공간-시간 분리**: 인코더(공간) + Mamba(시간). DreamerV3 CNN-GRU 패턴의 확장

### 수정 후 적용 가능
- **h_t snapshot → Branch 평가 → 복귀**: DreamerV3의 imagination 패턴을 Branch-in-CRS에 적용. 단, GRU h_t clone 대신 Mamba InferenceParams deepcopy 필요 (Drama에서 검증)
- **Symlog 예측**: Q_abs 등 넓은 범위의 물리량에 대한 출력 안정화. 현재 log_zscore와 효과 중복 가능, 시험 후 판단

### 부적합
- **RSSM 이중 상태**: 우리 문제는 결정론적. 확률적 잠재 상태 + KL balancing은 불필요한 복잡성
- **Actor-Critic**: Branch GT가 있으므로 정책 학습이 불필요. 행동 최적화는 직접 Branch 비교로 수행
- **Cat(32×32) 이산 잠재**: 연속 벡터가 더 적합 (물리량의 연속적 변화 표현)
