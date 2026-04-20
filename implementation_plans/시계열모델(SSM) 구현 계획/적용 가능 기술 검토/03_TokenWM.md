# TokenWM — 토큰화된 잠재 상태로 세계를 모델링

> **출처**: Zhai, G., Zhang, X., & Navab, N. (2025). *Recurrent World Model with Tokenized Latent States.* ICLR 2025 Workshop. [OpenReview](https://openreview.net/forum?id=xmwcdUdcWz)
> **공개 수준**: 완전공개

---

## 1. 시스템 개요

### 직관

DreamerV3와 Drama는 관측을 **하나의 벡터 h_t**로 압축한다. 이것은 마치 노심 전체의 상태를 "평균 온도 하나"로 요약하는 것과 같다 — 전역 추세는 포착하지만 공간적 세부 사항(어느 연료봉이 더 뜨거운지)을 잃는다.

TokenWM은 이 문제를 **토큰 집합 h_t ∈ R^(N_h × D)**로 해결한다. 상태를 하나의 벡터가 아니라 여러 개의 토큰으로 표현하면, 각 토큰이 공간의 서로 다른 부분을 담당할 수 있다.

**우리 문제에서의 핵심 발견**: 우리 인코더 출력은 이미 720개 cell token × 128차원이다. 이것은 TokenWM의 "토큰화된 잠재 상태"와 구조적으로 동일하다. 우리는 의식하지 못한 채 TokenWM 아키텍처를 이미 구현하고 있었다.

### 아키텍처 다이어그램

```
[관측 O_t]
     ↓
[ViT Encoder] → z_t ∈ R^(N_z × D_z)   (패치 토큰 집합)
     ↓
[Cross-Attention(z_t, h_{t-1})] → h_t ∈ R^(N_h × D)   (Posterior 업데이트)

[Prior — imagination]:
h_t → [+ Positional Encoding]
    → [FiLM(a_t): scale, shift = MLP(a_t)]    ← 행동 주입
    → [Post-Norm Self-Attention]               ← 토큰 간 상호작용
    → [Memory Bank Cross-Attention]            ← 과거 맥락 조회
    → h_{t+1} ∈ R^(N_h × D)
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 (Temporal Dynamics)

#### Self-Attention 기반 시간 전이

**직관**: GRU나 Mamba는 상태를 "하나의 벡터를 갱신"하는 방식으로 시간 전이를 수행한다. TokenWM은 다르다 — **토큰들이 서로 대화(Self-Attention)하면서** 다음 시점의 상태를 결정한다.

**작동 방식**:
```
h_t의 N_h개 토큰이 Self-Attention으로 상호작용
→ 토큰 i가 토큰 j의 정보를 참조하여 자신을 업데이트
→ 모든 토큰이 동시에 업데이트되어 h_{t+1} 생성
```

**Post-Norm (Pre-Norm이 아님)**: TokenWM은 의도적으로 Post-Norm을 사용한다. 이유:
- 장기 rollout에서 Pre-Norm은 잔차 연결을 통해 초기 정보가 너무 강하게 유지됨
- Post-Norm은 각 레이어의 출력을 정규화하여 시간이 지남에 따라 새로운 정보가 잘 반영됨
- 이것은 autoregressive rollout 안정성에 중요

**우리 문제와의 대응**: 우리는 시간 전이에 Mamba(cell-wise 독립)를 사용한다. TokenWM의 Self-Attention 방식과 비교하면:

| | TokenWM Self-Attention | 우리 cell-wise Mamba |
|---|---|---|
| 토큰 간 상호작용 | O (매 시점 전체 토큰이 소통) | X (각 cell이 독립 처리) |
| 전역 상태 파악 | Self-Attention이 자동으로 수행 | 인코더 attention에만 의존 |
| 계산 비용 | O(N_h²) per step | O(N_h) per step |
| 장기 의존성 | Attention 범위 내에서 포착 | Mamba h_t에 압축 |

**핵심 질문**: 우리 cell-wise 독립 Mamba에서 전역 상태(keff 추세, 전체 Xe 축적량 등)는 어떻게 유지되는가?
- 현재: 720개 cell의 개별 h_i(t)에 **분산 저장**. 인코더 attention이 매 시점 전역 정보를 전달하지만, 시간적 전역 추세는 추적이 어려움
- TokenWM 접근: 매 시점 Self-Attention으로 토큰이 상호 참조 → 전역 패턴 자연스럽게 포착
- **절충안**: Global token 4~8개를 추가하여, Mamba에서도 시스템 수준 상태를 유지 (아래 §5 참조)

### 2.2 상태 표현 (State Representation)

#### 토큰 집합 잠재 상태

**TokenWM의 핵심 혁신**:
- h_t가 **단일 벡터가 아니라 토큰 집합** (N_h × D)
- 각 토큰이 관측의 서로 다른 측면을 담당
- 확률적 잠재 변수 없음 (**결정적**) → KL balancing 불필요

**우리 설계와의 직접 대응**:

| TokenWM | 우리 |
|---|---|
| h_t ∈ R^(N_h × D) | 인코더 출력 (720 × 128) |
| N_h: 학습된 토큰 수 | 720: 물리적 cell 수 (20×6×6 halo) |
| D: 잠재 차원 | 128: 인코더 출력 차원 |
| 토큰의 의미: 학습됨 | 토큰의 의미: **물리적 위치에 고정** |

**중요한 차이**: TokenWM의 토큰은 의미가 학습에 의해 결정되지만, 우리 토큰은 물리적 cell 위치에 고정되어 있다. 이것은 장점이다:
- 각 토큰의 물리적 의미가 명확 → 해석 가능성
- 공간 구조가 보존됨 → 디코더가 공간 재구성에 활용 가능
- LAPE + STRING이 위치 정보를 정확히 인코딩

#### Memory Bank

TokenWM은 과거 토큰 표현을 **Memory Bank**에 저장하고, Cross-Attention으로 조회한다:
```
h_{t+1} = SelfAttn(h_t) + CrossAttn(h_t, MemoryBank)
```
이는 장기 기억을 유지하는 메커니즘으로, 긴 시퀀스에서 과거 정보를 참조할 수 있게 한다.

**우리 문제에서의 대응**: Mamba의 h_t 자체가 과거 정보의 압축이므로 별도 Memory Bank가 불필요. 단, 575 step의 매우 긴 시퀀스에서 초기 정보가 h_t에서 소실될 가능성 → Mamba-3의 복소 상태가 이를 완화할 수 있음 (06_Mamba_계열.md 참조).

### 2.3 제어/조건 주입 (Action/Condition Injection)

#### FiLM Conditioning — 토큰별 변조

**직관**: Concat은 "행동 정보를 옆에 붙여서 같이 처리하는 것"이라면, FiLM은 "행동이 토큰의 성질 자체를 바꾸는 것"이다.

**FiLM (Feature-wise Linear Modulation)** 수식:
```
scale, shift = MLP(a_t)
h_modulated = scale ⊙ h_t + shift
```

- 행동 a_t에서 scale과 shift를 생성
- 토큰의 각 feature를 곱셈(scale)과 덧셈(shift)으로 변조
- **토큰별** 변조: 모든 토큰에 동일한 scale/shift 적용 (토큰 위치와 무관)

**Concat과 FiLM 비교**:

| | Concat | FiLM | AdaLN |
|---|---|---|---|
| 메커니즘 | 입력 벡터에 행동을 이어붙임 | 입력 feature에 곱하고 더함 | LayerNorm의 γ, β를 행동으로 생성 |
| 행동 반영 시점 | 입력 단계 | 중간 feature 단계 | 정규화 단계 |
| 행동의 영향 | 후속 연산이 알아서 활용 | 직접적으로 feature 크기/위치 변경 | 분포 자체를 조절 |
| 사용 예시 | DreamerV3, Drama | **TokenWM** | DiT, 우리 디코더 |

**우리 문제에서의 적용 가능성**:
- rod_map은 공간 분포 → FiLM(스칼라)으로는 부적합. 인코더 입력(21ch)이 적합
- p_load는 스칼라 → Concat(현재), FiLM, AdaLN 모두 적용 가능
- 현재 설계: Mamba에 Concat, 디코더에 AdaLN-Zero. FiLM은 추가 검토 가치 있으나, 기존 방식이 이미 학술적으로 검증됨

### 2.4 시공간 결합 (Spatiotemporal Coupling)

#### TokenWM: 시간 전이에서 공간 결합을 수행

DreamerV3/Drama와 달리, TokenWM은 **시간 전이 단계에서 공간 토큰 간 상호작용**을 수행한다:
```
시간 전이 = Self-Attention(토큰 집합) + FiLM(행동)
```
Self-Attention이 토큰 간 관계를 학습하므로, 시간 전이가 공간 구조를 반영한다.

**우리 문제에서의 시사점**: 현재 cell-wise 독립 Mamba는 시간 전이에서 공간 결합을 하지 않는다. 대안:

1. **현재 설계 유지**: 공간 결합은 인코더/디코더 attention에만 의존. Mamba는 순수 시간 전이
2. **Global token 추가**: 720 cell token 외에 4~8개 시스템 토큰을 추가. Mamba에서 함께 처리하되, 인코더 attention에서 global token이 전체 cell 정보를 집약
3. **Mamba 후 가벼운 Attention**: 각 시점에서 Mamba 출력에 1-layer attention을 적용하여 cell 간 정보 교환. 계산 비용 증가

→ 2번(Global token)이 비용 대비 효과가 가장 높을 것으로 예상. 이미 기존 검토에서 제안된 방안.

### 2.5 롤아웃/추론 안정화

#### Post-Norm + Self-Attention의 장기 안정성

TokenWM이 Post-Norm을 선택한 이유가 정확히 이것이다:
- Pre-Norm: `x + Attn(Norm(x))` → 잔차가 정규화되지 않은 x를 직접 전달 → 초기 값이 끝까지 유지
- Post-Norm: `Norm(x + Attn(x))` → 잔차 합산 후 정규화 → 매 레이어에서 분포 재조정

장기 rollout(수백 step)에서 Post-Norm이 더 안정적이라는 보고.

**우리 문제**: 575 step rollout에서 Post-Norm vs Pre-Norm의 선택은 디코더 설계 시 검토 대상. 현재 인코더는 Pre-LN (ViT 표준). Mamba 자체는 Norm 구조가 다름 (RMSNorm 사용이 일반적).

### 2.6 분기/다중경로 평가

해당 없음. TokenWM은 단일 궤적 상상만 수행.

---

## 3. 역공학 추론

해당 없음 (완전공개).

---

## 4. 우리 문제 적용성 평가

| 기능 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| 토큰 집합 잠재 상태 | **O** | 우리 720 cell token이 이미 이것. 확인 및 정당화 | - |
| Self-Attention 시간 전이 | X | 720 token → O(720²) = 518,400 연산/step. 비용 과다 | - |
| Global token 추가 | **O** | 4~8개 시스템 토큰으로 전역 상태 유지. 최소 비용 | 하 |
| Memory Bank | △ | Mamba h_t가 대체. 극히 긴 시퀀스에서만 검토 가치 | 중 |
| FiLM conditioning | △ | p_load에 적용 가능하나 기존 Concat+AdaLN로 충분 | 하 |
| Post-Norm 안정성 | △ | 디코더 설계 시 검토. 575-step rollout 안정성 기여 가능 | 하 |
| 결정적 잠재 상태 (KL 없음) | **O** | 우리와 동일한 결정. TokenWM이 정당화 사례 | - |

---

## 5. 핵심 차용 후보

### 즉시 적용 가능
- **토큰 집합 잠재 상태 정당화**: 우리 720 cell token × 128D가 TokenWM의 아이디어와 구조적으로 동일하다는 것을 문서화. 아키텍처 타당성의 학술적 근거
- **결정적 잠재 상태**: TokenWM도 확률적 잠재 없이 결정적 토큰만 사용. 우리 설계 방향 확인

### 수정 후 적용 가능
- **Global token (4~8개)**: 인코더 attention에서 720 cell + 4~8 global token을 함께 처리. Global token은 시스템 수준 요약(keff 추세, 총 Xe 축적량, AO 등)을 학습. Mamba에서도 cell token과 함께 시간 전이. **구현 비용 최소, 잠재적 효과 높음**
  ```
  인코더: [720 cell token + 8 global token] → Attention → [728 token × 128D]
  Mamba:  [728 token × T] cell-wise + global-wise 시간 전이
  디코더: global token → 전역 정보 주입, cell token → 공간 재구성
  ```

### 부적합
- **Self-Attention 시간 전이**: 720 토큰에 대한 매 시점 Self-Attention은 계산 비용이 너무 높음 (575 step × O(720²)). Mamba의 O(N) 복잡도가 우리 규모에 적합
- **Memory Bank**: Mamba의 h_t가 동일한 역할을 더 효율적으로 수행
