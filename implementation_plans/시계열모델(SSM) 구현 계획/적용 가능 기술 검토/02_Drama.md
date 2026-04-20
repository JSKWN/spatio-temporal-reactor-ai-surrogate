# Drama — Mamba 기반 모델 기반 강화학습

> **출처**: Wang, W., Dusparic, I., Shi, Y., Zhang, K., & Cahill, V. (2025). *Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient.* ICLR 2025. [arXiv:2410.08893](https://arxiv.org/abs/2410.08893)
> **공개 수준**: 완전공개 (논문 + 코드)

---

## 1. 시스템 개요

### 직관

Drama는 "DreamerV3의 GRU를 Mamba-2로 교체하면 어떻게 될까?"에 대한 답이다. 결과: **파라미터 수 10분의 1, 학습 데이터 절반으로도 동등한 성능**. 핵심은 Mamba-2의 입력 선택적(selective) 시간 전이가 GRU의 고정 게이트보다 환경 동역학을 더 효율적으로 모델링한다는 것이다.

**우리 문제에서의 중요성**: Drama는 Mamba를 세계모델의 시간 전이 모듈로 사용한 **최초의 검증된 사례**이다. 우리 설계(Mamba 기반 시간 모델)의 직접적 근거가 되며, InferenceParams snapshot을 통한 hidden state 관리가 Branch-in-CRS 문제의 해법을 제공한다.

### 아키텍처 다이어그램

```
[학습 — parallel mode]

z_t (인코더 출력)  +  a_t (행동)
         ↓
   [concat + Linear]
         ↓
   x_t ∈ (B, T, 512)
         ↓
   [Mamba-2 Block × 2]      ← 학습 시 parallel scan (전 시점 한번에)
         ↓
   d_t (dynamics output)
         ↓
   ┌─────────┬──────────┐
   [Prior MLP]  [Reward]  [Term]
   z_hat_{t+1}   r_hat    c_hat

   ★ 디코더는 d_t가 아니라 z_t에서 재구성 ★
   z_t → [CNN Decoder] → O_hat_t
```

```
[추론 — autoregressive step mode]

1. Real prefix 8 steps → Mamba-2 parallel → InferenceParams (SSM state) 구축
2. Step loop:
     z_hat_t + d_t → [Actor] → a_t
     concat(z_hat_t, a_t) → [Linear] → [Mamba-2 step(InferenceParams)] → d_{t+1}
     d_{t+1} → [Prior] → z_hat_{t+1}
     d_{t+1} → [Reward/Term] → r, c
```

### 핵심 수치

| 항목 | 값 |
|------|------|
| 전체 파라미터 | ~7M (DreamerV3의 ~10%) |
| Mamba-2 hidden | 512 |
| Mamba-2 layers | 2 |
| state_dim (N) | 16 |
| heads | 4 |
| head_dim | 128 |
| Prefix burn-in | 8 steps |
| Imagination horizon | 15 steps |

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 (Temporal Dynamics)

#### Mamba-2 (SSD) 시간 전이

**직관**: GRU가 "고정된 공식으로 정보를 섞는 믹서"라면, Mamba-2는 "매 시점마다 공식 자체를 입력에 맞게 바꾸는 적응형 믹서"다. 이 선택성(selectivity)이 환경의 변화를 더 섬세하게 포착한다.

**Drama에서 Mamba-2가 GRU를 대체하는 방식**:

| DreamerV3 (GRU) | Drama (Mamba-2) |
|---|---|
| `h_t = GRU(x_t, h_{t-1})` | `d_t = Mamba2(x_t; SSM_state)` |
| h_t는 hidden state 텐서 | SSM_state는 InferenceParams 객체 |
| 학습: TBPTT (짧은 윈도우 순차) | 학습: **parallel scan** (전 시퀀스 한번에) |
| 추론: 1-step 순차 | 추론: 1-step 순차 (InferenceParams 업데이트) |

**학습 속도 이점**: 
- GRU는 학습 시에도 순차적 (또는 TBPTT로 짧게 끊어서 처리)
- Mamba-2는 학습 시 **chunked matmul (SSD)**로 시간 축을 완전 병렬화
- Drama 논문 보고: 동일 환경에서 wall-clock 학습 시간 2-3배 감소

**우리 문제에서의 핵심 이점**:
- 575 step CRS 시퀀스를 순차로 처리하면 학습이 극도로 느림
- Mamba-2의 parallel scan으로 575 step 전체를 한번에 처리 가능
- 추론 시에는 어차피 자기회귀 (한 step씩)이므로, 학습-추론 모드 전환이 자연스러움

#### 2-layer 설계의 의미

Drama는 Mamba-2를 **단 2층**만 쌓았다. 이유:
- 세계모델의 dynamics는 상대적으로 단순 (Atari, DMC 등의 게임/로봇 환경)
- 인코더/디코더가 공간 처리를 담당하므로, 시간 모델은 가볍게 유지
- 2층으로도 DreamerV3의 GRU 성능을 넘었다는 것은 Mamba의 효율성을 보여줌

**우리 문제**: 원자로 동역학(Xe 축적, 온도 피드백)이 게임보다 복잡할 수 있으나, cell-wise 독립 처리(720개 병렬)가 각 cell의 부담을 줄여주므로 2-4층이면 충분할 가능성이 높다. 시작은 2층, 성능 부족 시 확장.

### 2.2 상태 표현 (State Representation)

#### DreamerV3와 동일한 RSSM 구조, Mamba로 교체

- **d_t** (deterministic): Mamba-2 출력. DreamerV3의 h_t에 대응
- **z_t** (stochastic): Cat(32×32) 이산 잠재. DreamerV3와 동일하게 유지

**중요한 설계 결정: 디코더는 d_t가 아니라 z_t에서 재구성**

```
DreamerV3: 디코더 입력 = concat(h_t, z_t)   → 순차 의존 존재
Drama:     디코더 입력 = z_t만              → 순차 의존 제거 → 완전 병렬화 가능
```

이 결정의 의미:
- 학습 시 인코더가 z_t를 생성하면 디코더가 **즉시** 재구성 loss를 계산할 수 있음
- Mamba의 parallel scan과 결합하면, 학습 전체가 시간축에서 병렬화
- 대신 디코더가 시간 맥락(d_t)을 보지 못하는 제약 → Prior/Reward head가 d_t 활용

**우리 문제와의 대응**: 우리 디코더는 Mamba 출력(d_t에 해당)을 받아 공간 재구성을 수행한다. Drama처럼 디코더에서 시간 맥락을 제거할 이유는 없다 — 우리는 CRS GT가 있으므로 직접 지도학습이 가능하고, 병렬화의 이점보다 디코더의 시간 맥락 활용이 더 중요하다.

### 2.3 제어/조건 주입 (Action/Condition Injection)

#### DreamerV3와 동일한 Concat 패턴

```python
x_t = Linear(concat(z_t, a_t))  # 잠재 상태 + 행동 결합
d_t = Mamba2(x_t)                # Mamba-2 처리
```

- 행동을 잠재 상태와 concat → Linear → Mamba 입력
- Mamba 내부에서 A/B/C/Delta가 이 결합된 입력에 data-dependent하게 반응
- 별도의 conditioning 메커니즘(FiLM, AdaLN 등) 없이 단순 concat만 사용

**우리 문제에서의 적용**: p_load concat → Mamba 입력. Drama와 정확히 동일한 패턴. p_load가 스칼라이므로 concat은 자연스럽다.

### 2.4 시공간 결합 (Spatiotemporal Coupling)

DreamerV3와 동일한 **완전 분리** 전략:
- 공간: CNN encoder/decoder (시간 축 없음)
- 시간: Mamba-2 (공간 구조 없음, 잠재 벡터만 처리)

**우리와의 유사성/차이**: 위 01_DreamerV3.md 참조.

### 2.5 롤아웃/추론 안정화

#### Prefix Burn-in — 핵심 개념

**직관**: 자동차 엔진을 바로 고속으로 돌리지 않고, 먼저 워밍업하는 것과 같다. Mamba의 SSM state를 "워밍업"시켜야 제대로 된 상태 전이가 가능하다.

**작동 방식**:
```
1. 실제 궤적의 첫 8 step을 Mamba-2에 parallel로 통과
2. 이 과정에서 SSM state (InferenceParams)가 구축됨
3. 8번째 step의 SSM state를 초기 상태로 사용
4. 이후부터 imagination (autoregressive step) 시작
```

**왜 8 step인가**: 
- 너무 짧으면 SSM state가 불충분 (초기 상태 정보가 부족)
- 너무 길면 실제 데이터를 많이 소모 (imagination 구간이 줄어듦)
- Drama 논문에서 8 step이 경험적으로 최적

**우리 문제에서의 적용**:
- Phase 2 CRS 학습에서, Mamba의 h_t를 구축하는 과정이 바로 prefix burn-in
- CRS 575 step 전체를 parallel scan → 중간 h_1...h_575 추출
- 각 시점의 h_t를 detach하여 Branch 학습에 사용 (Drama 패턴 직접 적용)

#### DFS (Dynamic Frequency-based Sampling)

Drama 고유의 학습 안정화 기법:
- 환경별로 방문 빈도가 불균형할 때, 드물게 방문한 상태를 더 자주 샘플링
- 우리 문제에서는 프로파일별 데이터가 균등하므로 직접적 필요성은 낮지만, 특정 운전 조건(극저출력, 급변 등)의 불균형이 있을 경우 참고 가능

### 2.6 분기/다중경로 평가

#### InferenceParams Snapshot — Branch-in-CRS 해법

**이것이 Drama에서 가장 중요한 기여**. Mamba의 hidden state를 snapshot하고 복원하는 구체적 메커니즘을 검증했다.

**InferenceParams의 구조** (Mamba-2 기준):
```python
class InferenceParams:
    ssm_state: Tensor  # (batch, d_model, d_state)  — SSM의 h_t
    conv_state: Tensor # (batch, d_model, d_conv)   — 1D Conv의 버퍼
```

**Snapshot/Restore 패턴**:
```python
# CRS 자기회귀 도중, 시점 t에서 Branch 평가
params_backup = copy.deepcopy(inference_params)  # snapshot

# Branch 평가 (29개 rod offset)
for offset in rod_offsets:
    branch_input = encode(state_t, rod_map_offset)
    branch_output = mamba_step(branch_input, inference_params)  # h_t가 변경됨!
    # branch_output → decoder → loss

inference_params = params_backup  # 원래 h_t로 복귀
# CRS 계속 진행
```

**왜 deepcopy가 가능한가**:
- InferenceParams는 Python 객체이므로 `copy.deepcopy()` 사용 가능
- ssm_state와 conv_state 모두 일반 텐서이므로 복사에 문제 없음
- Drama 논문에서 이 패턴을 실험적으로 검증 완료

**주의점**:
- deepcopy는 GPU 텐서를 복사하므로 메모리 사용량이 늘어남
- 29개 Branch 평가 동안 29번의 Mamba step이 필요 (추론 시간 증가)
- 학습 시에는 detached h_t를 사용하므로 deepcopy 불필요 (추론 시에만 필요)

**우리 문제 적용 시나리오**:

```
[학습 — Phase 2]
  CRS 전체 시퀀스 → Mamba parallel scan → h_1...h_575 추출
  각 시점 t의 h_t.detach() → Branch 학습의 초기 상태로 주입
  → deepcopy 불필요 (detach된 텐서를 초기 상태로 사용)

[추론 — 575-step 자기회귀]
  시점 t에서 Branch 평가 필요 시:
    params_backup = deepcopy(inference_params)
    29개 Branch 평가
    inference_params = params_backup
    CRS 다음 step 계속
```

---

## 3. 역공학 추론

해당 없음 (완전공개).

---

## 4. 우리 문제 적용성 평가

| 기능 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| Mamba-2 시간 전이 | **O** | 현재 설계의 직접 근거. 7M params 규모 검증됨 | 중 |
| 2-layer Mamba 구성 | **O** | 시작점으로 적합. 필요시 확장 | 하 |
| Parallel scan 학습 | **O** | 575-step CRS의 학습 속도에 필수적 | 중 |
| Action concat 주입 | **O** | p_load concat의 근거 (이미 채택) | 하 |
| Prefix burn-in | **O** | Phase 2 CRS → Branch 전환의 핵심 메커니즘 | 중 |
| InferenceParams snapshot | **O** | Branch-in-CRS h(t) 관리의 검증된 해법 | 중 |
| 디코더에서 d_t 제외 | X | 우리는 직접 지도학습이므로 시간 맥락 유지가 유리 | - |
| DFS 불균형 샘플링 | △ | 특정 운전 조건 불균형 시 참고 가능 | 하 |

---

## 5. 핵심 차용 후보

### 즉시 적용 가능
- **Mamba-2 dynamics backbone**: hidden=512, 2 layers, state_dim=16. 우리 규모(인코더 0.78M + Mamba)와 호환
- **Action concat**: p_load → Mamba concat. 이미 채택
- **Prefix burn-in**: Phase 2에서 CRS parallel scan → h_t 추출 → Branch 초기 상태 주입
- **InferenceParams deepcopy**: 추론 시 Branch-in-CRS h(t) 오염 방지

### 수정 후 적용 가능
- **Parallel scan 구현**: TensorFlow 2에서 Mamba-2 parallel scan을 직접 구현해야 함. PyTorch `mamba_ssm` 라이브러리의 로직을 참고하되, TF2 ops로 변환 필요
- **DFS**: 운전 프로파일별 데이터 불균형 정도를 분석 후, 필요 시 적용

### 부적합
- **디코더 d_t 제외**: 우리 디코더는 시간 맥락(Mamba 출력)을 받아야 정확한 공간 재구성 가능. Drama의 "병렬화를 위한 타협"은 우리에게 불필요
- **Cat(32×32) 확률적 잠재**: DreamerV3에서와 동일한 이유로 불필요
