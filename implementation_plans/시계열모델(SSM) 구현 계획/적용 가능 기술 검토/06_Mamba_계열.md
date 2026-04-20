# Mamba 계열 — S4에서 Mamba-3까지의 SSM 진화

> **출처**: Gu & Dao (2023-2026). S4 → S5 → S6(Mamba) → Mamba-2(SSD) → Mamba-3
> **공개 수준**: 완전공개
> **우리 아키텍처 맥락**: A안에서 Mamba가 시간 전이 모듈. cell-wise 독립 처리 (720 cells × 128D). 인코더/디코더 FullAttn이 공간 결합 담당

---

## 1. 시스템 개요

### 직관

SSM(State Space Model) 계열은 "연속 시간 동역학을 이산 시퀀스에 적용하는 모델"이다. 원래 제어 이론의 상태 공간 모델 `h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)`를 신경망으로 학습 가능하게 만든 것이다.

우리 문제에서 이것이 자연스러운 이유: 원자로의 Xe-135 축적, 온도 변화 등은 실제로 **미분 방정식(ODE)**으로 기술된다. SSM의 상태 전이 `h(t+1) = Āh(t) + B̄x(t)`는 이 ODE를 이산화한 것과 수학적으로 동일한 형태다.

### 진화 계보

```
S4 (2021) → S5 (2022) → S6/Mamba (2023) → Mamba-2/SSD (2024) → Mamba-3 (2026)
  고정 A       고정+MIMO    입력 의존       입력 의존+병렬      입력 의존+복소+MIMO
```

---

## 2. 변종별 상세 비교

### 2.1 S4 (Structured State Spaces for Sequence Modeling, 2021)

> Gu et al. ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

**핵심**: A 행렬을 **HiPPO 초기화**로 고정. 장기 의존성 포착에 탁월.

```
h(t+1) = Ā·h(t) + B̄·x(t)     Ā, B̄는 고정 (입력에 무관)
y(t) = C·h(t)                  C도 고정
```

- A는 HiPPO 행렬 (과거 입력의 다항식 근사를 최적으로 유지)
- **입력에 무관한 고정 파라미터** → 모든 시점에서 동일한 전이 규칙
- 병렬화: 컨볼루션으로 변환 가능 → O(T log T)

**우리 문제와의 관련**: S4의 고정 A는 "시스템의 동역학이 입력과 무관하게 일정하다"고 가정한다. 원자로에서는 Xe 농도, 출력 수준에 따라 동역학이 변하므로 이 가정이 부적합. → S6/Mamba의 입력 의존 파라미터가 필수.

### 2.2 S5 (Simplified State Space, 2022)

> Smith et al. ICLR 2023. [arXiv:2208.04933](https://arxiv.org/abs/2208.04933)

**S4 대비 변경**: A를 대각화, **MIMO (Multiple Input Multiple Output)** 지원.

```
h(t+1) = diag(ā)·h(t) + B̄·x(t)     B̄ ∈ R^(N×D), MIMO
y(t) = C·h(t)                         C ∈ R^(D×N)
```

- 대각 A: 계산 단순화, 병렬 prefix scan으로 O(T) 학습
- MIMO: 상태 벡터의 각 차원이 여러 입력 차원에 연결

### 2.3 S6 / Mamba (Selective State Spaces, 2023)

> Gu & Dao. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

**핵심 혁신: 입력 의존(data-dependent) 파라미터**

```
B_t = Linear_B(x_t)     ← 매 시점 입력에서 생성
C_t = Linear_C(x_t)
Δ_t = softplus(Linear_Δ(x_t))

Ā_t = exp(A · Δ_t)      ← A는 학습 파라미터, Δ_t가 이산화 제어
B̄_t = (Ā_t - I) · A^(-1) · B_t · Δ_t     (ZOH 이산화)

h(t+1) = Ā_t · h(t) + B̄_t · x_t
y(t) = C_t · h(t)
```

**직관적으로**: 매 시점마다 "이번에는 과거를 얼마나 유지하고(Ā_t), 현재 입력을 얼마나 반영할지(B̄_t), 상태에서 무엇을 읽어낼지(C_t)"를 **입력 자체가 결정**한다.

**물리적 비유**: 제어봉이 삽입되면 Δ가 커져서 "이전 상태를 빨리 잊고 새 상태로 전환"하고, 안정 운전 중에는 Δ가 작아서 "이전 상태를 천천히 갱신"하는 것과 유사하다.

**Mamba 블록 구조**:
```
x → [Linear ×2 (확장)] → [Conv1D (d_conv=4, causal)] → [SSM (S6)] → [×gate] → [Linear (축소)] → y
         ↓                                                    ↑
         └─────────── SiLU gate ──────────────────────────────┘
```

**우리 문제에서의 적합성**:
- 입력 의존 파라미터가 p_load/운전 조건에 따른 동역학 변화를 자연스럽게 포착
- Conv1D (d_conv=4): 인접 시점의 지역 패턴 포착. 우리 575 step 시퀀스에서 단기 추세 감지
- 학습: selective scan (병렬 불가 → 순차적). **Mamba-2에서 해결됨**

### 2.4 Mamba-2 / SSD (Structured State Space Duality, 2024)

> Dao & Gu. ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

**핵심 혁신**: SSM과 Attention의 **수학적 이중성(duality)** 발견. 병렬 학습 가능.

```
SSM의 recurrence를 행렬로 풀어쓰면:
y = (L ⊙ (C·B^T)) · x     ← L은 하삼각 마스크, C·B^T는 "attention score"에 해당

이것은 인과적(causal) 선형 attention과 구조적으로 동일!
```

**S6 대비 변경**:
- A를 **스칼라** (a_t ∈ R)로 단순화 → 행렬 연산으로 변환 가능
- **Chunked matmul**: 시퀀스를 청크로 나누어 청크 내는 matmul, 청크 간은 recurrence
- **중간 상태 접근**: 청크 경계에서 h(t) 저장 → Drama의 InferenceParams snapshot 가능

**속도**: S6 대비 2-8배 빠른 학습

**우리 문제에서의 핵심 이점**:
1. **병렬 학습**: 575 step CRS 시퀀스를 chunked matmul로 한번에 처리
2. **중간 상태 접근**: 학습 시 h_1...h_575 추출 가능 → Phase 2 Branch 학습에 detached h_t 제공
3. **InferenceParams**: ssm_state + conv_state를 Python 객체로 관리 → deepcopy로 Branch-in-CRS snapshot 가능
4. **Drama에서 MBRL 검증 완료**: ~7M params 규모에서 DreamerV3 대등/상회

### 2.5 Mamba-3 (2026)

> Lahoti et al. ICLR 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)

**3가지 핵심 개선**:

#### (a) Exponential-Trapezoidal 이산화

```
Mamba-2: h(t) = a_t · h(t-1) + γ_t · B_t · x_t          (현재 입력만)
Mamba-3: h(t) = a_t · h(t-1) + β_t · B_{t-1} · x_{t-1} + γ_t · B_t · x_t   (이전+현재 입력)
```

- Mamba-2의 Euler 이산화(O(Δ²) 오차)를 Trapezoidal(O(Δ³) 오차)로 개선
- **현재와 이전 입력을 모두 사용** → 부드러운 동역학에 더 정확
- 물리적 의미: Xe-135 축적처럼 천천히 변하는 물리량의 시간 전이가 더 정확해짐
- 부수 효과: 이전 입력을 자동으로 참조하므로, Mamba-1/2의 Conv1D를 **대체 가능** → Conv1D 제거로 구조 단순화

#### (b) 복소수 상태 + Data-dependent RoPE

```
상태를 복소수로 확장: h(t) ∈ C^N (실수부 + 허수부)
실수 구현: 2×2 블록 대각 회전 행렬로 표현
         ┌ cos θ_t  -sin θ_t ┐
    R_t = │                   │    θ_t는 data-dependent
         └ sin θ_t   cos θ_t ┘
```

- Mamba-2까지는 상태가 "줄이거나 유지"만 가능 (실수 스칼라 곱)
- Mamba-3는 상태를 **회전** 가능 → 위상 정보를 내장
- 물리적 의미: Xe-135 진동의 **위상 추적**에 잠재적 이점. Xe 축적이 "올라가는 중"인지 "내려가는 중"인지를 상태 자체에 인코딩 가능

#### (c) MIMO (Multi-Input Multi-Output)

- rank R=2~4로 여러 입력 채널이 하나의 상태를 공유
- 상태 크기를 절반으로 줄여도 동일 성능 (효율성 향상)

**우리 문제에서의 평가**:

| 특성 | Mamba-2 | Mamba-3 | 우리 문제 관련성 |
|------|---------|---------|---------------|
| 이산화 정밀도 | O(Δ²) | **O(Δ³)** | Xe/온도의 부드러운 변화에 유리 |
| 위상 추적 | 불가 | **복소 상태** | Xe 진동 위상 인코딩 가능 |
| MIMO | SISO | **rank R** | 상태 효율성 향상 |
| MBRL 검증 | Drama에서 완료 | **미검증** | 위험 요인 |
| TF2 구현 | 직접 구현 가능 | 복소수 연산 추가 부담 | 구현 난이도 증가 |

---

## 3. 우리 아키텍처에서의 Mamba 역할 상세 분석

### 3.1 A안에서의 위치

```
Encoder(FullAttn ×3)         Mamba(cell-wise)              Decoder(FullAttn + AdaLN)
    ↓                            ↓                              ↓
공간 결합 + 표현 학습      시간 전이 (cell별 독립)         공간 재결합 + 물리 정합성
z(t) ∈ (720, 128)          d(t) ∈ (720, 128)              pred(t+1) ∈ (720, 10)
                                ↑
                          p_load concat 주입
```

### 3.2 Cell-wise 독립 처리의 물리적 정당화

Mamba가 720 cell을 독립 처리하는 것이 물리적으로 타당한 이유:

1. **Xe-135 Bateman 방정식은 지역적**: 각 cell의 Xe 생성/소멸은 해당 cell의 flux, yield, 붕괴상수에만 의존. 이웃 cell의 Xe가 직접 확산하지 않음
2. **온도 변화도 지역적**: 각 cell의 연료/냉각재 온도는 해당 cell의 출력에 주로 의존
3. **공간 결합의 원천은 중성자 확산**: 이것은 "flux 분포"를 결정하는 고유값 문제이며, Xe/온도 변화가 flux를 변경 → flux가 다른 cell의 출력을 변경하는 **간접 경로**

따라서: cell-wise Mamba가 "각 cell의 지역 동역학(Xe 축적, 온도 변화)"을 처리하고, 디코더 FullAttn이 "cell 간 결합(flux 재분배)"을 처리하는 역할 분담은 **물리적으로 자연스러운 분리**이다.

### 3.3 p_load concat 주입의 효과

```python
mamba_input = concat(z_t, p_load_t)  # (720, 128+1) → Linear → (720, 128)
d_t = Mamba(mamba_input)
```

p_load는 전역 스칼라이므로 모든 720 cell에 동일한 값이 주입된다. 이것이 "간접적 전역 동기화" 역할을 한다: 모든 cell이 동일한 p_load를 받으므로, p_load 변화에 대한 응답이 일관성을 유지한다.

---

## 4. 우리 문제 적용성 평가

| 변종 | 적용 가능 | 근거 | 우선순위 |
|------|----------|------|---------|
| **S4 (고정 A)** | X | 입력 무관 동역학. 운전 조건 변화 반영 불가 | - |
| **S5 (고정+MIMO)** | X | S4와 동일한 한계 | - |
| **S6/Mamba (입력 의존)** | △ | 입력 의존 선택성 확보. 단, 병렬 학습 불가 | 낮음 |
| **Mamba-2 (SSD)** | **O** | 병렬 학습 + 중간 상태 접근 + Drama MBRL 검증. **1순위 채택** | **높음** |
| **Mamba-3** | 보류 | Xe 위상 추적(복소 상태), 정밀 이산화에 이론적 이점. MBRL 미검증 + TF2 구현 부담 | 향후 |
| **Gated DeltaNet** | 보류 | OLMo에서 Mamba-2 상회. LLM 전용 검증. TF2 미확인 | 향후 |

---

## 5. 핵심 차용 후보

### 즉시 적용 (1순위)
- **Mamba-2 (SSD)**: hidden=512, 2 layers, state_dim=16, 4 heads. Drama 검증 규모
- **Chunked matmul 병렬 학습**: 575 step CRS를 청크로 나누어 학습 속도 확보
- **InferenceParams**: ssm_state + conv_state 관리. Branch-in-CRS deepcopy 지원

### A안 보강 시 적용
- **잠재 예측 loss (CDP)**: Mamba 출력에서 다음 시점 인코더 출력을 직접 예측. 디코더 경유 gradient의 보조
- **p_load concat**: 이미 채택. 전역 동기화 효과

### 향후 전환 검토
- **Mamba-3**: Xe 진동의 위상 추적에 이론적 이점. Mamba-2 성능 확인 후 전환 시험
- **Gated DeltaNet**: Mamba-2 대체 후보. PDE 도메인 검증 후 검토
- **Conv1D 제거**: Mamba-3 채택 시 trapezoidal 이산화가 Conv1D를 대체. 구조 단순화

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| S4 | Gu et al. ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) |
| S5 | Smith et al. ICLR 2023. [arXiv:2208.04933](https://arxiv.org/abs/2208.04933) |
| Mamba (S6) | Gu & Dao. 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Mamba-2 (SSD) | Dao & Gu. ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| Mamba-3 | Lahoti et al. ICLR 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) |
| Gated DeltaNet | ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| Drama | Wang et al. ICLR 2025. [arXiv:2410.08893](https://arxiv.org/abs/2410.08893) |
| MNO | Cheng et al. JCP 2025. [arXiv:2410.02113](https://arxiv.org/abs/2410.02113) |
