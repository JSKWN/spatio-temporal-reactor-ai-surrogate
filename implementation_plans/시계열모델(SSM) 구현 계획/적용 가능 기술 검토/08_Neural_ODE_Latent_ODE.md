# Neural ODE / Latent ODE — 연속시간 잠재 동역학

> **출처**: Chen et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS 2018 Best Paper. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366). + Rubanova et al. (2019). *Latent ODEs for Irregularly-Sampled Time Series.* NeurIPS 2019.
> **공개 수준**: 완전공개
> **우리 아키텍처 맥락**: 연속시간 관점에서 Mamba/SSM과의 이론적 연결. 직접 채택보다는 설계 관점 참고

---

## 1. 시스템 개요

### 직관

Neural ODE는 "이산 시간 스텝 대신, **연속 시간 미분 방정식**으로 상태를 진화시키자"는 아이디어다:

```
이산 (ResNet, RNN, Mamba):  h_{t+1} = h_t + f(h_t, t)      ← 고정 시간 간격
연속 (Neural ODE):           dh/dt = f_θ(h(t), t)           ← 임의 시간 간격
                             h(t₁) = h(t₀) + ∫_{t₀}^{t₁} f_θ(h(τ), τ) dτ
```

적분은 ODE solver (Euler, RK4, Dormand-Prince 등)로 수치 수행.

### 우리 문제와의 자연스러운 연결

원자로의 Bateman 방정식은 실제 ODE이다:
```
dN_Xe/dt = γ_Xe·Σ_f·φ + λ_I·N_I - (λ_Xe + σ_a^Xe·φ)·N_Xe
dN_I/dt = γ_I·Σ_f·φ - λ_I·N_I
```

Neural ODE가 이 ODE를 직접 학습할 수 있다는 점은 매력적이나, 실용적 한계가 있다.

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 — Adjoint Method

**장점**: 역전파 시 메모리 O(1) — 전체 궤적을 저장하지 않고, 역방향 ODE를 풀어서 gradient 계산.

**단점 (우리 문제에 치명적)**:
- **Stiff ODE 문제**: Xe-135 반감기 (~9.2h)와 중성자 수명 (~10⁻⁵s)의 시간 스케일이 10⁸ 이상 차이. 이런 stiff system에서 ODE solver가 극히 작은 시간 스텝을 사용해야 하며, 수렴이 느림
- **이산 제어 입력**: 제어봉 위치는 이산 시점에서 변경. 연속 ODE와 이산 제어의 결합이 복잡
- **병렬화 불가**: ODE solver는 본질적으로 순차적

### 2.2 Latent ODE — 잠재 공간에서의 ODE

Latent ODE는 **인코더로 잠재 초기 상태 z₀를 추론**하고, 잠재 공간에서 ODE를 풀어 미래를 예측:

```
관측 시퀀스 → [RNN Encoder] → z₀ (잠재 초기 상태)
z₀ → [Neural ODE: dz/dt = f_θ(z)] → z(t₁), z(t₂), ...
z(tₖ) → [Decoder] → 관측 예측
```

**우리 문제에서의 시사점**: Latent ODE의 구조는 우리 파이프라인과 유사:
- RNN Encoder ↔ 우리 Spatial Encoder
- Neural ODE ↔ 우리 Mamba
- Decoder ↔ 우리 Spatial Decoder

차이: Latent ODE는 연속 시간, 우리는 이산 시간 (Mamba의 이산화).

---

## 3. Mamba와 Neural ODE의 이론적 연결

### SSM은 이산화된 연속 시간 시스템

Mamba의 원형은 연속 시간 SSM이다:
```
연속:  dh/dt = A·h(t) + B·x(t)        ← Neural ODE의 선형 특수 사례
이산:  h_{t+1} = exp(A·Δ)·h_t + ...   ← ZOH 이산화 → Mamba
```

따라서 Mamba는 **Neural ODE의 구조화된(선형) 변형을 이산화한 것**으로 볼 수 있다. Mamba-3의 Trapezoidal 이산화(O(Δ³))는 이 연결을 더 정밀하게 만든다.

### 이 연결의 실용적 의미

- Neural ODE의 **연속 시간 관점**은 우리 Mamba의 Δ 파라미터에 물리적 의미를 부여: Δ가 큰 시점은 "빠르게 변하는 구간", Δ가 작은 시점은 "안정 구간"
- **가변 시간 간격**: 만약 우리 데이터의 시간 간격이 불균일하다면, Δ가 이를 자연스럽게 처리. 현재는 균일 간격(575 step)이므로 즉각적 이점은 제한적

---

## 4. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| Neural ODE 직접 사용 | X | Stiff ODE 문제, 병렬화 불가, 이산 제어 결합 어려움 |
| Latent ODE 구조 | △ | 우리 파이프라인(인코더→시간→디코더)과 구조적 유사. 설계 관점 참고 |
| SSM-ODE 이론적 연결 | O | Mamba의 Δ에 물리적 의미 부여. 이산화 정밀도 선택 근거 |
| Adjoint method | X | 575 step에서 계산 비용 과도. Mamba parallel scan이 더 실용적 |

### 핵심 차용
- **이론적 정당화**: Mamba가 Neural ODE의 이산화된 특수 사례라는 관점은 "SSM이 물리 시스템의 시간 전이를 모델링하기에 적합하다"는 주장의 이론적 기반
- **Δ의 물리적 해석**: 입력 의존적 Δ가 "이 시점에서 시간 동역학의 속도를 조절"하는 역할

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| Neural ODE | Chen et al. NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| Latent ODE | Rubanova et al. NeurIPS 2019. [arXiv:1907.03907](https://arxiv.org/abs/1907.03907) |
| Augmented Neural ODE | Dupont et al. NeurIPS 2019. [arXiv:1904.01681](https://arxiv.org/abs/1904.01681) |
