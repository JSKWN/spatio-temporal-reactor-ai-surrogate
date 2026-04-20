# MNO — Mamba Neural Operator (PDE 대리모델)

> **출처**: Cheng et al. (2025). *Mamba Neural Operator.* Journal of Computational Physics. [arXiv:2410.02113](https://arxiv.org/abs/2410.02113)
> **공개 수준**: 완전공개
> **우리 아키텍처 맥락**: Mamba를 PDE 대리모델에 적용한 유일한 주요 사례. cell-wise Mamba의 PDE 해석에서의 유효성 근거

---

## 1. 시스템 개요

### 직관

FNO(Fourier Neural Operator)와 Transformer 기반 신경연산자는 PDE 해를 학습하는 데 성공했지만, 시간 축이 길어지면 O(T²) 비용이 문제가 된다. MNO는 Mamba의 O(T) 복잡도를 활용하여 **시간 축에서 효율적인 PDE 대리모델**을 구축한다.

**핵심 결과**: 4개 2D PDE 벤치마크에서 Transformer 기반 대비 **~90% 오차 감소**. 특히 자기회귀 롤아웃에서 누적 오차가 현저히 적음.

### 아키텍처

```
u(x, t=0:T_in) → [Spatial Encoder (MLP/Conv)] → latent tokens
    → [Mamba blocks: 격자점별 독립 시간 처리] → evolved latent
    → [Spatial Decoder (MLP)] → u_pred(x, t=T_in+1:T_out)
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링

- 공간 격자점을 **독립적으로** 시간 축에서 Mamba로 처리 (우리의 cell-wise 독립과 동일)
- Mamba의 selective scan이 시간 축에서 "어떤 시점의 정보를 유지하고 버릴지" 결정
- **자기회귀 롤아웃 안정성**: Mamba가 Transformer 대비 롤아웃에서 오차 누적이 현저히 적음을 실증

### 2.2 우리 문제와의 직접 대응

| MNO | 우리 |
|---|---|
| 2D PDE (Navier-Stokes, Darcy 등) | 3D 중성자 확산 + Bateman ODE |
| 격자점별 독립 Mamba | cell-wise 독립 Mamba (720 cells) |
| MLP 인코더/디코더 | FullAttn 인코더/디코더 |
| 물리 제약 없음 (데이터 Loss만) | Physical Loss (L_diff_rel, L_Bateman) |

**핵심 차이**: MNO는 공간 결합을 **인코더/디코더에서도 수행하지 않는다** (MLP만 사용). 그럼에도 격자점별 독립 Mamba가 PDE를 잘 학습했다는 것은, **학습 과정에서 데이터 Loss gradient가 Mamba에 공간 결합 정보를 간접 전달**한다는 것을 시사한다.

우리는 MNO보다 **더 강한 공간 결합 메커니즘**(인코더/디코더 FullAttn + Physical Loss)을 가지므로, cell-wise Mamba가 공간 결합을 학습할 수 있는 조건이 MNO보다 유리하다.

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| **격자점별 독립 Mamba** | O | MNO에서 PDE 대리모델로 검증. 우리 cell-wise 설계의 직접적 근거 |
| **시간축 Mamba** | O | 자기회귀 롤아웃 안정성이 Transformer 대비 우수 |
| **MLP 인코더/디코더** | X | 공간 결합이 필요한 우리 문제에는 FullAttn이 필수 |

### 핵심 차용
- **Cell-wise 독립 Mamba의 PDE 유효성 근거**: MNO가 MLP만으로도 격자점별 독립 처리가 작동함을 증명. 우리는 FullAttn 인코더/디코더가 있으므로 더 유리
- **자기회귀 안정성**: Mamba가 Transformer 대비 긴 롤아웃에서 안정적. 575 step에 유리한 근거

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| MNO | Cheng et al. JCP 2025. [arXiv:2410.02113](https://arxiv.org/abs/2410.02113) |
| FNO | Li et al. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |
