# FNO — Fourier Neural Operator

> **출처**: Li et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
> **공개 수준**: 완전공개
> **우리 아키텍처 맥락**: 스펙트럼 공간에서의 전역 혼합. 인코더/디코더의 FullAttention 대안으로 검토. 해상도 불변 학습

---

## 1. 시스템 개요

### 직관

FNO는 "PDE의 해 연산자를 주파수 공간에서 학습"한다. 핵심 아이디어: 공간 격자의 **FFT → 주파수 필터링(학습) → IFFT**로 전역 공간 혼합을 O(N log N)에 달성한다. Attention의 O(N²)보다 효율적이며, 격자 해상도에 무관하게 학습 가능 (해상도 불변).

### Fourier Layer 구조

```
입력 v(x) ∈ R^(N×D)
    ↓
[FFT] → V(k) ∈ C^(K×D)    (주파수 공간, K ≤ N/2+1 모드)
    ↓
[R(k) · V(k)]               (학습된 주파수 필터 R(k) ∈ C^(K×D×D))
    ↓
[IFFT] → 전역 혼합된 feature
    ↓
[+ W·v(x)]                  (지역 선형 변환, skip connection)
    ↓
[σ (활성화)]                 → 출력 u(x)
```

- **R(k)**: 주파수별 학습 가능한 필터. 저주파 모드만 유지 (K < N)하여 효율화
- **W·v(x)**: 지역(per-point) 선형 변환. 전역 혼합과 지역 처리를 결합

---

## 2. 우리 문제와의 관련성

### 2.1 FullAttention vs Fourier 혼합

| | FullAttention | FNO Fourier Layer |
|---|---|---|
| 복잡도 | O(N²) | O(N log N) |
| 공간 혼합 | 학습된 attention weight | 학습된 주파수 필터 |
| 해상도 불변 | X (위치 인코딩에 의존) | **O** (주파수 공간에서 학습) |
| N=720일 때 비용 | 720² = 518,400 | 720·log(720) ≈ 6,900 |

**비용 비교**: FNO가 ~75배 효율적. **그러나** 720은 충분히 작아서 FullAttention이 실용적 범위 내. FourCastNet이 100만+ 격자점에서 AFNO를 사용하는 이유는 비용 때문이며, 720 격자점에서는 이 이점이 미미.

### 2.2 해상도 불변 — 잠재적 이점

FNO는 저해상도에서 학습하고 고해상도에서 추론할 수 있다 (주파수 필터가 해상도에 무관). 우리 문제에서:
- 현재: Quarter 5×5 (19 fuel cells) → Halo 6×6 (36 positions/layer)
- 향후: Full-core 9×9 (57 fuel cells) 또는 서브노드 18×18 확장 가능성
- FNO라면 5×5에서 학습 → 9×9에서 추론 가능 (이론적으로)

**그러나**: 우리 문제에서 해상도 변경은 연료 위치 변경을 의미하므로, 단순한 해상도 외삽이 물리적으로 유의미한지 의문. Attention + LAPE가 위치별 학습을 하므로 해상도 불변의 이점이 제한적.

### 2.3 FNO의 시간 처리

FNO 원래 논문: 시간 축도 입력에 포함하거나, 자기회귀로 처리:
```
방법 1: u(x, t=0:T_in) → FNO → u(x, t=T_in+1:T_out)   (시간 일괄)
방법 2: u(x, t) → FNO → u(x, t+Δt)                     (자기회귀)
```

우리 파이프라인은 방법 2에 해당 (Mamba가 시간 자기회귀).

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| Fourier 공간 전역 혼합 | △ | 720 격자에서 FullAttention 대비 이점 미미. 대규모 격자 시 재검토 |
| 해상도 불변 학습 | △ | 격자 변경 시 유용. 현재 고정 격자(5×5 quarter)에서 이점 없음 |
| 주파수 필터링 + 지역 변환 결합 | △ | FNO의 R(k)·V(k) + W·v(x) 패턴이 우리 인코더에 참고 가능 |
| 자기회귀 시간 처리 | O (이미 채택) | Mamba 자기회귀와 동일 개념 |

### 핵심 차용
- **이론적 참고**: FNO의 "주파수 공간에서 PDE 연산자 근사" 관점은 우리 인코더의 FullAttention이 사실상 **학습된 공간 연산자**라는 해석을 뒷받침

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| FNO | Li et al. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |
| AFNO | Guibas et al. NeurIPS 2022. [arXiv:2111.13587](https://arxiv.org/abs/2111.13587) |
