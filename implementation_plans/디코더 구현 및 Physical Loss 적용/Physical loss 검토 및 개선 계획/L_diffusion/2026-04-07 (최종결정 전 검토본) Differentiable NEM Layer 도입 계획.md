# Differentiable NEM Layer 도입 계획

> **작성일**: 2026-04-07
> **목적**: L_diffusion의 본질적 floor(NEM 4차 곡률 정보 부재, ~2-3%)를 우회하기 위한 NEM correction 학습 방안 설계
> **상태**: **계획 단계 — 선결 조건 검증 (작업 B) 미실행**. 본 문서는 설계 옵션 풀(pool)이며, 최종 노선은 B 분석 결과 후 §10에서 확정
> **관련 문서**:
> - `2026-04-07 (도입 고려 필요) L_diffusion Cross-Attention 이론 해석.md` §5.5, §9
> - `2026-04-01 JNET0 활용 방안 계획.md` 작업 B (D̂ 역산)
> - `2026-03-30 Physical Loss 통합 레퍼런스.md` §3.2 (CMFD/NEM 식)

---

## 0. 한 줄 요약

> NEM 알고리즘을 직접 재현하지 않고, **NEM correction 항 $\hat{D}_{\text{assy}}$ 를 모델 파라미터로 학습**시켜 L_diffusion 잔차를 NEM 식 형태로 부과한다.
>
> ⚠️ **단위 불일치 주의**: MASTER JNET0는 노드(어셈블리 2×2 분할) 단위 D̂를 함의하지만, 우리 모델은 어셈블리 단위. 따라서 **D̂ 직접 GT supervision은 부적합**.
>
> 메인 노선은 **A' (D̂_assy를 trainable variable로 두고 self-consistency로만 학습)**, 조건부로 **A'' (노드 D̂를 어셈블리 평균으로 매핑해 GT supervision)** 병기.

---

## 1. NEM = Nodal Expansion Method (확인)

방법론 §2.1.4 (p.20) 직접 인용:

> "In the NNEM, both the **CMFD method** and the **two-node NEM** are used for the solution of the multi-group diffusion equation."
> "The CMFD problem incorporates the **global coupling** of the nodes while the two-node problems incorporates **local higher order coupling**."

**우리 NEM = 핵공학 표준 Nodal Expansion Method** (Finnemann–Smith–Lawrence 계열). MASTER 채택 알고리즘은 **NNEM (Nonlinear Nodal Expansion Method)** = CMFD + 2-node NEM 반복.

### 1.1 핵심 식 (방법론 Eq. 2.1-37, 38)

**CMFD net current** (1차 근사):

$$
J^{\text{CMFD}}_{u} = -\tilde{D}_u\cdot\frac{\bar\varphi_2 - \bar\varphi_1}{h_u}, \quad \tilde{D}_u = \frac{2D_1 D_2}{D_1 + D_2}
$$

**NEM net current** (2-node NEM 보정):

$$
J^{\text{NEM}}_{u} = -\tilde{D}_u\cdot\frac{\bar\varphi_2 - \bar\varphi_1}{h_u} - \hat{D}_u\cdot(\bar\varphi_2 + \bar\varphi_1)
$$

→ **$\hat{D}_u$ = NEM correction**. 어셈블리(또는 노드) 내부 4차 곡률의 효과를 면 current에 반영하는 핵심.

---

## 2. 데이터에 D̂ 정보가 있다 — 그러나 단위 불일치 ★

### 2.1 JNET0 = MASTER NEM net current

`MAS_NXS`의 JNET0 필드는 MASTER NEM의 진짜 net current. 즉:

$$
J^{\text{NEM}}_u = \text{JNET0}_u \times 2 \quad\text{(방법론 단위 환산)}
$$

이미 [`2026-04-01 JNET0 활용 방안 계획.md`](2026-04-01 JNET0 활용 방안 계획.md) 작업 B에 노드 단위 D̂ 역산 식이 명시되어 있다:

$$
\hat{D}^{\text{node}}_u = \frac{2\,\text{JNET0}_u + \tilde{D}_u\,(\bar\varphi_2 - \bar\varphi_1)/h_u}{-(\bar\varphi_2 + \bar\varphi_1)}
$$

### 2.2 ★ 단위 불일치 — 직접 supervision 부적합

| | MASTER NEM (JNET0) | 우리 모델 (CMFD) |
|---|---|---|
| 격자 단위 | **노드** (어셈블리를 2×2로 분할) | **어셈블리** (5×5 quarter) |
| Flux | 노드별 체적평균 | 어셈블리별 체적평균 |
| Net current | 노드 간 면 | 어셈블리 간 면 |
| D̂ 정의 | **노드 간 면별** | (없음, 도입 대상) |
| 면 수 | 노드별 6면, 총 면 수 ~4배 | 어셈블리별 6면 |

→ **노드 D̂ ≠ 어셈블리 D̂**. JNET0의 노드 D̂를 우리 어셈블리 D̂에 1:1 supervision으로 사용하는 것은 부적합.

가능한 길은 **두 가지로 좁혀짐**:

| 옵션 | 설명 | 가능성 |
|---|---|:---:|
| (1) D̂ 직접 예측 (노드 단위 supervision) | 부적합 (단위 불일치) | ❌ |
| **(2) D̂_assy를 trainable variable / loss 계수로 활용** | self-consistency로 학습 | ✅ **메인** |
| **(3) 노드 D̂의 어셈블리 평균을 GT로** | 평균화가 NEM 의미를 보존하면 가능 | △ **조건부** |

→ 본 문서는 (2)를 메인 노선 A', (3)을 조건부 노선 A''로 병기.

### 2.3 정보 floor 재평가

§5.5에서 본 "정보 이론적 floor ~2-3%"는 **implicit 학습 가정** 하의 floor였다. NEM 식 형태의 잔차를 부과하면 floor를 어느 정도 낮출 수 있으나:

- **노선 A' (self-consistency)**: floor가 약간 낮아짐 (NEM 식 형태가 cross-attention 추가 표현력 제공)
- **노선 A'' (GT 평균 매핑)**: 노드↔어셈블리 평균화의 정확도에 따라 floor 추가 감소 가능. 단, 평균화가 NEM 의미를 부분적으로만 보존하므로 0%는 불가능

---

## 3. D̂의 데이터 의존성 — 동적 변수 ★

> **사용자 지적 (2026-04-07)**: "D̂도 데이터 driven 아닌가? 계산한 계수들이 일반적인 상수가 아니라 계산에 따라 달라지는 것 아닌가."

**정확하다.** D̂는 운전 조건/시점/노드/면마다 다르게 결정되는 **동적 양**.

### 3.1 정적 vs 동적 변수 분류

| 항목 | 상수성 | 의존 | 비고 |
|---|:---:|---|---|
| $D_g$ (확산계수) | 노드별 고정 | xs_fuel (BOC fixed) | LP당 1회 |
| $\tilde{D}_u$ (조화평균) | 면별 고정 | 양쪽 노드 $D$ | LP당 1회 |
| $\Sigma_a, \nu\Sigma_f, \Sigma_s$ | 노드별 고정 | xs_fuel (BOC fixed) | LP당 1회 |
| **$\hat{D}_u$ (NEM correction)** | **❌ 동적** | $\bar\varphi$, transverse leakage, 인접 노드 분포, $k_{\text{eff}}$ | **시점·노드·면별 변동** |

### 3.2 D̂가 동적인 물리적 이유

NEM correction은 어셈블리 내부 4차 flux 분포가 **현재 운전 상태에 따라 모양이 바뀌기** 때문에 발생:

| 운전 변화 | 4차 곡률 변화 | D̂ 변화 |
|---|---|---|
| 제어봉 삽입 → 흡수 증가 → 국소 flux 함몰 | 어셈블리 내부 곡률 ↑ | D̂ 크기 변화 |
| 출력 변화 → T_f, ρ_m 변화 → XS 변화 | flux 분포 재조정 | D̂ 재계산 |
| Xe 동역학 비평형 → 흡수 분포 변화 | 국소 곡률 시간 변동 | D̂(t) 시간 의존 |
| 인접 어셈블리 power 차이 → 누설 비대칭 | 면별 곡률 비대칭 | 면별 D̂ 비대칭 |

### 3.3 동적이라는 사실의 의미

이 사실은 **나쁜 소식이 아니라 좋은 소식**:

- ❌ "D̂ 테이블을 한 번 만들어서 곱하면 끝" — 불가능
- ✅ **모델이 학습할 가치가 있는 패턴이 존재**한다는 의미
- ✅ 만약 D̂가 단순 상수면 hardcode로 끝났을 것
- ✅ D̂의 동적성이 곧 **물리(NEM correction)의 본질** — 모델이 이를 학습하면 진짜 NEM 효과를 일반화 가능

→ D̂_assy를 단순 trainable scalar로 두는 것 (옵션 4.1.a)은 표현력 부족. 입력에 의존하는 함수형 (옵션 4.1.d)이 본질적.

학습 가능한 패턴인지는 §5의 작업 B에서 분포 분석으로 확인.

---

## 4. 설계 노선

### 4.0 노선 비교

| | A' (Trainable + SC) | A'' (어셈블리 평균 GT) | B (Hardcoded NEM) | C (Hybrid) |
|---|:---:|:---:|:---:|:---:|
| 단위 호환성 | ✅ 어셈블리 | △ 평균화 검증 필요 | ❌ 노드 가정 | ✅ 어셈블리 |
| GT 사용 | self-consistency만 | 평균화된 GT | 미사용 | 부분 |
| 구현 복잡도 | 낮음 | 중간 (검증 필요) | 매우 높음 | 중간 |
| 정확도 (이론) | 높음 | 매우 높음 (조건부) | 가장 높음 | 높음 |
| 학습 안정성 | 높음 | 중간 (GT 노이즈) | 낮음 | 중간 |
| 학술 가치 | 중간 | 중간 | 높음 | 중간 |
| **종합** | **★★★ 메인** | **★★ 조건부 병기** | ★ (보류) | ★★ 후순위 |

### 4.1 노선 A' — Trainable D̂ + Self-Consistency (메인)

**아이디어**: 어셈블리 면별 $\hat{D}_{\text{assy}}$ 를 학습 가능한 양으로 도입하고, **L_diffusion 잔차 자체를 줄이는 self-consistency로만 학습**. JNET0 GT는 사용하지 않음.

#### NEM 잔차 (학습된 $\hat{D}^{\text{pred}}_{\text{assy}}$ 사용)

$$
J^{\text{NEM,pred}}_u = -\tilde{D}_u\cdot\frac{\bar\varphi_2-\bar\varphi_1}{h_u} - \hat{D}^{\text{pred}}_{\text{assy},u}\cdot(\bar\varphi_2+\bar\varphi_1)
$$

$$
R_g^{\text{NEM}} = \sum_{\text{faces}} J^{\text{NEM,pred}}_u\cdot A + \Sigma_{r,g}\,\bar\varphi_g\,V - S_g\,V
$$

$$
\mathcal{L}_{\text{diffusion}}^{\text{NEM}} = \frac{1}{B\cdot N_{\text{fuel}}}\sum_{b,v}(R_{g1}^2 + R_{g2}^2)
$$

#### D̂_assy의 자유도 옵션

| 옵션 | 구조 | 파라미터 수 | 표현력 | 비고 |
|---|---|---|---|---|
| (a) 공간 균일 | trainable scalar 1개 (면 방향당, 군당) → 12개 | ~12 | 매우 낮음 | 시작점, 디버깅 |
| (b) 면 방향별 | 6 × 2군 = 12 trainable | ~12 | 낮음 | (a)와 유사 |
| (c) 어셈블리별 | $20\times5\times5\times6\times2$ trainable | ~6,000 | 중간 | 위치 패턴 학습 |
| **(d) 작은 NN으로 함수형** | $f_\theta(\bar\varphi, \text{xs\_fuel}, k_{\text{eff}}) \to \hat{D}_{\text{assy}}$ | ~수만 | **높음** | **권장 — D̂의 동적성 반영** |

→ §3에서 본 D̂의 동적성을 고려하면 **(d)가 본질적으로 필요**. (a)~(c)는 baseline 비교용.

#### 장점
- **단위 호환**: 어셈블리 단위로 정의되므로 우리 모델과 자연 결합
- **JNET0 GT 의존성 제거**: 노드/어셈블리 단위 불일치 문제 회피
- 모든 연산 미분 가능
- 기존 L_diffusion 구조 유지, 잔차 식만 NEM 형태로 교체
- 입력 변경 없음

#### 단점
- self-consistency만으로는 학습 신호가 약함 (잔차를 줄이는 방향이 여러 개 존재 — under-determined)
- 가중치 튜닝 민감
- A''에 비해 잔차 감소 폭 작을 수 있음
- 학습된 D̂_assy가 진짜 NEM correction과 무관한 양으로 수렴할 위험 (regularization 필요)

#### Regularization 고려사항
- D̂_assy의 부호/크기 제약: 물리적으로 의미 있는 범위 (~ -D/h to D/h)
- 인접 노드 D̂ smoothness penalty
- 학습 초기 D̂_assy = 0으로 zero-init → CMFD부터 시작

### 4.2 노선 A'' — 어셈블리 평균 D̂ GT (조건부 병기)

**아이디어**: 노드 D̂를 어셈블리 면 D̂로 적절히 평균화한 $\hat{D}^{\text{GT}}_{\text{assy}}$ 를 정의 가능하면 supervision 추가.

#### 노드 → 어셈블리 평균화 후보

| 방법 | 식 | 가정 |
|---|---|---|
| (i) 단순 산술 평균 | $\hat{D}^{\text{GT}}_{\text{assy}} = \frac{1}{N_{\text{node}}}\sum_n \hat{D}^{\text{node}}_n$ | 노드 D̂가 등가중치 |
| (ii) 면적 가중 평균 | $\hat{D}^{\text{GT}}_{\text{assy}} = \frac{\sum_n A_n \hat{D}^{\text{node}}_n}{\sum_n A_n}$ | 면적 비례 기여 |
| (iii) Flux 가중 평균 | $\hat{D}^{\text{GT}}_{\text{assy}} = \frac{\sum_n (\bar\varphi_n) \hat{D}^{\text{node}}_n}{\sum_n \bar\varphi_n}$ | 반응률 비례 기여 |
| (iv) NEM 식 어셈블리 재유도 | 2-node NEM을 어셈블리 단위로 다시 풀어 등가 D̂ 정의 | 가장 엄밀, 수학적 검증 부담 |

#### 수학적 검증 필요

→ 단순 평균이 NEM 보정의 의미를 보존하는지는 **수학적으로 자명하지 않음**. 핵심 질문:

> "노드 단위 NEM이 만든 net current 4개를 어셈블리 면 net current 1개로 통합할 때, 어떤 평균화가 등가 NEM correction을 정의하는가?"

이 검증은 작업 B의 B3 단계에서 수행 (§5).

#### 장점 (검증 통과 시)

- D̂_assy_GT가 supervision 신호로 작동 → A' 대비 학습 안정성 ↑
- NEM 잔차 자체도 더 빠르게 수렴
- 학습된 D̂_assy가 진짜 NEM correction과 일관

#### 단점

- 평균화 방법의 정당성 검증 필요 (B3, 1-2일 작업)
- 평균화 자체가 노드 단위 정보 손실 발생 → A''도 노드 단위 NEM의 정확도를 완전히 보존하지 못함
- 평균화가 의미를 잃는 영역 (예: 어셈블리 내부 강한 비대칭)에서는 GT 노이즈

#### A'와의 결합

A''가 검증을 통과하면 **A' + A'' 동시 사용**:

$$
\mathcal{L}_{\text{NEM total}} = \lambda_1 \cdot \mathcal{L}_{\text{diffusion}}^{\text{NEM}}\bigl(\hat{D}^{\text{pred}}_{\text{assy}}\bigr) + \lambda_2 \cdot \text{MSE}\bigl(\hat{D}^{\text{pred}}_{\text{assy}}, \hat{D}^{\text{GT}}_{\text{assy}}\bigr)
$$

self-consistency (A') + GT supervision (A'') 둘 다 동시 부과.

### 4.3 노선 B — Hardcoded NEM Two-Node Solver (보류)

NEM Two-Node 풀이 자체를 미분 가능 layer로 구현. 단위 문제 + 구현 복잡도 + 횡방향 누설 정보 부재로 **권장 안 함** (학술 가치만).

### 4.4 노선 C — Hybrid (후순위)

D̂의 함수형을 NEM 이론에서 유도된 form으로 제한하고 자유 파라미터만 학습. 노선 A'의 옵션 (d)와 유사하나 함수형이 더 강하게 제약. 노선 A' 진행 후 추가 검토.

---

## 5. 선결 조건 — 작업 B 재정의 ★

### 5.1 작업 B 실행 여부

[`2026-04-01 JNET0 활용 방안 계획.md`](2026-04-01 JNET0 활용 방안 계획.md) 작업 B (D̂ 역산 + 분포 분석)는 piecewise-test 폴더에 해당 스크립트/결과 파일이 없으므로 **미실행 추정**.

### 5.2 재정의된 작업 B 단계

원래 작업 B는 노드 단위 D̂ 역산만 포함했으나, 본 문서의 노선 A'/A''를 결정하려면 **어셈블리 평균화 검증 추가 필요**.

| 단계 | 작업 | 산출물 | 시간 추정 |
|---|---|---|---|
| **B1** | 노드 단위 D̂ 역산 (JNET0 + 노드 flux + D̃) | `dhat_node_inverse.py/.txt` | 0.5-1일 |
| **B2** | D̂_node 분포 분석 (통계, 위치/면/조건 의존) | `dhat_node_distribution.md` | 0.5일 |
| **B3** | 어셈블리 평균화 검증 (단순 평균 vs 면적 가중 vs 등가식 유도) | `dhat_assy_averaging.md` | 1-2일 |
| **B4** | 어셈블리 단위 D̂_assy 사용 시 잔차 영향 측정 | `dhat_assy_residual_test.md` | 0.5-1일 |
| **B5** | 결과 정리, 노선 A' vs A'' vs 보류 결정 | `dhat_analysis_summary.md` | 0.5일 |
| **전체** | | | **3-5일** |

### 5.3 B1, B2 분석 항목

| 항목 | 측정 | 해석 |
|---|---|---|
| 전체 분포 | 평균, 분산, median, p95 | D̂_node 크기의 일반적 범위 |
| 부호 분포 | 양/음 비율 | 부호 일관성 (방향성) |
| 위치 의존성 | 노드별 D̂_node 평균 (Z, qy, qx) | 공간 패턴 (경계 vs 내부) |
| 면 방향 의존성 | 6면(±Z, ±Y, ±X)별 통계 | 방향성 비대칭 |
| 운전 조건 의존성 | rod_map, p_load, T_f별 D̂ | **동적 패턴 (§3 가설 검증)** |
| 노드 조성 의존성 | xs_fuel(D, Σ_a)와 D̂ 상관 | 물리적 설명력 |
| 시점 의존성 | t에 따른 D̂(t) 자기상관 | Mamba가 학습 가능한지 |
| 2군 결합 | g1 vs g2 D̂ 상관 | 군별 분리 학습 필요성 |

### 5.4 B3 어셈블리 평균화 검증 항목

| 평균화 방법 | 검증 항목 |
|---|---|
| (i) 단순 산술 평균 | 평균 D̂_assy의 부호 일관성, 노드 4개 D̂의 분산 |
| (ii) 면적 가중 평균 | (i)와 차이가 있는가 |
| (iii) Flux 가중 평균 | 비대칭 어셈블리에서 (i)/(ii)와 차이 |
| (iv) NEM 식 재유도 | 어셈블리 단위 NEM 식이 닫힌 형태로 유도되는가 (방법론 §2.1.4 참조) |

**검증 기준**: 평균화된 $\hat{D}^{\text{GT}}_{\text{assy}}$ 를 우리 어셈블리 CMFD 잔차에 부과했을 때 잔차가 의미 있게 감소하는가 (B4에서 측정).

### 5.5 단계별 의사결정 기준

| 단계 | 결과 | 다음 단계 |
|---|---|---|
| B1 후 | D̂_node가 0에 가깝거나 무작위 | NEM 보정 자체가 미미 → **노선 보류**, 7점 유지 |
| B1 후 | D̂_node가 체계적 패턴 | B2~B3 진행 |
| B2 후 | D̂가 운전 조건과 무관 (정적) | §3 가설 부정 → 노선 A' 옵션 (a)/(b) 충분 |
| B2 후 | D̂가 운전 조건 의존 (동적) | §3 가설 확인 → 노선 A' 옵션 (d) 필요 |
| B3 후 | 어셈블리 평균 D̂_assy가 의미 있는 양으로 정의 가능 | B4 진행 (옵션 A'' 가능) |
| B3 후 | 어셈블리 평균이 NEM 의미를 잃음 | **노선 A'만 진행** (A'' 폐기) |
| B4 후 | D̂_assy 사용 시 잔차 6%→3% 수준 개선 확인 | 노선 A' 또는 A'+A'' 채택 → SD-Phase 진입 |
| B4 후 | D̂_assy 사용해도 잔차 개선 미미 | 노선 보류, 7점 유지 |

---

## 6. 노선 A'/A'' 구현 단계 (조건부 — 작업 B 통과 후)

| 단계 | 작업 | 검증 | Phase |
|---|---|---|---|
| **D̂-0** | **선결**: 작업 B 통과 (§5) | 학습 가능 패턴 확인 | piecewise-test |
| D̂-1 | $\hat{D}^{\text{pred}}_{\text{assy}}$ 생성기 (옵션 a~d 중 선택) | shape 테스트, gradient 흐름 | SD-Phase |
| D̂-2 | NEM 잔차 기반 L_diffusion 식 구현 | 단일 샘플 과적합 | SD-Phase |
| D̂-3 (조건부) | A'' 채택 시 L_dhat_assy_GT 추가 | GT supervision 효과 | SD-Phase |
| D̂-4 | warm-up schedule (D̂_assy zero-init 활성화) | 학습 안정성 | SD-Phase |
| D̂-5 | Regularization (smoothness, 부호 제약) | 학습된 D̂_assy 패턴 진단 | SD-Phase 평가 |
| D̂-6 | 가중치 튜닝 (L_data, L_diffusion(NEM), L_dhat) | 최종 power 예측 정확도 | SD-Phase 평가 |

### 6.1 7점 + NEM 병행 — Loss schedule

사용자 지적대로 **7점 유지와 NEM 병행이 완성형**. 두 신호의 역할 분리:

| 신호 | 역할 | 활성 시점 |
|---|---|---|
| **L_data** (MASTER GT) | 정확도 보장 | 메인, 항상 |
| **L_diffusion (CMFD, 7점)** | 빠른 공간 결합 prior | 학습 초기 (warm-up) |
| **L_diffusion (NEM, $\hat{D}^{\text{pred}}_{\text{assy}}$ 사용)** | NEM 정합성 강제 | 학습 중기 이후 |
| **L_dhat_assy_GT** (조건부, A'' 채택 시) | NEM correction supervision | A'' 채택 시 |

권장 schedule:

```
Epoch 0 ~ 10:    L_data + L_diffusion(CMFD, 7점)                            ← warm-up
Epoch 10 ~ 30:   L_data + L_diffusion(CMFD) + L_diffusion(NEM, D̂_pred)     ← D̂_assy 활성화
                 (A'' 채택 시: + L_dhat_assy_GT)
Epoch 30 ~ end:  L_data + L_diffusion(NEM) [+ L_dhat_assy_GT]                ← CMFD 항 제거
```

가중치 시작점 (튜닝 필요):
- L_data: 1.0
- L_diffusion(CMFD): 0.1 → 0
- L_diffusion(NEM): 0 → 0.1
- L_dhat_assy_GT (A''): 0 → 0.05 (조건부)

### 6.2 D̂_assy 생성기 옵션 (D̂-1)

§4.1의 (a)~(d) 중 선택. 권장 진행:

1. **(a) 공간 균일** 1-2개 변수로 시작 → baseline 잔차 확인
2. **(c) 어셈블리별** trainable로 확장 → 위치 패턴 학습 가능 여부
3. **(d) 작은 NN** ($\bar\varphi$, xs_fuel 입력) → 동적 패턴 학습

각 단계에서 학습된 D̂_assy 시각화로 물리적 합리성 진단.

### 6.3 우려사항 정리

| 우려 | 대응 |
|---|---|
| Self-consistency만으로 under-determined | Regularization (smoothness, 부호 제약, 영역 제약) |
| 학습 초기 잔차 폭주 | warm-up schedule + D̂_assy zero-init |
| L_diffusion(NEM)이 L_data 압도 | 가중치 튜닝 (~0.01-0.1) |
| 학습된 D̂_assy가 비물리적 값 수렴 | A'' 검증 통과 시 GT supervision으로 anchoring |
| 추론 시 안정성 | D̂_assy가 (d) NN인 경우 입력 분포 밖에서 불안정 가능 → 추론 시 클리핑 |

---

## 7. 기대 효과 + 정직한 한계

### 7.1 잔차 감소 추정

| 항목 | 현재 (CMFD only) | 노선 A' (예상) | 노선 A'+A'' (예상) | 본질 한계 |
|---|---:|---:|---:|---:|
| g1 median | 2.24% | ~1.5-2% | ~1-1.5% | ~0.5% |
| g2 median | 6.44% | ~4-5% | ~2-3% | ~1.5% |
| g1 max | ~21% | ~15% | ~8-12% | — |
| g2 max | ~36% | ~25% | ~15-20% | — |

**주의**: 위 추정은 매우 보수적이고, 실제 결과는 작업 B (특히 B4)에서 측정.
**A'은 self-consistency만이라 A''보다 보수적 추정**.

### 7.2 부수적 이득

- 학습된 D̂_assy 시각화로 물리 진단 (§6.2)
- 추후 반사체 albedo BC와 자연 결합
- MASTER 의존성 감소

### 7.3 잔존 한계

- **어셈블리 평균 표현 자체의 정보 손실** (어셈블리 내부 분포는 여전히 복원 불가)
- D̂의 운전 조건 의존성이 학습 데이터 분포 밖에서 일반화될지 불확실
- 노드 NEM의 정확도는 어셈블리 단위 모델이 본질적으로 도달 불가
- A''의 평균화 자체가 의미 손실 동반

---

## 8. 리스크 및 대안

### 8.1 D̂_node 분포가 무작위일 경우 (B1 결과)

**확률**: 낮음 (NEM 이론상 D̂는 물리적으로 의미 있는 보정)

**대응**:
- 노선 A'/A'' 모두 보류
- §5.5의 단계적 권장으로 후퇴 (1단계: 7점 유지)

### 8.2 어셈블리 평균화가 의미 잃는 경우 (B3 결과)

**대응**:
- 노선 A'' 폐기
- 노선 A' (self-consistency)만 진행
- (d) NN 옵션으로 동적 D̂_assy 학습

### 8.3 D̂_assy가 학습 안 되는 경우 (구현 후)

**원인 추정**: under-determined, regularization 부족, schedule 부적절

**대응**:
- (a) → (c) → (d)로 자유도 단계적 확장
- Regularization 강화 (smoothness, 부호 제약)
- L_diffusion(NEM) 가중치 증가
- A''가 가능하면 GT supervision 추가

### 8.4 추론 시 D̂_pred 품질 저하

**대응**:
- 학습 시 D̂_pred를 사용한 NEM 잔차도 부과해서 self-consistency 강제 (§6.1 schedule)
- D̂_assy 출력에 클리핑/정규화 적용

### 8.5 학습된 D̂_assy가 비물리적 값으로 수렴

**대응**:
- 부호 제약 ($\hat{D}_{\text{assy}}$의 물리적 가능 범위)
- A'' 검증 통과 시 GT anchoring
- 학습 후 시각화로 진단 → 비물리적이면 regularization 강화

---

## 9. 다음 단계

| 순서 | 작업 | 산출물 | 의사결정 |
|---|---|---|---|
| **1** | **작업 B (B1+B2 우선) — 별도 대화에서 진행** | piecewise-test 스크립트/결과 | D̂_node 패턴 확인 |
| 2 | B3 (어셈블리 평균화 검증) | A'' 가능 여부 결정 | A''/A' 분기 |
| 3 | B4 (잔차 영향 측정) | 잔차 감소 정량 | go/no-go |
| 4 | B5 (결과 정리) | §10 결론 채움 | SD-Phase 진입 |
| 5 (조건부) | D̂-1~D̂-6 (구현) | NEM Layer 코드 | SD-Phase |

본 문서는 작업 B 결과 보고 후 §10 결론 절을 채워 업데이트.

---

## 10. 결론 (작업 B 분석 후 작성 예정)

> 본 절은 §9 1단계 (작업 B B1+B2) 완료 후 채울 예정.
> - D̂_node 분포 패턴 요약
> - 동적 vs 정적 판정
> - 어셈블리 평균화 가능 여부 (B3 결과)
> - 노선 A' / A''/ 보류 결정
> - 가중치 schedule 1차 안
> - SD-Phase 진입 일정

---

## 부록 A. 핵심 식 정리

### CMFD net current
$$
J^{\text{CMFD}}_u = -\tilde{D}_u\cdot\frac{\bar\varphi_2-\bar\varphi_1}{h_u}, \quad \tilde{D}_u = \frac{2D_1D_2}{D_1+D_2}
$$

### NEM net current (일반)
$$
J^{\text{NEM}}_u = -\tilde{D}_u\cdot\frac{\bar\varphi_2-\bar\varphi_1}{h_u} - \hat{D}_u\cdot(\bar\varphi_2+\bar\varphi_1)
$$

### 노드 단위 D̂ 역산 (작업 B 입력)
$$
\hat{D}^{\text{node}}_u = \frac{2\,\text{JNET0}_u + \tilde{D}_u\,(\bar\varphi_2-\bar\varphi_1)/h_u}{-(\bar\varphi_2 + \bar\varphi_1)}
$$

### 어셈블리 단위 NEM 잔차 (노선 A')
$$
J^{\text{NEM,pred}}_u = -\tilde{D}_u\cdot\frac{\bar\varphi_2-\bar\varphi_1}{h_u} - \hat{D}^{\text{pred}}_{\text{assy},u}\cdot(\bar\varphi_2+\bar\varphi_1)
$$

$$
R_g^{\text{NEM}} = \sum_{\text{faces}} J^{\text{NEM,pred}}_u\cdot A + \Sigma_{r,g}\,\bar\varphi_g\,V - S_g\,V
$$

### Loss (노선 A')
$$
\mathcal{L}_{\text{diffusion}}^{\text{NEM}} = \frac{1}{B\cdot N_{\text{fuel}}}\sum_{b,v}(R_{g1}^2 + R_{g2}^2)
$$

### Loss 추가 (노선 A'' — 조건부)
$$
\mathcal{L}_{\text{dhat\_assy}} = \text{MSE}\bigl(\hat{D}^{\text{pred}}_{\text{assy}}, \hat{D}^{\text{GT}}_{\text{assy}}\bigr)
$$

### Total (병기)
$$
\mathcal{L}_{\text{NEM total}} = \lambda_1 \cdot \mathcal{L}_{\text{diffusion}}^{\text{NEM}} + \lambda_2 \cdot \mathcal{L}_{\text{dhat\_assy}}
$$

---

## 부록 B. 참고 문서

- `2026-04-07 (도입 고려 필요) L_diffusion Cross-Attention 이론 해석.md` §5.5, §9
- `2026-04-01 JNET0 활용 방안 계획.md` 작업 B
- `2026-03-30 Physical Loss 통합 레퍼런스.md` §3.2 (CMFD/NEM 식)
- `piecewise-test/JNET0(MAS_NXS)의 공식 정합성 검증(오차0%확인)/` (JNET0 검증 완료 폴더)
- 핵공학 표준: Finnemann, Smith, Lawrence et al. — Nodal Expansion Method
- 방법론 §2.1.4 Eq. 2.1-37, 38
