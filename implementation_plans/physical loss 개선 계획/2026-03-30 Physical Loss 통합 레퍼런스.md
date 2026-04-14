# Physical Loss 통합 레퍼런스

> **작성일**: 2026-03-30
> **개정일**: 2026-04-08 — §3.6 신설 (L_diff 상대 잔차 형식 + redundancy 분석 + Consistency Barrier 참조), §5.1 전체 손실 함수 공식에 L_data_halo / L_diff_rel 반영.
> **개정일**: 2026-04-09 — 공간 형상 표기에 단계 구분 추가 (데이터 입력 vs 인코더 처리 vs L_diff 합산 도메인)
> **목적**: 산재된 Physical Loss 관련 설계/검증 문서를 단일 참조 문서로 통합
> **적용 대상**: 경수형 SMR의 부하추종 노심 해석 대리모델
> **공간 형상**: 데이터 입력 (20, 5, 5) [quarter, 500 inner cell] → halo expand → 인코더/Mamba/디코더 처리 (20, 6, 6) [720 cell, halo 11개 포함] → L_diff_rel 합산 도메인 (20, 5, 5) [inner 500 cell, halo는 stencil neighbor lookup] → 최종 외부 출력 (20, 5, 5). 1/4 대칭. 상세: `공간인코더 구현 계획/인코더 컴포넌트별 채용 이유/05_symmetry_mode.md`
> **시간 간격**: Δt = 300s (5분)

## 목차

0. [HDF5 데이터 스키마 Overview](#0-hdf5-데이터-스키마-overview)
1. [L_Bateman — Xe/I-135 Bateman ODE 잔차](#1-l_bateman--xei-135-bateman-ode-잔차)
   - 1.1 연립 방정식 / 1.2 물리 상수 / 1.3 핵분열 수율 / 1.4 평형 농도
   - 1.5 barn-cm 단위 / 1.6 Markov 상태 변수 / 1.7 수치 적분 방법 / 1.8 MASTER와의 차이
   - 1.9 손실 정의 / 1.10 TF2 구현 패턴
2. [L_sigma_a_Xe — 제논 미시단면적 일관성](#2-l_sigma_a_xe--제논-미시단면적-일관성)
   - 2.1 MASTER 내부 계산 메커니즘 / 2.2 데이터 취득 / 2.3 검증 결과
   - 2.4 손실 정의 / 2.5 TF2 구현 패턴
3. [L_diffusion — 확산방정식 PDE 잔차 (확장용)](#3-l_diffusion--확산방정식-pde-잔차-확장용)
   - 3.1 물리 근거 / 3.2 면 neutron current / 3.3 Loss 정의 / 3.4 사전 검증 결과
   - 3.5 반사체 경계 처리 — 학습 가능 Albedo
   - **3.6 L_diffusion 상대 잔차 형식 (L_diff_rel) — CMFD-only Bias 우회 (개정 2026-04-08)**
4. [L_keff — K-eff Rayleigh 몫 (확장용)](#4-l_keff--k-eff-rayleigh-몫-확장용)
5. [가중치 전략](#5-가중치-전략)
6. [물리량-데이터 매핑 요약](#6-물리량-데이터-매핑-요약)
7. [구현 시 주의사항](#7-구현-시-주의사항)
- [Appendix A. Bateman ODE — Euler vs 해석해 오차 검증](#appendix-a-bateman-ode--euler-vs-해석해-오차-검증)
- [Appendix B. σ_a^Xe 검증 상세 (E0-d/e/f, E1)](#appendix-b-σ_axe-검증-상세-e0-def-e1)

---

## 0. HDF5 데이터 스키마 Overview

> 원본 코드: `data_preprocess/lf_preprocess/dataset_builder.py`

현재 전처리 코드가 생성하는 HDF5 데이터셋의 전체 구조. 모델 학습 및 Physical Loss 계산에 사용되는 데이터의 출처.

### 0.1 고정 데이터 (`fixed/`) — BU=0 기준, 연소도 미반영

**현재 상태**: MAS_XSL에서 **BU=0(BOC) 인덱스 1개만** 추출하여 고정값으로 저장.
연소도에 따른 변화를 반영하지 않으며, LP가 결정되면 BOC 48h 운전 기간 내 불변 취급.
추후 Phase F에서 BU 전체 격자(σ₀: 72점, 미분계수: 14점)를 저장하고 런타임에 현재 연소도로 보간하는 구조로 확장 예정.

| HDF5 경로 | Shape | dtype | 물리량 | 단위 | 출처 |
|-----------|-------|-------|--------|------|------|
| `fixed/xs_fuel` | (Z,qH,qW,10) | float32 | 10채널 거시단면적 (νΣf1, Σf1, Σc1, Σtr1, Σs12, νΣf2, Σf2, Σc2, Σtr2, Σs21) | /cm | MAS_XSL 파싱 |
| `fixed/taylor_xe35/sigma_0` | (Z,qH,qW,2) | float64 | σ₀ 기준 미시단면적 (2군) | barn | MAS_XSL XE35 블록 |
| `fixed/taylor_xe35/d_sigma_dP` | (Z,qH,qW,2) | float64 | ∂σ/∂PPM | barn/ppm | MAS_XSL |
| `fixed/taylor_xe35/d_sigma_dTf` | (Z,qH,qW,2) | float64 | ∂σ/∂√T_f | barn/√K | MAS_XSL |
| `fixed/taylor_xe35/d_sigma_dM` | (Z,qH,qW,2,6) | float64 | ∂σ/∂ρ_m (6구간 piecewise) | barn/(g/cc) | MAS_XSL |
| `fixed/taylor_xe35/ref_conditions` | (4,) | float64 | [REFPPM, REFTF, REFTM, REFDM] | ppm, °C, °C, g/cc | MAS_XSL 헤더 |
| `fixed/taylor_xe35/dmod_delta` | (6,) | float64 | ΔM 밀도 편차 격자 | g/cc | MAS_XSL COMP 헤더 |

> **연소도 미반영 항목**: xs_fuel, Taylor 계수 전부 — BU=0 단일 인덱스만 저장된 상태. Phase F에서 BU 보간 동적 갱신 예정.

### 0.2 시나리오 시계열 데이터 (`scenarios/{profile}/`)

매 스텝(Δt=5min)·매 노드별로 MASTER 해석 결과에서 추출.

**Critical 상태 (CRS 궤적, 시점 t의 노심 상태)**:

| HDF5 필드 | Shape | dtype | 물리량 | 단위 | MAS_NXS 컬럼 | Loss 사용 |
|-----------|-------|-------|--------|------|:------------:|:---------:|
| `critical_xe` | (T,Z,qH,qW) | float64 | N_Xe 수밀도 | #/barn-cm | `DEN-XEN` | L_Bateman |
| `critical_sm` | (T,Z,qH,qW) | float64 | N_Sm 수밀도 | #/barn-cm | `DEN-SAM` | — |
| `critical_i135` | (T,Z,qH,qW) | float64 | N_I 수밀도 | #/barn-cm | MAS_OUT `$NUCL3D` | L_Bateman |
| `critical_sigma_a_xe` | (T,Z,qH,qW,2) | float64 | σ_a^Xe 미시 흡수단면적 (2군) | barn | `ABS-XEN` | L_Bateman, L_σXe |
| `critical_flux` | (T,Z,qH,qW,2) | float64 | φ 중성자속 (2군) | n/cm²/s | `FLX` | L_Bateman |
| `critical_Sigma_f` | (T,Z,qH,qW,2) | float64 | Σ_f 핵분열 거시단면적 (2군) | /cm | `FIS` | L_Bateman |
| `critical_yield_xe` | (T,Z,qH,qW) | float64 | γ_Xe 핵분열 수율 | 무차원 | `FYLD-XE135` | L_Bateman |
| `critical_yield_i` | (T,Z,qH,qW) | float64 | γ_I 핵분열 수율 | 무차원 | `FYLD-I135` | L_Bateman |

**쿼리 (입력 조건)**:

| HDF5 필드 | Shape | dtype | 물리량 | 단위 |
|-----------|-------|-------|--------|------|
| `query_rod_map_3d` | (T,31,Z,qH,qW) | float32 | 제어봉 3D 삽입 분율 맵 | 0~1 |
| `query_rod_offsets_1d` | (T,31) | float32 | 1D rod offset | step |
| `query_pload` | (T,) | float32 | 목표 출력 수준 | 0~1 |

**해석 결과 (GT 타겟, 31-way: index 0=critical, 1~30=branch)**:

| HDF5 필드 | Shape | dtype | 물리량 | 단위 | Loss 사용 |
|-----------|-------|-------|--------|------|:---------:|
| `result_power` | (T,31,Z,qH,qW) | float32 | 절대 열출력 | MW/node | L_data |
| `result_tcool` | (T,31,Z,qH,qW) | float32 | 냉각재 온도 | °C | L_data |
| `result_tfuel` | (T,31,Z,qH,qW) | float32 | 연료 온도 | °C | L_data, L_σXe(예측) |
| `result_rhocool` | (T,31,Z,qH,qW) | float32 | 냉각재 밀도 | g/cc | L_data, L_σXe(예측) |
| `result_keff` | (T,31) | float64 | 유효증배계수 | — | L_data, L_keff |
| `result_ao` | (T,31) | float32 | 축방향 출력 편차 | — | L_data |
| `result_max_pin_power` | (T,31) | float32 | 핀 출력 peaking factor | — | — |
| `result_max_pin_loc` | (T,31,3) | int32 | 핀 피크 위치 (z,y,x) | — | — |

**Branch 핵종 데이터 (31-way)**:

| HDF5 필드 | Shape | dtype | 물리량 | 비고 |
|-----------|-------|-------|--------|------|
| `branch_xe` | (T,31,Z,qH,qW) | float64 | N_Xe | Frozen Xenon: CRS와 거의 동일 |
| `branch_sm` | (T,31,Z,qH,qW) | float64 | N_Sm | |
| `branch_i135` | (T,31,Z,qH,qW) | float64 | N_I | |
| `branch_sigma_a_xe` | (T,31,Z,qH,qW,2) | float64 | σ_a^Xe (2군) | branch 조건(T_f, ρ_m 변동)에 따라 미세 차이 |
| `branch_flux` | (T,31,Z,qH,qW,2) | float64 | φ (2군) | |
| `branch_Sigma_f` | (T,31,Z,qH,qW,2) | float64 | Σ_f (2군) | |
| `branch_yield_xe` | (T,31,Z,qH,qW) | float64 | γ_Xe | |
| `branch_yield_i` | (T,31,Z,qH,qW) | float64 | γ_I | |

> **node_fullcore 데이터** (분석/참조용): 위 critical/branch 필드 중 일부가 quarter crop 전 풀코어(Z,18,18) 형태로도 저장됨. 모델 학습에는 사용하지 않음.

> **dtype**: 전처리 코드 전체를 **float32로 통일 완료** (2026-03-30). MASTER 원본 출력의 유효숫자(4~6자리)상 float32(~7자리)로 충분하며 정밀도 손실 없음. normalizer 내부 연산(Welford 누적, log/pcm 변환)만 float64 유지 (수치 안정성), 최종 통계 출력은 float32. **기존 HDF5 파일은 재생성 필요.**

### 0.3 차원 정의

| 기호 | 값 | 의미 |
|:----:|:---:|------|
| T | 576 | 시계열 스텝 수 (48h ÷ 5min) |
| Z | 20 | 축방향 연료 평면 (반사체 K=1, K=22 제외) |
| qH, qW | 5, 5 | 1/4 대칭 크롭 후 반경 방향 |
| 2 | 2 | 에너지군 (g=1: fast, g=2: thermal) |
| 31 | 31 | CRS(1) + Branch(30) |
| 10 | 10 | xs_fuel 채널 수 (νΣf1, Σf1, Σc1, Σtr1, Σs12, νΣf2, Σf2, Σc2, Σtr2, Σs21) |

---

## 1. L_Bateman — Xe/I-135 Bateman ODE 잔차

### 1.1 연립 방정식

**I-135:**

$$\frac{dN_I}{dt} = \gamma_I \,\Sigma_f \,\phi - \lambda_I \,N_I \quad \text{[Eq.1]}$$

**Xe-135:**

$$\frac{dN_{Xe}}{dt} = \gamma_{Xe}\,\Sigma_f\,\phi + \lambda_I\,N_I - \lambda_{Xe}\,N_{Xe} - \sigma_a^{Xe}\,\phi\,N_{Xe} \quad \text{[Eq.2]}$$

### 1.2 물리 상수 (불변)

| 기호 | 의미 | 값 | 단위 |
|------|------|-----|------|
| λ_I | I-135 붕괴상수 | 2.875×10⁻⁵ | s⁻¹ (T½≈6.70h) |
| λ_Xe | Xe-135 붕괴상수 | 2.0916×10⁻⁵ | s⁻¹ (T½≈9.17h) |
| Δt | 시간 간격 | 300 | s |

### 1.3 핵분열 수율 (MASTER 출력, 시계열)

| 기호 | 의미 | 대표값 | MAS_NXS 컬럼 | HDF5 필드 |
|------|------|:------:|:------------:|-----------|
| γ_I | I-135 누적 핵분열 수율 | ~0.0639 | `FYLD-I135` | `critical_yield_i` (T,Z,qH,qW) |
| γ_Xe | Xe-135 직접 핵분열 수율 | ~0.00228 | `FYLD-XE135` | `critical_yield_xe` (T,Z,qH,qW) |

> **주의**: γ_I, γ_Xe는 물리 상수가 아님. MASTER가 매 계산 시점·노드별로 내부 산출하여 MAS_NXS에 출력한 값.
> 정확한 내부 의존성(연소도, 조성 등)은 추가 확인 필요하나, 노드별·스텝별 변동이 존재하므로 상수 취급하지 않음.
> 학습 시에는 HDF5 시계열 값을 사용하고, 추론 시에는 대표값 또는 고정 라이브러리 값 사용 가능.

### 1.4 평형 Xe-135 농도 (참고)

$$N_{Xe}^{eq} = \frac{(\gamma_I + \gamma_{Xe})\,\Sigma_f\,\phi}{\lambda_{Xe} + \sigma_a^{Xe}\,\phi} \quad \text{[Eq.3]}$$

고출력에서 σ_a^Xe·φ ≫ λ_Xe → 평형 Xe 농도가 출력에 거의 무관해지는 포화 특성.

### 1.5 barn-cm 단위계 적용

> 원본: `2026-03-26 physical loss 적용단계 계획.md` §3-1

HDF5 데이터의 수밀도 N은 [#/barn-cm] 단위. Σ_f[/cm]×φ[n/cm²/s] = [반응/cm³/s]이므로,
N [#/barn-cm] 단위와 맞추려면 **×1e-24** 필요 (∵ 1/cm³ = 1e-24/barn-cm).

```
dN_Xe/dt [#/barn-cm/s] =
   γ_Xe × (Σ_f_g1×φ_g1 + Σ_f_g2×φ_g2) × 1e-24     ← 생산 (cm³→barn-cm 변환)
 + λ_I × N_I                                          ← I-135 붕괴 → Xe 유입
 - λ_Xe × N_Xe                                        ← Xe 자연 붕괴
 - (σ_g1×φ_g1 + σ_g2×φ_g2) × N_Xe × 1e-24           ← 중성자 흡수 (cm³→barn-cm)

dN_I/dt [#/barn-cm/s] =
   γ_I × (Σ_f_g1×φ_g1 + Σ_f_g2×φ_g2) × 1e-24       ← 핵분열 생산
 - λ_I × N_I                                           ← I-135 붕괴
```

**λ×N 항은 변환 불필요**: λ[/s]×N[#/barn-cm] = [#/barn-cm/s]로 이미 동일 단위.

### 1.6 Markov 상태 변수

#### 1.6.1 마르코프 성질 요건

**대상**: Bateman ODE (Xe-135/I-135 연립방정식, Eq.1~2)의 시간 진화.
**요건**: 현재 상태 s_t만으로 dN/dt의 우변이 완전히 결정되어야 함 — 즉 Eq.1~2의 우변에 나타나는 **모든 물리량**이 s_t에 포함되어야 한다.
**대표적 위반**: I-135 누락 → Xe 생성의 ~95%가 I-135 붕괴 경유이므로 마르코프 성질 깨짐.

#### 1.6.2 필수 상태 변수

| 카테고리 | 변수 | Shape | 마르코프 필요 이유 |
|----------|------|-------|--------------------|
| 독물질 | N_Xe(t) | (Z,qH,qW) | Eq.2 직접 상태변수 |
| 독물질 | N_I(t) | (Z,qH,qW) | Xe 간접 생성원 |
| 중성자속 | φ(t) | (Z,qH,qW,2) | 생성항·소멸항 계수 (2군) |
| 단면적 | Σ_f(t) | (Z,qH,qW,2) | 핵분열 생산항 (2군) |
| 단면적 | σ_a^Xe(t) | (Z,qH,qW,2) | Xe 소멸항 계수 (2군). MASTER에서 Σ_a^Xe(거시) 직접 출력 불가 → σ [barn] × N으로 계산 |
| 열수력 | T_fuel(t) | (Z,qH,qW) | Doppler → σ_a^Xe 보정 |
| 열수력 | ρ_mod(t) | (Z,qH,qW) | 감속능 → 스펙트럼 → σ_a^Xe |
| 연소도 | Bu | (Z,qH,qW) | σ_a^Xe 보간 기준축 |
| 제어봉 | rod_map(t) | (Z,qH,qW) | 국소 중성자속 분포 변화 |

#### 1.6.3 물리량 분류 (학습/Loss 관점)

| 분류 | 물리량 | HDF5 위치 | 비고 |
|------|--------|-----------|------|
| **시계열** (scenarios/) | N_Xe, N_I, σ_a^Xe, Σ_f, φ, γ_Xe, γ_I, T_f, ρ_m, P, keff, AO | critical_*, branch_* | 매 스텝·노드별 변동 (MAS_NXS에서 파싱) |
| **LP 고정** (fixed/) | xs_fuel, Taylor 계수 (σ_0, ∂σ/∂Tf, ∂σ/∂M) | fixed/taylor_xe35/ | Loss 파라미터로만 사용 |
| **물리 상수** | λ_Xe, λ_I, Δt | 하드코딩 | 불변 (§1.2) |
| **BU 의존** (BOC 고정) | xs_fuel, Taylor 계수 | Phase F에서 보간 갱신 | BOC 48h 내 상수 취급 |

> **γ_Xe, γ_I 분류 근거**: §1.3 참조. MAS_NXS `FYLD-XE135`, `FYLD-I135` 컬럼에서 파싱되며 노드별 변동이 존재하므로 시계열로 분류.

### 1.7 수치 적분 방법

**공통 가정**: Δt(=300s) 구간 내에서 φ(t), σ_a^Xe(t), Σ_f(t), γ(t) 등 ODE 계수를 **시점 t의 값으로 고정**(piecewise-constant 가정). 이 가정 하에서 Bateman ODE는 상수 계수 선형 ODE가 되어 아래 두 방법이 모두 적용 가능하다.

**Euler forward** (1줄):

$$N(t+\Delta t) = N(t) + \Delta t \cdot \frac{dN}{dt}\bigg|_t$$

1차 절단 오차 O(Δt²). Δt/τ_Xe ≈ 0.006이므로 국소 절단 오차 ~0.09%.

**해석해** (행렬 지수함수):

$$\mathbf{N}(t+\Delta t) = e^{A\Delta t}\,\mathbf{N}(t) + A^{-1}(e^{A\Delta t} - I)\,\mathbf{S} \quad \text{[Eq.6]}$$

piecewise-constant 가정 하에서 닫힌 해. φ를 Δt 내 상수로 고정한 뒤 행렬 지수함수로 적분.

**E1-T5b 비교** (Appendix A 참조): 두 방법의 차이는 mean 0.02%p, max 0.04%p. Δt=300s에서 적분 방법 차이는 무시할 수 있는 수준이나, **두 방법이 수학적으로 동일하지는 않음** — 해석해가 이론적으로 더 정확하나 실질적 개선은 미미.

### 1.8 MASTER와의 구조적 차이

MASTER(EXE_DEP)는 해석해 + **Full Predictor-Corrector** 방식을 사용한다 (매뉴얼 확인: idepl 파라미터 Not Working, 항상 Full PC 강제. 방법론 34080142.pdf §10.3.1 참조):

1. **Predictor**: 시점 t의 φ(t), σ(t)로 해석해 적분 → N̄(t+1) 예측
2. **중성자 수송 + 열수력 재계산**: N̄(t+1)을 반영하여 φ(t+1), σ(t+1) 갱신
3. **Corrector**: 갱신된 계수의 가중평균으로 해석해 재적분 → N(t+1) 최종값

**우리 모델의 차이**: 시점 t의 GT 값만 사용하여 1회 적분 → 중간 재계산(step 2~3) 없음.

이 차이 + 열수력 피드백 미반영 등의 **복합 요인**이 MASTER GT 대비 ~0.9% 바닥 오차를 유발 (E1-T5c에서 PC를 모방해도 0.905%로 수렴 → 적분법 개선만으로는 해소 불가).

> **향후 검토**: 이 구조적 차이를 어떻게 해소할지는 모델 학습 결과를 확인한 후 결정. L_data(MSE)가 보완할 것으로 기대.

### 1.9 손실 정의

$$\mathcal{L}_{Bateman} = \frac{1}{B \cdot V} \sum_{b,v} \left[ \left(\hat{N}_{Xe,b,v}(t\!+\!1) - N_{Xe,b,v}^{phys}\right)^2 + \left(\hat{N}_{I,b,v}(t\!+\!1) - N_{I,b,v}^{phys}\right)^2 \right]$$

- B = batch size, V = voxels (연료 영역 노드 수, fuel_mask 적용 후)
- N^{phys}: 현재 스텝 GT 값으로 ODE 적분한 결과
- N̂: 모델 예측값
- Bateman ODE는 **노드 로컬 방정식** (공간 결합 없음) → 셀별 독립 계산 유효

### 1.10 TF2 구현 패턴

```python
import tensorflow as tf

GAMMA_I  = 0.0639
GAMMA_XE = 0.00228
LAMBDA_I  = 2.878e-5  # s^-1
LAMBDA_XE = 2.092e-5  # s^-1
DT = 300.0            # 초 (5분)

def bateman_analytical(N_xe, N_I, phi, sigma_a_xe, Sigma_f_total):
    """해석해 기반 Bateman ODE 적분기 (Δt 내 phi 고정 가정)"""
    S_I   = GAMMA_I  * Sigma_f_total * phi * 1e-24
    S_Xe  = GAMMA_XE * Sigma_f_total * phi * 1e-24

    lambda_eff = LAMBDA_XE + sigma_a_xe * phi  # Xe 유효 소멸 상수

    # I-135 해석해
    exp_I   = tf.exp(-LAMBDA_I * DT)
    N_I_next = N_I * exp_I + (S_I / LAMBDA_I) * (1.0 - exp_I)

    # Xe-135 해석해 (I-135 붕괴항 포함)
    exp_eff = tf.exp(-lambda_eff * DT)
    denom   = lambda_eff - LAMBDA_I + 1e-30  # 분모 0 방지
    N_xe_next = (
        N_xe * exp_eff
        + (S_Xe + LAMBDA_I * N_I - S_I * LAMBDA_I / LAMBDA_I) / lambda_eff * (1.0 - exp_eff)
        + LAMBDA_I * (N_I - S_I / LAMBDA_I) / denom * (exp_I - exp_eff)
    )
    return N_xe_next, N_I_next
```

---

## 2. L_sigma_a_Xe — 제논 미시단면적 일관성

### 2.1 MASTER 내부 σ_a^Xe 계산 메커니즘

> 원본: `Xe-135 동역학...MASTER 연동.md` §3~4, 방법론 문서 34080142.pdf §10.3

#### 2.1.1 이론적 정의 (원 공식)

2-군 유효 미시 흡수 단면적은 에너지 스펙트럼 가중 평균으로 정의된다:

$$\sigma_{a,g}^{Xe} = \frac{\int_{E \in g} \sigma_a^{Xe}(E)\,\phi(E)\,dE}{\int_{E \in g}\phi(E)\,dE} \quad \text{[Eq.4]}$$

φ(E)는 온도·밀도·연소도에 따라 변동 → **원시 ENDF 핵자료를 직접 대입하려면 격자 수송 코드를 구현해야 함** (사실상 불가능).

#### 2.1.2 MASTER의 실제 계산 방식: MAS_XSL Taylor 전개

MASTER는 Eq.4를 직접 풀지 않고, **격자 물리 코드(KARMA, DeCART2D)가 사전 계산한 Taylor 전개 계수**를 MAS_XSL 파일에 저장한 뒤, 매 계산 시점마다 1차 Taylor 전개로 σ_a^Xe를 근사한다 (방법론 문서 34080142.pdf §10.3, MAS_XSL 매뉴얼 p.196-198):

$$\sigma_a^{Xe}(B_u, T_f, \rho_m) \approx \sigma_0(B_u) + \frac{\partial\sigma}{\partial\sqrt{T_f}}\left(\sqrt{T_f}-\sqrt{T_{f,0}}\right) + \frac{\partial\sigma}{\partial\rho_m}(\rho_m - \rho_{m,0}) \quad \text{[Eq.5]}$$

- σ_0, 편미분 계수: MAS_XSL 파일에 저장 (격자 물리 코드 사전 계산 결과)
- 기준 상태(T_{f,0}, ρ_{m,0}): MAS_XSL 고정 상수 (**이전 타임스텝 값이 아님**)

#### 2.1.3 스펙트럼 경화 효과 — σ_a^Xe 변동 인자

| 인자 | 스펙트럼 효과 | σ_a^Xe 영향 |
|------|-------------|-------------|
| T_mod 상승 | 경화 (고에너지 이동) | **감소** (0.084eV 공명에서 멀어짐) |
| ρ_mod 감소 | 감속능 저하 → 경화 | **감소** |
| T_fuel 상승 | Doppler broadening | 간접적, 미미 |
| Bu 증가 | FP 축적 → 경화 | 감소 경향 |

→ σ_a^Xe는 **T_mod, ρ_mod에 가장 민감**. Taylor 전개의 입력 변수가 이를 반영.

### 2.2 데이터 취득

#### 2.2.1 MAS_NXS 출력 (학습 데이터)

`%EDT_OPT inxs=1` 설정 시, MASTER가 매 시점 Taylor 전개로 산출한 σ_a^Xe를 MAS_NXS 파일에 노드별로 출력한다. **이 값은 독립적 Ground Truth가 아니라 MASTER 내부 Taylor 계산의 진단 출력**이다.

**MAS_NXS 실제 컬럼명과 용도:**

| 컬럼명 | 매뉴얼 명칭 | 물리량 | 단위 | 비고 |
|--------|-----------|--------|------|------|
| `ABS-XEN` | XSXE35 | σ_a^Xe 미시 흡수단면적 | barn | 매뉴얼에 "/cm"(거시)으로 오기. **E1/E3 삼중 검증으로 barn(미시) 확정** |
| `DEN-XEN` | NDXE | N_Xe 수밀도 | #/barn-cm | |
| `FIS` | XSF | Σ_f 핵분열 거시단면적 | /cm | 2군 |
| `FLX` | — | φ 중성자속 | n/cm²/s | 2군 |
| `FYLD-XE135` | — | γ_Xe 핵분열 수율 | 무차원 | |
| `FYLD-I135` | — | γ_I 핵분열 수율 | 무차원 | |

> **Σ_a^Xe(거시)는 MASTER에서 직접 출력 불가** (E0-e 확인: `$XS3D`의 `ABS`는 총 흡수 Σ_a^total이며 Xe 전용 아님). 필요 시 `ABS-XEN` [barn] × `DEN-XEN` [#/barn-cm] = Σ_a^Xe [/cm]로 계산.

#### 2.2.2 추론 시: Taylor 전개 재현

학습 데이터의 σ_a^Xe가 MASTER의 Taylor 전개 결과이므로, 추론 시에도 동일한 Taylor 전개(Eq.5)를 모델 예측 T̂_f, ρ̂_m으로 수행하면 일관성이 유지된다.

- MAS_XSL에서 파싱한 Taylor 계수: HDF5 `fixed/taylor_xe35/` (§7 및 `제논 미시단면적 Taylor 계수 HDF5 스키마.md` 참조)
- E3 검증: Taylor σ vs MAS_NXS `ABS-XEN` → g1: 0.25%, g2: 0.92% 차이

### 2.3 검증 결과

> 원본: `2026-03-25_sigma_a_xe_검증.md` (E0-d/e/f, E1). 상세: Appendix B 참조.

#### 2.3.1 핵심 발견: N_Xe 약분

Bateman ODE 소멸항에서:

$$\sigma_a^{Xe} \cdot \phi \cdot N_{Xe} = \frac{\Sigma_a^{Xe}}{N_{Xe}} \cdot \phi \cdot N_{Xe} = \Sigma_a^{Xe} \cdot \phi$$

→ **σ_a^Xe(미시) 역산·저장 불필요**. Σ_a^Xe(거시, MAS_NXS `ABS-XEN`×`DEN-XEN`)를 직접 사용 가능.

단, **이 Loss(L_σ_a^Xe)에서 σ_a^Xe 자체가 필요하므로**, HDF5에는 σ_a^Xe [barn]으로 저장 유지.

#### 2.3.2 MASTER 어셈블리 집계 (E0-d)

`$XESM3D`(어셈블리) vs MAS_NXS 노드 평균: max 0.044%, mean 0.013%
→ MASTER 내부 = **노드 단순 평균**

#### 2.3.3 ABS-XEN 단위 확정 (E1)

| 근거 | 결과 |
|------|------|
| E1-T1: ABS-XEN > XABS 전 노드 | 거시(/cm) **기각** |
| E1-T3: σ×N 재구성 < XABS | 미시(barn) **적합** |
| E3: MAS_XSL Taylor σ vs ABS-XEN | 0.25%/0.92% 일치 |
| 방법론 문서 p.93: "microscopic" 명시 | barn 확인 |

→ `ABS-XEN` = σ_a^Xe [barn] **3중 검증으로 최종 확정**.

#### 2.3.4 Bateman Euler 전수 검증 (E1-T5)

barn 가설 하에서 Euler forward 1스텝:

| 항목 | 값 |
|------|:---:|
| 비교 노드 | 2,800개 |
| mean |상대오차| | 1.10% |
| max |상대오차| | 4.36% |
| < 5% 비율 | **100%** |

→ 단위가 틀렸다면 10⁸% 이상 발산 → **Euler 적분으로 단위 최종 확인**.

### 2.4 손실 정의

Taylor 1차 전개는 **MASTER 코드가 내부적으로 단면적을 보간하는 방법과 동일한 수학적 구조**이다 (방법론 문서 34080142.pdf §6.1~6.2, MAS_XSL 매뉴얼 p.196-198). 이 Loss는 모델의 열수력 예측값(T̂_f, ρ̂_m)으로 **MASTER와 동일한 방식으로 σ_a^Xe를 재계산**한 뒤, MASTER가 출력한 σ_a^Xe(GT)와 일관성을 검증한다.

#### 2.4.1 Taylor 1차 전개 공식

$$\sigma_{Taylor}(g) = \sigma_0(g) + \frac{\partial\sigma}{\partial\sqrt{T_f}}(g) \cdot \left(\sqrt{\hat{T}_f + 273.15} - \sqrt{T_{f,ref} + 273.15}\right) + \text{interp}\!\left(\frac{\partial\sigma}{\partial M}(g,:),\, \hat{\rho}_m\right) \cdot \left(\hat{\rho}_m - \rho_{m,ref}\right)$$

**MAS_XSL에서 참조하는 값** (HDF5 `fixed/taylor_xe35/`에 저장):

| HDF5 경로 | Shape | 물리량 | 단위 |
|-----------|-------|--------|------|
| `sigma_0` | (Z,qH,qW,2) | σ₀ 기준 단면적 | barn |
| `d_sigma_dTf` | (Z,qH,qW,2) | ∂σ/∂√T_f | barn/√K |
| `d_sigma_dP` | (Z,qH,qW,2) | ∂σ/∂PPM | barn/ppm |
| `d_sigma_dM` | (Z,qH,qW,2,6) | ∂σ/∂ρ_m (6구간 piecewise) | barn/(g/cc) |
| `ref_conditions` | (4,) | [REFPPM, REFTF, REFTM, REFDM] | ppm, °C, °C, g/cc |
| `dmod_delta` | (6,) | ΔM 밀도 편차 격자 | g/cc |

- T̂_f, ρ̂_m: **모델 예측값** (시점 t+1)
- 스키마 상세: `제논 미시단면적 Taylor 계수 HDF5 스키마.md` 참조

#### 2.4.2 손실 수식

$$\mathcal{L}_{Taylor} = \frac{1}{B \cdot V \cdot G}\sum_{b,v,g} \left(\sigma_{Taylor,b,v}^{(g)} - \sigma_{a,b,v}^{Xe,GT,(g)}\right)^2$$

- **B** = batch size (미니배치 내 샘플 수)
- **V** = voxels (연료 영역 노드 수, fuel_mask 적용 후)
- **G** = energy groups (G=2: fast, thermal)

**핵심 차이**: L_bateman은 **GT 입력 → ODE 타겟**, L_taylor는 **모델 예측값 → Taylor 계산 → MASTER GT σ 비교**.

### 2.5 TF2 구현 패턴

```python
def taylor_sigma_xe(T_f_pred, rho_m_pred, sigma0, dsigma_dTf, dsigma_dM,
                    T_f_ref, rho_m_ref):
    """1차 Taylor 전개 기반 sigma_a^Xe 재계산"""
    delta_sqTf = tf.sqrt(T_f_pred + 273.15) - tf.sqrt(T_f_ref + 273.15)
    delta_rho  = rho_m_pred - rho_m_ref
    return sigma0 + dsigma_dTf * delta_sqTf + dsigma_dM * delta_rho
```

---

## 3. L_diffusion — 확산방정식 체적 적분 잔차 (확장용)

**L_Bateman과의 비교**:

| | L_Bateman | L_diffusion |
|--|--|--|
| **무엇을 검사** | 시간 미분 (ODE 잔차) | 공간 밸런스 (PDE 잔차) |
| **시점** | t → t+1 (두 시점) | 단일 시점 t (한 시점) |
| **핵심 질문** | "모델 예측 Xe/I 변화량이 ODE를 만족하는가?" | "모델 예측 flux 분포가 확산방정식을 만족하는가?" |

L_diffusion은 시간 전파(t→t+1)가 아닌, **단일 시점에서 모델이 예측한 flux 공간 분포의
물리적 정합성**을 검사하는 잔차. 모델 예측 φ̂로 확산방정식 밸런스를 계산하여,
밸런스가 0에 가까울수록 물리적으로 일관된 flux 분포를 학습한 것.

### 3.1 물리 근거: 노드 체적 적분 균형식

> 근거: 방법론 Eq. 2.1-1a (2군 확산방정식) → Eq. 2.1-2a (노드 체적 적분)

미분 형태의 확산방정식을 노드 체적 V에 대해 적분하면, 가우스 발산 정리에 의해 누설항이 6면의 면적분으로 변환:

```
R_g = [6면 순 누설] + [소멸×V] - [소스×V] = 0  (이상적)

g=1: R₁ = Σ_u(J₁,right - J₁,left)·A_u + Σ_r1·φ̂₁·V - (1/k_eff)·(νΣ_f1·φ̂₁ + νΣ_f2·φ̂₂)·V
g=2: R₂ = Σ_u(J₂,right - J₂,left)·A_u + Σ_a2·φ̂₂·V - Σ_s12·φ̂₁·V
     (χ₂=0이므로 핵분열 소스 없음)
```

**단면적 유도** (방법론 Eq. 2.1-1a, Σ_ag = "absorption cross section in group g"):
- Σ_r1 = Σ_a1 + Σ_s12 = (Σ_c1 + Σ_f1) + Σ_s12 = xs_fuel[2]+[1]+[4]
- Σ_a2 = Σ_c2 + Σ_f2 = xs_fuel[7]+[6]
- D_g = 1/(3·Σ_tr,g): xs_fuel[3],[8]에서 유도

### 3.2 면 neutron current: CMFD 방식

> 근거: 방법론 §2.1.4 (p.20)
> "In the NNEM, **both the CMFD method and the two-node NEM** are used for the solution of the multi-group diffusion equation."
> "The CMFD problem incorporates the **global coupling** of the nodes while the two-node problems incorporates **local higher order coupling**."

MASTER의 NNEM = CMFD(전역 커플링) + NEM 보정(국소 고차 커플링) 반복.
L_diffusion에서는 **CMFD 부분만 재현**.

**Flux 출처**: MAS_OUT flux_3d는 **어셈블리(또는 노드) 체적 평균 중성자속**:
```
φ̄_g = (1/V) ∫∫∫ φ_g(r) dV  [/cm²/s]
```

**CMFD 면 neutron current** (인접 노드 체적평균 flux 간 차분, 방법론 Eq. 2.1-37):
```
J = D̃ × (φ̄_이웃 - φ̄_중심) / h
D̃ = 2·D_a·D_b / (D_a + D_b)  (조화평균)
→ 인접 노드 간 직렬 통과의 유효 확산계수 (작은 쪽의 병목 반영)
```

**체적 적분 밸런스 잔차** (R=0이면 확산방정식 만족):
```
g=1: R₁ = Σ_faces(J₁×A) + Σ_r1×φ̄₁×V − (1/keff)(νΣf₁φ̄₁ + νΣf₂φ̄₂)×V
g=2: R₂ = Σ_faces(J₂×A) + Σ_a2×φ̄₂×V − Σ_s12×φ̄₁×V
```
- Σ_faces(J×A): 6면 누설 합 [n/s]
- Σ_r×φ̄×V: 제거 반응률 [n/s]
- Source×V: 생성 반응률 [n/s] (g1=핵분열원 χ₁=1, g2=산란전입원 χ₂=0)

NEM 보정(Eq. 2.1-38)의 D̂ 항은 MASTER 내부 iteration 결과로 **외부 재현 불가**:
> "The second term of RHS of Eq. (2.1-38) can be regarded as a **correction term** which corrects the error of the linear flux used to obtain the first term."

### 3.3 Loss 정의

$$\mathcal{L}_{diffusion} = \frac{1}{B \cdot N_{fuel}} \sum_{b,v} (R_{g1}^2 + R_{g2}^2)$$

- B = batch size, N_fuel = 전체 연료 노드 수
- 내부 노드: CMFD 면 neutron current 사용 (6면 이웃 모두 연료)
- 경계 노드 (반사체 인접): 학습 가능 Albedo BC 적용 (§3.5 참조)

### 3.4 사전 검증 결과

#### 3.4.1 초기 검증 (2026-03-30, 어셈블리 단위, 피치 21.504 cm)

CMFD 체적 적분 잔차, 2 LP × CRS 10스텝 + Branch 10개, 5760 유효 노드:

| 통계 | g=1 (fast) | g=2 (thermal) |
|------|:----------:|:-------------:|
| median | 3.50% | 6.99% |
| mean | 7.01% | 7.17% |
| < 10% | 70.7% | 82.6% |

#### 3.4.2 Step 0: MAS_NXS ABS/JNET0 검증 (2026-03-31)

MAS_NXS 물리량 정의를 JNET0 밸런스 역산으로 확정:
- **ABS = Σ_c + Σ_f** (총 흡수 거시단면적 [/cm])
- **JNET0 = half net current** (net current의 절반으로 출력됨)
  - per-unit-area [n/cm²/s], positive = outward
  - **밸런스 시 ×2 필요**: leak = **2** × Σ(JNET0_face × A_face)
- **WIDE = 21.60780 cm** (기존 21.504 cm은 출처 불명 오류, core_geometry.py:147에서 확인)
- ndivxy = 2, 노드 피치 = 10.804 cm

JNET0 밸런스 잔차 (α=2 적용, 4560 연료 노드):
- **g1 median 0.0002%, g2 median 0.0002%** → 사실상 0 (밸런스 완벽 성립)
- α=2 발견 과정: 최적 스케일 팩터 탐색 → median α=2.0000, std=0.03
- ADF = 모든 연료 노드에서 1.0 (SET 적용 확인)

상세: `piecewise-test/2026-03-31_step0_abs_jnet0_result.txt`
V&V 최종 검증: `v&v/01_jnet0_direction/verification_report.md` (median 0.000147%, 전류 연속 exact 0)

#### 3.4.3 노드 vs 집합체 CMFD 비교 (2026-03-31)

6면 이웃 모두 연료인 노드만 평가, 2 LP × CRS 10스텝, 피치 수정(21.608 cm):

| 메트릭 | 노드 CMFD (10.8cm, MAS_NXS) | 집합체 CMFD (21.6cm, xs_fuel) |
|--------|:---:|:---:|
| g1 median | **2.02%** | 2.29% |
| g2 median | **1.23%** | 7.54% |
| g1 mean | 3.08% | 4.15% |
| g2 mean | 2.10% | 8.64% |
| 유효 노드 | 61,920 | 13,320 |

- g2: 노드 단위에서 **6배 개선** (7.54% → 1.23%) — 열중성자 확산거리가 짧아 메시 의존성 큼
- g1: 유사 수준 (XS 출처 차이로 순수 격자 효과 분리 어려움)
- **집합체 단위 L_diffusion의 CMFD 구조적 한계**: g1 ~2.3%, g2 ~7.5%

상세: `piecewise-test/2026-03-31_diffusion_residual_comparison_output.txt`

#### 3.4.4 JNET0 스케일 팩터 발견 및 밸런스 검증 (2026-03-31)

초기 JNET0 밸런스에서 g1 3.2%, g2 2.0% 잔차가 발생하여 공간 분포 분석 수행.
Z축 상·하부 및 XY 체커보드 패턴이 관찰되었으나, **최적 스케일 팩터 탐색 결과 α=2.0000**
(median, std=0.03)에서 잔차가 정확히 0%로 수렴.

**경험적 결과**: α=2 적용 시 잔차 정확히 0% (median α=2.0000, std=0.03)
- α=1 사용 시 관찰된 Z축/XY 패턴은 모두 **스케일 오류의 아티팩트**
- α=2 적용 후 모든 Z 평면(K=2~21)에서 median < 0.001%, 공간 패턴 완전 소멸

**α=2 적용 후 Z축 프로파일 (g=1 median)**:
```
K=2(하부): 0.0003%    K=10(중앙): 0.0002%    K=21(상부): 0.0002%
→ 전 영역에서 균일하게 ~0% (밸런스 완벽 성립)
```

#### 사용한 밸런스 수식 vs 방법론 공식 대조

**검증에 사용한 수식** (α=2로 밸런스 성립 확인):
```
leak = α × Σ_faces(JNET0_face × A_face)      (6면 합산, α=2)
     = 2 × (JNET0_N×A_xz + JNET0_S×A_xz
           + JNET0_E×A_yz + JNET0_W×A_yz
           + JNET0_T×A_xy + JNET0_B×A_xy)

밸런스: leak + Σ_r × φ̄ × V = Source × V
```
- JNET0: MAS_NXS 출력값 (per-unit-area, positive=outward)
- A_face: 면적 (A_xz=dx×dz, A_yz=dy×dz, A_xy=dx×dy)

**방법론 Eq. 2.1-2a** (노드 체적당 밸런스, [n/cm³/s]):
```
Σ_{u=x,y,z} (1/a_u) × [(j_gul^{-m} + j_gur^{+m}) - (j_gul^{+m} + j_gur^{-m})]
  + (Σ_ag + Σ_{gg'})×φ̄_g = Σ_{g'<g} Σ_{g'g}×φ̄_{g'} + (1/k)×χ×νΣ_f×φ̄
```
- j_gus^{±m}: 면 s에서 ±u 방향으로 흐르는 partial current [n/cm²/s]
- a_u = mesh size in direction u [cm]

**Eq. 2.1-2a 누설항의 물리적 의미**:
```
좌면(l):  j_gul^{-m} = 유출(-u방향), j_gul^{+m} = 유입(+u방향)
우면(r):  j_gur^{+m} = 유출(+u방향), j_gur^{-m} = 유입(-u방향)

(j_l^- + j_r^+) = 총 유출 (outgoing)
(j_l^+ + j_r^-) = 총 유입 (incoming)
차이 = 순 누설 = J_net_l + J_net_r
```
→ **표준 net current 합산과 정확히 일치** (입자 보존 법칙 성립)

**Eq. 2.1-2b** (Fick's law): `j⁺ - j⁻ = -D × (∂ψ/∂u)|_s`
**방법론 p.16**: `ψ_gus = (j⁺ + j⁻) / 2` (면 평균 flux)

#### α=2에 대한 고찰

경험적으로 α=2 적용 시 밸런스가 완벽히 성립 (잔차 0.0002%).

**Eq. 2.1-2a를 체적 적분**하면 표준 밸런스:
```
Σ_faces(J_net_face × A_face) + Σ_r × φ̄ × V = Source × V
```
- J_net = 면의 순 neutron current (양수 = 노드 외부 방향) [n/cm²/s]
- 누설 = 6면 net current × 면적의 **합산** (입자 보존)

**α=2의 원인 — JNET0 = J_net/2 저장 규약**:

경험적 검증: **JNET0 = J_net / 2** (median α=2.0000, std=0.03)
- 따라서: `leak = Σ(J_net × A) = Σ(2 × JNET0 × A)` → **α=2**

**왜 JNET0 = J_net/2 인가**: 확산 이론에서 partial current의 정의:
```
j⁺ = φ/4 + J_net/2
j⁻ = φ/4 - J_net/2
```
J_net에는 기본적으로 1/2 상수가 따라다님.
노달 코드 개발자들은 내부 연산 최적화를 위해(매 이터레이션마다 부동소수점
나눗셈을 피하기 위해) J_net/2 형태 자체를 하나의 변수로 치환하여 저장.
이는 원자로 물리 코드(MASTER, PARCS 등)에서 흔하게 발견되는 프로그래밍 관습.

→ α=2는 물리 공식의 인자가 아니라, **코드 내부 변수 치환 관습**을 보정하는 것

상세: `piecewise-test/2026-03-31_jnet0_residual_spatial_output.txt`

#### 3.4.5 검증 결과 종합 및 L_diffusion 설계 시사점

**JNET0 ×2 밸런스 검증으로 확인된 사실**:
- MASTER의 노드 밸런스(Eq. 2.1-2a)는 MAS_NXS 출력 물리량으로 **정확히 재현 가능**
- MAS_NXS의 ABS, SCA, NFS, FLX, JNET0(×2) 간에 **불일치 없음**

**집합체 단위 L_diffusion에 대한 시사점**:
1. 집합체 CMFD 잔차(g1 ~2.3%, g2 ~7.5%)는 **순수하게 CMFD FD 근사 오차**
   - MASTER 내부 물리량 자체에는 문제 없음 (JNET0 ×2로 밸런스 완벽 성립)
   - 잔차 원인: 조화평균 D × 선형 flux 가정 vs NEM 고차 flux 분포
2. 이 잔차는 NEM을 재구현하지 않는 한 **구조적 한계**
   - L_diffusion loss 가중치 설계 시 이 noise floor 고려 필요
3. JNET0 ×2 자체는 L_diffusion 구현에 직접 사용되지 않음
   - L_diffusion은 CMFD FD로 누설을 계산하므로 JNET0 불필요
   - JNET0은 검증/진단 도구로만 활용

#### 3.4.6 집합체 단위 CMFD 밸런스 잔차 — 전체 연료 노드 검증 (2026-04-01)

기존 §3.4.1/§3.4.3의 inner-only 평가를 **전체 연료 노드**로 확장.
Albedo BC(반사체면) + Mirror CMFD(대칭면)를 포함한 6종 면 유형 전부를 반영.

**검증 내용**: 밸런스 잔차 `R_g = [6면 누설] + [제거×V] - [생성×V]` (이상적 = 0)
- **모든 입력은 MASTER GT**: φ(center/neighbor), 거시단면적, keff
- **누설 계산만 CMFD FD 근사**: 내부면 `D̃×(φ_center−φ_neighbor)/h`, 반사체면 Marshak α/C, 대칭면 Mirror
- R_g ≠ 0의 원인: (1) CMFD FD vs NEM 고차 보정(D̂) 차이, (2) 거시단면적 출처에 따른 피드백 반영 여부

> **스케일 구분**: §3.4.2 JNET0 밸런스는 **노드 단위(10.8cm)** MAS_NXS 물리량으로 검증 → 잔차 0.0002%.
> 여기서의 잔차는 **집합체 단위(21.6cm)** CMFD FD 근사에서 발생하며, 노드 단위와는 스케일이 다름.

**[A] MAS_XSL 기반 (BOC 고정 xs_fuel)** — 실제 L_diffusion이 사용할 단면적

10LP × CRS 10스텝 = 100 시나리오, R5 Albedo 확정값 적용, 전체 연료 38,000노드:

| 구분 | N | g1 median | g1 mean | g2 median | g2 mean |
|------|---|:---------:|:-------:|:---------:|:-------:|
| 내부 노드 | 10,800 | 1.91% | 2.77% | 6.75% | 7.48% |
| 경계 노드 (Albedo BC) | 27,200 | 2.49% | 4.02% | 6.22% | 8.20% |
| **전체 연료** | 38,000 | **2.25%** | **3.67%** | **6.39%** | **7.99%** |

- §3.4.1(초기) 대비 변경: 피치 21.608cm, 3개 버그 수정(CMFD 부호, zmesh, Mirror), Albedo BC 적용
- g2 잔차 ~6.4%에는 BOC XS와 실시간 피드백 XS의 불일치가 포함됨 (아래 [B] 참조)

**[B] MAS_NXS 기반 (스텝별 피드백 반영 단면적)** — XS 불일치 제거 후 순수 CMFD 한계 측정

동일 조건에서 거시단면적만 MAS_NXS(Xe/Sm/온도 피드백 반영)로 교체:

| 구분 | N | g1 median | g2 median | g2 max |
|------|---|:---------:|:---------:|:------:|
| 내부 노드 | 10,800 | 2.03% | **1.79%** | 10.7% |
| 경계 노드 | 27,200 | 2.74% | **2.46%** | 11.2% |
| **전체 연료** | 38,000 | **2.45%** | **2.19%** | **11.2%** |

**[C] 비교 및 분석**

| 메트릭 | MAS_XSL (BOC) | MAS_NXS (피드백) | 변화 |
|--------|:---:|:---:|------|
| g2 median | 6.39% | **2.19%** | **-65.7% (3배 개선)** |
| g2 max | 35.6% | 11.2% | -68.6% |
| g1 median | 2.25% | 2.45% | +8.5% (미미) |

- **g2 잔차의 주 원인은 XS 불일치**: BOC xs_fuel이 Xe-135 축적에 의한 νΣ_f 감소를 미반영 (NFS g2 최대 25.7% 차이)
- **순수 CMFD FD 한계** (NXS 기반 = D̂ 오차만): g1 ~2.5%, g2 ~2.2%
- g1 max 증가(21→32%): Gd 함유 B5 어셈블리에서 XSL의 ABS 과소평가가 D̂ 오차를 우연히 상쇄하고 있었으며, NXS로 교정 시 숨겨진 CMFD 한계가 노출된 것
- **실제 L_diffusion은 MAS_XSL (BOC xs_fuel)을 사용**하므로 [A]가 운용 조건이나, 모델이 열수력 피드백을 정확히 예측할수록 [B] 수준에 접근 가능

상세: `piecewise-test/L_diffusion XS 검증/XS검증_T002_결과.md`, `piecewise-test/2026-04-01_L_diffusion_endtoend_결과.md`

### 3.5 반사체 경계 처리 — 학습 가능 Albedo (Learnable α)

#### 3.5.1 문제

Inner-only 평가(6면 모두 연료인 노드만)는 반사체 인접 연료 노드를 제외:
- 집합체 기준: 37/57 = 65% 노드만 평가 (**35% 제외**)
- → 경계 영역에 L_diffusion gradient signal 부재, 공간 커플링 학습 불균형

#### 3.5.2 해결: Marshak BC + 학습 가능 파라미터 (R5 캘리브레이션 확정)

반사체 인접면에 Albedo BC를 적용. 각 면의 유형에 따라 처리 방식이 다름:

```
(1) 내부면 (이웃=연료): J = D̃ × (φ̂_nb - φ̂_center) / h              (CMFD)
(2) Radial 반사체면:    J_g = α_g × D_g / (α_g×h/2+D_g) × φ̂_g      (Marshak BC)
(3) Axial 반사체면:     [J_g1, J_g2] = C × [φ̂_g1, φ̂_g2]            (행렬 BC)
(4) 대칭면:             J = D̃ × (φ̂_center - φ̂_mirror) / h            (Mirror CMFD, REFLECT)
    ※ mirror neighbor 매핑: qy=-1→qy=1, qx=-1→qx=1 (mirror symmetry 기준)
    ※ rotational symmetry의 경우 transpose 매핑으로 변경 필요
```

**Radial은 스칼라 α**, **Axial은 행렬 C**를 사용하는 이유:
- Radial(SS+H₂O): 군간 결합 C₂₁이 C₂₂의 0.7~7% → 무시 가능 → 스칼라 충분
- Axial(순수 경수): 군간 결합 C₂₁이 C₂₂의 32~46% → 무시 불가 → 행렬 필수
  (경수 반사체에서 고속→열 감속 반환 효과가 큼)

**학습 가능 파라미터 (12개)** — R5 캘리브레이션 확정 초기값 (40LP):
```python
# Radial — ortho(R1,R2 직각 방향) / diag(R3~R6 모서리)
alpha_ortho_g1 = tf.Variable(0.108, trainable=True)   # β=0.805, R²=0.998
alpha_ortho_g2 = tf.Variable(0.453, trainable=True)   # β=0.377, R²=0.998
alpha_diag_g1  = tf.Variable(0.082, trainable=True)   # β=0.849, R²=0.988
alpha_diag_g2  = tf.Variable(0.513, trainable=True)   # β=0.322, R²=0.997

# Axial — 행렬 C (2×2), J = C × [φ̂_g1, φ̂_g2]
C_bottom = tf.Variable([[+0.155, -0.135],    # R²=(0.999, 0.992)
                         [-0.025, +0.078]], trainable=True)
C_top    = tf.Variable([[+0.174, -0.097],    # R²=(0.993, 0.925)
                         [-0.036, +0.080]], trainable=True)
```

- 초기값 출처: **R5 캘리브레이션 (40LP × CRS 10스텝 = 400 시나리오)**
- R0~R3은 JNET0 N/S 매핑 오류(J↑=South 미인지) 영향 → R4부터 수정 매핑 적용.
  상세: `physical loss 개선 계획/2026-04-01 JNET0 N-S 매핑 오류 발견 및 해결.md`
- R4(2LP) 대비: g1 안정 (<1%), g2 6~9% 감소, top C₁₂ 부호 반전
- 훈련 중 L_diffusion + L_data가 α/C를 자동 조정
- 추론 시 학습된 α/C로 경계 누설 계산 (GT 불필요)
- α 범위 제약: `tf.clip_by_value(alpha, 0.01, 2.0)` 등으로 물리적 범위 유지

상세: `Albedo 캘리브레이션 종합 보고서 (확정).md`

#### 3.5.3 적용 시 Loss 수식 변경

기존: N_inner (내부 연료만) → 변경: **N_fuel (전체 연료, 경계 포함)**

각 연료 노드에서 6면을 boundary_mask로 분류 후 유형별 누설 계산:
- 내부면 → CMFD, 반사체면 → Marshak α 또는 행렬 C, 대칭면 → Mirror CMFD (REFLECT)

→ **L_diffusion을 공간 커플링 학습의 보조 gradient 신호로 도입 가능**.

---

### 3.6 L_diffusion 상대 잔차 형식 (L_diff_rel) — CMFD-only Bias 우회 (개정 2026-04-08)

> **개정 배경**: CMFD-only L_diffusion은 NEM 보정 누락으로 인한 g2 ~6.4% bias floor 존재 (§3.4.6 [A] 참조). 이는 모델이 GT에 수렴해도 손실이 0이 안 되는 systematic bias의 원인. ML 검토 plan: `C:\Users\Administrator\.claude\plans\compressed-wandering-stroustrup.md`

#### 3.6.1 절대 잔차의 한계

기존 절대 잔차 형식:
```
L_diff_abs = ‖R_CMFD(φ_pred, xs_BOC)‖²
```

문제점:
- φ_GT (= MASTER NNEM 출력) 자체가 R_CMFD = 0 을 만족하지 않음 (NEM 보정량만큼 잔차 존재)
- 즉 `R_CMFD(φ_GT) = ε(t) ≠ 0`, ε는 NEM gap (g2 median ~6.4%)
- L_diff_abs의 최소점은 R_CMFD(φ*) = 0 인 φ* 인데, 이는 φ_GT 와 다름 (NEM 보정량만큼 어긋남)
- 결과: L_data가 dominate해도 L_diff_abs의 gradient는 φ_pred를 *NEM gap 방향으로 systematic bias* 시킴

이 현상은 PINN 문헌의 **Consistency Barrier** 와 동일한 메커니즘 (§3.6.6 참조).

#### 3.6.2 상대 잔차 형식 (개정안)

```
L_diff_rel = MSE(R_CMFD(φ_pred, xs_BOC), R_CMFD(φ_GT, xs_BOC))
           = ‖R_CMFD(φ_pred − φ_GT, xs_BOC)‖²    (R이 phi에 대해 선형)
```

핵심 메커니즘:
- target이 0이 아닌 `R_CMFD(φ_GT, xs_BOC) = ε_total(t)` 으로 재정의됨
- ε_total(t)는 NEM gap + XS staleness + 제어봉 XS 변화 등 *모든 systematic bias의 합* 인 시점별 상수
- L_diff_rel 최소점: `R_CMFD(δ) = 0` (δ = φ_pred − φ_GT)
- R_CMFD는 (적절한 BC 하에) injective → null space ≈ {0} → δ = 0 → φ_pred = φ_GT
- **bias 항이 사라짐**

#### 3.6.3 XS 사용 규칙

- **양변 모두 BOC `xs_fuel` (10채널, 시점/branch 무관) 사용**
  - `xs_fuel`의 형상 (20, 5, 5, 10), LP-level lattice physics 결과
  - 출처: `data_preprocess/lf_preprocess/xs_voxel_builder.py`
  - 채널: [νΣf1, Σf1, Σc1, Σtr1, Σs12, νΣf2, Σf2, Σc2, Σtr2, 0.0]
- **`crit_Sigma_f` / `branch_Sigma_f` 미사용** — cheating 회피
  - 이들은 시점별 effective XS이며 GT에 종속된 정보 → loss 계산 시 사용 시 누출 우려
  - 시점별 정확도가 필요하지 않음 — bias가 양변에서 cancel되므로
- **rod 효과**: `rod_map_3d`는 입력 채널로 NN에 들어가나, R_CMFD 계산에는 사용 안 함
  - rod로 인한 실제 흡수단면적 증가는 ε_total(t)에 흡수되어 cancel
  - 수학적 근거: φ_GT는 MASTER 시뮬레이션의 effective flux이므로 rod 효과 이미 반영됨

#### 3.6.4 ε_total(t) 사전 계산

학습 시 매 batch마다 R_CMFD(φ_GT)를 다시 계산하는 것은 비효율적이므로 전처리에서 1회 계산 후 저장:

```
Phase G (또는 신규 Phase H) 전처리:
  for each timestep t in dataset:
      ε_total(t) = R_CMFD(φ_GT(t), xs_fuel_BOC)   # (20, 5, 5, 2)
      save to HDF5 as 'cmfd_residual_GT'
```

- 저장 형상: 시점별 (20, 5, 5, 2), φ_GT와 동일
- 저장 비용: 데이터셋 총 크기의 ~5~10% 추가
- 학습 dataloader가 (φ_GT, xs_fuel, **ε_total**) 를 함께 반환

#### 3.6.5 L_diff_rel의 redundancy 분석 — 무엇을 보강하는가

> 본 분석은 논문 작성 시 L_diff_rel 도입 근거의 주춧돌이 되므로 명확히 기록해 둠.

**L_data와 L_diff_rel의 수학적 비교**:

`δ := φ_pred − φ_GT` 라 하면:
```
L_data     = ‖δ‖²                      ← pointwise L² norm
L_diff_rel = ‖R_CMFD(δ)‖² = ‖L δ‖²    ← L = 7점 stencil 행렬 (Laplacian 이산화 + pointwise 항)
```

`L δ` 는 R_CMFD의 leakage 항(공간 미분)과 removal/source 항(pointwise scaling)의 합. 이는 함수 공간에서 **Sobolev seminorm** (특히 H¹/H² 류) 과 동형:
```
‖L δ‖² ≈ ‖−∇·D∇δ + (Σ_r − source/φ)·δ‖² × V²
```
즉 L_diff_rel은 단순 phi 차이가 아니라 **phi의 공간 곡률(Laplacian)에 가중치를 둔 norm**. 이런 부류의 손실 함수를 PINN 문헌에서 **Sobolev regularizer** 라 부름.

**오차 패턴별 민감도**:

| 오차 패턴 | L_data 패널티 | L_diff_rel 패널티 | 비고 |
|---|:---:|:---:|---|
| Uniform constant offset | 1 단위 | (Σ_r − source) × V (작음) | 둘 다 잡음 |
| Smooth slope error | 1 단위 | 약함 | L_data가 주된 신호 |
| **High-frequency oscillation** | 1 단위 | **수십~수백 단위** | **L_diff_rel만 강하게 잡음** |
| Boundary discontinuity | 1 단위 | 큼 | L_diff_rel이 더 강함 |
| Inter-cell incoherence | 1 단위 | 큼 | L_diff_rel이 인접 cell 일관성 강제 |

**Inner cell에서의 Redundancy**:
- L_data가 inner cell 25개를 직접 supervise하고, L_data_halo (λ=0.3) 가 halo 11개도 cover하면, L_diff_rel의 **고유 가치는 inner 영역에서 거의 사라짐**
- 두 손실의 최소점이 동일 (R injective 가정 하)
- 단지 다른 gradient 방향 (Sobolev vs L²) 으로 인한 미세한 optimization 차이만 남음
- 즉 inner cell 자체의 phi 정확도 측면에서는 L_diff_rel은 *대부분 redundant*

**그럼에도 L_diff_rel을 유지하는 이유** — 4가지 고유 가치:

| # | 가치 | 설명 |
|:---:|---|---|
| **1** | **Albedo BC 학습** | §3.5의 12개 학습 가능 파라미터 (α_ortho, α_diag, C_bottom, C_top)는 R_CMFD 안에 내재됨. L_diff_rel 없이는 이 파라미터들에 gradient가 흐르지 않음. **L_diff_rel의 가장 명확한 raison d'être** |
| **2** | **Sobolev / 공간 매끄러움 regularization** | 모델 출력의 high-frequency artifact를 차단. inner cell에서도 노이즈 억제 효과. 유한 데이터셋에서 overfitting 감소 잠재 효과 |
| **3** | **Cross-cell coupling (인접 cell 일관성)** | 7점 stencil이 인접 cell의 phi를 동시에 참조 → 출력 phi 분포가 cell-independent 가 아닌 인접 일관성을 만족하도록 강제. quarter core 대칭 구조의 boundary 영역에서 특히 의미 있음 |
| **4** | **물리적 inductive bias** | 모델에 "이 시스템은 (CMFD 근사 하의) 확산방정식을 만족함"을 알려주는 prior. attention 기반 모델은 본질적으로 cell-permutation invariant에 가까우나 L_diff_rel이 PDE 구조를 주입 |

**구체적 경계조건/대칭조건 정보 흐름**:

L_diff_rel을 통해 모델에 전달되는 정보 중 L_data로는 전달 불가능한 부분:

| 정보 | 전달 경로 (L_diff_rel 안에서) |
|---|---|
| **Albedo BC (반사체면)** | R_CMFD의 face_type 1~4 분기에서 α/C 적용 → 학습 가능 파라미터 학습 |
| **Mirror CMFD (대칭면)** | R_CMFD의 face_type 5 분기에서 ghost neighbor 매핑 → halo cell phi와 inner cell phi의 대칭 정합 강제 |
| **Albedo와 Mirror의 quarter-core 정합성** | ε_total(t) 사전 계산 시 quarter (5,5) 도메인의 BC 정합성이 ε에 인코딩 → 학습 시 그 정합 조건이 L_diff_rel 잔차에 반영 |
| **인접 어셈블리 결합 강도 (D̃ harmonic mean)** | leakage 항의 D̃ 가중치 → 어셈블리 간 결합이 강한 영역과 약한 영역의 차이가 R 안에 인코딩 |

이 4가지 가치 중 **#1 (Albedo BC 학습) 이 L_diff_rel 도입의 가장 명확한 정당화**이며, #2~#4는 부수적 효과. 만약 Albedo BC가 별도 calibration으로 fix되어 학습 대상이 아니라면, L_diff_rel의 한계비용/한계효익 비율은 더 신중히 평가 필요.

#### 3.6.6 Consistency Barrier — 짧은 참조

**참조 논문**: 2026 arXiv "On the Role of Consistency Between Physics and Data in PINNs"

**핵심 주장**: PINN에서 데이터(GT)가 지배방정식을 정확히 만족하지 않을 때, PDE loss는 데이터-PDE 불일치 항 ε에 의해 구조적 하한 오차로 수렴함. PDE만으로는 ε를 극복 못 함.

**우리 문제와의 관계**:
- 우리의 ε = NEM 보정량 (CMFD-only operator로 NNEM 결과를 평가했을 때의 잔차) ≈ g2 median 6.4%
- L_diff_abs를 그대로 쓰면 정확히 이 Consistency Barrier에 갇힘
- **본 안 (physics + data 결합 = L_diff_rel) 이 이를 해소**: ε를 target에서 빼버림으로써 PDE 잔차가 데이터 일치 방향으로만 작동하도록 만듦
- 즉 우리는 physics(R_CMFD operator)와 data(φ_GT)를 곱셈/치환이 아닌 **차분 형식으로 결합**하여 ε에 의한 bias를 cancel

상세한 effective loss 분해 line-by-line 검증은 별도 분석 사안 (필요 시 추후 정밀 분석).

#### 3.6.7 가중치 권고 (개정)

상대 잔차 채택 시 bias가 cancel되므로 weight를 약간 강화 가능:

| Loss | 절대 잔차 (기존) | 상대 잔차 (개정) | 비고 |
|---|:---:|:---:|---|
| L_data (inner) | 1.0 | 1.0 | 변경 없음 |
| L_data_halo | 없음 | **0.3** | 신설 (ML plan 권고 1) |
| L_diffusion | **drop** | — | 절대 잔차 폐기 |
| **L_diff_rel** | — | **0.05~0.1** | 신설. 주된 가치는 Albedo BC 학습 |

L_diff_rel 시작값 0.05 권장:
- inner cell에서는 대부분 redundant이므로 작아도 충분
- Albedo BC 학습에 필요한 minimum 신호 수준
- 학습 안정 후 0.1까지 증가 가능

#### 3.6.8 Optimization 보조 옵션 (rod cell 처리)

L_diff_rel 채택 후 추가로 고려할 optimization 안정화 옵션:

**옵션 A: Rod-aware Σ_a 학습 가능 보정**
```
Σ_a_eff(z, y, x, g) = Σ_a_BOC(z, y, x, g) + α_g(z) × rod_frac(z, y, x)
```
- α_g(z): 채널별 × axial 학습 가능 파라미터 (~40개)
- 효과: rod 삽입 cell에서 R_CMFD가 더 정확 → ε_total(t)의 rod 기여 감소 → optimization landscape 안정
- 비용: 매우 낮음 (수십 파라미터)

**옵션 B: Rod cell L_diff 가중치 감쇠**
```
weight_cell(z, y, x) = exp(−β × rod_frac(z, y, x))
L_diff_rel = Σ_cells weight_cell × ‖R_pred − R_GT‖²
```
- β 시작값 = 1.0, 학습 진행에 따라 점진적 감쇠 (1.0 → 0.3 → 0)
- 효과: rod 삽입 cell의 큰 잔차가 noise로 작용하는 초기 단계에서 학습 안정화
- 비용: 1줄 코드

상세 적용 시점 및 ablation: ML plan Phase 6 참조

---

## 4. L_keff — K-eff Rayleigh 몫 (확장용)

Rayleigh 몫:

$$k_{pred} = \frac{\sum_{v,g}\nu\Sigma_{f,g,v}\hat{\phi}_{g,v}^2}{\sum_{v,g}(D_g|\nabla\hat{\phi}_{g,v}|^2 + \Sigma_{a,g,v}\hat{\phi}_{g,v}^2)}$$

$$\mathcal{L}_{keff} = \left(k_{pred} - k_{GT}\right)^2$$

---

## 5. 가중치 전략

### 5.1 전체 손실 함수 (개정 2026-04-08)

$$\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{data\_halo}\mathcal{L}_{data\_halo} + \lambda_{Bateman}\mathcal{L}_{Bateman} + \lambda_{Taylor}\mathcal{L}_{Taylor} + \lambda_{diff\_rel}\mathcal{L}_{diff\_rel} + \lambda_{keff}\mathcal{L}_{keff}$$

- **L_data_halo 신설**: halo cell phi에 직접 supervision 부과 (`05_symmetry_mode.md` §3.1, ML plan 권고 1)
- **L_diffusion → L_diff_rel 전환**: 절대 잔차 폐기, 상대 잔차로 교체 (§3.6 참조)

### 5.2 Warm-up / Ramp-up 전략

> 이 전략은 PINN(Physics-Informed Neural Network) 문헌에서 표준적 — Mamba 고유가 아님.

**Warm-up**: 학습 초기 에폭에서 physical loss 가중치를 매우 낮게(~0.01) 설정.
- **이유**: 모델이 아직 물리적으로 의미 있는 출력을 내지 못하는 초기 단계에서, physical loss의 gradient가 MSE loss를 방해(gradient conflict)하는 것을 방지.

**Ramp-up**: Warm-up 이후 에폭마다 physical loss 가중치를 선형으로 증가시켜 목표 가중치에 도달.

**Full training**: 목표 가중치로 고정하여 학습 진행.

**가중치 예시** (하이퍼파라미터 튜닝 필요, 확정값 아님):

| 단계 | λ_data | λ_data_halo | λ_Bateman | λ_σXe | λ_diff_rel | λ_keff |
|------|:-----:|:----------:|:---------:|:--------:|:-----------:|:------:|
| Warm-up | 1.0 | 0.3 | 0.01 | 0.01 | 0 | 0 |
| Full training (예시) | 1.0 | 0.3 | 0.5 | 0.3 | 0.05~0.1 | 0.1 |

- λ_data_halo = 0.3: cell 비율 기반 중립값 0.44(=11/25)에서 redundancy 고려한 임의 선택값. ablation 권장 (0.1, 0.3, 0.5)
- λ_diff_rel = 0.05~0.1: 절대 잔차의 0.1보다 낮춘 것은 inner cell에서 redundancy가 높아 작아도 충분하기 때문. 주된 가치는 §3.6.5에 명시된 4가지 (특히 Albedo BC 학습)

> **주의**: 가중치의 합이 1일 필요 없음 — 각 항의 상대적 크기 비율이 중요. 위 값은 참고용 예시이며 실제 학습에서 loss 스케일에 따라 조정 필요.

### 5.3 정규화 주의

N_Xe ~ 10⁻⁷ barn-cm vs φ ~ 10¹³ n/cm²/s → 7~8자릿수 차이.
각 Loss 내부에서 물리량을 노드별 평균으로 정규화 후 계산.

---

## 6. 물리량-데이터 매핑 요약

| Loss 항 | 필요 물리량 | MASTER 가용 | 추가 계산 |
|---------|-----------|:-----------:|:---------:|
| L_data | GT 전체 필드 | ✅ | 없음 |
| L_Bateman | N_Xe, N_I, φ(2군), Σ_f(2군), σ_a^Xe(2군), γ, λ | ✅ | ODE 적분 |
| L_σXe | T̂_f, ρ̂_m (예측), σ_a^Xe(GT), Taylor 계수 | ✅ | 격자 라이브러리 편미분 계수 |
| L_diffusion | φ̂(2군), D_g, Σ_a, νΣ_f, k_GT | ✅ 부분 | ∇²φ 유한차분 |
| L_keff | φ̂, νΣ_f, D_g, Σ_a, k_GT | ✅ | Rayleigh 몫 연산 |

---

## 7. 구현 시 주의사항

1. **수치 안정성**: Bateman 해석해에서 λ_eff ≈ λ_I (즉 σ_a^Xe·φ ≈ 0)일 때 분모 → 0. `tf.where` 또는 ε 클리핑 처리 필수.

2. **2군 처리**: 흡수/생산항은 g1·φ_g1 + g2·φ_g2 합산. 별도 군이 아닌 **합산 반응률** 사용.

3. **Δ 계산의 기준값**: Taylor 전개의 ΔT_m = T_m^현재 − T_{m,0}에서 T_{m,0}은 **MAS_XSL 기준 상태** (파일 고정값). 이전 타임스텝 값이 아님.

4. **Mamba 연동**: Mamba SSM 자체가 선택적 상태공간 모델로 Markov ODE를 자연스럽게 표현 → Bateman ODE residual을 SSM 전이 행렬 초기화에 반영하는 방식도 고려 가능.

5. **L2 vs L∞**: ODE 잔차는 L2가 기본. 국소 플럭스 피킹(power peaking) 구간에서 잔차 급증 시 gradient clipping 고려.

---

## Appendix A. Bateman ODE — Euler vs 해석해 오차 검증

> 검증 시점: 2026-03-25 (E1-T5, E1-T5b)
> 검증 데이터: MASTER GT (MAS_NXS s0001~s0005)
> 결과 파일: `unit_tests/modifying_plan_phase_E/E1_T5_euler_analytic_results.txt`, `E1_T5b_multistep_results.txt`

### A.1 단일 스텝 (s0001→s0002, 2800노드)

| 기법 | mean |상대오차| | max |상대오차| | <1% 비율 |
|------|:---:|:---:|:---:|
| Euler forward | 1.1034% | 4.3632% | 55.3% (1547/2800) |
| Analytic | 1.1017% | 4.3219% | 54.9% (1538/2800) |

Analytic이 Euler보다 정확한 노드: 1370/2800 (48.9%) — **거의 반반**

### A.2 연속 5스텝 (11,200 노드-스텝)

| 기법 | mean | max | <1% | <5% |
|------|:---:|:---:|:---:|:---:|
| Euler | 0.9170% | 4.3632% | 64.5% | **100%** |
| Analytic | 0.9068% | 4.3219% | 65.3% | **100%** |
| Analytic+PC | 0.9052% | 4.3179% | 65.5% | **100%** |

### A.3 대표 노드 비교

```
노드 (11,11,12):
  N_Xe(t) = 2.316930E-09, N_I(t) = 5.638000E-09
  Euler:    2.304094E-09  (오차 1.2551%)
  Analytic: 2.304321E-09  (오차 1.2454%)
  MASTER GT: 2.333380E-09
```

### A.4 결론

1. **Euler vs Analytic 차이: mean 0.02%p, max 0.04%p** — Δt=5min에서 실질적으로 동등
   - Δt/τ_Xe ≈ 300/47600 ≈ 0.006, 1차 절단 오차 O(Δt²) ≈ 0.09%
2. **오차의 주요 원인은 MASTER Full Predictor-Corrector** 미반영 (매뉴얼 확인: 항상 Full PC 사용)
   - GT φ(t)만 사용하므로 MASTER 내부 φ 갱신 효과가 미반영됨
3. **Physical Loss 적분 방법: 둘 다 허용**
   - Euler: 단순 (1줄), 해석해: tf.exp 기반 (5줄)

---

## Appendix B. σ_a^Xe 검증 상세 (E0-d/e/f, E1)

> 원본: `2026-03-25_sigma_a_xe_검증.md` 전체

### B.1 검증 대상 4대 논점

| 논점 | 내용 | 결과 |
|------|------|------|
| A | Bateman 소멸항 σ·φ·N = Σ·φ 약분 유효성 | ✅ 학습 시 항등, 추론 시 Σ·φ 직접 사용 가능 |
| B | MAS_NXS ABS-XEN 단위 (barn vs /cm) | ✅ barn (미시) — E1-T1/T3/E3 삼중 검증 |
| C | 노드→어셈블리 집계 오차 | ✅ 중앙 <0.01%, 모서리 최대 2.6% (방법1 vs 방법2) |
| D | MASTER 내부 집계 방식 | ✅ `$XESM3D` = 노드 단순 평균 (max 0.044%) |

### B.2 E0-d 결과: MASTER 어셈블리 집계

`iprcon=2` + `inxs=1` 동시 실행:

| 위치 | `$XESM3D` | MAS_NXS ⟨N_Xe⟩ | 차이 |
|------|-----------|-----------------|:---:|
| center K=12, 중앙 (5,E) | 2.317E-9 | 2.317E-9 | 0.003% |
| center K=12, 모서리 (1,D) | 1.486E-9 | 1.486E-9 | 0.011% |

전체 1140개: max 0.044%, mean 0.013%

### B.3 E0-e 결과: Σ_a^Xe 어셈블리 출력 불가

- `$XS3D`의 `ABS` = 총 흡수 Σ_a^total (Xe 전용 아님)
- Σ_a^Xe는 MAS_NXS `ABS-XEN` × `DEN-XEN`으로만 산출 가능 (노드 단위)

### B.4 E0-f 결론: σ_a^Xe 역산 불필요

$$\sigma_a^{Xe} \cdot \phi \cdot N_{Xe} = \Sigma_a^{Xe} \cdot \phi$$

N_Xe 약분 → **Bateman ODE에서 σ_a^Xe 자체 불필요** (소멸항이 N_Xe에 비의존).
단, Taylor Loss에서는 σ_a^Xe 필요 → HDF5에 barn 단위 저장 유지.

### B.5 E1 단위 확정 경로

1. **E1-T1**: ABS-XEN(7.19) > XABS(0.009) → 거시 기각
2. **E1-T3**: σ×N 재구성 → 총 흡수의 3~3.5% (g=2) → 물리적 타당
3. **E3**: MAS_XSL Taylor σ vs ABS-XEN → 0.25%/0.92% 일치
4. **E1-T5**: Euler forward 전수 검증 → 2800노드 100% < 5% → 단위 확정

### B.6 E1-T5b 연속 5스텝

| 구간 | Euler mean | Euler max | Analytic mean | Analytic max | <5% |
|------|:---:|:---:|:---:|:---:|:---:|
| s0001→s0002 | 1.10% | 4.36% | 1.10% | 4.32% | 100% |
| s0002→s0003 | 0.85% | 4.16% | 0.84% | 4.12% | 100% |
| s0003→s0004 | 0.85% | 4.19% | 0.84% | 4.15% | 100% |
| s0004→s0005 | 0.86% | 4.22% | 0.84% | 4.18% | 100% |
| **전체 11,200** | **0.92%** | **4.36%** | **0.91%** | **4.32%** | **100%** |

3기법(Euler/Analytic/PC) 모두 ~0.9%로 수렴 → **적분법 무관 바닥 오차**. 원인: MASTER 내부 수송+열수력 재계산(PC) 미반영.
