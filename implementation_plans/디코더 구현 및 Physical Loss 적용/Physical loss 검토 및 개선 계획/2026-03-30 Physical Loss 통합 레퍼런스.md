# Physical Loss 통합 레퍼런스

> **작성일**: 2026-03-30
> **개정일**: 2026-04-08 — §3.6 신설 (L_diff 상대 잔차 형식 + redundancy 분석 + Consistency Barrier 참조), §5.1 전체 손실 함수 공식에 L_data_halo / L_diff_rel 반영.
> **개정일**: 2026-04-09 — 공간 형상 표기에 단계 구분 추가 (데이터 입력 vs 인코더 처리 vs L_diff 합산 도메인)
> **개정일**: 2026-04-20 — §0.2/§0.3 정정 (branch shape T,30 + dtype float32 + Frozen Xenon 표기 → 5분 진화 반영), §1 (L_Bateman) 상단에 Phase 1 적용 가능성 검증 결과 링크 추가
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
4. [L_keff — K-eff Rayleigh 몫 (2026-04-22 폐기, 3D 부적합)](#4-l_keff--k-eff-rayleigh-몫-폐기)
5. [L_pload — 출력 총량 self-consistency (2026-04-22 초기 누락 정정)](#5-l_pload--출력-총량-self-consistency)
6. [L_AO — 축방향 출력 편차 consistency (2026-04-22 초기 누락 정정)](#6-l_ao--축방향-출력-편차-consistency)
7. [가중치 전략](#7-가중치-전략)
8. [물리량-데이터 매핑 요약](#8-물리량-데이터-매핑-요약)
9. [구현 시 주의사항](#9-구현-시-주의사항)
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
| `critical_xe` | (T,Z,qH,qW) | float64 | N_Xe 수밀도 | #/barn-cm | `DEN-XEN` | L_data (모델 state ch 4), L_Bateman (초기 조건 GT) |
| `critical_sm` | (T,Z,qH,qW) | float64 | N_Sm 수밀도 | #/barn-cm | `DEN-SAM` | — (state 제외) |
| `critical_i135` | (T,Z,qH,qW) | float64 | N_I 수밀도 | #/barn-cm | MAS_OUT `$NUCL3D` | L_data (모델 state ch 5), L_Bateman (초기 조건 GT) |
| `critical_sigma_a_xe` | (T,Z,qH,qW,2) | float64 | σ_a^Xe 미시 흡수단면적 (2군) | barn | `ABS-XEN` | **L_data 배제 (Option D, 2026-04-22)** — L_σXe (Taylor 공식이 σ̂ 미사용, T̂_f·ρ̂_m 로 재조립), L_Bateman (예측값 입력 — σ̂ direct gradient 경로) |
| `critical_flux` | (T,Z,qH,qW,2) | float64 | φ 중성자속 (2군) | n/cm²/s | `FLX` | L_data (모델 state ch 6, 7), L_Bateman (예측값 입력) |
| `critical_Sigma_f` | (T,Z,qH,qW,2) | float64 | Σ_f 핵분열 거시단면적 (2군) | /cm | `FIS` | **L_data_Sigma_f (모델 state ch 10, 11, 2026-04-22 신설), L_Bateman (예측값 입력)** |
| `critical_yield_xe` | (T,Z,qH,qW) | float64 | γ_Xe 핵분열 수율 | 무차원 | `FYLD-XE135` | L_Bateman (GT 시계열 입력, state 부재) |
| `critical_yield_i` | (T,Z,qH,qW) | float64 | γ_I 핵분열 수율 | 무차원 | `FYLD-I135` | L_Bateman (GT 시계열 입력, state 부재) |

**쿼리 (입력 조건)**:

| HDF5 필드 | Shape | dtype | 물리량 | 단위 |
|-----------|-------|-------|--------|------|
| `query_rod_map_3d` | (T,31,Z,qH,qW) | float32 | 제어봉 3D 삽입 분율 맵 | 0~1 |
| `query_rod_offsets_1d` | (T,31) | float32 | 1D rod offset | step |
| `query_pload` | (T,) | float32 | 목표 출력 수준 | 0~1 |

**해석 결과 (GT 타겟, 30-way: b=0=critical, b=1~29=rod_offset 분기)**:

> **2026-04-20 정정**: 이전 표기 "31-way: index 0=critical, 1~30=branch"은 실제 데이터와 불일치. 실제 second axis는 30 (b=0~29).

| HDF5 필드 | Shape | dtype | 물리량 | 단위 | Loss 사용 |
|-----------|-------|-------|--------|------|:---------:|
| `result_power` | (T,30,Z,qH,qW) | float32 | 절대 열출력 | MW/node | L_data |
| `result_tcool` | (T,30,Z,qH,qW) | float32 | 냉각재 온도 | °C | L_data |
| `result_tfuel` | (T,30,Z,qH,qW) | float32 | 연료 온도 | °C | L_data, L_σXe(예측) |
| `result_rhocool` | (T,30,Z,qH,qW) | float32 | 냉각재 밀도 | g/cc | L_data, L_σXe(예측) |
| `result_keff` | (T,30) | float64 | 유효증배계수 | — | L_data 만 (2026-04-22 L_keff physics loss 폐기 — 3D 문제에 부적합, §4 참조) |
| `result_ao` | (T,30) | float32 | 축방향 출력 편차 | — | **L_AO 만 (Option D 확장, 2026-04-23)** — AO 는 power 3D 의 상부/하부 절반 비율 로 유도 (derived), 직접 예측 헤드 부재, L_data direct fit 배제 |
| `result_max_pin_power` | (T,30) | float32 | 핀 출력 peaking factor | — | — |
| `result_max_pin_loc` | (T,30,3) | int32 | 핀 피크 위치 (z,y,x) | — | — |

**Branch 핵종 데이터 (30-way: b=0=critical, b=1~29=rod_offset 분기)**:

> **2026-04-20 정정**: 이전 표기 "31-way, dtype float64, Frozen Xenon"은 실제 데이터와 불일치하여 정정.
> 실제 데이터: shape (T, **30**, ...), dtype **float32**, b=0이 critical 자체이며 b=1~29가 rod_offset 변경 + 5분 연소 진화 결과 (frozen 아님).
> 검증: [piecewise-test/2026-04-20_branch_xenon_evolution_결과.md](../../../piecewise-test/2026-04-20_branch_xenon_evolution_결과.md), 관련 결정: [2026-04-20 Branch Xenon 진화 검증.md](../2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md)

| HDF5 필드 | Shape | dtype | 물리량 | Loss 사용 | 비고 |
|-----------|-------|-------|--------|----------|------|
| `branch_xe` | (T,30,Z,qH,qW) | float32 | N_Xe | L_data, L_Bateman (초기 조건 GT) | b=0=critical, b=1~29=rod_offset 분기 + **5분 진화 반영** (검증 ~1% 변화, frozen 아님) |
| `branch_sm` | (T,30,Z,qH,qW) | float32 | N_Sm | — (state 제외) | b=0=critical, b=1~29=분기 |
| `branch_i135` | (T,30,Z,qH,qW) | float32 | N_I | L_data, L_Bateman (초기 조건 GT) | b=0=critical, b=1~29=분기 |
| `branch_sigma_a_xe` | (T,30,Z,qH,qW,2) | float32 | σ_a^Xe (2군) | **L_data 배제 (Option D, 2026-04-22)**, L_σXe, L_Bateman (예측값 입력) | branch 조건(T_f, ρ_m 변동)에 따라 미세 차이 |
| `branch_flux` | (T,30,Z,qH,qW,2) | float32 | φ (2군) | L_data, L_Bateman (예측값 입력) | |
| `branch_Sigma_f` | (T,30,Z,qH,qW,2) | float32 | Σ_f (2군) | **L_data_Sigma_f (신설), L_Bateman (예측값 입력)** | 2026-04-22 state ch 10, 11 신설 |
| `branch_yield_xe` | (T,30,Z,qH,qW) | float32 | γ_Xe | L_Bateman (GT 시계열 입력) | state 부재 |
| `branch_yield_i` | (T,30,Z,qH,qW) | float32 | γ_I | L_Bateman (GT 시계열 입력) | state 부재 |

> **node_fullcore 데이터** (분석/참조용): 위 critical/branch 필드 중 일부가 quarter crop 전 풀코어(Z,18,18) 형태로도 저장됨. 모델 학습에는 사용하지 않음.

> **dtype**: 전처리 코드 전체를 **float32로 통일 완료** (2026-03-30). MASTER 원본 출력의 유효숫자(4~6자리)상 float32(~7자리)로 충분하며 정밀도 손실 없음. normalizer 내부 연산(Welford 누적, log/pcm 변환)만 float64 유지 (수치 안정성), 최종 통계 출력은 float32. **기존 HDF5 파일은 재생성 필요.**

### 0.3 차원 정의

| 기호 | 값 | 의미 |
|:----:|:---:|------|
| T | 576 | 시계열 스텝 수 (48h ÷ 5min) |
| Z | 20 | 축방향 연료 평면 (반사체 K=1, K=22 제외) |
| qH, qW | 5, 5 | 1/4 대칭 크롭 후 반경 방향 |
| 2 | 2 | 에너지군 (g=1: fast, g=2: thermal) |
| 30 | 30 | Branch 필드의 두 번째 축 — b=0=critical 자체, b=1~29=rod_offset 분기 (실제 분기는 29개). critical 필드는 별도 (T, ...) shape로 저장됨. **2026-04-20 정정**: 기존 표기 "31 = CRS(1) + Branch(30)"은 데이터 구조와 불일치 |
| 10 | 10 | xs_fuel 채널 수 (νΣf1, Σf1, Σc1, Σtr1, Σs12, νΣf2, Σf2, Σc2, Σtr2, Σs21) |

---

## 1. L_Bateman — Xe/I-135 Bateman ODE 잔차

> **2026-04-20 추가 검증**: Branch 데이터의 Xe 진화 가능성 검증 완료. Branch는 frozen Xenon이 아니라 5분 시간 진화를 정확히 반영하므로, **L_Bateman을 Phase 1 (Branch 단일 step) 학습에서도 적용 가능**.
> 검증 상세: [L_Bateman/2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md](L_Bateman/2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md)
> 검증 코드/결과: [piecewise-test/2026-04-20_branch_xenon_evolution_결과.md](../../../../piecewise-test/2026-04-20_branch_xenon_evolution_결과.md)

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

| 카테고리 | 변수 | Shape | 모델 state 포함 | 마르코프 필요 이유 |
|----------|------|-------|:---:|--------------------|
| 독물질 | N_Xe(t) | (Z,qH,qW) | ✅ ch 4 | Eq.2 직접 상태변수 |
| 독물질 | N_I(t) | (Z,qH,qW) | ✅ ch 5 | Xe 간접 생성원 |
| 중성자속 | φ(t) | (Z,qH,qW,2) | ✅ ch 6, 7 | 생성항·소멸항 계수 (2군) |
| 단면적 | Σ_f(t) | (Z,qH,qW,2) | **✅ ch 10, 11 (2026-04-22 신설)** | 핵분열 생산항 (2군) |
| 단면적 | σ_a^Xe(t) | (Z,qH,qW,2) | ✅ ch 8, 9 | Xe 소멸항 계수 (2군). MASTER에서 Σ_a^Xe(거시) 직접 출력 불가 → σ [barn] × N으로 계산 |
| 열수력 | T_fuel(t) | (Z,qH,qW) | ✅ ch 2 | Doppler → σ_a^Xe 보정 |
| 열수력 | ρ_mod(t) | (Z,qH,qW) | ✅ ch 3 | 감속능 → 스펙트럼 → σ_a^Xe |
| 연소도 | Bu | (Z,qH,qW) | ❌ (BOC 고정, Phase F 확장) | σ_a^Xe 보간 기준축 |
| 제어봉 | rod_map(t) | (Z,qH,qW) | ❌ (쿼리 입력) | 국소 중성자속 분포 변화 |

> **2026-04-22 갱신**: Σ_f 를 모델 state 에 추가 (ch 10, 11 신설) — L_Bateman 에 예측 물리량 사용 결정에 따라. 자세한 내용은 §1.9, §1.10 참조.

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

### 1.9 손실 정의 — A+B Hybrid (2026-04-22 최종 확정)

#### 1.9.1 수식

$$\mathcal{L}_{Bateman} = \lambda_A \cdot \mathcal{L}_A + \lambda_B \cdot \mathcal{L}_B$$

$$\mathcal{L}_A = \frac{1}{B V} \sum_{b,v} \left[\left(\tilde{\hat{N}}_{Xe} - \tilde{N}_{Xe}^{phys}\right)^2 + \left(\tilde{\hat{N}}_I - \tilde{N}_I^{phys}\right)^2\right]$$

$$\mathcal{L}_B = \frac{1}{B V} \sum_{b,v} \left[\left(\tilde{N}_{Xe}^{phys} - \tilde{N}_{Xe}^{GT}(t\!+\!1)\right)^2 + \left(\tilde{N}_I^{phys} - \tilde{N}_I^{GT}(t\!+\!1)\right)^2\right]$$

- **모든 항 정규화 공간 (z-score) 에서 계산** (§1.9.3 참조) — tilde (~) 기호 는 정규화값 표기
- **A 항**: N̂ 에 physics constraint (외삽 / covariate shift 대응)
- **B 항**: 예측 계수 (φ̂, σ̂, Σ̂_f) 에 GT anchor (model misspecification 보정)
- λ_A, λ_B: 동적 가중치 — §1.9.4 참조

| 변수 | 설명 |
|------|------|
| N̂_Xe, N̂_I (ch 4, 5) | 모델 직접 예측 at t+1 |
| N^phys | Bateman ODE 적분 결과 (GT 초기조건 + 예측 계수 φ̂/σ̂/Σ̂_f + GT γ + 상수 λ, 아래 입력 정책 참조) |
| N^GT(t+1) | MASTER GT (관측 타겟) |
| B, V | batch size, voxels (연료 영역 노드 수, fuel_mask 적용 후) |

Bateman ODE 는 **노드 로컬 방정식** (공간 결합 없음) → 셀별 독립 계산 유효.

#### 1.9.2 A+B Hybrid 채택 근거 (2026-04-22 최종)

초기 설계 (A 단독) → Path B 검토 (`MSE(N_phys, N_GT)`) → **A+B Hybrid 확정**. A/B 단독 안 모두 철회.

**A+B 가 단독 안 대비 우월한 이유**:

| # | 관점 | Hybrid 이점 |
|---|------|-------------|
| 1 | **Labeled 정보 활용** | B 항 이 N_GT(t+1) 의 "공짜 정보" 를 gradient 로 직접 활용. A 단독 은 내부 consistency 만 강제 (약한 신호) |
| 2 | **Model misspecification 보정 (결정적)** | Bateman 공식 은 근사 — 공간 비균질성 / 고차 항 / 근사 λ / Full PC 부재 (~0.9%) 등. A 단독 은 공식 맹신 → 공식 오차 가 예측 에 직접 전이. B 항 이 GT 와 직접 비교 하여 공식 bias 흡수 |
| 3 | **외삽 오차 출처 별 역할 분담** | (a) Model misspecification → **B 필수**. (b) Covariate shift (GT 부재 영역) → **A 필수**. 실제 데이터 는 (a)+(b) 동시 존재 |
| 4 | **PINN 실증** | Karniadakis et al. 2021 (Nat. Rev. Phys.) 및 다수 benchmark: hybrid (data + physics) > pure physics, 외삽 성능 일관 우위 |

**단독 안 의 한계**:
- **A 단독**: 공식 맹신, 0.9% 구조 오차 가 N̂ 에 직접 전이, labeled 정보 미활용
- **B 단독**: GT 없는 영역 에서 N̂ 에 physics constraint 없음 → OOD dynamics drift

**상호 보완 구조**:
- 두 항 공동 최소점: N̂ ≈ N_phys ≈ N_GT — 공식 consistency + data fidelity 동시 만족
- N̂ 경로: A (direct) + L_data (direct) 중복 anchor
- 예측 계수 경로: B (ODE 경유, 강) + A (ODE 경유, 약) + L_data (채널 별 direct) 중복 anchor
- 과제약 (over-determined) → robustness 상승

#### 1.9.3 Normalization 처리 — **방법 1 (정규화 공간 loss) 채택**

**문제 배경**:
- N_Xe 스케일 ~10^15 #/barn-cm → z-score μ ~10^15, σ ~10^14
- 원 공간 에서 loss 계산 시: `(N̂ - N_phys)² ~ (σ·Δ)² ~ 10^28` → 다른 loss (state loss 는 정규화 공간 O(1)) 와 **28 자릿수 차이** → 합산 의미 상실
- Chain rule: ∂L/∂θ = 2(N̂ - N_phys)·σ·∂Ñ̂/∂θ — σ 인자 로 gradient 폭주

**방법 1 절차 (원칙)**:

```
1. 모델 출력 (정규화 공간):  Ñ̂, φ̂_tilde, Ñ_I, σ̂_tilde, Σ̂_f_tilde
2. 역변환 (원 공간):          N̂ = Ñ̂·σ_N + μ_N,  φ̂, N̂_I, σ̂, Σ̂_f (각 normalizer 적용)
3. Bateman 계산 (원 공간):   N^phys = bateman(N_xe_GT(t), N_I_GT(t), γ_GT, φ̂, σ̂, Σ̂_f)
4. 재정규화:                Ñ^phys_Xe = (N^phys_Xe - μ_N)/σ_N,  Ñ^phys_I = (N^phys_I - μ_I)/σ_I
5. Loss (정규화 공간):       L_A = MSE(Ñ̂, Ñ^phys),  L_B = MSE(Ñ^phys, Ñ^GT(t+1))
```

**효과**: 모든 loss 항 O(1) 스케일 → λ 가중치 튜닝 용이 / gradient 불균형 해소.

**방법 2 (상대오차, 참고)**: `L = ((N̂ - N_phys)/N_ref)²` — 물리량 자릿수 가 시점별 크게 변할 때 대안. 본 프로젝트 는 **방법 1 채택**.

#### 1.9.4 가중치 운용 — λ_A > λ_B (rollout 시)

**Phase 별 기본값**:

| Phase | λ_state (L_data) | λ_A | λ_B | 비고 |
|-------|:---:|:---:|:---:|------|
| 1a (L_data only) | 1.0 | — | — | L_Bateman 비활성 |
| 1b (annealing) | 1.0 | 0 → 0.3 | 0 → 0.1 | linear ramp-up over **Phase 1 epoch 의 10%** (예: Phase 1 = 50 epochs → annealing = 5 epochs, Phase 1 = 100 → 10 epochs). 전체 학습 epoch 변화 에 비례 자동 조정 (2026-04-23 ratio-based 결정) |
| 2a/2b (1-step, teacher forcing) | 1.0 | 0.3 | 0.1 | ReLoBRaLo 활성 시 adaptive (Phase 2b 중반 가능) |
| 3 (K-curriculum, rollout) | **1.0** | **0.3** | **0.1** | **λ_s ≳ λ_A > λ_B** 원칙 유지. **K-step 1/K 평균화 적용** (2026-04-23 결정): `L_Bateman_total = (1/K)·Σ_k L_Bateman(step_k)` — K 와 무관 하게 L_data 와 동일 스케일 유지. 다른 시간 의존 loss (L_data, L_σXe, L_diff_rel, L_pload, L_AO) 도 동일 1/K 평균. **ReLoBRaLo 그룹 단위 활성** (7 개 항, 계층 별, Phase 2b 중반 부터) |

**λ_A > λ_B 근거**:
- Rollout / 외삽 시 **A 항 의 self-consistency 가 dynamics stability 핵심** — N̂ 이 공식 을 만족해야 multi-step feedback 안정
- B 항 은 "데이터 보정" 역할 — training distribution 내 bias 제거 기능이 주. 과도 하면 OOD 에서 무효
- 전형적 수치: **λ_s = 1.0, λ_A = 0.3, λ_B = 0.1** (경험적 시작값, ReLoBRaLo 로 adaptive 조정)

**실무 체크리스트** (학습 초기):
1. **각 loss 항 크기 로깅**: L_data / L_A / L_B / L_σXe / L_diff_rel / L_pload / L_AO 를 별도 스칼라 로 찍어 자릿수 비교. 한 항 이 다른 항 대비 10^5 이상 크면 정규화 재설계.
2. **Gradient L2 norm 비교**: 각 loss 가 생성 하는 gradient norm — 한 쪽 압도 시 스케일 문제.
3. **초기 loss 추이**: L_A 만 급락 / 다른 정체 → 스케일 불균형 신호.

**ODE 입력 정책 (2026-04-22 L_Bateman 에 예측 물리량 사용 결정)**:

| 입력 변수 | 출처 | 비고 |
|---------|------|------|
| N_xe(t), N_I(t) | **GT** (초기 조건, t 시점) | Phase 1 은 Branch GT, Phase 2+ 는 CRS/Branch 혼합 sampling |
| φ(t) (2군) | **모델 예측값** | state ch 6, 7 |
| σ_a^Xe(t) (2군) | **모델 예측값** | state ch 8, 9 |
| Σ_f(t) (2군) | **모델 예측값** | state ch 10, 11 (2026-04-22 신설) |
| γ_Xe(t), γ_I(t) | **GT** (시계열) | state 부재, MASTER HDF5 시계열 값 사용 |
| λ_Xe, λ_I, Δt | 물리 상수 | §1.2 참조 |

**근거**:
- Phase 1a 의 L_data-only 수렴 (§4 학습 방법론 DG-PINN 2-stage) 이 초기 data anchor 제공
- Phase 1b 에서 L_Bateman 도입 시점에 모델 예측이 이미 합리적 값 → ODE 입력으로 예측 물리량 사용 시 trivial drift 위험 완화
- 예측값을 L_Bateman 에 주입함으로써 N_xe, N_I 뿐 아니라 φ, σ_a^Xe, Σ_f 예측 에도 물리 일관성 gradient signal 제공

**Trivial drift 완화책** (다중 GT anchor):
- L_data (N_xe, N_I, φ, σ_a^Xe 각 채널의 GT fit) — 가장 강력한 anchor
- L_data_Sigma_f (Σ_f 의 GT fit, 2026-04-22 신설)
- L_σXe (σ_a^Xe 의 Taylor 일관성 — 추가 물리 제약)
- L_diff_rel (φ 의 공간 curvature — 추가 물리 제약)

→ 예측값이 Bateman 을 "속이도록" drift 하려면 위 anchor 들을 동시에 위배해야 함.

### 1.10 TF2 구현 패턴 — A+B Hybrid + 정규화 공간 Loss

**2026-04-22 갱신**: A+B Hybrid 수식 (§1.9.1) + 방법 1 정규화 공간 loss (§1.9.3) + λ_A > λ_B 가중치 (§1.9.4) 반영.

#### 1.10.1 Bateman ODE 적분기 (원 공간 계산)

```python
import tensorflow as tf

# 물리 상수만 하드코딩
LAMBDA_I  = 2.878e-5  # s^-1
LAMBDA_XE = 2.092e-5  # s^-1
DT = 300.0            # 초 (5분)

def bateman_analytical(
    # GT 입력 (HDF5 로부터 DataLoader 가 제공)
    N_xe_GT, N_I_GT,          # 초기 조건 (t 시점, 원 공간)
    gamma_I_GT, gamma_Xe_GT,  # 핵분열 수율 시계열 GT (critical_yield_*/branch_yield_*)
    # 모델 예측값 (원 공간, 역정규화 완료된 상태)
    phi_pred, sigma_a_xe_pred, Sigma_f_pred,  # (Z,qH,qW,2) 2-group
):
    """해석해 기반 Bateman ODE 적분기 (Δt 내 계수 고정 가정).

    2026-04-22 A+B Hybrid + 방법 1 (정규화 공간 loss):
    - 모든 입력 은 **원 공간** (역정규화 완료) 값
    - 출력 N_phys(t+1) 은 원 공간 값 → 호출 측 에서 재정규화 후 loss 계산

    입력 정책:
    - 초기조건 N_xe_GT, N_I_GT: GT (t 시점)
    - 계수 φ_pred, σ_a^Xe_pred, Σ_f_pred: 모델 예측값 (state ch 6~11, Σ_f 2ch 신설)
    - γ_I_GT, γ_Xe_GT: HDF5 시계열 GT (state 부재, 보조 입력)
    - λ, DT: 물리 상수

    출력: N_Xe_phys(t+1), N_I_phys(t+1) — A+B Hybrid loss 계산 의 공통 중간 산물
    """
    # 2-group 핵분열 생산항 (Σ_f·φ 합산 + barn-cm 단위 변환 1e-24)
    Sigma_f_phi = tf.reduce_sum(Sigma_f_pred * phi_pred, axis=-1)  # (...,Z,qH,qW)
    S_I   = gamma_I_GT  * Sigma_f_phi * 1e-24
    S_Xe  = gamma_Xe_GT * Sigma_f_phi * 1e-24

    # 2-group Xe 중성자 흡수항 (σ_a·φ 합산 + 단위 변환)
    sigma_phi = tf.reduce_sum(sigma_a_xe_pred * phi_pred, axis=-1) * 1e-24
    lambda_eff = LAMBDA_XE + sigma_phi  # Xe 유효 소멸 상수

    # I-135 해석해
    exp_I   = tf.exp(-LAMBDA_I * DT)
    N_I_next = N_I_GT * exp_I + (S_I / LAMBDA_I) * (1.0 - exp_I)

    # Xe-135 해석해 (I-135 붕괴항 포함)
    exp_eff = tf.exp(-lambda_eff * DT)
    denom   = lambda_eff - LAMBDA_I + 1e-30  # 분모 0 방지
    N_xe_next = (
        N_xe_GT * exp_eff
        + S_Xe / lambda_eff * (1.0 - exp_eff)
        + LAMBDA_I * (N_I_GT - S_I / LAMBDA_I) / denom * (exp_I - exp_eff)
    )
    return N_xe_next, N_I_next
```

#### 1.10.2 A+B Hybrid Loss (정규화 공간)

```python
def l_bateman_hybrid(
    # 모델 출력 (정규화 공간)
    N_xe_hat_tilde, N_I_hat_tilde,        # 모델 직접 예측 N̂ (state ch 4, 5)
    phi_hat_tilde, sigma_hat_tilde, Sigma_f_hat_tilde,  # 예측 계수 (정규화 공간)
    # GT (원 공간, DataLoader 제공)
    N_xe_GT_t, N_I_GT_t,                  # 초기 조건 N(t)
    N_xe_GT_next, N_I_GT_next,            # 관측 타겟 N(t+1)
    gamma_I_GT, gamma_Xe_GT,
    # Normalizer (각 변수 별 μ, σ)
    norm,                                  # dict: {'N_xe': (μ,σ), 'N_I': (μ,σ), 'phi': ..., ...}
    fuel_mask,                             # (Z,qH,qW) 0/1
    # 가중치
    lambda_A=0.3, lambda_B=0.1,
):
    """A+B Hybrid Bateman loss, 정규화 공간 에서 계산.

    절차 (방법 1):
    1. 예측 계수 역정규화 (원 공간)
    2. Bateman ODE 적분 → N_phys (원 공간)
    3. N_phys 재정규화 → Ñ_phys (정규화 공간)
    4. L_A = MSE(Ñ̂, Ñ_phys),  L_B = MSE(Ñ_phys, Ñ_GT_next)
    """
    # Step 1: 예측 계수 역정규화 (Bateman 은 원 공간 에서 적분)
    phi_pred         = denormalize(phi_hat_tilde,         norm['phi'])
    sigma_a_xe_pred  = denormalize(sigma_hat_tilde,       norm['sigma_a_xe'])
    Sigma_f_pred     = denormalize(Sigma_f_hat_tilde,     norm['Sigma_f'])

    # Step 2: Bateman 적분 (원 공간)
    N_xe_phys, N_I_phys = bateman_analytical(
        N_xe_GT_t, N_I_GT_t, gamma_I_GT, gamma_Xe_GT,
        phi_pred, sigma_a_xe_pred, Sigma_f_pred,
    )

    # Step 3: 재정규화 (loss 계산 공간 통일)
    N_xe_phys_tilde = normalize(N_xe_phys, norm['N_xe'])
    N_I_phys_tilde  = normalize(N_I_phys,  norm['N_I'])
    N_xe_GT_next_tilde = normalize(N_xe_GT_next, norm['N_xe'])
    N_I_GT_next_tilde  = normalize(N_I_GT_next,  norm['N_I'])

    # Step 4: A+B loss (정규화 공간, fuel_mask 적용)
    def masked_mse(a, b):
        sq = tf.square(a - b) * fuel_mask[..., None] if a.shape.rank > fuel_mask.shape.rank else tf.square(a - b) * fuel_mask
        return tf.reduce_sum(sq) / (tf.reduce_sum(fuel_mask) + 1e-12)

    L_A = masked_mse(N_xe_hat_tilde, N_xe_phys_tilde) + masked_mse(N_I_hat_tilde, N_I_phys_tilde)
    L_B = masked_mse(N_xe_phys_tilde, N_xe_GT_next_tilde) + masked_mse(N_I_phys_tilde, N_I_GT_next_tilde)

    L_bateman = lambda_A * L_A + lambda_B * L_B
    return L_bateman, {'L_A': L_A, 'L_B': L_B}   # 모니터링 용 반환
```

#### 1.10.3 주요 변경 (2026-04-22)

1. **A+B Hybrid** — L_A (N̂ ↔ N_phys) + L_B (N_phys ↔ N_GT) 두 항 결합
2. **정규화 공간 loss (방법 1)** — Bateman 만 원 공간, loss 는 모두 정규화 공간 (O(1) 스케일 통일)
3. **호출 인자 분리**: `_GT` suffix (GT 입력) vs `_hat_tilde` suffix (모델 예측 정규화) vs `_pred` (역정규화 후)
4. **`GAMMA_I`, `GAMMA_XE` 상수 제거** → `gamma_I_GT`, `gamma_Xe_GT` 시계열 인자화 (§1.3 결정 일관)
5. **2-group 합산 로직 명시**: `tf.reduce_sum(Sigma_f_pred * phi_pred, axis=-1)`
6. **기본 가중치** λ_A=0.3, λ_B=0.1 (§1.9.4 rollout 권장치, ReLoBRaLo 활성 시 adaptive)
7. **모니터링 반환값** `{'L_A', 'L_B'}` — §1.9.4 실무 체크리스트 (각 항 크기 로깅) 지원

**추론 시 주의사항**:
- L_Bateman 은 **학습 시간 loss 만** — 추론 forward 에는 호출 안 함
- 만약 추론 시 검증 지표로 L_Bateman 계산 할 경우: γ 는 **대표값 (GAMMA_I=0.0639, GAMMA_XE=0.00228) 사용** (§1.3 방침)

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

### 2.5 TF2 구현 패턴 — 역정규화-재정규화 (2026-04-23 갱신, 방법 1 일괄 적용)

#### 2.5.1 Taylor 공식 (원 공간)

```python
def taylor_sigma_xe(T_f_pred, rho_m_pred, sigma0, dsigma_dTf, dsigma_dM,
                    T_f_ref, rho_m_ref):
    """1차 Taylor 전개 기반 sigma_a^Xe 재계산.

    입력 은 모두 **원 공간** (역정규화 완료 된 물리 단위):
    - T_f_pred: °C
    - rho_m_pred: g/cc
    - sigma0, dsigma_dTf, dsigma_dM: MAS_XSL 계수 (barn, barn/√K, barn/(g/cc))

    출력: σ_Taylor (barn, 원 공간) — 호출 측 에서 재정규화 후 loss 계산
    """
    delta_sqTf = tf.sqrt(T_f_pred + 273.15) - tf.sqrt(T_f_ref + 273.15)
    delta_rho  = rho_m_pred - rho_m_ref
    return sigma0 + dsigma_dTf * delta_sqTf + dsigma_dM * delta_rho
```

#### 2.5.2 L_σXe Loss (정규화 공간)

```python
def l_sigma_xe(
    # 모델 출력 (정규화 공간)
    T_f_hat_tilde, rho_m_hat_tilde,       # state ch 2, 3
    # GT (원 공간, DataLoader 제공)
    sigma_a_xe_GT,                         # MAS_NXS ABS-XEN (barn, 원 공간)
    # Taylor 계수 (HDF5 fixed/taylor_xe35/, 원 공간)
    sigma0, dsigma_dTf, dsigma_dM,
    T_f_ref, rho_m_ref,
    # Normalizer
    norm,                                  # dict: {'T_f': (μ,σ), 'rho_m': (μ,σ), 'sigma_a_xe': (μ,σ)}
    fuel_mask,
):
    """L_σXe, 방법 1 정규화 공간 에서 계산.

    절차:
    1. T_f, ρ_m 역정규화 (원 공간, Taylor 공식 입력)
    2. Taylor 공식 → σ_Taylor (원 공간, barn)
    3. σ_Taylor 및 σ_GT 재정규화 (공통 σ_a^Xe normalizer 사용)
    4. Loss = MSE(σ̃_Taylor, σ̃_GT) (정규화 공간)
    """
    # Step 1: 역정규화
    T_f_pred   = denormalize(T_f_hat_tilde,   norm['T_f'])       # °C
    rho_m_pred = denormalize(rho_m_hat_tilde, norm['rho_m'])     # g/cc

    # Step 2: Taylor 공식 (원 공간)
    sigma_Taylor = taylor_sigma_xe(
        T_f_pred, rho_m_pred, sigma0, dsigma_dTf, dsigma_dM,
        T_f_ref, rho_m_ref,
    )  # (B, Z, qH, qW, 2) barn

    # Step 3: 재정규화 (σ_Taylor 와 σ_GT 는 같은 물리량 → 공통 normalizer)
    sigma_Taylor_tilde = normalize(sigma_Taylor, norm['sigma_a_xe'])
    sigma_GT_tilde     = normalize(sigma_a_xe_GT, norm['sigma_a_xe'])

    # Step 4: MSE (정규화 공간, fuel_mask 적용)
    sq = tf.square(sigma_Taylor_tilde - sigma_GT_tilde) * fuel_mask[..., None]
    L = tf.reduce_sum(sq) / (tf.reduce_sum(fuel_mask) * 2 + 1e-12)  # G=2 평균
    return L
```

**Normalizer 공유 원칙**: σ_Taylor 와 σ_a^Xe_GT 는 **동일 물리량** (σ_a^Xe, barn) → 공통 통계 (`norm['sigma_a_xe']`) 사용. T_f, ρ_m 은 **Taylor 공식 입력 전용 으로 만** 역정규화 — loss 계산 은 σ 공간.

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

#### 3.5.2 해결: Marshak BC + 학습 가능 파라미터 (Albedo 캘리브레이션 확정)

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

**학습 가능 파라미터 (12개)** — Albedo 캘리브레이션 (least-squares fitting) 기존 결과 초기값 (40LP, 재캘리브레이션 권장):
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

- 초기값 출처: **Albedo 캘리브레이션 (least-squares fitting of Marshak boundary coefficients) — 기존 결과 40LP × CRS 10스텝 = 400 시나리오, 현재 데이터셋 (mirror sym + rotational sym 추가) 으로 재캘리브레이션 권장**
- R0~R3은 JNET0 N/S 매핑 오류(J↑=South 미인지) 영향 → R4부터 수정 매핑 적용.
  상세: `physical loss 개선 계획/2026-04-01 JNET0 N-S 매핑 오류 발견 및 해결.md`
- R4(2LP) 대비: g1 안정 (<1%), g2 6~9% 감소, top C₁₂ 부호 반전
- 훈련 중 L_diffusion + L_data가 α/C를 자동 조정
- 추론 시 학습된 α/C로 경계 누설 계산 (GT 불필요)
- α 범위 제약: `tf.clip_by_value(alpha, 0.01, 2.0)` 등으로 물리적 범위 유지

#### 3.5.3 α_init 저장 구조 (HDF5 attribute, 2026-04-23 갱신)

**원칙**: α_init 은 **training set HDF5** 의 flux 를 사용 해 산출 → **동일 HDF5 파일** 의 attribute / dataset 으로 저장. 별도 `stats.json` 등 분리 안 함 (데이터셋 ↔ α_init 일관성 자동 보장).

**HDF5 그룹 구조**: `/albedo_calibration/`

| 항목 | 위치 | shape / type | 비고 |
|------|------|------------|------|
| **Metadata (추적용)** | | | |
| `source` | `albedo_calibration.attrs` | str | 예: `"Least-squares fitting of Marshak boundary coefficients"` |
| `n_LP_used` | attribute | int | 40 |
| `n_CRS_steps` | attribute | int | 10 |
| `n_scenarios` | attribute | int | 400 |
| `calibration_date` | attribute | str | 캘리브레이션 수행 일 (예: `"2026-04-01"`) |
| `calibration_script` | attribute | str | 산출 스크립트 경로 / 이름 |
| `pitch_cm` | attribute | float | 21.608 |
| **Radial scalar α** (ortho / diag × g1 / g2) | | | |
| `alpha_ortho_g1` | attribute | float | 캘리브레이션 결과 (예시: 기존 0.1082, **재캘리브레이션 시 갱신**) |
| `alpha_ortho_g2` | attribute | float | 캘리브레이션 결과 (예시: 기존 0.4527, 갱신 예정) |
| `alpha_diag_g1`  | attribute | float | 캘리브레이션 결과 (예시: 기존 0.0820, 갱신 예정) |
| `alpha_diag_g2`  | attribute | float | 캘리브레이션 결과 (예시: 기존 0.5126, 갱신 예정) |
| **Axial matrix C** | | | |
| `C_bottom` | dataset | (2, 2) float | 행렬 BC, 캘리브레이션 결과 (재캘리브레이션 시 갱신) |
| `C_top`    | dataset | (2, 2) float | 행렬 BC, 캘리브레이션 결과 (재캘리브레이션 시 갱신) |
| **Quality 지표** (`quality` 하위 그룹) | | | |
| `R2_ortho_g1` | `quality.attrs` | float | 결정 계수 R² (예시: 기존 0.9975) |
| `R2_ortho_g2` | attribute | float | (예: 0.9977) |
| `R2_diag_g1`  | attribute | float | (예: 0.9878) |
| `R2_diag_g2`  | attribute | float | (예: 0.9971) |
| `R2_bottom_g1` | attribute | float | (예: 0.9994) |
| `R2_bottom_g2` | attribute | float | (예: 0.9915) |
| `R2_top_g1`   | attribute | float | (예: 0.9928) |
| `R2_top_g2`   | attribute | float | (예: 0.9246) |

> **참고용 코드 스니펫** (전체 아키텍처 미반영, 구조 이해 목적 한정 — 실제 구현 시 전처리 파이프라인 전체 설계 와 정합 필요):
>
> ```python
> # 전처리 신규 단계 (참고용)
> with h5py.File('train.h5', 'r+') as f:
>     grp = f.require_group('albedo_calibration')
>     grp.attrs['source']            = 'Least-squares fitting of Marshak boundary coefficients'
>     grp.attrs['n_LP_used']         = 40
>     grp.attrs['n_CRS_steps']       = 10
>     grp.attrs['n_scenarios']       = 400
>     grp.attrs['calibration_date']  = '2026-04-01'
>     grp.attrs['calibration_script']= 'albedo_calibration.py'
>     grp.attrs['pitch_cm']          = 21.608
>     grp.attrs['alpha_ortho_g1']    = 0.1082
>     grp.attrs['alpha_ortho_g2']    = 0.4527
>     grp.attrs['alpha_diag_g1']     = 0.0820
>     grp.attrs['alpha_diag_g2']     = 0.5126
>     grp.create_dataset('C_bottom', data=C_bottom_matrix)
>     grp.create_dataset('C_top',    data=C_top_matrix)
>     quality = grp.require_group('quality')
>     quality.attrs['R2_ortho_g1'] = 0.9975
>     # ... (나머지 R2_* 동일 패턴)
> ```

**사용 시점**:

| 시점 | 사용 방법 |
|------|----------|
| **전처리 신규 단계** | training set HDF5 의 flux 로 Albedo 캘리브레이션 (least-squares fitting) → α_init 산출 → HDF5 attribute 로 저장. 직후 α_init 으로 R_GT 매 시나리오 계산 → R_g1, R_g2 통계 산출 (§3.6.4 B 참조). 통계 도 같은 HDF5 attribute 로 저장 (`stats/R_g1`, `stats/R_g2`). 상세: `v-smr_load_following/data_preprocess/lf_preprocess/docs/2026-04-23 전처리 파이프라인 상세 설계 — 신규 단계 추가.md` |
| **학습 시작** | HDF5 attribute 에서 α_init 읽어 학습 가능 파라미터 의 초기값 으로 사용 (`tf.Variable(initial_value=alpha_init_from_h5, trainable=True)`) |
| **학습 중** | α 가 학습 으로 변화하나 R 통계 (μ_R, σ_R) 는 α_init 기반 값 고정 사용 (§3.6.4 B 채택 안) |
| **추론** | 학습 종료 시점 checkpoint 의 α 사용 (HDF5 의 α_init 은 더 이상 참조 안 함) |

상세: `Albedo 캘리브레이션 종합 보고서 (확정).md`, 전처리 절차: `v-smr_load_following/data_preprocess/lf_preprocess/docs/2026-04-23 전처리 파이프라인 상세 설계 — 신규 단계 추가.md`

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

#### 3.6.4 R_GT 운용 — 매 batch 동적 계산 + α_init 기반 통계 고정 (2026-04-23 갱신)

**원래 안 폐기**: 초안 (2026-04-08) 은 ε_total(t) = R_CMFD(φ_GT(t), xs_fuel_BOC) 를 전처리 에서 1회 계산 후 HDF5 에 저장 하려 함. **그러나 Albedo (α) 가 학습 가능 파라미터** (§3.5 Marshak BC) 이므로 R_CMFD 가 α 의 함수 → α 가 학습 중 변할 때마다 R_GT 도 달라짐 → **사전 저장 불가**.

```
ε_total = R_CMFD(φ_GT, xs_fuel_BOC, geometry, α_albedo, k_eff)
                                              ↑
                                   학습 가능 → 매 step 변동
```

**채택 안**:
- **R_pred, R_GT 모두 매 학습 batch 에서 직접 계산** (사전 저장 안 함)
- **R 통계 (μ_R, σ_R) 는 α_init 기준 데이터셋 초기 통계량 으로 1회 산출 후 고정**

##### A. 학습 중 매 batch — R_pred, R_GT 동일 α 적용

```python
def compute_L_diff_rel(phi_hat_norm, sample, stats, geometry, albedo_module):
    # 1. φ̂ 역정규화 (Bateman 방법 1 동일)
    phi_hat = denormalize(phi_hat_norm, stats['phi'])
    phi_GT  = sample['phi_GT']  # 원 공간 저장

    # 2. 현재 학습 α 가져오기
    alpha = albedo_module()  # 학습 가능 파라미터

    # 3. 같은 operator 를 양쪽 flux 에 적용 (동일 α 필수)
    R_pred = R_CMFD(phi_hat, sample['xs_fuel'], sample['k_eff'], geometry, alpha)
    R_GT   = R_CMFD(phi_GT,  sample['xs_fuel'], sample['k_eff'], geometry, alpha)

    # 4. 정규화 + MSE (§3.6.9 참조)
    R_pred_tilde = (R_pred - stats['R']['mu']) / stats['R']['sigma']
    R_GT_tilde   = (R_GT   - stats['R']['mu']) / stats['R']['sigma']
    L = masked_mse(R_pred_tilde, R_GT_tilde, fuel_mask)
    return L
```

**핵심**: R_pred 와 R_GT 양쪽 에 **동일한 현재 α** 적용 → α 가 어떤 값 이든 두 잔차 의 차이 비교 일관.

##### α 의 loss 기여 — 반영 되지만 제한적

**α 가 R 에 들어가는 위치**: **반사체 인접 노드 의 boundary current 항 한정** (§3.5 Marshak BC 의 face_type 1~4 분기). 코어 내부 노드 의 face current 는 α 무관 (D̃ harmonic mean / interior diffusion 만 사용).

**Loss 영향 메커니즘**:
- 동일 α 를 양쪽 에 적용 하더라도 **φ̂ ≠ φ_GT 인 동안** 은 R_refl(φ̂, α) ≠ R_refl(φ_GT, α)
- 두 값 의 차이 가 loss 에 들어가므로, **α 값 에 따라 loss 크기 가 달라짐**
- 예: α 가 크면 반사체 current 항 이 양쪽 모두 커지고 차이 도 비례 해서 커질 수 있음

→ α 는 loss 계산 에 **분명히 참여 (사라지지 않음)** 하나, 영향 범위 는 반사체 인접 노드 (전체 도메인 의 일부) 로 한정.

##### Gradient 흐름

$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial R_{pred}}\frac{\partial R_{pred}}{\partial \alpha} + \frac{\partial L}{\partial R_{GT}}\frac{\partial R_{GT}}{\partial \alpha}$$

α 가 두 R 을 동시에 움직이고 loss 는 차이 를 봄 → α 가 학습 가능 파라미터 로 등록 되어 있는 한 gradient 신호 자체 는 흐름. 다만 위 의 "loss 영향 = 반사체 인접 노드 한정" 제약 을 함께 고려 하여 학습 강도 / 수렴 양상 평가 필요 (실험 모니터링 권장 — §3.6.9 체크리스트 참조).

##### B. R 정규화 — 옵션 δ (per-sample, removal·V 분모) (2026-04-30 갱신)

> **이전 안 (z-score 통계 산출, Option 1) 폐기**: 2026-04-23 채택안 ("α_init 기반 R 통계 1회 산출 후 학습 내내 고정") 은 자체 검토 결과 다음 4가지 사유로 폐기 (2026-04-30):
>
> 1. **μ_R cancel + σ_R 단순 스케일링**: `L_diff_rel = (1/σ²)·raw_MSE` — z-score 의 본래 효과 (0 중심 평행이동) 무효, σ_R 만 남는 단순 λ 가중치와 등가
>
> 2. **PINN 표준 부재**: 다중군 중성자 확산 PINN, R²-PINN, Direct Term Scaling, Hierarchical Normalization 등 PINN 문헌에 R 잔차 z-score 통계 정규화 사례 없음
>
> 3. **학습 단계 흐름과 부조화 (DG-PINN 2-stage)**: 본 모델 학습은 ① Phase 1a 에서 L_data 만으로 모델을 어느 정도 수렴시킨 뒤 ② Phase 1b 에서 L_diff_rel 을 추가로 도입하는 구조이다. 그러나 σ_R 은 "전체 데이터셋의 R_GT 변동 폭" 으로 학습 진행 여부와 무관하게 큰 값으로 고정된다. Phase 1b 시점에는 이미 모델이 어느 정도 수렴한 상태이므로 R_pred 와 R_GT 의 차이 (분자) 가 원래도 작은데, 큰 σ_R (분모) 로 나누면 loss 값이 거의 0 으로 계산되어 학습 신호가 매우 약해진다.
>
> 4. **데이터 다양성과 학습 신호의 역설**: σ_R 은 시나리오 간 R_GT 의 표준편차이므로, 학습 데이터셋이 다양할수록 (LP 종류, 시점, 출력 수준 등의 변동이 클수록) σ_R 도 커진다. 결과적으로 같은 R_pred 오차에 대해서도 분모가 커서 loss 가 작아지는 현상이 발생한다. 직관적으로는 "다양한 데이터일수록 PDE 일관성을 더 강하게 강제해야" 하는데, σ_R 정규화는 정반대로 작동하여 다양한 데이터에서 오히려 학습 신호가 약해진다.

**채택 안 — 옵션 δ (per-sample, removal·V 분모, φ_GT 사용)**:

매 batch 에서 R_pred, R_GT 와 함께 S_ref 도 같은 텐서에서 1줄로 산출 (사전 통계 저장 불필요). **분모 S_ref 는 φ_GT 만으로 산출** — 분모를 통한 gradient 흐름 차단으로 loss hacking 방지.

```python
# 학습 중 매 batch (의사 코드)
R_pred = R_CMFD(phi_pred, xs_fuel, k_eff, geometry, alpha)
R_GT   = R_CMFD(phi_GT,   xs_fuel, k_eff, geometry, alpha)

# per-sample 정규화 — phi_GT 만 사용 (loss hacking 방지)
# 분모에 phi_pred 가 등장하지 않으므로 분모를 통한 gradient 흐름 차단
S_ref_g1 = mean(abs(Sigma_r1 * phi_GT[..., 0] * V), fuel_mask)   # scalar per sample
S_ref_g2 = mean(abs(Sigma_a2 * phi_GT[..., 1] * V), fuel_mask)

L_g1 = masked_mse((R_pred[..., 0] − R_GT[..., 0]) / S_ref_g1, fuel_mask)
L_g2 = masked_mse((R_pred[..., 1] − R_GT[..., 1]) / S_ref_g2, fuel_mask)
```

**옵션 δ 채택 사유**:
- **removal 분모는 기존 검증 코드 5개와 일관** — 검증 표시 `R/|removal·V| × 100%` 와 정확히 같은 무차원 비율
- **per-sample → LP·시점·branch 모든 운전 조건 변동 자동 정규화** — 가장 물리적
- **사전 통계 산출 불필요** → 전처리 비용 0, HDF5 추가 저장 없음
- **α 가 학습 중 변동해도** R_GT 와 S_ref 가 같은 batch 에서 산출 → 자동 일관성
- **GT 기반 분모 — 다른 loss 와 동일 원칙**: L_data (σ_φ), L_Bateman (σ_N), L_σXe (σ_σXe) 모두 GT 통계 분모. 옵션 δ 의 S_ref^GT 도 같은 패턴 (다만 글로벌 통계가 아닌 per-sample 산출)
- **Loss hacking 방지**: 분모 S_ref 가 φ_GT 만으로 산출되어 model output 무관 → 옵티마이저가 *"분모를 키워 loss 를 줄이는"* 우회 경로 수학적으로 불가능. (대안: φ_pred 사용 시 `.detach()` 필수, 그러나 본 안은 φ_GT 직접 사용으로 더 명시적)
- **스케일링 일관성**: 분자 |L·δ| 와 분모 S_ref 모두 φ 자릿수에 1차 비례 → 운전 조건 자릿수 cancel → loss 가 운전 조건 무관 O(1)

**저장 비용**: 0. R_pred, R_GT, S_ref 모두 매 batch 동적 산출.

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
| **Albedo와 Mirror의 quarter-core 정합성** | R_GT(t) 매 batch 동적 계산 (현재 학습 α 사용) — quarter (5,5) 도메인의 BC 정합성이 R 에 인코딩 → 학습 시 그 정합 조건이 L_diff_rel 잔차에 반영 (§3.6.4 갱신) |
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

#### 3.6.9 Normalization — 옵션 δ per-sample 정규화 (2026-04-30 갱신, 폐기 후 재정의)

> **이전 안 (R 전용 z-score 통계) 폐기**: 본 절의 2026-04-23 신설 안 ("R 전용 z-score 통계, φ 통계 사용 금지") 은 옵션 δ (per-sample, removal·V 분모) 채택으로 폐기. **폐기 사유 4가지 + 옵션 δ 상세 명세는 §3.6.4 B 참조**.

**핵심 원칙 (옵션 δ)**:

- 정규화 분모: per-sample `S_ref_g^GT = mean(abs(Σ_r,g · φ_g^GT · V), fuel_mask)` (g1, g2 별도, **φ_GT 만 사용**)
- 사전 통계 산출 / HDF5 저장 없음 — 매 batch 동적 산출
- removal 분모 채택 근거: 기존 검증 코드 5개의 % 표시 방식과 정확히 일관 (`R/|removal·V| × 100` = 노드별 PDE 위반 비율)
- 코드 의사 형식: §3.6.4 B 참조

**다른 loss 와의 일관성 — GT 기반 분모 원칙**:

| Loss | 분모 형태 | 출처 | model output 의존? |
|------|----------|------|-------------------|
| L_data | σ_φ (전체 데이터셋 표준편차) | φ_GT 통계 | ❌ |
| L_Bateman | σ_N_xe, σ_N_I | N_xe^GT, N_I^GT 통계 | ❌ |
| L_σXe | σ_σXe | σ_a^Xe^GT 통계 | ❌ |
| **L_diff_rel (옵션 δ)** | **S_ref^GT (per-sample)** | **φ_GT (per-sample)** | **❌** |

→ 모든 loss 가 GT 기반 분모를 사용 (model output 의 분모 진입 차단). 옵션 δ 만의 차이는 per-sample 산출이라는 점 — 데이터 다양성 역설 (전역 σ 가 데이터셋 다양성에 비례하여 학습 신호 약화) 을 회피하기 위함.

**Loss 수식**:

$$\mathcal{L}_{diff\_rel} = \frac{1}{|F|} \sum_{g \in \{1, 2\}} \sum_{v \in F} \left(\frac{R_{g,v}^{pred} - R_{g,v}^{GT}}{S_{ref,g}^{GT}}\right)^2$$

$$S_{ref,g}^{GT} = \frac{1}{|F|} \sum_{v \in F} \left| \Sigma_{r,g} \cdot \phi_{g,v}^{GT} \cdot V_v \right| \quad \text{(per-sample, g 별 별도, } \phi^{GT} \text{ 만 사용)}$$

- F = 연료 노드 집합, |F| = fuel cell 개수
- R_pred, R_GT, S_ref^GT 모두 매 batch 동적 계산 (§3.6.4 B 의사 코드)
- 사전 통계 (μ_R, σ_R) 산출 / 저장 / 로드 코드 모두 폐기

**Loss hacking 방지 원칙**: 분모 S_ref 는 **φ_GT 만으로 산출**, φ_pred 는 분자 (R_pred) 에만 등장. 분모를 통한 gradient 흐름 차단 → 옵티마이저가 *"분모를 키워 loss 를 줄이는"* 우회 경로 수학적으로 불가능. 추론 시 S_ref 는 필요 없으므로 φ_GT 부재 문제 없음.

**φ 통계 재사용 금지 원칙은 그대로 유효**: φ 와 R 은 단위·부호·분포가 다르므로 서로의 정규화 통계를 공유 불가. 단 본 안에서는 R 통계 자체를 산출하지 않으므로 이 우려는 자연 해소됨.

**실무 체크리스트** (Bateman §1.9.4 와 동일 원칙):

1. **각 loss 항 크기 로깅**: L_state / L_diff_rel / L_A / L_B / L_σXe — 자릿수 비교, 한 항이 다른 항 대비 10^5 이상 크면 가중치 재설계
2. **Gradient L2 norm 비교**: 각 loss 의 gradient — 한 쪽 압도 시 스케일 문제 신호
3. **S_ref^GT 분포 모니터링**: per-sample S_ref 가 시나리오별로 어느 정도 변동하는지 로깅 (저출력 transient 시점에서 S_ref 가 비정상적으로 작아지는 케이스 확인)

---

## 4. L_keff — K-eff Rayleigh 몫 (폐기)

> **2026-04-22 폐기 결정**: L_keff (Rayleigh 몫) 은 **절대 도입하지 않음**. **3D 문제에 부적합** (자문 의견). k_eff 학습은 **L_data scalar GT fitting 만으로 처리** (result_keff 를 별도 헤드 출력으로 학습). 아래 Rayleigh 몫 공식은 참고용으로만 유지. 다른 k-eigenvalue self-consistency / inverse power iteration 등 검토도 본 프로젝트 범위 외.

Rayleigh 몫:

$$k_{pred} = \frac{\sum_{v,g}\nu\Sigma_{f,g,v}\hat{\phi}_{g,v}^2}{\sum_{v,g}(D_g|\nabla\hat{\phi}_{g,v}|^2 + \Sigma_{a,g,v}\hat{\phi}_{g,v}^2)}$$

$$\mathcal{L}_{keff} = \left(k_{pred} - k_{GT}\right)^2$$

### 4.1 k_pred 를 L_diff_rel 에 사용하는 제안 — 검토 기록 (2026-04-22)

**제안 내용**: 현 L_diff_rel 의 R_CMFD 계산 시 k_GT 대신 k_pred (Rayleigh 몫 또는 별도 헤드 출력) 사용.

```
L_diff_rel_new = ‖ M·φ̂ − (1/k_pred) F·φ̂ ‖ / scale    (현재 k_GT → k_pred)
```

**검토 결과: 현 설계 (k_GT 고정) 유지 권고**

**부적합 근거**:

| 쟁점 | 현행 k_GT | 제안 k_pred |
|------|---------|----------|
| CMFD bias 해결 | ✅ 상대 잔차 (§3.6.2) 로 이미 해결 | ❌ k 값 변경은 operator bias (M, F vs MASTER NEM) 와 무관 |
| ε_total(t) 사전 계산 최적화 (§3.6.4) | ✅ 전처리 1회 저장 | ❌ 매 iteration 재계산 필요 (k_pred 가 모델 출력이라 매번 변동) |
| Fundamental mode 수렴 보장 | ✅ k_GT 고정 | ❌ Rayleigh 몫 은 임의 eigenvector 에 대해 최소화 가능 — trivial solution 위험 |
| 구현 복잡도 | 낮음 | 높음 (k_pred 헤드 + gradient path + precomputation 상실) |
| 문헌 표준성 (eigenvalue PINN 단독) | — | ✅ (단 k 와 φ 를 **동시** unknown 취급, L_keff physics loss 와 병용 필수) |
| 본 프로젝트 적합성 | ✅ | ❌ — L_keff physics loss **미도입** 상태에서는 k_pred 자체의 gradient pressure 부재 |

**핵심 논점**:
- CMFD-only bias 는 **operator 불일치** (NEM 보정 누락) 에서 발생 — k 값과 무관하므로 k_pred 로 바꿔도 해결되지 않음
- 현 L_diff_rel (상대 잔차) 이 이미 이 문제를 완벽히 해결 — ε_total(t) = R_CMFD(φ_GT, k_GT) 를 target 으로 삼아 bias 상쇄
- Eigenvalue PINN 문헌에서 k_pred 사용 은 **L_keff physics loss (Rayleigh 몫 잔차) 와 병용** 이 전제 — 본 프로젝트는 L_keff physics loss 미도입 상태

**재검토 시점**: 향후 L_keff physics loss 도입 결정 시 **함께 재검토**. 그 시점에 (a) k_pred 를 L_diff 에 주입하여 self-consistency 강제, (b) L_keff 로 k_pred 의 fundamental mode 수렴 보장 — 두 효과가 동시에 성립하므로 전환 타당성 확보 가능.

**대안 (ablation 실험 용)**: L_diff_rel 변형 으로 `k_pred` 버전을 추후 TR-R-XX 신설 실험 대상 검토 가능. 현재 default 는 **k_GT 유지**.

---

## 5. L_pload — 출력 총량 self-consistency

> **2026-04-22 초기 누락 정정**: 2026-03-30 공간인코더 구현 계획 에 정의되어 처음부터 사용 의도 였으나 본 레퍼런스 작성 시 **누락되어 추가 보충**. L_physics 로 분류 (원래 "3D → scalar 일관성 loss" 범주) — 로드맵 §2 `losses.L_pload.{enabled, weight}` 로 config 화.

### 5.1 물리적 의미

노심 전체 절대 출력 (열출력 MW) 의 상대 비율 (= 설계 출력 대비) 이 **쿼리 입력 `p_load`** 와 일치해야 함 — query-prediction self-consistency.

- 예측 3D 출력분포 `P̂_3d(t)` 에서 전체 fullcore 합산 = 현재 절대 열출력 (단위 MW)
- 이를 설계 출력 `P_design` 으로 나눈 값이 **목표 상대 출력** (= `p_load_input`)
- 즉 "모델 예측 총량" = "쿼리 지정 부하" 일관성 강제

### 5.2 손실 정의

$$\mathcal{L}_{pload} = \left( \frac{P_{total}(\hat{P}_{3d})}{P_{design}} - p_{load,input}(t) \right)^2$$

where `P_total(P̂_3d)` = quarter → fullcore 변환 후 전체 합산 (**sym_type 에 따라 환산 계수 다름**, 아래 §5.4 참조).

- **GT 불필요**: 입력 쿼리 `query_pload` (HDF5 `scenarios/{profile}/query_pload`, shape (T,)) 만으로 계산
- **self-consistency loss**: 별도 GT 타겟 없이 입력-출력 간 제약

### 5.3 적용 Phase

- **Phase 1b 부터 활성** (L_physics 5종 중 하나로)
- 1-step 예측으로 계산 가능 (시간축 불필요)
- Phase 1b annealing 0 → 0.1 동일 적용 (로드맵 §2 참조)

### 5.4 sym_type 에 따른 fullcore 환산 주의

**핵심 주의**: quarter 도메인 (20, 5, 5) 합산을 단순 × 4 로 fullcore 환산하면 **sym_type 에 따라 틀림**. 대칭축 위의 cell (중앙 row/col) 이 중복 계수되거나 1회만 세지는 경우가 있어, sym_type 별로 가중 합산 계수가 다름.

- 기존 전처리 / 후처리 코드의 `quarter_to_fullcore()` helper 를 재사용 (lf_preprocess `symmetry_utils.py` 또는 동등 구현)
- sym_type 분류: mirror / rotation (quarter-core 로드맵 §1 참조)
- 각 sym_type 별 환산 계수는 원 helper 에 내장 — loss 계산 시 이를 경유해야 함

### 5.5 TF2 구현 패턴 (sym_type-aware)

```python
def loss_pload(P_pred_3d, P_design, p_load_query, sym_type):
    """
    P_pred_3d : (B, Z, qH, qW) or (B, T, Z, qH, qW) — quarter 도메인 예측 3D 출력 (MW/node)
    P_design  : scalar (설계 출력, MW)
    p_load_query : (B,) or (B, T) — 쿼리 입력 p_load (0~1)
    sym_type  : str — mirror / rotation 등 (config 로부터)
    """
    # sym_type-aware quarter → fullcore 환산 (helper 재사용)
    # 주의: 단순 × 4 금지 — 대칭축 cell 중복/누락 처리가 sym_type 에 따라 다름
    total_power = quarter_to_fullcore_sum(P_pred_3d, sym_type)  # (B,) or (B, T)
    relative_power = total_power / P_design
    return tf.square(relative_power - p_load_query)
```

`quarter_to_fullcore_sum()` 은 기존 코드베이스의 symmetry helper 를 이용 (구체 API 는 구현 시 확인).

### 5.6 원 출처

- 2026-03-30 공간인코더 구현 계획 line 122 (`L_pload = (Σ(P̂_3d) / P_design − p_load_input)² — 총 출력 보존`)
- 2026-04-22 L_physics 로 통합 결정 (사용자 확정)
- 2026-04-22 sym_type 환산 주의 추가 (사용자 지적)

---

## 6. L_AO — 축방향 출력 편차 consistency

> **2026-04-22 초기 누락 정정**: 2026-03-30 공간인코더 구현 계획 에 정의되어 처음부터 사용 의도 였으나 본 레퍼런스 작성 시 **누락되어 추가 보충**. L_physics 로 분류 — 로드맵 §2 `losses.L_AO.{enabled, weight}` 로 config 화.

### 6.1 물리적 의미

예측 3D 출력 에서 정의식 으로 계산한 **축방향 출력 편차 (Axial Offset, AO)** 가 MASTER 가 계산한 scalar GT (`result_ao`) 와 일치해야 함.

- **AO 정의** (원자로 공학 표준):
$$AO = \frac{P_{top} - P_{bot}}{P_{top} + P_{bot}}$$
  - `P_top` = 상반부 (축방향 상위 절반) 적분 출력
  - `P_bot` = 하반부 (축방향 하위 절반) 적분 출력
  - `AO ∈ [-1, 1]` — 음수 = 하반부 편향, 양수 = 상반부 편향
- 3D 분포 → scalar 변환 의 대표 지표 — Xe 진동, 축방향 출력 분포 추적 에 중요
- **축방향 지표이므로 H/W 평면 대칭 (sym_type) 과 무관** — quarter 도메인 에서 직접 계산 가능

### 6.2 손실 정의

$$\mathcal{L}_{AO} = \left( AO(\hat{P}_{3d}) - AO_{GT} \right)^2$$

- **GT 필요**: HDF5 `scenarios/{profile}/result_ao` (shape (T, 30), branch 축 포함)
- 3D power 예측이 올바르면 AO(P̂_3d) 가 자동으로 맞지만, **3D 분포 다르지만 scalar AO 만 맞는 trivial solution** 을 방지하기 위한 추가 제약

### 6.3 적용 Phase

- **Phase 1b 부터 활성** (L_physics 5종 중 하나로)
- 1-step 예측으로 계산 가능
- Phase 1b annealing 0 → 0.1 동일 적용

### 6.4 TF2 구현 패턴

```python
def loss_ao(P_pred_3d, ao_gt):
    """
    P_pred_3d : (B, Z, qH, qW) or (B, T, Z, qH, qW) — 예측 3D 출력 (quarter 도메인)
    ao_gt     : (B,) or (B, T) — result_ao

    주: AO 는 축방향 (Z축) 분리 이므로 H/W 평면 sym_type 과 무관.
        Quarter 도메인 (Z, qH, qW) 에서 직접 P_top / P_bot 계산 시,
        AO = (P_top − P_bot) / (P_top + P_bot) 는 quarter vs fullcore 동일 값
        (분자/분모 모두 동일 sym 계수 곱해져 상쇄)
    """
    Z = tf.shape(P_pred_3d)[-3]
    half = Z // 2  # Z=20 → half=10
    P_top = tf.reduce_sum(P_pred_3d[..., half:, :, :], axis=[-3, -2, -1])
    P_bot = tf.reduce_sum(P_pred_3d[..., :half, :, :], axis=[-3, -2, -1])
    ao_pred = (P_top - P_bot) / (P_top + P_bot + 1e-12)  # 분모 안정화용 ε (scalar division)
    return tf.square(ao_pred - ao_gt)
```

### 6.5 L_data 와의 차이

| 항목 | L_data (result_ao 직접 fit) | L_AO |
|------|---------------------------|------|
| 학습 대상 | 별도 scalar 헤드 의 AO 출력 | 3D power 예측 에서 파생 AO |
| 구조적 제약 | 없음 — scalar 만 맞으면 됨 | 3D 분포 → scalar 정의식 일관성 강제 |
| Trivial solution 위험 | ✅ (3D 틀려도 scalar 만 맞추면 loss = 0) | 방지 |

→ **병용 권장**: L_data (result_ao) 는 별도 scalar 헤드 의 calibration, L_AO 는 3D 예측 의 축방향 균형 강제. 두 경로 가 서로 보완.

### 6.6 원 출처

- 2026-03-30 공간인코더 구현 계획 line 123 (`L_AO = (AO(P̂_3d) − AO_GT)² — 축방향 분포 보존`)
- 2026-04-22 L_physics 로 통합 결정 (사용자 확정)

---

## 7. 가중치 전략

### 7.1 전체 손실 함수 (개정 2026-04-22)

$$\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{data\_halo}\mathcal{L}_{data\_halo} + \lambda_{Bateman}\mathcal{L}_{Bateman} + \lambda_{Taylor}\mathcal{L}_{Taylor} + \lambda_{diff\_rel}\mathcal{L}_{diff\_rel} + \lambda_{pload}\mathcal{L}_{pload} + \lambda_{AO}\mathcal{L}_{AO}$$

- **L_data_halo 신설** (2026-04-08): halo cell phi에 직접 supervision 부과 (`05_symmetry_mode.md` §3.1, ML plan 권고 1)
- **L_diffusion → L_diff_rel 전환** (2026-04-08): 절대 잔차 폐기, 상대 잔차로 교체 (§3.6 참조)
- **L_pload, L_AO 초기 누락 정정** (2026-04-22): 2026-03-30 공간인코더 구현 계획 에 정의되어 처음부터 사용 의도 였으나 누락 → L_physics 로 통합 (§5, §6 참조)
- **L_keff 제거** (2026-04-22): physics loss 도입 미정 → 현재 L_data scalar GT 로 처리, 별도 항 아님 (§4 참조)

### 7.2 Warm-up / Ramp-up 전략

> 이 전략은 PINN(Physics-Informed Neural Network) 문헌에서 표준적 — Mamba 고유가 아님.
> **2026-04-22 재구성**: 로드맵 §2 의 Phase 1b annealing (0 → 0.1) 과 정합. "Warm-up" ≈ Phase 1a, "Ramp-up" ≈ Phase 1b annealing 구간, "Full training" ≈ Phase 1b annealing 종료 이후.

**Warm-up (Phase 1a)**: L_data + L_data_halo 만 학습, physics loss 는 λ=0.
- **이유**: DG-PINN 2-stage 1a 단계 (Yuan 2024) — data fitting 부터 먼저, physics 는 후순위.

**Ramp-up (Phase 1b annealing)**: L_physics 5종 (L_Bateman, L_σXe, L_diff_rel, L_pload, L_AO) 를 linear 하게 λ = 0 → 0.1 로 증가.

**Full training (Phase 1b 이후)**: 목표 가중치 (0.1) 로 고정, 이후 Phase 2 / 3 까지 유지 (ReLoBRaLo 활성화 이전까지).

**가중치 예시** (2026-04-22 갱신):

| 단계 | λ_data | λ_data_halo | λ_Bateman | λ_σXe | λ_diff_rel | λ_pload | λ_AO |
|------|:-----:|:----------:|:---------:|:-----:|:----------:|:-------:|:----:|
| Warm-up (Phase 1a) | 1.0 | 0.3 | 0 | 0 | 0 | 0 | 0 |
| Ramp-up (Phase 1b annealing) | 1.0 | 0.3 | 0 → 0.1 | 0 → 0.1 | 0 → 0.1 | 0 → 0.1 | 0 → 0.1 |
| Full training (Phase 1b end → Phase 2b 중반) | 1.0 | 0.3 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Phase 2b 중반 이후 | ReLoBRaLo 자동 조정 (기준, 로드맵 §2 참조) | | | | | | |

- λ_data_halo = 0.3: cell 비율 기반 중립값 0.44(=11/25)에서 redundancy 고려한 임의 선택값. ablation 권장 (0.1, 0.3, 0.5)
- λ_diff_rel = 0.1: inner cell redundancy 감안. 주된 가치는 §3.6.5에 명시된 4가지 (특히 Albedo BC 학습)
- λ_pload, λ_AO = 0.1: 다른 L_physics 와 동일 수준 — 추후 ablation 으로 조정

> **주의**: 가중치의 합이 1일 필요 없음 — 각 항의 상대적 크기 비율이 중요. 위 값은 참고용 예시이며 실제 학습에서 loss 스케일에 따라 조정 필요.
> **λ_keff 제거**: 2026-04-22 결정으로 L_keff physics loss 미도입 — result_keff 는 L_data scalar GT 로만 fitting (별도 λ 불필요).

### 7.3 정규화 주의

N_Xe ~ 10⁻⁷ barn-cm vs φ ~ 10¹³ n/cm²/s → 7~8자릿수 차이.
각 Loss 내부에서 물리량을 노드별 평균으로 정규화 후 계산.

---

## 8. 물리량-데이터 매핑 요약

| Loss 항 | 필요 물리량 | MASTER 가용 | 추가 계산 |
|---------|-----------|:-----------:|:---------:|
| L_data | GT 전체 필드 (result_power, tcool, tfuel, rhocool, keff, ao 등) | ✅ | 없음 |
| L_data_halo | halo cell φ supervision | ✅ | halo expand |
| L_Bateman | N_Xe, N_I, φ(2군), Σ_f(2군), σ_a^Xe(2군), γ, λ | ✅ | ODE 적분 |
| L_σXe | T̂_f, ρ̂_m (예측), σ_a^Xe(GT), Taylor 계수 | ✅ | 격자 라이브러리 편미분 계수 |
| L_diff_rel | φ̂(2군), D_g, Σ_a, νΣ_f, k_GT, ε_total(t) 사전계산 | ✅ 부분 | ∇²φ 유한차분, CMFD residual |
| **L_pload** (2026-04-22 초기 누락 정정) | P̂_3d, P_design, query_pload, sym_type | ✅ (입력 쿼리) | quarter→fullcore 환산 (sym-aware) |
| **L_AO** (2026-04-22 초기 누락 정정) | P̂_3d, result_ao | ✅ | 축방향 상/하반부 적분 |
| L_keff (확장용, physics 도입 미정) | φ̂, νΣ_f, D_g, Σ_a, k_GT | ✅ | Rayleigh 몫 연산 (현재 L_data scalar 로 대체) |

---

## 9. 구현 시 주의사항

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
