# L_diffusion 노드 vs 집합체 CMFD 잔차 비교 테스트

## Context

집합체 단위 L_diffusion을 physical loss로 사용하고 싶으나, MASTER는 CMFD + NEM(D̂ 보정)을 노드 단위로 계산하므로 구조적 불일치 존재.

**기존 어셈블리 CMFD 결과**: g1 median 3.50%, g2 median 6.99% (FAIL)
**Step 0 JNET0 밸런스**: α=2 적용 시 g1/g2 median ~0% (밸런스 완벽 성립)
  ※ 초기 α=1에서 3.19%/1.96% 잔차는 JNET0=half current 스케일 오류 아티팩트였음

**목적**: 노드 단위(10.8 cm) vs 집합체 단위(21.6 cm) CMFD 잔차를 비교하여, 격자 세분화가 얼마나 오차를 줄이는지 확인 → 집합체 단위 L_diffusion 정밀화 전략 결정

## Step 0 확정 사항 (2026-03-31 완료)

- ABS = Σ_c + Σ_f (총 흡수 거시단면적)
- JNET0 = per-unit-area, positive = outward, 면적 곱셈 필요
- WIDE = 21.60780 cm (기존 21.504 cm은 오류)
- ndivxy = 2, 노드 피치 = 10.804 cm

### Step 0 JNET0 밸런스 잔차에 대한 고찰

MASTER 수렴 전류(JNET0)를 사용했음에도 g1 3.2%, g2 2.0% 잔차 존재.
이론적으로 JNET0_sum + Σ_r × φ̄ × V = Source × V 가 정확히 0이어야 하나,
실제로는 잔차 발생. 가능한 원인:
1. MAS_NXS 출력 시 부동소수점 정밀도 손실
2. MASTER 내부 추가 보정(ADF 등)이 출력에 완전히 반영되지 않을 가능성
3. MASTER 내부 밸런스 형태가 가정과 미묘하게 다를 가능성

정확한 원인은 MASTER 소스코드 분석이 필요하며 현 범위 밖.

---

## 구현 계획

### 비교 테스트 스크립트 작성

**파일**: `piecewise-test/2026-03-31_diffusion_residual_comparison_node_vs_assembly.py`

하나의 스크립트에서 동일 데이터셋에 대해 두 가지 해상도로 CMFD 잔차를 계산하고 비교:

### CMFD 밸런스 수식

모든 물리량은 노드(또는 어셈블리) **체적 평균값**을 사용:
- φ̄_g = (1/V) ∫∫∫ φ_g(r) dV  (체적 평균 중성자속 [/cm²/s])
- Σ_r, νΣ_f 등도 노드/어셈블리 평균 거시단면적

면 전류 (CMFD 핵심 — 인접 노드 체적평균 flux 간 차분):
```
J_face = D̃ × (φ̄_neighbor - φ̄_center) / h
D̃ = 2·D_a·D_b / (D_a + D_b)  (조화평균)
```

체적 적분 밸런스 잔차 (R=0이면 확산방정식 만족):
```
g=1: R₁ = Σ_faces(J₁×A) + Σ_r1×φ̄₁×V − (1/keff)(νΣf₁φ̄₁ + νΣf₂φ̄₂)×V
g=2: R₂ = Σ_faces(J₂×A) + Σ_a2×φ̄₂×V − Σ_s12×φ̄₁×V
```
- Σ_faces(J×A): 6면 누설 합 [n/s]
- Σ_r×φ̄×V: 제거 반응률 [n/s]
- Source×V: 생성 반응률 [n/s] (g1=핵분열원, g2=산란전입원)

상대잔차: |R_g| / |Removal_g| × 100 [%]

평가 대상: 자기 + 6면 이웃이 모두 연료(νΣf > 0)인 노드만

---

#### A. 노드 단위 CMFD (MAS_NXS 기반, 22×22 중 연료 영역)
- **Flux 출처**: MAS_NXS FLX — **노드 체적 평균** 중성자속
  - V_node = 10.804 × 10.804 × dz [cm³]
- **XS 출처**: MAS_NXS (DIF, ABS, SCA, NFS) — MASTER 내부 계산값
- 노드 피치 dx = dy = 10.804 cm, dz = ZMESH
- Σ_r1 = ABS + SCA, Σ_a2 = ABS (Step 0 확정)

#### B. 집합체 단위 CMFD (xs_fuel 기반, 9×9)
- **Flux 출처**: MAS_OUT flux_3d — **어셈블리 체적 평균** 중성자속
  - V_asm = 21.608 × 21.608 × dz [cm³]
- **XS 출처**: xs_fuel (MAS_XSL 기반) — AI 모델이 실제 사용하는 물리량
  - xs_fuel 10채널: [νΣf₁, Σf₁, Σc₁, Σtr₁, Σs₁₂, νΣf₂, Σf₂, Σc₂, Σtr₂, 0]
  - D_g = 1/(3×Σ_tr), Σ_r1 = (Σ_c1+Σ_f1)+Σ_s12, Σ_a2 = Σ_c2+Σ_f2
- 집합체 피치 dx = dy = 21.60780 cm (**기존 21.504에서 수정**)

#### C. 비교 표 출력
| 메트릭 | 노드 CMFD (10.8cm) | 집합체 CMFD (21.6cm) | 기존 테스트 (xs_fuel) |
|--------|:---:|:---:|:---:|
| g1 median | ? | ? | 3.50% |
| g2 median | ? | ? | 6.99% |

### 데이터 흐름
```
[노드 단위]
  MAS_NXS → parse (DIF,ABS,SCA,NFS,FLX at node level 18×18)
    └─ 노드 CMFD: (20,18,18) → inner [1:-1] → 잔차

[집합체 단위]
  xs_fuel (MAS_XSL) → xs_voxel_builder → (20,5,5,10) → quarter→full 복원
  MAS_OUT → flux_3d (20,9,9,2)
    └─ 집합체 CMFD: (20,9,9) → inner [1:-1] → 잔차

[공통]
  MAS_OUT → keff
```

### 핵심 파일
- `piecewise-test/2026-03-30_diffusion_residual_test_assembly.py` — 기존 어셈블리 CMFD 로직
- `piecewise-test/2026-03-31_step0_abs_jnet0_verification.py` — MAS_NXS 파싱 패턴
- `lf_preprocess/xs_voxel_builder.py` — xs_fuel 10채널 구성
- `lf_preprocess/mas_out_parser.py` — keff, flux 추출

### 주의사항
- 반사체 제외 (K=2~21 연료만, I/J 연료 범위 동적 탐지)
- 피치 = WIDE / ndivxy = 21.60780 / 2 = 10.80390 cm
- MAS_NXS의 DIF를 직접 사용 (이 테스트 한정, xs_fuel Σ_tr 유도와 별개)
- LP 2개 × CRS 10스텝으로 통계 확보

### 검증
- 집합체 CMFD 결과가 기존 테스트와 유사한지 확인 (피치 수정분 제외)
- 노드 vs 집합체 잔차 비율로 O(h²) 스케일링 확인
