# 08. Symmetry mode 처리 — Halo cell 채용 이유

> **결정**: 입력 함수가 quarter LP + sym option → halo (6,6) expand 1회. 인코더부터 디코더까지 모델 전체가 (6,6) feature space 처리. 최종 출력만 (5,5) crop. L_diffusion은 디코더 (6,6) 출력을 그대로 사용하며 합산 도메인은 5×5 core cell.
> **결정 일자**: 2026-04-08
> **개정 (2026-04-08, ML 위협 검토 후)**: §3.1, §5, §8 보강 — halo cell에 L_data_halo (λ=0.3) 부여, encoder/decoder (6,6) 격자 통일 명시, L_diff 형식을 상대 잔차 (L_diff_rel) 로 전환. ML 위협 검토 plan: `C:\Users\Administrator\.claude\plans\compressed-wandering-stroustrup.md`

---

## 1. 확정 파이프라인

```
[Input function]  quarter LP (Z, 5, 5, C) + sym option ("mirror" | "rotational")
       │
       ├── halo expand 1회 (분기문 한 곳)
       │
       ▼
   halo tensor (Z, 6, 6, C)
       │
       ▼
[Encoder]  input (B, 20, 6, 6, 21)
   CellEmbedder Conv3D(1,1,1) 21→128
   → LAPE add (Z, 6, 6, 128)
   → flatten (B, 720, 128)
   → (Pre-LN + FullAttention3D + STRING + Pre-LN + FFN3D) × 3 stage
   → reshape (B, 20, 6, 6, 128)
       │
       ▼
[Mamba]  latent (6,6) 그대로
       │
       ▼
[Decoder]  latent (6,6) → 디코더 전체 (6,6) feature space
   → output (B, 20, 6, 6, 10)
       │
       ├──> [L_diffusion]
       │      디코더 (6,6) 출력 그대로 사용 (별도 expand 없음)
       │      합산 도메인: 5×5 core cell
       │      halo cell은 stencil neighbor lookup 용도
       │
       └──> [Final crop]
              (B, 20, 6, 6, 10) → (B, 20, 5, 5, 10)
              외부 인터페이스 한 곳에서만 slicing

[Data loss]  quarter (5,5) cell만 GT 비교
```

---

## 2. 이전 검토 안과 현행 안

설계 과정에서 3차에 걸쳐 정정:

| 차수 | 안 | 폐기 사유 |
|---|---|---|
| 1차 | Quarter input + L_diffusion 직전 halo expand | 인코더가 대칭 정보 인지 못 함. halo expand 위치가 loss 함수 안에 갇힘 |
| 2차 | Fullcore (9,9) input + 인코더 quarter crop + 디코더 halo crop | Halo expand 2회. 인코더-디코더 격자 불일치. crop 위치 2개 |
| **3차 (현행)** | **Halo (6,6) all the way + 최종 출력만 (5,5) crop** | — |

3차 안의 핵심 이점:
- Halo expand 함수 호출 1회 (입력 단계)
- 인코더-디코더 격자 일치 ((6,6))
- L_diffusion이 (6,6)을 그대로 받음 (별도 expand 없음)
- 인코더가 대칭 정보 직접 학습 (boundary cell이 halo cell을 attention input으로 봄)
- Crop 위치 1개 (외부 인터페이스 한 곳)

---

## 3. 학습 신호 흐름 — Halo cell의 역할

### 3.1. Halo cell의 의미

- Halo cell의 값 = quarter LP를 대칭 변환한 view (mirror 또는 rotational)
- 물리적으로는 fullcore의 진짜 인접 cell (대칭 너머 위치)
- Ground truth가 *존재* (대칭 관계로 quarter cell GT에서 자동 도출)
- **L_data_halo (λ=0.3) 부과 — 개정**:
  - 그냥 (6,6) 형태로 halo 위치를 유지할 경우, halo cell에 도달하는 학습 신호는 **L_diff stencil neighbor lookup 한 경로뿐**
  - L_diff 잔차가 커지면 (CMFD 절대 잔차의 g2 ~6.4% bias floor) halo cell이 비물리적인 값으로 학습될 수 있음
  - **해소**: `L_data_halo = MSE(pred_halo_phi, sym(GT_inner_phi))` 부과
  - **λ 결정**: cell 비중 기준 중립값 = `N_halo / N_inner = 11/25 = 0.44`. 0.44~0.0 범위에서 **0.3으로 임의 선택** (정보 redundancy 일부 고려, ablation 대상)

### 3.2. 인코더가 받는 효과

- 인코더 attention이 (6,6) 격자 처리 → boundary cell이 halo cell을 attention input으로 직접 봄
- "boundary cell의 한쪽 이웃은 sym 매핑된 cell" 이라는 정보가 attention input feature에 *암묵적*으로 표현
- 인코더 attention/feature 처리가 이 패턴을 학습 → 대칭조건 inductive bias 흡수

### 3.3. 디코더가 받는 효과

- 디코더도 (6,6) 격자 처리 → 동일한 attention input 효과
- 디코더 출력 halo cell의 학습 신호:
  1. **L_data_halo 직접 부과** (λ=0.3, 개정 후) — 가장 강한 신호
  2. L_diffusion stencil의 neighbor lookup → backward chain
  3. 디코더 attention의 feature mixing → 다른 cell로의 기여

### 3.4. L_diffusion 사용

- 디코더 (6,6) 출력을 그대로 받음 (별도 expand 없음)
- 합산 도메인: 5×5 core cell (qy_h=1..5, qx_h=1..5)
- Halo cell residual은 합산 안 함
- Halo cell은 core cell stencil의 neighbor lookup용
- **잔차 형식 (개정)**: 절대 잔차 → **상대 잔차 (L_diff_rel)**
  - `L_diff_rel = MSE(R_CMFD(φ_pred, xs_BOC), R_CMFD(φ_GT, xs_BOC))`
  - CMFD-only가 가진 g2 ~6.4% bias floor (NEM 보정 누락분) 가 양변에서 자동 cancel
  - NEM gap, XS staleness, 제어봉 XS 변화 모두 ε_total(t)에 흡수되어 해소
  - 양변 모두 BOC `xs_fuel` (10채널, 시점/branch 무관) 사용. `crit_Sigma_f`/`branch_Sigma_f` 미사용 (cheating 회피)
  - `ε_total(t) = R_CMFD(φ_GT(t), xs_fuel_BOC)` 를 Phase G 전처리에서 사전 계산
  - 상세: `physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md` §3.6 신설

---

## 4. 채널과 파라미터

### 4.1. 텐서 형상

| 텐서 | 형상 | grad |
|---|---|---|
| Quarter LP (입력 함수 전) | (B, 20, 5, 5, 21) | ✗ |
| Halo input (입력 함수 후) | (B, 20, 6, 6, 21) | ✗ |
| Encoder feature (각 layer) | (B, 20, 6, 6, 128) | ○ |
| Encoder output latent | (B, 20, 6, 6, 128) | ○ |
| Mamba output latent | (B, 20, 6, 6, 128) | ○ |
| Decoder feature (각 layer) | (B, 20, 6, 6, 128) | ○ |
| Decoder output | (B, 20, 6, 6, 10) | ○ |
| Final cropped output | (B, 20, 5, 5, 10) | ○ |
| L_diffusion 입력 | (B, 20, 6, 6, 10) | ○ (디코더 출력 그대로) |

### 4.2. Attention 비용

- Encoder/Decoder 토큰 수: **720** (= 20·6·6)
- Quarter (5,5) 가정 대비: (720/500)² = **2.07×** FLOPs
- 절대치 trivial (현재 격자 작아 무시 가능)

### 4.3. Halo cell의 학습 가능 weight

- Halo cell도 디코더 출력 phi를 만들어야 함 — L_diffusion stencil이 그 값을 ghost neighbor로 사용
- 디코더가 halo cell 위치에서 phi 출력을 생성하려면 그 위치에서도 forward 계산이 필요
- 우리 안에서 이 forward 계산은 **cell 단위 공유 weight** 로 자동 처리:
  - CellEmbedder Conv3D(1,1,1): kernel이 cell-wise라 모든 cell에 동일 적용
  - Attention Q/K/V projection: token-wise dense라 모든 token에 동일 적용
  - FFN: token-wise dense라 동일
  - LAPE: 위치별 변수 (각 위치에 D=128개 학습 가능 변수). halo cell 위치에도 LAPE 변수 존재
- 즉 halo cell이 별도의 weight 세트를 가질 필요는 없지만, 모델 weight 자체는 halo cell 위치에서도 작동
- LAPE 추가 변수: (6,6) 격자 → 23,040개 vs (5,5) → 16,000개. 차이 **7,040개** (인코더 0.66M의 1.07%). LAPE만이 위치별 변수이므로 격자 확장의 추가 파라미터 비용은 LAPE에만 발생

### 4.4. 그래디언트 흐름 (개정 후)

- Halo cell의 GT 정의: 데이터셋에서 얻은 (5,5) quarter GT를 입력 단계의 `halo_expand()` 함수로 (6,6)으로 변환할 때, halo cell 위치에 함께 채워진 값. 즉 halo GT는 quarter GT의 deterministic copy (mirror 또는 rotational 매핑된 inner cell 값)
- Halo cell의 phi 출력에 도달하는 학습 신호는 **3 경로**:
  1. **L_data_halo (λ=0.3)** — halo GT 직접 비교, 가장 강함
  2. **L_diff_rel stencil neighbor lookup** — inner cell residual의 backward chain (간접)
  3. Decoder attention의 token-wise feature mixing — 부수효과
- 권고 1 (L_data_halo) 채택 전에는 경로 2/3만 존재했으며, L_diff 절대 잔차의 ~6.4% bias floor가 halo cell 출력을 비물리적인 boundary 값으로 수렴시킬 위험이 있었음
- 개정 후: L_data_halo가 dominant 신호로 halo cell phi를 halo GT에 직접 정합시키며, L_diff_rel은 **인접 cell과의 공간 일관성** 을 보강

> **참고 — Sobolev regularizer 란?**
> L² norm (= L_data가 사용하는 단순 MSE) 은 각 cell의 phi 값과 GT 값의 차이만 본다 (pointwise 비교).
> 반면 L_diff_rel은 R_CMFD라는 7점 stencil 연산자를 통과한 차이를 본다. R_CMFD는 인접 cell의 phi를 함께 참조하는 공간 미분 연산자(Laplacian 이산화)이므로, R_CMFD를 거친 차이의 norm은 **단순 phi 차이가 아니라 phi의 공간 곡률(2차 미분)에 가중치를 둔 norm** 이 된다.
> 수학적으로는 함수의 미분 norm을 가중하는 Sobolev space의 H¹/H² norm과 같은 부류이며, 이를 손실 함수로 사용하면 모델 출력의 공간 매끄러움(smoothness)과 인접 cell 간 일관성(coherence)을 강제하는 효과가 있다.
> 이런 류의 손실 함수를 통칭 **Sobolev regularizer** 라고 부른다.

---

## 5. STRING / LAPE 적용

| 모듈 | 격자 | 토큰 수 |
|---|---|---|
| Encoder STRING/LAPE | halo (6,6) | 720 |
| Decoder STRING/LAPE | halo (6,6) | 720 |

- **개정 (2026-04-08)**: 기존 표는 Encoder를 quarter (5,5), Decoder를 halo (6,6)로 분리 표기했으나 §1 (전체 파이프라인) 및 §4.1 (텐서 형상)과 모순. encoder도 (6,6) 처리로 통일
- 통일 사유:
  - §1 의도: encoder 입력부터 (B, 20, 6, 6, 21) → encoder가 boundary cell에서 halo를 attention input으로 직접 봄 → §3.2 ("대칭 inductive bias 흡수") 메커니즘 작동
  - 만약 encoder (5,5), decoder (6,6) 분리 시: Mamba 출력 (5,5)와 decoder 입력 기대 (6,6) 간 shape mismatch → halo expand 2회 필요 → §2 폐기 사유 (1차 안)와 동일한 함정
- 디코더의 STRING은 (6,6) 좌표 기준. halo cell도 STRING/LAPE 위치에 포함됨
- 디코더 LAPE 변수가 인코더 LAPE와 별도/공유는 디코더 plan에서 결정 (본 plan 범위 외). 단 별도 변수일 경우 두 격자의 좌표축 정렬 (encoder LAPE의 (h,w)=(0,0)과 decoder LAPE의 (h,w)=(0,0)이 동일 halo cell을 가리킴) 보장 필수

---

## 6. Symmetry 정보 흐름

| 단계 | sym 사용 | 어떻게 |
|---|---|---|
| Input function | ✓ | quarter → fullcore 매핑 분기문 |
| Encoder | ✗ | fullcore에 이미 매핑된 값 |
| Decoder | ✗ | 동일 |
| L_diffusion | ✓ | quarter 출력 → output halo 매핑 분기문 |

- Config: `data.symmetry` — HDF5 metadata의 `symmetry_type` 에서 읽음
- 두 곳에서 동일 함수 (`halo_expand`) 호출
- 모델(encoder/decoder)은 sym을 명시적 입력으로 받지 않으나, **halo cell의 값 패턴으로부터 대칭 유형을 암묵적으로 학습 가능** — 수학적 증명: [05a_symmetry_distinguishability_proof.md](05a_symmetry_distinguishability_proof.md)
- **대칭 유형 정보 보존 (Critical)**: `crop_type` (quarter crop 방식) 과 `symmetry_type` (LP 기하 대칭 유형) 은 별개 정보이며, 혼동하면 halo_expand 오작동. 상세: [05a §8.3a](05a_symmetry_distinguishability_proof.md)
  - 현재 100LP: `symmetry_type = 'quarter_mirror'` (생산 코드 변경 전 생산, 2026-03-26)
  - 향후 신규 LP: `symmetry_type = 'quarter_rotational'` (생산 코드 변경 후, 2026-03-30~)
  - HDF5 metadata에 `symmetry_type` 필드 추가 필요 (현재 미기록)

---

## 7. 무한 확장 우려 답

- L_diffusion 합산 도메인 = 5×5 core cell (총 500) 고정
- Output halo cell은 stencil neighbor lookup 용도. residual 합산 안 함
- Halo cell의 stencil 계산 자체를 안 하므로 halo의 halo 불필요
- 재귀 없음

기존 코드 검증 (`piecewise-test/2026-04-01_L_diffusion_endtoend_test.py:142-247`):
- `for qy in range(5): for qx in range(5):` 로 5×5 core cell만 순회
- mirror 분기 (line 187-195)는 inline neighbor lookup용
- 본 안의 output halo는 이 inline lookup을 별도 텐서로 분리한 것 — 수학적으로 등가

---

## 8. 검토한 다른 옵션

| 옵션 | 거부 사유 |
|---|---|
| Loss 함수 안 분기문 (`if symmetry == ...` inline) | 분기가 loss 코드 곳곳에 흩어짐 |
| Full-core 모델 입력 (인코더도 fullcore) | Attention FLOPs 10.5×, ~75% redundancy. `01_input_grid.md` 거부 사유 동일 |
| D4-equivariant attention | 학술적, 구현 복잡도 높음, over-engineering |
| 디코더 출력 (6,6) (halo cell까지 출력, GT 없음) | **개정 후 폐기**: 본 plan은 halo cell까지 출력하되 L_data_halo (λ=0.3)로 직접 supervision 부과하는 형태로 채택. "GT 없음 → 학습 신호 0 → 임의 값 생성" 우려는 L_data_halo 부과로 해소 (§3.1 참조) |

---

## 9. 본 plan 적용 범위

본 plan은 인코더 SE-A/B만 다룸. halo expand 구현은 별도 plan(L_diffusion / dataset 단계).

본 plan에서 처리할 사항:
- `configs/model.yaml`에 `data.symmetry: "mirror"` 항목 추가 (SE-B2)
- 인코더 입력 형식: 사용자 확정 파이프라인 상 fullcore (9,9)이지만 인코더 내부에서 quarter (5,5) crop. 본 plan의 SE-A/B 코드는 quarter (5,5)를 받는 형태로 유지하고, 모델 wrapper가 fullcore → quarter crop을 수행하는 형태로 분리 (SE-B1에서 명시)
- 본 보고서 작성

---

## 10. 결정 과정

- 2026-04-08 사용자 의문: "L_diffusion ghost cell 매핑이 mirror인지 rotational인지 데이터에 따라 달라짐. 추론 시점에 모델이 어떻게 알 수 있는가?"
- 검토 시나리오:
  - A1: Quarter + loss 안 분기문
  - B2: Full-core 입력 (인코더가 fullcore 처리)
  - D-clarified (제 변형): Quarter 입력 + L_diffusion 직전 halo expand만
- **사용자 정정 (2026-04-08)**: D-clarified 폐기. 사용자 확정 안:
  1. 입력 함수가 quarter + sym → fullcore 만듦
  2. 모델이 fullcore 받음
  3. 인코더는 quarter crop, 디코더는 halo crop
- 사용자 4가지 우려 ((a) input halo 의미, (b) 대칭조건 채널, (c) 디코더 layer 범위, (d) L_diffusion 시점) 전부 답변 후 본 보고서 작성

---

## 11. 참고

- 기존 L_diffusion 코드 (mirror 분기 inline): `piecewise-test/2026-04-01_L_diffusion_endtoend_test.py:142-247`
- 입력 격자 결정: 본 폴더 `01_input_grid.md`
- BC mask 미사용: 본 폴더 `06_omitted_options.md`
