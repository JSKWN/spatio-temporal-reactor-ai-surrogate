# spatio-temporal-reactor-ai-surrogate 모델 구현 계획

> **작성일**: 2026-03-29
> **개정일**: 2026-04-09 — §2.1 신설 (단계별 처리 형상), §3 아키텍처 + §5 폴더 구조 + §6 Phase 표를 halo (6,6) 처리 + L_diff_rel + L_data_halo 결정 반영하여 갱신. 상세 결정 이력: `공간인코더 구현 계획/인코더 컴포넌트별 채용 이유/05_symmetry_mode.md`, `physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md` §3.6, ML 위협/일관성 plan `C:\Users\Administrator\.claude\plans\compressed-wandering-stroustrup.md`

## Context

데이터 생산(gen_master_load_follow) 및 전처리(lf_preprocess) 패키지가 완성되어 HDF5 데이터셋 생성이 가능한 상태.
`spatio-temporal-reactor-ai-surrogate` 저장소에 모델 코드를 구현한다.
현재 이 저장소에는 설계 문서 5개 + LICENSE + requirements.txt만 존재하며 Python 코드는 전무.

---

## 1. 관련 문서

### 통합 레퍼런스 (2026-03-30, 1순위 참조)

> 아래 3개 문서가 산재된 원본 파일들의 내용을 통합·정리한 최신 참조 문서임.

| 문서 | 핵심 내용 |
|------|----------|
| **`2026-03-30 Physical Loss 통합 레퍼런스.md`** | §0 HDF5 스키마, §1 L_Bateman(ODE 수식/Markov 상태/수치적분/MASTER PC 차이), §2 L_sigma_a_Xe(MASTER Taylor 메커니즘/검증/손실), §3~4 확장용(L_diffusion/L_keff), §5 가중치, Appendix A/B |
| **`2026-03-30 Mamba SSM 수학 레퍼런스.md`** | 연속 SSM → ZOH 이산화 → S6 선택적 스캔, 원자로 물리 대응, 훈련/추론 이중성, Mamba-2(SSD)/Mamba-3 |
| **`2026-03-30 제논 미시단면적 Taylor 계수 HDF5 스키마.md`** | Taylor σ_a^Xe HDF5 저장 구조 (Z,qH,qW,2,6), ref_conditions, dmod_delta, Phase F multi-BU 확장, 런타임 보간 |
| **`2026-03-30 모델 구현 계획(공간 인코더).md`** | 공간 인코더 세부 설계: 기존 코드 분석(Prob-1~11), 향상 후보(Enh-1~4), 아키텍처, L_diffusion 가용성, IIET PC 루프, p_load 처리 |
| **`2026-03-30 Neural PC(IIET) 도입 고찰.md`** | MASTER Full PC vs IIET 비교, 도입 경로 A/B/C 분석, 구현 우선순위 |

### Physical Loss 개선 계획 (2026-03-31 ~ 04-01, 통합 레퍼런스 보충)

| 문서 | 핵심 내용 |
|------|----------|
| **`physical loss 개선 계획/2026-03-31 L_diffusion 노드 vs 집합체 CMFD 비교 계획.md`** | 노드(10.8cm) vs 집합체(21.6cm) CMFD 잔차 비교 설계 |
| **`physical loss 개선 계획/2026-04-01 JNET0 N-S 매핑 오류 발견 및 해결.md`** | FACE_DIR N/S 방향 반전 오류 발견(R0~R3 영향) 및 수정 |
| **`physical loss 개선 계획/2026-04-01 JNET0 활용 방안 계획.md`** | 작업 A: Albedo α 역산, 작업 B: D̂ 보정항 역산, 작업 C: irreducible floor |
| **`physical loss 개선 계획/2026-04-01 L_diffusion 연료 및 반사체 지점 loss 반영 계획.md`** | boundary_mask 설계 (6종 면 유형), 전체 연료 L_diffusion 적용, 의사코드 |
| **`physical loss 개선 계획/Albedo 캘리브레이션을 통한 초기값 확정/`** | R0~R5 캘리브레이션 종합 보고서 + 확정값 |

### V&V 검증 스위트 (2026-04-02 추가)

| 디렉토리 | 핵심 내용 |
|----------|----------|
| **`v&v/01_jnet0_direction/`** | JNET0 방향/크기 최종 검증: α=2 밸런스 median 0.000147%, 전류 연속 exact 0 |
| **`v&v/02_cmfd_consistency/`** | CMFD 방향 규약 + JNET0 대조 검증 (진행중) |

### 원본 파일 (deprecated / 대체됨 — 통합 문서에 흡수)

| 파일 | 상태 | 통합 대상 |
|------|:----:|:---------:|
| `2026-03-29 Physical Loss 설계 보고서 (대체됨).md` | 대체됨 | 문서 A |
| `Xe-135 동역학...MASTER 연동 (대체됨).md` | 대체됨 | 문서 A |
| `archive/2026-03-23 구현 계획 초안 (deprecated).md` | deprecated | 문서 B |
| `archive/2026-03-23 구현 계획 초안 추가 (deprecated).md` | deprecated | — |
| `archive/2026-03-17 vit3d_mamba_architecture_report (deprecated).md` | deprecated | 문서 B |
| v-smr: `physical loss 적용단계 계획 (대체됨).md` | 대체됨 | 문서 A |
| v-smr: `Taylor 계수 HDF5 스키마 구성 방안 (대체됨).md` | 대체됨 | 문서 C |
| v-smr: `이산화 수치해석 공식 (deprecated).md` | deprecated | 문서 A |
| v-smr: `sigma_a_xe_검증 (대체됨).md` | 대체됨 | 문서 A Appendix B |
| v-smr: `Perplexity 조사자료 (deprecated).md` | deprecated | — |

### 데이터 생산/전처리 이력 (간접 참조)

| 파일 | 용도 |
|------|------|
| `2026-03-24_lf_gen_preproc_package_modifying_plan.md` | Phase A~E 이력, HDF5 스키마 확정 경위 |
| `data_preprocess/lf_preprocess/dataset_builder.py` | HDF5 실제 스키마 (그룹 경로, shape) — **구현 시 코드 직접 참조** |
| `data_preprocess/lf_preprocess/normalizer.py` | 정규화 방법 참조 구현 — **런타임 normalizer와 동일하게 구현 필요** |

---

## 2. 확정 사항

- **공간 형상**: (20, 5, 5) [데이터 입력] → halo expand → (20, 6, 6) [인코더/Mamba/디코더 처리] → crop → (20, 5, 5) [최종 출력] (1/4 대칭, 상세 §2.1)
- **TF 버전**: 2.14 고정 (Keras 내장, ONNX 호환)
- **C++20 연동**: SavedModel + ONNX 병행 (사전 빌드 라이브러리 호출, 직접 빌드 불필요)
- **데이터 포맷**: HDF5 유지 (TFRecord 불필요)
- **순수 TF ops만 사용** (custom CUDA kernel 금지 — ONNX 변환 호환성)

### 2.1 단계별 처리 형상 (개정 2026-04-09)

`05_symmetry_mode.md` 결정에 따라 데이터 입력 / 인코더 처리 / 최종 출력 단계가 서로 다른 격자 형상을 사용. 본 절은 각 단계의 형상을 명시.

| 단계 | 형상 | 의미 | 결정 출처 |
|---|---|---|---|
| **데이터 입력** | (B, T, 20, 5, 5, 21) | 데이터셋에서 로드되는 quarter core 형상. state(10) + xs_fuel(10) + rod_map(1) | `01_input_grid.md` |
| **halo expand 후** | (B, T, 20, 6, 6, 21) | 입력 함수 `halo_expand(sym)` 적용 후, 인코더 진입 전 (mirror 또는 rotational 매핑) | `05_symmetry_mode.md` §1 |
| **인코더 처리** | (B, 20, 6, 6, 128) | CellEmbedder ~ attention stage 모두 (6,6) feature space (per timestep) | `05_symmetry_mode.md` §1, §4.1 |
| **Mamba 처리** | (B×720, T, 128) → (B×720, T, 128) | (6,6) latent를 cell 축으로 flatten 후 시간 시퀀스 처리 | `05_symmetry_mode.md` |
| **디코더 처리** | (B, 20, 6, 6, 128) → (B, 20, 6, 6, C_out) | 디코더도 (6,6) feature space | `05_symmetry_mode.md` |
| **L_diff_rel 입력** | (B, 20, 6, 6, C_out) | 디코더 출력 그대로, 합산 도메인은 5×5 inner cell, halo는 stencil neighbor lookup | `Physical Loss 통합 레퍼런스 §3.6` |
| **L_data_halo 입력** | halo cell 11개 (= (6,6) − 5×5 inner) | halo cell phi에 직접 supervision (λ=0.3) | `05_symmetry_mode.md` §3.1 |
| **최종 외부 출력** | (B, T, 20, 5, 5, C_out) | crop 1회 (외부 인터페이스 한 곳에서만) | `05_symmetry_mode.md` §1 |

각 단계의 컴포넌트:
- **halo_expand()**: 입력 함수 (data preprocessing 단계 또는 model wrapper 진입 시) - 1회 호출
- **CellEmbedder, FullAttention3D, FFN3D, Mamba, Decoder**: 모두 (6,6) feature space 처리, halo cell이 (5,5) inner cell과 동등하게 attention/state 처리에 참여
- **Final crop**: 외부 인터페이스 한 곳에서만 (B, 20, 6, 6, C_out) → (B, 20, 5, 5, C_out)

---

## 3. 아키텍처

**핵심 원칙**: Pooling 없이 720개 cell 공간 해상도 완전 보존 (= 500 inner + 220 halo expand)

```
[데이터 입력] (B, T, 20, 5, 5, C_in)               ← 데이터셋 quarter

[halo_expand(sym)] mirror 또는 rotational 매핑 (1회)
  (B, T, 20, 5, 5, C_in) → (B, T, 20, 6, 6, C_in)

[공간 인코더] 시간스텝마다 (6,6) feature space에서 attention 처리
  (B, T, 20, 6, 6, C_in) → per timestep → (B, T, 20, 6, 6, D_latent)
  ├ CellEmbedder Conv3D(1,1,1)
  ├ LearnedAbsolutePE3D add (1회)
  └ Stage 1~3: Pre-LN + FullAttention3D (Q/K STRING 회전) + FFN3D

[reshape] cell 축을 batch에 합침
  (B, T, 20, 6, 6, D_latent) → (B×720, T, D_latent)

[Mamba] 하나의 Mamba가 모든 cell의 시간 시퀀스를 처리 (가중치 공유)
  (B×720, T, D_latent) → (B×720, T, D_hidden)

[reshape] 다시 3D로 복원
  (B×720, T, D_hidden) → (B, T, 20, 6, 6, D_hidden)

[디코더] (6,6) feature space에서 물리량 복원 ← L_diff_rel + L_data_halo 적용 지점
  (B, T, 20, 6, 6, D_hidden) → (B, T, 20, 6, 6, C_out)

[Final crop] 외부 인터페이스 한 곳에서만
  (B, T, 20, 6, 6, C_out) → (B, T, 20, 5, 5, C_out)
```

**물리적 근거**:
- 인접 노드 커플링(중성자 확산 누설항)은 인코더/디코더의 attention/Conv가 담당
- halo cell이 quarter core 대칭 너머의 fuel cell 정보를 직접 인코더에 전달 → boundary cell의 attention input에 대칭 inductive bias 자동 흡수
- 각 cell의 시간 변화(Xe 축적 등)는 local 물리량에 주로 의존 → cell별 독립 Mamba가 물리적으로 타당
- L_diff_rel (CMFD 상대 잔차) 가 디코더 출력에 인접 cell 커플링 + Albedo BC 학습 신호 추가
- L_data_halo (λ=0.3) 가 halo cell phi에 직접 supervision → 비물리적 발산 차단

---

## 4. Physical Loss 설계 (개정 2026-04-09)

### 핵심 4종 (우선 구현)

| Loss | 수식 | 타겟 | 비고 |
|------|------|------|------|
| **L_data** | 다중 출력 가중 MSE (inner 5×5) | GT 전체 필드 | 필수 |
| **L_data_halo** | 다중 출력 가중 MSE (halo 11 cell) | halo_expand로 변환된 sym 매핑 GT | λ=0.3 (cell 비율 0.44에서 임의 discount). halo cell 비물리적 발산 차단. 상세: `05_symmetry_mode.md` §3.1 |
| **L_Bateman** | Xe/I-135 Bateman ODE 잔차 | GT 기반 ODE 적분 결과 | Euler/해석해 모두 허용 (차이 mean 0.02%p). MASTER Full PC와의 구조적 차이로 ~0.9% 바닥 오차 (Physical Loss만으로 해소 불가, L_data 담당). 상세: 통합 레퍼런스 §1 |
| **L_sigma_a_Xe** | Taylor 1차 전개 σ_a^Xe 일관성 | GT σ_a^Xe | **모델 예측** T_f, ρ_m으로 MASTER Taylor 재현. 상세: 통합 레퍼런스 §2 |

### 확장 (가능성 파악 후 추가)

| Loss | 수식 | 구현 난이도 | 비고 |
|------|------|-----------|------|
| **L_diff_rel** | `‖R_CMFD(φ_pred, xs_BOC) − R_CMFD(φ_GT, xs_BOC)‖²` (상대 잔차) | 중간 | CMFD-only g2 ~6.4% bias floor를 양변 cancel로 우회. Albedo BC (12 학습 가능 파라미터) 학습이 주된 가치. NEM gap + XS staleness + rod XS 변화 모두 ε_total(t)에 흡수. 상세: 통합 레퍼런스 §3.6 |
| **L_keff** | Rayleigh 몫 `k_pred vs k_GT` | 낮음 | L_diffusion의 부분집합 |

> **L_diffusion 절대 잔차 (기존) 폐기**: `‖R_CMFD(φ_pred)‖²` 형식은 NEM 보정 누락으로 인한 ~6.4% Consistency Barrier에 갇힘. 본 안 (L_diff_rel) 으로 대체. 상세: 통합 레퍼런스 §3.6.1, §3.6.6

### 가중치 전략 (개정 2026-04-09)

- Warm-up: λ_data=1.0, λ_data_halo=0.3, λ_Bateman=0.01, λ_σXe=0.01, λ_diff_rel=0, λ_keff=0
- Ramp-up: λ_diff_rel, λ_keff 선형 증가
- Full training: λ_data=1.0, λ_data_halo=0.3, λ_Bateman=0.5, λ_σXe=0.3, λ_diff_rel=0.05~0.1, λ_keff=0.1
- λ_data_halo = 0.3: cell 비율 기반 중립값 0.44 (= 11/25) 에서 임의 discount. ablation 권장 (0.1, 0.3, 0.5)
- λ_diff_rel = 0.05~0.1: inner cell에서 redundant이므로 작아도 충분. 주된 가치는 Albedo BC 학습

---

## 5. 폴더 구조

```
spatio-temporal-reactor-ai-surrogate/
├── LICENSE, requirements.txt        # (기존)
├── configs/
│   ├── model.yaml                   # 인코더/Mamba/디코더 하이퍼파라미터
│   ├── train.yaml                   # 학습 설정, loss weights, warm-up
│   └── data.yaml                    # HDF5 경로, split, 시퀀스 설정
├── src/
│   ├── models/
│   │   ├── spatial/                 # 공간 인코더 (attention 기반)
│   │   │   ├── halo_expand.py       # quarter (5,5) → halo (6,6) mirror/rotational
│   │   │   ├── layers3d.py          # CellEmbedder, LearnedAbsolutePE3D, FFN3D
│   │   │   ├── attention3d.py       # FullAttention3D, STRINGRelativePE3D
│   │   │   ├── encoder3d.py         # (20,6,6,C_in) → (20,6,6,D_latent) [halo expand는 wrapper]
│   │   │   └── decoder3d.py         # (20,6,6,D_hidden) → (20,6,6,C_out)
│   │   ├── temporal/                # Mamba (cell별 공유 가중치)
│   │   │   ├── ssm_layer.py         # S6 selective scan
│   │   │   ├── mamba_block.py       # Linear → Conv1D → SSM → Gate
│   │   │   └── temporal_model.py    # N블록 스택
│   │   └── surrogate.py             # 전체 파이프라인 조립 (halo_expand → enc → mamba → dec → final crop)
│   ├── data/
│   │   ├── hdf5_loader.py
│   │   ├── sequence_dataset.py      # ε_total(t) 사전 계산값 함께 반환
│   │   ├── normalizer.py
│   │   └── constants.py             # 물리상수 (λ_Xe, λ_I, γ, Δt 등)
│   ├── losses/
│   │   ├── data_loss.py             # L_data (inner) + L_data_halo (halo)
│   │   ├── bateman_loss.py          # Xe+I 통합, 해석해 기반
│   │   ├── taylor_sigma_loss.py
│   │   ├── diffusion_loss.py        # L_diff_rel: compute_R(φ_pred), compute_R(φ_GT) 차분
│   │   ├── keff_loss.py             # (확장용, Rayleigh 몫)
│   │   └── composite_loss.py
│   └── training/
│       ├── trainer.py
│       ├── scheduler.py
│       └── callbacks.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
└── notebooks/
```

---

## 6. 구현 단계

| 단계 | 내용 | 검증 |
|------|------|------|
| **Phase 0** | halo_expand 함수 + ε_total(t) 사전 계산 (Phase G 또는 신규 Phase H) | (B,T,20,5,5,C_in) → (B,T,20,6,6,C_in) 변환, ε_total HDF5 저장 |
| **Phase 1** | 공간 인코더 — Stem(Conv3D 1×1×1) + LAPE add + FullAttention3D × 3 stages (STRING Q/K 회전) | (B,20,6,6,C_in) → (B,20,6,6,D_latent) shape, gradient 흐름, halo cell 학습 신호 |
| **Phase 2** | Mamba temporal (ssm_layer, mamba_block, temporal_model) | (B×720,T,D) forward pass |
| **Phase 3** | 디코더 + surrogate 조립 (halo_expand → enc → mamba → dec → final crop) | full forward: (B,T,20,5,5,C_in) → (B,T,20,5,5,C_out), 중간 단계는 (6,6) |
| **Phase 4** | configs + data pipeline (hdf5_loader, normalizer, sequence_dataset, constants) + Loss 4종 (L_data + L_data_halo + Bateman해석해 + Taylor) + trainer | HDF5 로드 → tf.data shape, 학습 곡선 확인 |
| **Phase 5** | L_diff_rel + Albedo BC (12 학습 가능 파라미터) 구현 | compute_R(φ_pred), compute_R(φ_GT) 차분, ε_total(t) cancel 검증 |
| **Phase 6** | evaluate + inference (autoregressive) | test split MAPE, halo cell 모니터링 metric (M1~M6) |

> **Phase 4 후순위 사유**: 데이터 전처리(100LP)가 아직 완료되지 않았으므로, configs + data pipeline은 모델 구조(Phase 1~3) 확정 후 전처리 완료 시점에 맞춰 구현.
> **Phase 1 전략**: 기존 CustomMaxViT3D(`참고 파일/custom_voxel_attention_alpha_remove_preproc_layers_250808.ipynb`)를 처음부터 새로 작성하지 않고, 필요한 구성요소(MBConv3D, Attention3D 등)를 선별 추출하여 `모델 구현 계획(공간 인코더).md`의 Prob-1~11 수정사항을 적용. 상세: `모델 구현 계획(공간 인코더).md` §4~§7.

---

## 7. 검증 방법

1. `pytest tests/test_data_pipeline.py` — HDF5 로드, 정규화 왕복, tf.data shape
2. `pytest tests/test_surrogate_e2e.py` — dummy (B,T,20,5,5,C_in) → 올바른 최종 출력 shape (B,T,20,5,5,C_out). 중간 단계 (인코더/Mamba/디코더) 는 (6,6) halo 처리
3. `pytest tests/test_losses.py` — Bateman 해석해 vs 알려진 해, Taylor 일관성
4. `scripts/train.py` — TensorBoard loss 감소 확인
5. `scripts/evaluate.py` — test split MAPE < 5%

### 7.1 학습 전 사전 검증 (Phase 1~3 구현 후, 학습 없이 수행)

| 검증 | 방법 | 확인 대상 |
|------|------|----------|
| **Gradient 흐름** | 랜덤 초기화 → dummy forward → loss.backward → 전 파라미터 grad ≠ 0 | relative_bias_table 등 특정 레이어의 gradient 단절 여부 |
| **단일 샘플 과적합** | GT 1개 (입력→출력) 1쌍으로 수백 step 학습 → loss → 0 수렴 여부 | 모델 표현력 충분성. 채널별 loss 분리로 병목 채널 특정 |
| **조건 신호 감도 (p_load)** | p_load 값만 변경, 나머지 입력 고정 → 출력 변화량 측정 | AdaLN-Zero가 실제로 출력을 변조하는지 |
| **xs_fuel 채널 기여도** | xs_fuel 10ch를 0으로 마스킹 → 출력 변화량 측정 | concat된 xs_fuel이 인코더에서 실제로 활용되는지 |

---

## 8. L_diffusion 사전 검증 테스트 (2026-03-30)

### 8.1 목적

L_diffusion(확산방정식 PDE 잔차 Loss) 도입 가능성을 사전 검증.
MASTER GT φ에 유한차분 7점 스텐실을 적용하여 확산잔차 크기를 확인.
→ 결과에 따라 **MBConv3D 제거 여부** 결정 (`모델 구현 계획(공간 인코더).md` §2.3)

### 8.2 배경

MASTER는 노달법(CMFD 등 고차 방법)으로 확산방정식을 풀지만, 우리 L_diffusion은 **7점 유한차분 스텐실**로 ∇²φ를 근사.
노달법 결과를 유한차분으로 재평가하면 수치적 불일치가 발생할 수 있으며, 불일치가 크면 L_diffusion이 **잘못된 gradient 신호**를 줄 위험.

### 8.3 테스트 데이터

`D:\workspace_lf_20260326_40LP\LP_0000\t12_363_p50_power_lower\`
- MAS_NXS: φ(g1,g2), Σ_f(g1,g2), σ_a^Xe(g1,g2) → `mas_nxs_parser.py`로 추출
- MAS_XSL: xs_fuel 10채널 → Σ_tr → D_g = 1/(3·Σ_tr) → `xs_voxel_builder.py`로 구성
- MAS_OUT: keff, ZMESH(노드 높이) → `core_geometry.py`로 추출

### 8.4 테스트 항목

| 테스트 | 내용 | 데이터 |
|--------|------|--------|
| **T1** | 단일 스텝 확산잔차 | s0001 CRS |
| **T2** | 출력 변동 구간 10스텝 | s0001~s0010 CRS |
| **T3** | 경계 vs 내부 노드 | Z 상하단, XY 모서리 vs 중앙 |

**확산잔차 공식** (2군):
```
R_g(φ) = -∇·D_g∇φ_g + Σ_r,g·φ_g - (χ_g/k_eff)·Σ_g'(νΣ_f,g'·φ_g') - Σ_s,g'→g·φ_g'

여기서:
  D_g = 1/(3·Σ_tr,g)           ← xs_fuel[3], [8]
  Σ_a,g = Σ_c,g + Σ_f,g        ← xs_fuel[2]+[1], [7]+[6]
  Σ_r,g = Σ_a,g + Σ_s,g→g'     ← Σ_a + xs_fuel[4] (g=1), Σ_a + 0 (g=2)
  νΣ_f,g                        ← xs_fuel[0], [5]
  Σ_s,12 = xs_fuel[4], Σ_s,21 = 0 (2군 역산란 무시)
  χ_g = [1.0, 0.0]             ← 하드코딩
  ∇²φ: 7점 유한차분 스텐실 (Z: ZMESH Δz, XY: 균일 어셈블리 피치)
```

### 8.5 판정 기준 (2026-03-30 수정)

> **MASTER는 유한차분(FDM)이 아닌 노달법(NEM/NIM)**으로 확산방정식을 풀이.
> FDM 잔차 5~15%는 코드 오류가 아닌 **방법론 구조 차이** (ADF 불연속, 4차 다항식 전개, 부분전류 경계조건 등).
> 따라서 판정 기준을 재설정함.

| 잔차 범위 (median) | 판정 | 의미 |
|:---:|:---:|------|
| < 1% | PASS | FDM이 노달법을 잘 근사 |
| 1~5% | MARGINAL | ADF 보정 미반영 영향 |
| **5~20%** | **ACCEPTABLE** | FDM vs 노달법 구조 차이. L_diffusion을 물리 정규화 항으로 사용 가능 |
| > 20% | FAIL | 파싱 오류 또는 단면적 불일치 의심 |

### 8.5.1 테스트 방법론

**CMFD 체적 적분 잔차** (방법론 Eq. 2.1-2a + Eq. 2.1-37 기반):
- 면 중성자류: CMFD의 D̃ (계면 조화평균) 사용
- 소멸항: Σ_r = Σ_a + Σ_s (방법론 Eq. 2.1-1a 좌변 두 항 합산)
- Inner-only 평가: 경계 노드 제외

> 방법론 §2.1.4: "In the NNEM, both the **CMFD** method and the **two-node NEM** are used."
> 우리는 CMFD만 재현. NEM 보정(D̂, Eq. 2.1-38)은 MASTER 내부 iteration 결과로 재현 불가.
> 이것이 잔차 5~20%의 구조적 원인.

### 8.5.2 테스트 결과 (2026-03-30, 최종)

2 LP (LP_0000, LP_0001) × CRS 10스텝 + Branch 10개, 총 5,760 유효 노드:

| 통계 | g=1 (fast) | g=2 (thermal) |
|------|:----------:|:-------------:|
| mean | 7.01% | 7.17% |
| **median** | **3.50%** | **6.99%** |
| p95 | 19.06% | 13.31% |
| < 5% | 58.3% | 25.5% |
| < 10% | 70.7% | 82.6% |

**판정**: g1 **MARGINAL** (median 3.5%), g2 **ACCEPTABLE** (median 7.0%)
→ **L_diffusion을 공간 커플링 학습의 보조 gradient 신호로 도입 가능**.

### 8.5.3 JNET0 α=2 발견 + 피치 수정 (2026-03-31)

MAS_NXS 물리량 정의를 JNET0 밸런스 역산으로 확정:
- **JNET0 = J_net / 2** (코드 내부 변수 치환 관습, partial current 정의에서 유래)
- 밸런스 시 `leak = 2 × Σ(JNET0_face × A_face)` → α=2 적용 시 잔차 **정확히 0%** (median 0.0002%)
- **WIDE = 21.608 cm** (기존 21.504 cm은 출처 불명 오류, core_geometry.py에서 확인)
- 상세: `Physical Loss 통합 레퍼런스 §3.4.2~3.4.4`

### 8.5.4 노드 vs 집합체 CMFD 비교 (2026-03-31)

6면 이웃 모두 연료인 노드만 평가, 피치 21.608 cm:

| 메트릭 | 노드 CMFD (10.8cm) | 집합체 CMFD (21.6cm) |
|--------|:---:|:---:|
| g1 median | **2.02%** | 2.29% |
| g2 median | **1.23%** | 7.54% |

- g2: 노드 단위에서 6배 개선 → 열중성자 확산거리가 짧아 메시 의존성 큼
- **집합체 CMFD의 구조적 한계 (noise floor)**: g1 ~2.3%, g2 ~7.5%
- 상세: `Physical Loss 통합 레퍼런스 §3.4.3`

### 8.5.5 JNET0 N/S 매핑 오류 발견 및 수정 (2026-04-01)

- MASTER에서 **J 증가 = 남쪽(South)** → 기존 코드가 J 증가 = 북쪽으로 가정 (FACE_DIR N/S 반전)
- R0~R3 Albedo 캘리브레이션이 영향 받음. R4부터 수정 매핑으로 재수행
- 상세: `physical loss 개선 계획/2026-04-01 JNET0 N-S 매핑 오류 발견 및 해결.md`

### 8.5.6 Albedo R5 캘리브레이션 확정 (2026-04-01)

40LP × CRS 10스텝 = 400 시나리오 기반, **12 학습 파라미터** 확정:

```python
# Radial — ortho(R1,R2) / diag(R3~R6)
alpha_ortho = {g1: 0.108, g2: 0.453}   # R²=0.998
alpha_diag  = {g1: 0.082, g2: 0.513}   # R²=0.988~0.997

# Axial — 행렬 C (2×2)
C_bottom = [[+0.155, -0.135], [-0.025, +0.078]]  # R²=(0.999, 0.992)
C_top    = [[+0.174, -0.097], [-0.036, +0.080]]  # R²=(0.993, 0.925)
```

- trainable이므로 학습 중 자동 보정됨
- 상세: `Physical Loss 통합 레퍼런스 §3.5`, `physical loss 개선 계획/Albedo 캘리브레이션을 통한 초기값 확정/`

### 8.5.7 end-to-end 대규모 검증 (2026-04-01)

10LP × CRS 10스텝 = 100 시나리오, Albedo BC 적용 전체 연료 노드(38,000개):

| 구분 | g1 median | g1 mean | g2 median | g2 mean |
|------|:---------:|:-------:|:---------:|:-------:|
| 내부 노드 (10,800) | 1.91% | 2.77% | 6.75% | 7.48% |
| 경계 노드 (27,200) | 2.45% | 4.02% | 6.29% | 8.20% |
| **전체 (38,000)** | **2.24%** | **3.67%** | **6.44%** | **7.99%** |

- 초기 결과(§8.5.2, 2LP inner-only) 대비 대규모 + 경계 포함
- CMFD 부호 통일, zmesh 인덱싱, 대칭면 Mirror CMFD 3개 버그 수정 반영
- 상세: `piecewise-test/2026-04-01_L_diffusion_endtoend_결과.md`

### 8.5.8 boundary_mask 설계 — 전체 연료 노드 L_diffusion

기존 inner-only(내부 28%) → 전체 연료(100%) L_diffusion 적용. 면 유형 6종 분류:

| 유형 | 코드 | 누설 계산 |
|------|:---:|---------|
| 내부 | 0 | CMFD: D̃×(φ̂_nb - φ̂_center)/h × A |
| ortho 반사체 | 1 | Marshak: α_ortho×D/(αh/2+D)×φ̂ × A |
| diag 반사체 | 2 | Marshak: α_diag 동일 |
| bottom/top 반사체 | 3/4 | 행렬 C: C × [φ̂_g1, φ̂_g2] × A |
| 대칭면 | 5 | Mirror CMFD (REFLECT) |

- 상세: `physical loss 개선 계획/2026-04-01 L_diffusion 연료 및 반사체 지점 loss 반영 계획.md`

### 8.5.9 판정 업데이트

초기(§8.5.2): MARGINAL/ACCEPTABLE (inner-only, 2LP, 버그 수정 전)
→ **최종(§8.5.7)**: 전체 연료 + Albedo BC + 버그 수정 후 g1 2.24%, g2 6.44%
→ **L_diffusion 설계 확정**. 이 잔차는 집합체 CMFD의 구조적 한계이며, L_diffusion은 공간 커플링 학습의 **보조 gradient 신호**로 사용.

### 8.6 테스트 코드 및 결과 위치

- 초기 테스트: `piecewise-test/2026-03-30_diffusion_residual_test.py`, `_result.txt`, `_result.png`
- end-to-end: `piecewise-test/2026-04-01_L_diffusion_endtoend_test.py`, `_결과.md`
- V&V: `v&v/01_jnet0_direction/`, `v&v/02_cmfd_consistency/`

### 8.7 전처리 코드 병렬화 — ✅ 완료 (2026-04-02)

Phase G에서 전처리 파이프라인 병렬화 완료:
- ProcessPoolExecutor 기반 LP 수준 병렬 처리
- 다중 워크스페이스 지원 (D: LP_0000~0039, E: LP_0040~0099, 글로벌 LP ID 매핑)
- 임시 HDF5 per worker → 완료 후 병합

---

## Appendix A. Bateman ODE 수치 적분 방법 — Euler vs 해석해 오차 검증

### 검증 개요

- **검증 시점**: 2026-03-25 (E1-T5, E1-T5b 테스트)
- **검증 데이터**: MASTER GT (MAS_NXS s0001~s0005)
- **Δt**: 300s (5분)
- **검증 결과 파일**: `unit_tests/modifying_plan_phase_E/E1_T5_euler_analytic_results.txt`, `E1_T5b_multistep_results.txt`

### 단일 스텝 결과 (s0001→s0002, 2800노드)

| 기법 | mean |상대오차| | max |상대오차| | <1% 비율 |
|------|:---:|:---:|:---:|
| Euler forward | 1.1034% | 4.3632% | 55.3% (1547/2800) |
| Analytic | 1.1017% | 4.3219% | 54.9% (1538/2800) |

Analytic이 Euler보다 정확한 노드: 1370/2800 (48.9%) — **거의 반반**

### 연속 5스텝 결과 (11,200 노드-스텝 합산)

| 기법 | mean | max | <1% 비율 | <5% 비율 |
|------|:---:|:---:|:---:|:---:|
| Euler | 0.9170% | 4.3632% | 64.5% | **100%** |
| Analytic | 0.9068% | 4.3219% | 65.3% | **100%** |
| Analytic+PC | 0.9052% | 4.3179% | 65.5% | **100%** |

### 대표 노드 비교

```
노드 (11,11,12):
  N_Xe(t) = 2.316930E-09, N_I(t) = 5.638000E-09
  Euler:    2.304094E-09  (오차 1.2551%)
  Analytic: 2.304321E-09  (오차 1.2454%)
  MASTER GT: 2.333380E-09

노드 (13,13,12):
  N_Xe(t) = 2.297100E-09, N_I(t) = 7.387000E-09
  Euler:    2.302938E-09  (오차 0.5086%)
  Analytic: 2.302828E-09  (오차 0.5133%)
  MASTER GT: 2.314710E-09
```

### 결론

1. **Euler vs Analytic 차이: mean 0.02%p, max 0.04%p** — Δt=5min에서 실질적으로 동등
   - Δt/τ_Xe ≈ 300/47600 ≈ 0.006, 1차 절단 오차 O(Δt²) ≈ 0.09%
2. **오차의 주요 원인은 적분 방법이 아닌 MASTER Full Predictor-Corrector** (매뉴얼 확인: 항상 Full PC 사용)
   - 우리는 GT φ(t)만 사용하므로 MASTER 내부의 φ 갱신 효과가 미반영됨
   - 이것이 mean ~0.9% 오차의 원인 중 하나 (열수력 피드백 등 복합 요인)
3. **Physical Loss 적분 방법: 둘 다 허용**
   - 두 방법의 차이가 미미(mean 0.02%p)하므로 구현 시 선택
   - Euler: 단순 (1줄), 해석해: tf.exp 기반 (5줄)
