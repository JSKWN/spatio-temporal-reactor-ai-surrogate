# 04. 정규화 + 정밀도 + 미채용 옵션 종합

> **결정**: Pre-LN + LayerNorm + FP32(TF32) / 미채용 7개 항목 (QK-Norm, Register tokens, BC mask, Calibration-FiLM, Perceiver IO, MoNODE, ContextAR)
> **결정 일자**: 2026-04-04 ~ 2026-04-08
> **본 문서의 위치**: 이전 plan의 06/07 결정 사항을 통합·압축. "보조/제외" 결정으로 묶어 정리

---

## 1. 정규화 — Pre-LN + LayerNorm

### 1.1. 검토 옵션 표

| 정규화 기법 | 적용 위치 | 채택 |
|---|---|:---:|
| **Pre-LN + LayerNorm** | LN → Attention → residual add / LN → FFN → residual add | **✓** |
| Post-LN + LayerNorm | Attention → LN → residual add | ✗ |
| Pre-LN + BatchNorm | LN 대신 BN | ✗ |
| Pre-LN + RMSNorm | LN 대신 RMSNorm | ✗ |

### 1.2. 거부 사유

- **BatchNorm**: 미니배치 의존 + 추론 시 running mean/var 불일치 + B가 작을 수밖에 없는 3D 텐서 (B, 20, 6, 6, 21) 에서 통계 신뢰도 부족. Attention 기반 모델에서 표준적으로 퇴출
- **Post-Norm**: Pre-Norm 대비 학습 불안정 + 깊은 layer 에서 gradient 감소. Transformer 커뮤니티에서 Pre-Norm 이 표준 (GPT-2 이후)
- **RMSNorm**: LayerNorm 에서 mean centering 을 제거한 변형. TF2.14 의 기본 Keras `LayerNorm` 과의 호환성, 추가 이점 부재

### 1.3. 채택 근거

- **Attention 기반 모델 표준**: GPT-2, ViT, BERT 등 모든 주류 Transformer 가 Pre-LN 채택
- **추론 안정**: 학습/추론 시 동일한 LayerNorm 연산 (배치 통계 불필요)
- **Gradient flow**: Pre-Norm 의 residual stream 이 gradient 를 layer 전체에 안정적으로 전달
- **TF2.14 Keras 호환**: `tf.keras.layers.LayerNorm` 으로 직접 사용 가능, 추가 구현 불필요

### 1.4. 인코더 vs 디코더 분리

- **인코더**: 일반 Pre-LN (조건 변조 없음). 입력 조건 (state, xs_fuel, rod_map) 은 채널 concat 으로 직접 투입
- **디코더**: **AdaLN-Zero Pre-LN** — p_load (스칼라 조건) 로 LayerNorm 의 γ, β 를 변조. DiT (Peebles & Xie, 2023 "Scalable Diffusion Models with Transformers") 패턴
  - 인코더에 p_load 를 FiLM 으로 주입하지 않는 이유: p_load 는 *다음 시점의 출력* 을 조건부로 만드는 조건이지 *현재 상태의 공간 압축* 에는 무관. 인코더 역할은 "state(t) + rod_map(t+1) + xs_fuel → 전이 정보 추출" 이며 p_load 는 디코더/Mamba 에서 적용
- 디코더의 AdaLN-Zero 상세는 SD-Phase (공간 디코더) plan 에서 결정 (본 문서 범위 외)

상세: `2026-04-04 정밀도 및 경량화 검토.md` §5, `2026-03-30 모델 구현 계획(공간 인코더).md` line 334-337

---

## 2. 정밀도 — FP32 + TF32 (GPU 자동 가속)

### 2.1. 검토 옵션 표

| 포맷 | 가수 bit | 지수 bit | 유효숫자 (10진) | max 값 | 채택 |
|---|:---:|:---:|:---:|:---:|:---:|
| **FP32** (채택) | 23 | 8 | **~7.2자리** | 3.4×10³⁸ | **✓** |
| FP16 (half) | 10 | 5 | ~3.3자리 | 65,504 | ✗ |
| BF16 (bfloat16) | 7 | 8 | ~2.4자리 | 3.4×10³⁸ | ✗ |
| FP64 (double) | 52 | 11 | ~15.9자리 | 1.8×10³⁰⁸ | ✗ |
| INT8 | — | — | — | 127 | ✗ |

### 2.2. 거부 사유

- **FP16**: 유효숫자 ~3.3자리 → 정규화 전 φ~10¹³ n/cm²/s 표현 불가 (max 65,504). Bateman ODE 적분 시 중간 계산 (λ_eff ≈ λ_I 근처) 에서 누적 오차 위험
- **BF16**: 유효숫자 ~2.4자리. FP16 보다 더 나쁜 정밀도. 지수 범위 (max 3.4×10³⁸) 는 넓으나 유효숫자 부족
- **FP64**: 불필요. MASTER 원본 유효숫자 4~6자리 < FP32 7.2자리 → FP32 로 모든 물리량 손실 없이 표현 가능. GPU FP64 연산은 FP32 대비 2~32× 느림
- **INT8**: 회귀 출력 정밀도 부족. 핵/열수력 도메인 검증 사례 없음

### 2.3. 채택 근거

- **핵심**: MASTER 원본 유효숫자 (4~6자리) < FP32 유효숫자 (7.2자리) → **모든 물리량이 FP32 안에서 손실 없음**
- **TF32 의 의미**: 사용자 선택 포맷이 아님. Ampere+ GPU (A100 등) 가 FP32 matmul 을 내부적으로 가수 23bit → 10bit 로 근사하여 Tensor Core 가속. 결과는 FP32 반환. 즉 "FP32 를 쓴다" 가 답이며 TF32 는 GPU 내부 자동 최적화
- **Mixed Precision 초기 미도입**: 신경망 forward/backward 는 FP16 가속 가능하나, physical loss (L_Bateman, L_diff_rel) 의 중간 계산은 FP32 유지 필요. 속도 병목 확인 시 검토

### 2.4. NeurIPS 2025 경고

**"FP64 is All You Need"** (arXiv 2505.10949, NeurIPS 2025):
- PINN 도메인에서 FP32 조차 L-BFGS 수렴 저해 사례 보고
- 본 프로젝트는 GT 데이터 기반 학습 + physical loss 보조이므로 상황이 다르나, physical loss 항의 중간 계산 정밀도가 중요하다는 경고 반영
- 향후 L_diff_rel/L_Bateman 의 수렴 문제가 발견되면 FP64 전환 검토 대상

상세: `2026-04-04 정밀도 및 경량화 검토.md` §1~2, §5

---

## 3. 미채용 옵션 — 7개 항목

### 3.1. 미채용 옵션 요약 표

| # | 옵션 | 핵심 거부 사유 | 결정적 검토 문서 |
|:---:|---|---|---|
| 1 | **QK-Norm** | STRING + 좌표 스케일링으로 안정성 확보. ablation 후 필요 시 추가 | `2026-04-04 위치 인코딩` §3.3 끝부분 |
| 2 | **Register tokens** | 격자 720 토큰 충분히 작아 attention sink 위험 낮음 | 일반 Transformer 문헌 |
| 3 | **BC mask 채널** | LAPE 가 boundary 정체성 학습 → 별도 채널 불필요 | `2026-04-07 미결정` line 43 |
| 4 | **Calibration-FiLM** | xs_fuel 이 공간 비균일 (720 cell 각각 다른 어셈블리) → FiLM 채널별 스칼라 변조 표현 불가 | `2026-04-03 기법` §2~3 |
| 5 | **Perceiver IO** | 720 토큰 충분히 작아 latent 압축 불필요 | `2026-04-03 기법` §5 |
| 6 | **MoNODE** | 1D latent 가정, 3D 공간 구조 직접 처리 불가 | `2026-04-03 기법` §7 |
| 7 | **ContextAR** | 2D 이미지 생성용, 격자 크기에서 불필요한 복잡성 | `2026-04-03 기법` §8 |

### 3.2. 미채용 옵션 상세

#### 1. QK-Norm

STRING PE 적용 시 R(r)·Q 변환으로 벡터 크기 변동 가능 — QK-Norm (Q, K 를 dot product 전에 L2 정규화) 이 이를 방지. ViT-22B, Stable Diffusion 3 등에서 채택.

**미채용 이유**: STRING 의 생성자 노름 제약 (`L_k ← L_k · min(1, c/‖L_k‖_F)`) + 좌표 스케일링 (cm → 0~1) 으로 R(r) 의 크기 변동을 제어 가능. QK-Norm 은 추가 보험이며 ablation 후 필요 시 추가하는 *선택 사항*으로 보류.

#### 2. Register tokens

ViT 대규모 모델 (DINOv2 등) 에서 attention 초기 토큰에 정보가 과도하게 집중되는 "attention sink" 현상 방지를 위해 도입된 dummy 토큰.

**미채용 이유**: 격자 720 토큰은 ViT 의 수백~수천 패치 대비 작아 attention sink 위험이 낮음. L_data_halo (λ=0.3) 가 halo cell 에 supervision 을 부여하므로 unsupervised 토큰 (sink 후보) 이 존재하지 않음 (05_symmetry_mode 결정 후 해소).

#### 3. BC mask 채널

격자 경계 유형 (반사체 인접, 대칭면, 내부) 을 별도 입력 채널로 모델에 투입.

**미채용 이유**: LAPE 가 각 위치의 고유 정체성을 학습 가능 변수로 인코딩 → 경계 유형 정보가 학습으로 자동 흡수. 별도 채널은 불필요한 정보 중복. 추후 LAPE 가 boundary 식별에 실패하면 (ML 모니터링 metric M5 로 검출) 재도입 검토.

#### 4. Calibration-FiLM

느린 갱신 물성 (xs_fuel) 을 FiLM 변조 (γ, β) 로 빠른 CNN 에 전달하는 패턴 (Aitken et al.).

**미채용 이유**: FiLM 은 채널별 스칼라 (γ_c, β_c) 변조 — 공간 균일 물성에만 유효. xs_fuel 은 (20, 6, 6, 10) 으로 720 cell 각각 다른 어셈블리 조성 → 공간 비균일. 공간 해상도까지 유지한 (γ, β) 를 생성하면 (20, 6, 6, C) 크기가 되어 채널 concat 과 연산량 동일 → FiLM 분리의 이점 소멸.

#### 5. Perceiver IO

이종 입력 (스칼라 + 3D) 을 cross-attention 으로 고정 크기 latent 배열에 투사하는 아키텍처 (Jaegle et al. 2021).

**미채용 이유**: 핵심 장점인 "입력 크기 축소" 가 720 토큰 규모에서 불필요. Conv3D spatial inductive bias 를 잃는 단점 + cross-attention 추가 구현 비용. 격자가 full-core 18×18 로 커질 경우 재검토 대상.

#### 6. MoNODE (Mixture of Neural ODEs)

잠재 공간에서 Neural ODE 를 혼합하는 시계열 모델.

**미채용 이유**: 1D latent 공간 전제. 3D 공간 구조 (20, 6, 6) 를 직접 처리하지 못함. 본 프로젝트는 공간 인코더 → Mamba 시계열 → 공간 디코더 의 명시적 분리 구조이며, MoNODE 의 1D ODE 가정과 비호환.

#### 7. ContextAR (Context-Aware Attention Routing)

2D 이미지 생성에서 마스킹 기반 attention routing 을 수행하는 기법.

**미채용 이유**: 2D 이미지 생성 도메인 최적화. CCA 마스킹 아이디어 (xs_fuel ↔ state 어텐션 분리) 는 차용 가능하나 격자 (720 토큰) 에서는 불필요한 복잡성. xs_fuel 과 state 는 채널 concat 으로 충분히 전달 (`02_cell_embedder.md` 결정).

### 3.3. 미채용 옵션 7개의 공통 거부 패턴

모든 미채용 옵션은 다음 세 가지 기준 중 하나 이상에 해당:

1. **본 문제의 격자 크기 (720 토큰) 에서 불필요**: Perceiver IO, Register tokens
2. **공간 비균일 특성과 비호환**: Calibration-FiLM
3. **다른 메커니즘으로 이미 해결**: BC mask → LAPE, QK-Norm → STRING 좌표 스케일링
4. **도메인 미스매치**: MoNODE (1D), ContextAR (2D), GeoPE (quaternion)

상세: `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §1~8

---

## 4. 결정 일자 + 결정 과정 요약

### 4.1. 정규화 (이전 07 정규화)
- **2026-03-30**: 기존 코드의 BatchNorm 주석 처리 확인 → Pre-LN + LayerNorm 통일 확정
- **2026-04-08**: 디코더 AdaLN-Zero 분리 결정 (사용자 의문: "p_load 를 인코더에도 주입해야 하나?")
- 상세: `2026-03-30 모델 구현 계획(공간 인코더).md` line 334-337

### 4.2. 정밀도 (이전 07 정밀도)
- **2026-04-04**: FP16 / BF16 / FP32 / FP64 / INT8 비교 검토 → FP32 확정
- 핵심: MASTER 원본 유효숫자 (4~6자리) < FP32 (7.2자리)
- 상세: `2026-04-04 정밀도 및 경량화 검토.md` §1~2

### 4.3. 미채용 옵션 (이전 06)
- **2026-04-03**: 7가지 기법 적합성 검토. 각각 격자 크기 / 공간 비균일 / 도메인 미스매치 기준으로 거부
- **2026-04-08**: BC mask 최종 거부 (LAPE 도입으로 대체). QK-Norm 보류 (ablation 대상)
- 상세: `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §1~8

### 4.4. 통합 (본 문서, 2026-04-09)
- 06/07 결정을 "보조/제외" 결정으로 통합
- 정규화/정밀도는 핵심 결정, 미채용 옵션은 "왜 안 썼는가" 기록

---

## 5. 참고 문헌

### 학계 원문
1. **Ba, Kiros & Hinton. 2016**. "Layer Normalization". arXiv 1607.06450. (LayerNorm)
2. **Peebles & Xie. 2023**. "Scalable Diffusion Models with Transformers". ICCV. (DiT, AdaLN-Zero)
3. **arXiv 2505.10949. 2025**. "FP64 is All You Need". NeurIPS 2025. (PINN 정밀도 경고)
4. **Jaegle et al. 2021**. "Perceiver IO: A General Architecture for Structured Inputs & Outputs". ICML. (Perceiver IO)
5. **Darcet et al. 2024**. "Vision Transformers Need Registers". ICLR 2024. (Register tokens)

### 본 프로젝트 검토 문서
- `2026-04-04 정밀도 및 경량화 검토.md` §1~2, §5 (정밀도 + 정규화 상세)
- `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §1~8 (미채용 옵션 각 기법별 상세)
- `2026-03-30 모델 구현 계획(공간 인코더).md` line 334-337 (정규화 확정)

### 본 폴더 cross-reference
- `02_cell_embedder.md` (xs_fuel 채널 concat 결정 → Calibration-FiLM 거부 근거)
- `03_attention_and_position.md` (STRING/LAPE 결정 → QK-Norm 보류, BC mask 거부 근거)
- `05_symmetry_mode.md` (halo expand + L_data_halo → Register tokens 불필요 근거)
