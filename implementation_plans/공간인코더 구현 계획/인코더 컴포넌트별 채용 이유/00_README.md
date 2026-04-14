# 인코더 컴포넌트별 채용 이유 — 종합 인덱스

> **작성일**: 2026-04-08
> **목적**: 공간 인코더(Spatial Encoder)의 모든 design decision에 대한 옵션 비교 + 채용/미채용 근거를 이전 대화 세션들의 논의를 종합하여 정리. 논문 작성 시 직접 인용 가능한 형태.
> **계획 문서**: `C:\Users\Administrator\.claude\plans\breezy-herding-meadow.md`

---

## 본 문서의 위치

본 폴더는 *기능 테스트 보고서와 별도*로, 각 design decision의 *왜* 를 정리합니다.

| 폴더 | 용도 |
|---|---|
| `v&v/03_encoder_components/` | 컴포넌트별 단독 sanity check (shape, params, 수치 검증) |
| **`implementation_plans/공간인코더 구현 계획/인코더 컴포넌트별 채용 이유/`** | **본 폴더 — 옵션 비교 + 채용 근거** |
| `implementation_plans/공간인코더 구현 계획/2026-04-04 *.md`, `2026-04-07 *.md` | 결정에 이르기까지의 검토 과정 (역사 기록) |

---

## 결정 사항 요약 표

| # | 컴포넌트 | 결정 | 옵션 수 | 결정 일자 | 보고서 |
|:---:|---|---|:---:|:---:|---|
| 01 | 입력 격자 | Quarter 5×5 (데이터 입력, 500 cell) → halo expand (6,6) (인코더 처리, 720 cell) | 4 | 2026-04-08 | [01_input_grid.md](01_input_grid.md) |
| 02 | CellEmbedder | Conv3D(1,1,1), 21 → 128 (격자 형상 무관) | 4 | 2026-04-08 | [02_cell_embedder.md](02_cell_embedder.md) |
| 03 | Attention backbone + Position Encoding | Full Attention × 3, D=128, H=4 + LAPE (절대, 1회 add) + STRING (상대, Q/K 회전) | (Full vs Block+Grid) × PE 5종 × 조합 | 2026-04-07 ~ 2026-04-08 | [03_attention_and_position.md](03_attention_and_position.md) |
| 04 | 정규화 + 정밀도 + 미채용 옵션 | Pre-LN + LayerNorm + FP32(TF32) / 미채용 7개 항목 | 다수 | 2026-04-04 ~ 2026-04-08 | [04_normalization_omitted_options.md](04_normalization_omitted_options.md) |
| 05 | 대칭 모드 / halo expand | halo (6,6) all-the-way + 최종 (5,5) crop | 3 | 2026-04-08 | [05_symmetry_mode.md](05_symmetry_mode.md) |

---

## 아키텍처 한눈에 보기

```
Data input: (B, 20, 5, 5, 21)               ← 데이터셋 quarter
       │  state(10) + xs_fuel(10) + rod_map(1)
       ▼
[halo_expand(sym)]  mirror 또는 rotational 매핑 (05_symmetry_mode)
       │
       ▼ (B, 20, 6, 6, 21)                  ← halo expand 완료, 인코더 진입
[CellEmbedder] Conv3D(1,1,1), 21 → 128       ← Stem (no spatial mixing)
       │
       ▼ (B, 20, 6, 6, 128)
[LearnedAbsolutePE3D] +(20, 6, 6, 128) trainable embedding   ← LAPE 1회 add
       │                                      (이후 모든 layer는 residual stream으로 전달)
       ▼ (B, 20, 6, 6, 128)
[reshape] flatten spatial → (B, 720, 128)
       │
       ▼
┌────────────────────── Stage 1 ──────────────────────┐
│  [Pre-LN] → [FullAttention3D] + residual           │
│              ├ Q = W_Q · x                          │
│              ├ K = W_K · x                          │
│              ├ V = W_V · x                          │
│              ├ Q, K = STRING(Q, K, coords)  ← STRING은 Q/K projection 직후, V 제외
│              └ output = softmax(Q·K^T/√d) · V       │
│  [Pre-LN] → [FFN3D, expand=4] + residual           │
└──────────────────────────────────────────────────────┘
       │
       ▼ (B, 720, 128)
┌────────────────────── Stage 2 ──────────────────────┐
│  ... (동일 구조, STRING은 매 stage attention마다 재적용)  │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────── Stage 3 ──────────────────────┐
│  ... (동일 구조)                                      │
└──────────────────────────────────────────────────────┘
       │
       ▼ (B, 720, 128)
[reshape] → (B, 20, 6, 6, 128) latent       ← 인코더 출력 (Mamba/디코더 입력)

[Mamba]    latent (6,6) 그대로 통과
[Decoder]  (B, 20, 6, 6, 128) → (B, 20, 6, 6, C_out)
[L_diff_rel] 디코더 (6,6) 출력 그대로 사용 (합산 도메인은 5×5 inner cell)

[Final crop] (B, 20, 6, 6, C_out) → (B, 20, 5, 5, C_out)   ← 외부 인터페이스 한 곳

Total params: ~600K (encoder body) + LAPE (20·6·6·128 = 92,160 또는 D_lape에 따라 변동)
  - Halo (6,6) LAPE: 92,160 vs Quarter (5,5) LAPE: 64,000 → 차이 +28,160 (4.3%)
  - 정확한 LAPE 차원은 05_position_encoding_lape.md 결정 시 확정
```

**위치 인코딩 적용 위치 — 학계 표준 패턴**:

| PE 종류 | 위치 | 빈도 | 근거 |
|---|---|:---:|---|
| **LAPE** (절대) | CellEmbedder (Stem) 직후, flatten 전 | **1회만** | ViT, BERT, Vaswani 2017 (input 단계 1회 add → residual stream으로 전파) |
| **STRING** (상대) | FullAttention3D 내부, Q/K projection 직후, dot product 직전 | **매 attention layer** | RoPE/RoFormer (Su 2021), STRING (ICML 2025). R^T·R = R(상대거리) trick 성립 |
| **V 회전** | — | — | RoPE/STRING 모든 구현체에서 V는 미회전 (attention weight가 위치 정보 운반, V는 content 집계) |

상세 학계 근거 + 본 프로젝트 코드 정합성 검증: `2026-04-04 3D 위치 인코딩 기법 검토.md` §3.6

---

## 공통 보고서 구조

각 보고서는 다음 구조를 따릅니다:

1. **결정 사항** — 한 줄 요약
2. **검토 옵션** — 표 (옵션, 식/구조, 장단점)
3. **각 옵션이 *부적합한* 명시적 이유** — 미채용 옵션의 4가지 정도 명시 근거
4. **채택 옵션의 *명시적* 근거** — 4가지 정도의 채택 이유, 인용 포함
5. **결정 일자 + 참고 대화 세션 식별**
6. **참고문헌** — 관련 논문, 외부 문서

---

## 참고: 결정에 영향을 준 주요 검토 문서

| 문서 | 다룬 결정 |
|---|---|
| `2026-04-03 인공지능 모델 및 기법 검토(공간).md` | AdaLN-Zero, PeRCNN, PI-CRNN, normalization 후보 |
| `2026-04-04 3D 위치 인코딩 기법 검토.md` | RPE / 3D RoPE / STRING / LieRE / GeoPE 비교 |
| `2026-04-04 정밀도 및 경량화 검토.md` | FP32 vs FP16 vs BF16 vs TF32 vs INT8, mixed precision |
| `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md` | 기존 vs 신규 파라미터, Full vs Block+Grid Attention |
| `2026-04-07 (도입 고려 필요) L_diffusion Cross-Attention 이론 해석.md` | Cross-attention = Green's function, NEM 4차 곡률 floor |
| `2026-04-07 Differentiable NEM Layer 도입 계획.md` | NEM correction 학습 layer, JNET0 단위 불일치 |
| `2026-04-07 PDE Transformer 자문 참고문헌.md` | PDE-Transformer, IFactFormer-m, PINTO, Point-wise DiT |

본 폴더의 5개 보고서 (01~05) 는 위 문서들의 *결정 결과*만을 종합. 검토 과정은 원본 문서 참조.
