# 공간 인코더 미결정 항목 기록 + Attention 가중치 비교

> **작성일**: 2026-04-07
> **개정일**: 2026-04-09 — §2.4 Attention 비교표를 N=500 (quarter) → **N=720 (halo expand 후)** 으로 갱신. halo (6,6) 결정 (`05_symmetry_mode.md`) 반영. Block+Grid 분할 옵션 표를 (6,6) 격자 기준으로 재검토. Full Attention 채택 결정은 그대로 유지 (절대 FLOPs 여전히 trivial).
> **목적**: 데이터 전처리 완료 후 공간 인코더 구현 재개 — 미결정 항목 정리 및 Attention 전략 결정

---

## 1. 미결정/재검토 항목 요약

### 1.1 ★ 구현 전 결정 필요

| 항목 | 옵션 | 영향 범위 |
|------|------|---------|
| **Attention 전략** | Full Attention vs Block+Grid Attention | 가장 큰 구조 분기 — 전체 인코더 |
| **D_latent** | 64 (기존) vs 128 (정보 보존) | Mamba 인터페이스, 메모리 |
| **Stem 구조** | Conv3D 유지 vs Linear projection (ViT patch embedding 패턴) | 경계 처리, 파라미터 수 |
| **p_load 주입 기법** | ~~FiLM~~ → **AdaLN-Zero로 격상 권장** | 디코더 1차, Mamba 보조. 근거 §3 |
| **위치 인코딩** | RPE(gradient 수정) vs STRING vs **LieRE** | LieRE는 비균일 Z 메시 직접 지원 + Z-XY 결합. 근거 §4 |
| **백본 스타일** | 직접 구현 vs **PDE-Transformer 차용** | DiT 블록 + AdaLN-Zero 기성 검증. 근거 §4 |

### 1.2 사용자가 빠뜨린 항목 (재검토 권장)

| 항목 | 출처 | 비고 |
|------|------|------|
| **AdaLN-Zero (DiT)** | `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §2 | Attention+FFN 구조에 drop-in. DiT 검증. FiLM 결정을 격상할지 재고 |
| **IIET 반복 디코더 (Predictor-Corrector)** | `2026-03-30 모델 구현 계획(공간 인코더).md` §도입 검토 | EMNLP 2025. 디코더 r회 반복(가중치 공유) → MASTER Full PC 커플링 학습. ~0.9% 바닥 오차 개선 후보. SD-Phase이지만 인코더 latent 구조 영향 |
| **PeRCNN Π-block** | `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §4 | xs_fuel × state 곱셈 경로로 반응률 `Σ_f × φ` 구조 인코딩 |
| **PI-CRNN 분리 학습** | `2026-04-03 인공지능 모델 및 기법 검토(공간).md` §6 | 인코더 선학습 → 동결 → Mamba 학습. 학습 안정성 기여 |
| **사전 학습 검증 항목** | `2026-03-29 모델 구현 계획(통합).md` §7.1 | gradient flow, 단일 샘플 과적합, 조건 신호 감도, xs_fuel 채널 기여도 |

### 1.3 이미 결정된 항목 (확정)

- **입력 텐서**: 데이터 입력 (B, 20, 5, 5, 21ch) → halo expand → 인코더 본체 입력 (B, 20, 6, 6, 21ch). state 10 + rod_map 1 + xs_fuel 10, 채널 concat. 상세: `05_symmetry_mode.md` (개정 2026-04-09)
- **인코더 역할**: "현재 상태 압축"이 아니라 **"state(t) + rod_map(t+1) + xs_fuel → 전이 정보 추출"**
- **rod_map 시점**: rod_map(t+1) — 다음 상태 유도 제어 쿼리
- **출력 헤드**: k_eff, AO 별도 헤드 + L_pload/L_AO/L_keff 일관성 loss
- **정규화**: Pre-LN 통일 (BN 제거), 디코더만 AdaLN-Zero 분리
- **Mirror symmetry**: ~~인코더 범위 밖 (디코더 L_diffusion 전용)~~ → **개정 (2026-04-09)**: halo expand로 인코더 입력 단계에 sym 매핑된 cell 직접 부여. 인코더가 (6,6) feature space에서 boundary cell의 attention input으로 halo cell을 직접 봄. 근거: `05_symmetry_mode.md` §3
- **BoundaryPad**: 미사용
- **AxialSE**: 제거 확정 (Attention과 기능 중복)
- ~~**위치 인코딩**: RPE(gradient 수정)로 시작 → STRING 실험 (별도 검토 문서 참조)~~
- **위치 인코딩** *(2026-04-08 최종 결정)*: STRING(상대) + LAPE(절대) 동시 구현. RPE 단계 거치지 않음. BC mask 채널 미사용. 근거: `breezy-herding-meadow.md` plan
- **정밀도**: FP32 유지 (TF32 자동 가속)

---

## 2. Full vs Block+Grid Attention — 기존 Voxel Attention과 가중치 비교

### 2.1 기존 CustomMaxViT3D 파라미터 (역산)

**구성**: 입력 71ch (xs_fuel 10ch × 7 이웃 superposition + rod 1ch) → Stem3D → MaxViTBlock3D ×3 → 1ch power 출력
**hidden_size = 64, num_heads = 2, head_size = 32, expand_ratio = 4**

| 모듈 | 세부 | 파라미터 | 누적 |
|------|------|---:|---:|
| **Stem Conv3D(15,5,5) 71→128** | 15·5·5·71·128 + 128 | 681,728 | |
| **Stem Conv3D(15,3,3) 128→128** | 15·3·3·128·128 + 128 | 2,211,968 | |
| Stem 소계 | | **2,893,696** | 2.89M |
| **Stage 1 MBConv** (128→64) | expand 128→256 + DW(3³) + AxialSE + project 256→64 + LN×3 | ~90,880 | |
| **Stage 2,3 MBConv** (64→64) ×2 | expand 64→256 + DW + SE + project 256→64 | ~148,992 | |
| MBConv 소계 | | **239,872** | 0.24M |
| **BlockAttn ×3** | LN(128) + QKV(12,480) + out(4,160) + RPE table(126) | ~50,298 | |
| **FFN ×6** (Block, Grid 사이) | Dense 64→256 + Dense 256→64 | 198,528 | |
| **GridAttn ×3** | 동일 구조 | ~50,298 | |
| Attention/FFN 소계 | | **299,124** | 0.30M |
| **Final Conv3D(3,3,3) 64→1** | 27·64 + 1 | 1,729 | |
| **합계** | | | **~3.43M** |

> **핵심 관찰**:
> - **Stem이 84%(2.89M) 차지** — Conv3D(15,5,5) 커널이 비대 (Z축 커널 15는 D=24 기준 과대, Prob-4)
> - MBConv + Attention/FFN은 합쳐 0.54M에 불과
> - **Stem을 Linear projection으로 대체하면 ~0.5M 수준 모델로 축소 가능**

### 2.2 신규 인코더 (21ch 입력) 가중치 추정

**가정**:
- 인코더 본체 입력: (B, 20, 6, 6, 21ch) [halo expand 후, 개정 2026-04-09]. 데이터 입력은 (B, 20, 5, 5, 21ch). state 10 + rod_map 1 + xs_fuel 10
- Stem: **Linear projection 21→D** (Conv3D 제거, ViT patch embedding 패턴)
- MBConv 제거 (AxialSE 제거 확정 + L_diffusion 도입 시 Conv 역할 대체)
- Attention only: BlockAttn → FFN → GridAttn → FFN (Block+Grid) 또는 FullAttn → FFN
- num_heads = D / 32

#### Case A: D=64 (기존과 동일)

| 모듈 | 파라미터 |
|------|---:|
| Linear stem 21→64 | 1,408 |
| **Per attention block** (LN + QKV + out + FFN) | |
| ├ LN | 128 |
| ├ QKV Dense (3 × 64×64 + 3×64) | 12,480 |
| ├ Out Dense (64×64 + 64) | 4,160 |
| ├ RPE table (num_heads=2, (2·H-1)(2·W-1) for chosen window) | ~50~200 |
| ├ FFN Dense 64→256 + Dense 256→64 | 33,088 |
| └ FFN LN | 128 |
| **블록당 합** | **~50,000** |

| 전략 | 블록 구성 | Attention 블록 수 | Attention 파라미터 | 총 (Stem+Atten) |
|---|---|---:|---:|---:|
| **Full Attention 2-stage** | FullAttn+FFN ×2 | 2 | ~100K | **~0.10M** |
| **Full Attention 3-stage** | FullAttn+FFN ×3 | 3 | ~150K | **~0.15M** |
| **Block+Grid 2-stage** | (Block+FFN+Grid+FFN) ×2 | 4 | ~200K | **~0.20M** |
| **Block+Grid 3-stage** | (Block+FFN+Grid+FFN) ×3 | 6 | ~300K | **~0.30M** |

#### Case B: D=128 (정보 보존)

블록당 hidden=128, num_heads=4, FFN expand 4× → ~198K/block

| 전략 | 블록 수 | Attention 파라미터 | 총 |
|---|---:|---:|---:|
| Full Attention 2-stage | 2 | ~396K | **~0.40M** |
| Full Attention 3-stage | 3 | ~594K | **~0.60M** |
| Block+Grid 2-stage | 4 | ~792K | **~0.79M** |
| Block+Grid 3-stage | 6 | ~1,188K | **~1.19M** |

### 2.3 기존 vs 신규 가중치 비교

| 모델 | 파라미터 | 기존 대비 |
|------|---:|---:|
| **기존 CustomMaxViT3D** (Conv-heavy stem) | ~3.43M | 100% |
| 신규 D=64, Full 2-stage | ~0.10M | **3%** |
| 신규 D=64, Block+Grid 3-stage | ~0.30M | **9%** |
| 신규 D=128, Full 2-stage | ~0.40M | **12%** |
| 신규 D=128, Full 3-stage | ~0.60M | **17%** |
| 신규 D=128, Block+Grid 3-stage | ~1.19M | **35%** |

> **결론**: Stem을 Linear로 대체하고 AxialSE/MBConv를 제거하는 것만으로 **자동으로 1/3 이하**가 됨. Attention 전략(Full vs Block+Grid)의 차이는 약 2배 수준.

### 2.4 Attention 전략별 특성 비교

> **개정 2026-04-09**: 본 표는 N=720 (halo expand 후 인코더 본체 처리 형상) 기준으로 갱신. 원래 N=500 (quarter) 가정의 수치는 비교용으로 일부 보존. halo (6,6) 결정으로 Block+Grid 분할 옵션도 (6,6) 격자 기준으로 재검토.

| | **Full Attention** | **Block+Grid Attention** |
|---|---|---|
| 수용 영역 | 720 토큰 전체 (모든 쌍, halo 포함) | Block: 윈도우 내 / Grid: stride 샘플링 |
| Attention 연산량 (per stage) | N²·d = 720²·64 = 33.2M FLOPs (halo) / 500²·64 = 16M (quarter) | Block(N·16) + Grid(N·8) ≈ 1.10M FLOPs (halo) / 0.77M (quarter) |
| 메모리 (attention matrix) | 720×720 = 518K (halo) / 500×500 = 250K (quarter) | Block 16×16 ×블록 수 + Grid 8×8 ×그리드 수 |
| 구현 복잡도 | 단순 (reshape→attention) | 중간 (partition/reverse, padding) |
| Prob-7 (D=20, H=W=6 나눗셈 호환) | **무관** | (6,6) 격자에서 (2,2)/(3,3) 분할 자연스러움 |
| 위치 인코딩 결합 | RPE/RoPE/STRING 모두 자연 | RPE는 윈도우 내부 한정, STRING이 자연스러움 |
| 물리적 적합성 | 제어봉 삽입의 Z축 전역 영향 직접 반영 + halo cell도 attention에 참여 | 윈도우 + sparse로 간접 반영 |
| 검증된 사례 | ViT 표준 | 기존 모델 검증, MaxViT |

#### Block+Grid를 (20, 6, 6) halo 격자에 적용할 때 분할 옵션 (개정)

> halo expand 후 H=W=6 으로 짝수가 되어 (20,5,5) quarter 시점보다 분할 자유도가 높아짐.

| Block_size | Grid_size | 호환 | 비고 |
|---|---|:---:|---|
| (4, 2, 2) | (5, 3, 3) | ✅ | XY 4분할(2×2 블록), Z 5블록 (4cell씩). MaxViT 표준 패턴에 가장 가까움 |
| (4, 3, 3) | (5, 2, 2) | ✅ | XY 4분할(3×3 블록), Z 5블록 |
| (5, 6, 6) | (4, 1, 1) | ✅ | XY 평면 통째 1블록, Z만 분할 |
| (10, 6, 6) | (2, 1, 1) | ✅ | Block 절반-노심, Grid 상하 페어링 |

→ halo (6,6) 에서는 H=W=6 짝수이므로 **(4,2,2) + (5,3,3)** 분할이 MaxViT 표준 패턴 (block_size = grid_size = 윈도우 크기 일치) 에 가장 가까움. XY 분할이 가능해진 점이 quarter (5,5) 시점 대비 핵심 차이.

#### Quarter (5,5) 시점 분할 옵션 (참고용, DEPRECATED)

XY가 5×5로 작고 소수이므로 H/W 방향 분할이 어려웠음:

| Block_size | Grid_size | 호환 | 비고 |
|---|---|:---:|---|
| (5, 5, 5) | (4, 1, 1) | ✅ | Block은 quarter-XY 평면 1장, Grid는 Z축 4간격 sparse |
| (4, 5, 5) | (5, 1, 1) | ✅ | Z=20 → 5블록, Grid는 Z방향 stride 5 |
| (10, 5, 5) | (2, 1, 1) | ✅ | Block 절반-노심, Grid는 상하 페어링 |
| (4, 2, 2) | (4, 2, 2) | ❌ | 기존 코드 그대로는 H=W=5 나눗셈 불가 → padding 필요 |

### 2.5 결정 기준

| 기준 | Full 우위 | Block+Grid 우위 |
|------|:---:|:---:|
| 구현 단순성 | ✅ | |
| 기존 검증 | | ✅ |
| 파라미터 효율 | ✅ (블록 수 적음) | |
| 물리적 수용 영역 | ✅ (제어봉 Z 전역) | |
| Prob-7 회피 | ✅ | |
| 연산량 (현재 격자에서) | 차이 미미 | 차이 미미 |
| 위치 인코딩 자유도 | ✅ | |

> **현재 격자(N=720, halo expand 후)에서는 Full Attention이 연산/메모리 부담이 트리비얼**하므로, 구현 단순성·파라미터 효율·물리적 수용 영역 측면에서 **Full Attention이 우위**. (개정 2026-04-09: N=500 quarter 가정 → N=720 halo)
>
> Block+Grid의 유일한 우위는 "기존 모델 검증"인데, 이는 기존 모델이 Voxel Attention 전체 패키지(Stem+MBConv+SE+Block+Grid)로 검증된 것이지 Block+Grid 자체의 우위가 분리되어 검증된 것은 아님.

### 2.6 권장안

**1차 권장**: Full Attention, **D=128, 3-stage** (~0.60M)
- 기존 대비 17% 크기
- 21ch 입력의 정보 보존 (10ch state + 10ch xs_fuel + 1ch rod의 상호작용)
- N=720 (halo expand 후) 이므로 Full Attention도 트리비얼한 연산량 (FLOPs 33.2M/stage, 절대치 무시 가능)
- 위치 인코딩(LAPE + STRING) 자유, halo cell도 attention 처리에 자연 통합

**대안**: Block+Grid, **D=128, (4,2,2)+(5,3,3) 분할, 3-stage** (~1.19M) — halo (6,6) 기준
- 기존 검증 패턴 유지
- 35% 크기 (~3.43M의 약 1/3)
- halo (6,6) 격자에서는 H=W=6 짝수이므로 (4,2,2)+(5,3,3) 분할이 MaxViT 표준 패턴에 가장 가까움 (개정 2026-04-09)

---

## 3. 다음 결정 사항

1. Attention 전략 (Full vs Block+Grid) 최종 선택
2. D_latent (64 vs 128) — 정보 보존 vs Mamba 연산량
3. Stem 구조 (Linear vs 작은 Conv3D)
4. p_load 주입 기법 (FiLM vs AdaLN-Zero) — Attention 기반이므로 AdaLN-Zero 검토 권장
5. 사전 학습 검증 항목을 SE-B 단계 검증 기준으로 구체화

각 항목 결정 후 SE-A1(`layers3d.py`)부터 코드 작성 진입.
