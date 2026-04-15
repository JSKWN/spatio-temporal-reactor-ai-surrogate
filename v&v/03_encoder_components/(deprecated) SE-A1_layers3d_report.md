# SE-A1 컴포넌트 기능 테스트 보고서 — `layers3d.py`

> **작성일**: 2026-04-08
> **대상 파일**: `src/models/spatial/layers3d.py`
> **검증 단계**: SE-A1 (Spatial Encoder, Phase A, Step 1)
> **계획 문서**: `C:\Users\Administrator\.claude\plans\breezy-herding-meadow.md`

---

## 1. 목적

공간 인코더(Spatial Encoder)의 기본 layer 3종 — `CellEmbedder`, `FFN3D`, `LearnedAbsolutePE3D` — 의 단독 동작 검증.

각 layer가:
- 의도한 텐서 형상 변환을 수행하는가
- 파라미터 수가 설계 의도와 일치하는가 (수식 검증)
- 기본 forward 동작이 오류 없이 실행되는가

---

## 2. 검증 대상 컴포넌트

### 2.1. `CellEmbedder` — Per-cell channel projection

| 속성 | 값 |
|---|---|
| 역할 | Cell 내부 raw 채널을 latent 차원으로 projection (입력 적응) |
| 구현 | `Conv3D(kernel=(1,1,1), filters=d_latent, use_bias=True)` |
| Spatial mixing | **없음** (1×1×1 kernel) |
| 입력 가정 | 21채널 정규화 데이터 (state 10 + xs_fuel 10 + rod 1) |
| 후속 처리 | 모든 spatial mixing은 후속 `FullAttention3D`가 담당 |
| 설계 근거 | ViT patch embedding 패턴 + Conv3D 형식으로 5D 텐서 유지 (후속 LAPE/Attention 호환) |

### 2.2. `FFN3D` — Token-wise feed-forward

| 속성 | 값 |
|---|---|
| 역할 | Attention block 내부 token-wise 비선형 변환 (Vaswani 2017 표준) |
| 구조 | `Dense(D → D·expand)` → `GELU` → `Dense(D·expand → D)` |
| 활성 함수 | GELU |
| Default expand_ratio | 4 |
| 적용 위치 | 각 attention block (총 3 stage = 3회) |

### 2.3. `LearnedAbsolutePE3D` — 학습 가능 절대 위치 임베딩

| 속성 | 값 |
|---|---|
| 역할 | 500개 cell × D차원 정체성 부여, inner BC 식별 자유도 |
| 구현 | `(Z, qH, qW, d_latent)` trainable variable |
| 초기화 | `RandomNormal(stddev=0.02)` (ViT 표준) |
| Forward 동작 | broadcast 합산: `x + embedding[None, ...]` |
| 분석 hook | `get_norm_map() → (Z, qH, qW)` 위치별 L2 노름 |

---

## 3. 테스트 환경

| 항목 | 값 |
|---|---|
| Framework | TensorFlow 2.14 |
| CPU/GPU | CPU (sanity check만 수행, 대규모 학습 아님) |
| Batch size | B = 2 |
| 격자 | (Z, qH, qW) = (20, 5, 5), 총 500 cell |
| 입력 채널 | C_in = 21 |
| Latent 차원 | D = 128 |

---

## 4. 검증 항목 및 결과

### 4.1. 형상(Shape) 변환

| 컴포넌트 | 입력 형상 | 출력 형상 | 결과 |
|---|---|---|:---:|
| `CellEmbedder(d_latent=128)` | `(2, 20, 5, 5, 21)` | `(2, 20, 5, 5, 128)` | ✅ |
| `LearnedAbsolutePE3D(20, 5, 5, 128)` | `(2, 20, 5, 5, 128)` | `(2, 20, 5, 5, 128)` | ✅ |
| `FFN3D(d_latent=128, expand=4)` | `(2, 500, 128)` | `(2, 500, 128)` | ✅ |
| `LearnedAbsolutePE3D.get_norm_map()` | — | `(20, 5, 5)`, `float32` | ✅ |

### 4.2. 파라미터 수 — 수식 검증

| 컴포넌트 | 측정값 | 수식 | 기대값 | 결과 |
|---|---:|---|---:|:---:|
| `CellEmbedder` | **2,816** | `21·128 + 128` (Conv3D 1×1×1 weight + bias) | 2,816 | ✅ |
| `LearnedAbsolutePE3D` | **64,000** | `20·5·5·128` (단일 trainable variable) | 64,000 | ✅ |
| `FFN3D (expand=4)` | **131,712** | `(128·512 + 512) + (512·128 + 128)` | 131,712 | ✅ |

### 4.3. `LearnedAbsolutePE3D` 분석 hook 동작

| 항목 | 값 | 결과 |
|---|---|:---:|
| `get_norm_map()` 반환 형상 | `(20, 5, 5)` | ✅ |
| `get_norm_map()` 반환 dtype | `float32` | ✅ |

분석 hook은 학습 종료 후 callback에서 `np.save("lape_norm_map.npy", ...)` 형태로 저장되어 위치별 LAPE 학습 효과(BC 흡수 여부)를 시각화하기 위한 것. 본 단계에서는 형식만 검증.

---

## 5. 비교 박스 — 설계 의도 vs 측정

```
┌──────────────────────────────────────────────────────────────┐
│  CellEmbedder (Conv3D 1×1×1, 21 → 128)                      │
├──────────────────────────────────────────────────────────────┤
│  설계 의도: per-cell channel projection, spatial mixing 0   │
│  측정 결과:                                                   │
│    - shape (2,20,5,5,21) → (2,20,5,5,128)  ✓                 │
│    - params 2,816 = 21·128 + 128            ✓                │
│    - kernel_size (1,1,1)로 spatial 비고정    ✓                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  LearnedAbsolutePE3D ((20,5,5,128) trainable embedding)     │
├──────────────────────────────────────────────────────────────┤
│  설계 의도: 500 cell × 128차원 정체성 vector, BC 식별 자유   │
│  측정 결과:                                                   │
│    - shape (2,20,5,5,128) → (2,20,5,5,128)  ✓                │
│    - params 64,000 = 20·5·5·128             ✓                │
│    - get_norm_map() (20,5,5) float32         ✓                │
│    - broadcast 합산 동작                     ✓                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  FFN3D (Dense expand×4, GELU)                                │
├──────────────────────────────────────────────────────────────┤
│  설계 의도: token-wise 비선형 (attention 후 정보 처리)         │
│  측정 결과:                                                   │
│    - shape (2,500,128) → (2,500,128)        ✓                │
│    - params 131,712 = 2·(128·512+512)        ✓                │
│    - GELU 활성화                             ✓                │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 컴포넌트별 설계 결정 근거

### 6.1. CellEmbedder — 왜 Conv3D(1,1,1)인가

- **Dense와 수학적 동치**: `Conv3D(1,1,1, filters=D)` = 마지막 축에만 적용되는 `Dense(D)`. 같은 파라미터 수, 같은 결과
- **5D 텐서 형식 유지**: 후속 `LearnedAbsolutePE3D`와 `FullAttention3D`(reshape 전)가 5D 텐서를 기대. Dense를 쓰면 reshape 필요
- **명명 명확성**: 클래스명을 `Stem3D`가 아닌 `CellEmbedder`로 결정 — "한 cell의 21채널을 D차원 latent로 끌어올리는" 역할이 명확

### 6.2. FFN3D — CellEmbedder와의 역할 차이

| 항목 | CellEmbedder | FFN3D |
|---|---|---|
| 인코더 내 위치 | 진입부 1회 | 각 attention block 안 (3 stage) |
| 채널 변환 | 21 → 128 (입력 적응) | 128 → 512 → 128 (정보 처리) |
| 비선형 | 없음 | GELU 필수 |
| 역할 | 입력 raw → latent 공간 사상 | attention 후 token 내부 비선형 처리 |

Attention은 본질적으로 token 간 *선형* 결합이므로, FFN의 비선형 변환이 표현력의 핵심.

### 6.3. LearnedAbsolutePE3D — 왜 LAPE를 쓰는가

- **STRING(상대 PE)의 보완**: STRING은 token 간 상대 거리만 제공. 절대 위치(특정 cell이 boundary인지)를 STRING만으로 식별하기 어려움
- **BC mask 채널 대안**: 명시적 BC mask 입력 채널 없이 모델이 boundary 셀의 정체성을 LAPE로부터 학습
- **인용** (Dufter et al. 2022, *Position Information in Transformers: An Overview*, Computational Linguistics 48(3):733):
  > "absolute and relative position embeddings are complementary ... the combination outperforms either alone"

### 6.4. LAPE 분석 hook — 학습 후 BC 흡수 검증

학습 종료 후 `get_norm_map()`을 호출하여 위치별 임베딩 노름을 시각화:
- **기대 패턴**: boundary 셀(`qy ∈ {0, 4}`, `qx ∈ {0, 4}`, `z ∈ {0, 19}`)이 interior(중간 셀)보다 큰 노름 → "LAPE가 BC 구조 흡수했다" 신호
- **반대 패턴**: 노름이 거의 균일 → STRING + L_diffusion만으로 충분했다는 신호. LAPE 제거 ablation 후속 가능

---

## 7. 결론

| 항목 | 결과 |
|---|:---:|
| 형상 변환 정확성 | ✅ |
| 파라미터 수 정확성 | ✅ |
| 분석 hook 동작 | ✅ |
| Forward 오류 없음 | ✅ |

**판정**: SE-A1 모든 컴포넌트 단독 동작 검증 통과. SE-A2 (attention3d.py) 진입 가능.

---

## 8. 후속 검증 (SE-B 단계)

본 보고서는 컴포넌트 *단독* sanity check만 수행. 인코더 전체 조립 후 검증할 항목들:

- [ ] Gradient flow: 모든 trainable parameter에 backprop gradient 존재 확인
- [ ] 단일 샘플 과적합 (sanity check): loss → 0
- [ ] xs_fuel 채널 sensitivity: xs_fuel 채널 0 mask 시 출력 변화 확인
- [ ] 파라미터 수 총합 ≈ 660K (인코더 본체 + LAPE)

→ SE-B3 단계에서 `tests/encoder/` 하위에 unit test로 구현 예정.
