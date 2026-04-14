# 03. Attention Backbone + Position Encoding (LAPE + STRING) 채용 이유

> **결정**: Full Attention × 3 stages, D=128, H=4 + LAPE (절대) + STRING (상대) 동시 도입
> **결정 일자**: 2026-04-07 (Attention) ~ 2026-04-08 (PE 통합)
> **본 문서의 위치**: 이전 plan의 03/04/05 결정 사항을 통합·압축. "공간 정보 학습 backbone" 이라는 하나의 일관된 설계 의도로 묶어 정리

---

## 0. 결정 요약

| 컴포넌트 | 결정 | 핵심 이유 (1줄) |
|---|---|---|
| **Attention 종류** | Full Attention | N=720 halo에서 연산 비용이 trivial |
| **Stage 수** | 3 | ViT/MaxViT 표준, ~0.60M 파라미터 |
| **Hidden D** | 128 | 21ch 입력 정보 보존 + Mamba 인터페이스 폭 |
| **Heads H** | 4 | head_dim = 32, multi-head 표현력 |
| **상대 위치 (STRING)** | Block-diag 2×2 + 축별 독립 freq | Q/K projection 직후 회전, V 제외 |
| **절대 위치 (LAPE)** | (20, 6, 6, D) trainable embedding | CellEmbedder 직후 1회 add |

상세 다이어그램은 `00_README.md` 의 "아키텍처 한눈에 보기" 참조.

---

## 1. Attention Backbone — Full Attention × 3 stages

### 1.1. 채택 이유

**핵심 이유**: 현재 격자 N=720 (halo expand 후) 에서 Full Attention 의 연산 비용이 trivial.

- Per-stage attention FLOPs: 720² × 64 = **33.2M FLOPs**
- 전체: 33.2M × 3 stages × 576 timesteps × 2 (인코더+디코더) ≈ **1.8 GFLOPs/sample**
- 현대 GPU (A100, ~312 TFLOPs FP32) 기준 **~6 µs/sample** — 무시 가능
- Mamba/디코더/L_diff 등 다른 비용에 비해 미미

연산이 무시 가능하므로 단순한 Full Attention 을 채택. 윈도우 분할 (Block/Grid) 의 복잡도를 감수할 이유가 없음.

### 1.2. Block+Grid 거부 사유

| 사유 | 설명 |
|---|---|
| **분리 검증 부재** | 기존 CustomMaxViT3D 의 Block+Grid 검증은 전체 패키지 (Stem + MBConv + AxialSE + Block+Grid) 효과이며, Block+Grid 자체가 Full Attention 보다 우수함을 입증한 ablation 없음 |
| **Halo cell 격리** | 윈도우 분할 시 halo cell 이 어느 윈도우에 속하느냐에 따라 일부 inner cell 과만 상호작용. Full Attention 은 모든 cell 쌍 직접 참조 |
| **Prob-7** | Block_size 가 격자 차원의 약수여야 함. quarter (5,5) 에서는 5가 소수라 분할 자유도 거의 없음. halo (6,6) 에서는 짝수라 (2,2)/(3,3) 가능하나 여전히 추가 하이퍼파라미터 결정 부담 |

### 1.3. 하이퍼파라미터 결정 근거

| 항목 | 값 | 근거 |
|---|:---:|---|
| **D (hidden size)** | 128 | 21ch 입력 (state 10 + xs_fuel 10 + rod 1) 의 정보 보존. 기존 CustomMaxViT3D D=64 는 xs_fuel 10ch 전용이었으므로 21ch 입력에는 64~128 범위 → 정보 병목 회피 위해 128 채택 |
| **H (heads)** | 4 | head_dim = D/H = 128/4 = 32. multi-head 표현력 + RoPE/STRING 안정성 검증 범위 |
| **N_stages** | 3 | ViT/MaxViT 표준. D=128 + 3 stage 조합이 ~0.60M 파라미터로 표현력과 효율의 균형 |
| **expand_ratio (FFN)** | 4 | Transformer 표준. FFN hidden = 128 × 4 = 512 |

상세: `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md` §2.1~2.6

---

## 2. 상대 위치 인코딩 — STRING

### 2.1. 검토 옵션 표

| 기법 | 이동 불변성 | 비균일 메시 (반사체) | 축 간 결합 | gradient 안전 | 물리 좌표 입력 | 학계 검증 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **기존 RPE** (Bias Table) | △ | ❌ | ✅ | ❌ Prob-6 | ❌ 정수만 | 일반 |
| **3D RoPE** (축별 분리) | ✅ | △ | **❌** | ✅ | △ | 일반 |
| **STRING** (채택) | **✅ 엄밀** | **✅** | **✅** | **✅** | **✅** | ICML 2025 Spotlight |
| **LieRE** | △ 근사 | △ (실검증 균일) | ✅ | ✅ | ✅ | Stanford AIMI 2024 |
| **GeoPE** (Quaternion) | ✅ | ❌ | ✅ | ✅ | ❌ | 미성숙 |

상세 비교: `2026-04-04 3D 위치 인코딩 기법 검토.md` §1, §3.1~3.5

### 2.2. 거부 옵션의 핵심 사유

- **기존 RPE**: 정수 인덱스 기반 → 추후 비균일 메시 (반사체 30 cm + 연료 10 cm 혼합) 대응 불가. Prob-6 (gradient 단절) 별도 수정 필요
- **3D RoPE (축별 분리)**: feature 를 d/3 씩 분리 → 축 간 결합 부재 → 제어봉 Z 변화의 XY 영향 표현 불가
- **LieRE**: 비가환 생성자, 이동 불변성 근사적 (BCH 의 commutator 항으로 인한 오차). 균일 격자만 실검증
- **GeoPE**: 물리 좌표 미검증, feature 3 의 배수 제약, STRING 의 특수 사례에 포함됨

### 2.3. 채택 근거

- **물리 좌표 직접 입력**: 좌표 r_k 를 연속 실수값으로 입력 가능 → 추후 반사체 포함 (비균일 메시) 시 코드 변경 0
- **축 간 결합**: 생성자 {L_z, L_y, L_x} 가 모두 d×d dense matrix → 한 축의 회전이 다른 축의 feature 도 변환 (3D RoPE 의 d/3 분리와 본질적 차이)
- **이동 불변성 엄밀**: 교환 조건 [L_i, L_j] = 0 부과로 R(a)·R(b) = R(a+b) 보장 → R(r_i)^T·R(r_j) = R(r_j - r_i) 엄밀 성립 (§2.4 수학)
- **학계 검증**: ICML 2025 Spotlight, RoPE 의 수학적 일반화

### 2.4. STRING 의 수학적 핵심 — Q/K 회전 trick

STRING (RoPE 일반화) 의 핵심은 **Q/K projection 직후의 회전**:

```
위치 r_i 의 토큰 i:  Q'_i = R(r_i) · Q_i      ← Q 벡터 (d_head 차원) 를 위치에 따라 회전
위치 r_j 의 토큰 j:  K'_j = R(r_j) · K_j      ← K 벡터 (d_head 차원) 를 위치에 따라 회전
```

각 노드의 좌표가 다르므로 R(r_i) ≠ R(r_j). Attention score 계산:

```
Q'_i^T · K'_j = (R(r_i) Q_i)^T · (R(r_j) K_j)
              = Q_i^T · R(r_i)^T · R(r_j) · K_j
              = Q_i^T · R(-r_i) · R(r_j) · K_j         ← R 직교: R^T = R^{-1}
              = Q_i^T · R(r_j - r_i) · K_j             ← 교환 조건: R(a)·R(b) = R(a+b)
```

→ **절대 위치 r_i, r_j 는 사라지고 상대 거리 (r_j - r_i) 만 score 에 영향**

이 trick 이 작동하려면 (학계 표준 3가지 제약):

1. **회전 R 적용 시점이 W_Q/W_K projection *후* 여야 함**:
   - projection 전 (token feature x 자체) 에 R 을 곱하면 → W_Q · (R · x) 에서 R 과 W_Q 가 섞여 직교성 깨짐
   - projection 후 (Q, K 벡터) 에 R 을 곱해야 R^T · R = I 구조 보존
2. **dot product 직전 이어야 함**:
   - Q, K 회전 후 즉시 dot product 해야 위 식 전개 valid
   - 회전 후 추가 변환 (예: 또 다른 linear layer) 이 있으면 등식 깨짐
3. **매 attention layer 마다 새로 적용**:
   - 매 layer 마다 새로운 W_Q, W_K 가 있고 새로운 Q, K 가 생성
   - 각 layer 의 새 Q, K 에 대해 회전을 다시 적용해야 함
   - LAPE 처럼 1회 add 로는 불가능 (가산이 아닌 회전이므로 residual stream 으로 전달 안 됨)

### 2.5. 왜 V 에는 적용하지 않는가

- attention 출력: `output = softmax(Q K^T) · V`
- softmax(QK^T) 가 이미 위치 정보를 완전히 인코딩 (Q/K 회전 trick 으로)
- V 는 *content (값 내용)* 을 집계하는 역할이지 위치 정보의 운반체가 아님
- V 를 회전시키면 출력이 좌표계에 종속되어 translation invariance 깨짐
- RoPE 원문, STRING 원문, 모든 RoPE 계열 구현체에서 V 는 회전하지 않음 — **만장일치 표준**

상세: `2026-04-04 3D 위치 인코딩 기법 검토.md` §3.3 (STRING 수학적 구조), §3.6.2 (적용 위치 학계 근거)

### 2.6. 하이퍼파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| **회전 구조** | Block-diagonal 2×2 | RoPE 표준. 각 2×2 블록이 하나의 평면 회전. dense 생성자보다 파라미터 절약 + Cayley 변환 안정 |
| **축별 freq** | 독립 (Z, Y, X 각 별도) | 각 축의 물리적 스케일 차이 (Z=10cm, XY=21.6cm pitch) 반영 |
| **head 별 freq** | 독립 | multi-head 의 서로 다른 frequency band 학습 (RoPE 표준) |
| **base** | 10000 (RoPE 표준) | wide frequency range 확보 |
| **좌표 입력** | 정규화된 물리 cm (0~1) | 좌표 스케일링으로 R(r) 안정 + 추후 비균일 메시 자동 호환 |
| **V 적용** | 적용 안 함 | RoPE/STRING 표준 |

코드 위치: `src/models/spatial/attention3d.py:45-200` (STRINGRelativePE3D 클래스)

---

## 3. 절대 위치 인코딩 — LAPE

### 3.1. 검토 옵션 표

| 옵션 | 적용 위치 | 적용 빈도 | 학습 가능 변수 | 채택 |
|---|---|:---:|---|:---:|
| LAPE 미도입 | — | — | 없음 | ✗ |
| **LAPE, CellEmbedder 직후 1회 add** | flatten 전, attention 진입 직전 | **1회** | (20, 6, 6, D) | **✓** |
| LAPE, 디코더에만 | 디코더 attention 진입 직전 | 1회 | (20, 6, 6, D) | ✗ |
| LAPE, 매 attention layer 반복 | 각 stage 직전 | 3회 | 동일 변수 공유 | ✗ |

### 3.2. 거부 옵션의 핵심 사유

- **LAPE 미도입**: STRING 은 상대 거리만 인코딩 → 절대 위치의 고유 정체성 표현 불가 → boundary cell (반사체 인접, halo) 식별 불가
- **디코더에만**: 인코더가 boundary 정체성을 학습 못 하면 디코더가 받는 latent 에도 그 정보가 빠짐. 입력 단계에 위치 identity 가 새겨져야 인코더 attention 이 활용 가능
- **매 layer 반복**: residual stream 관점에서 1회 add 로 충분 (§3.4 참조). ViT/BERT 어디에서도 매 layer add 미채택

### 3.3. 채택 근거

- **Boundary cell 식별**: 각 (z, y, x) 위치에 학습 가능 변수 D 차원 → 위치별 *identity vector* 학습. boundary cell 의 LAPE 변수가 internal cell 과 다른 패턴으로 진화 → attention 이 식별
- **ViT/BERT 학계 표준**: Vaswani 2017, BERT, ViT 모두 patch/token embedding 직후 1회 add. 본 프로젝트는 동일 패턴
- **Dufter et al. 2022 상보성 정당화**: LAPE (절대) ⟂ STRING (상대) 의 직교 작용. 실험에서 두 PE 결합이 둘 중 하나만 사용보다 항상 우수
- **형상 비용 미미**: LAPE 변수 차이 ~28K (halo 6,6 vs quarter 5,5), 인코더 0.66M 대비 4.3% 추가

### 3.4. LAPE 의 수학적 핵심 — Residual Stream 관점

Pre-LN 트랜스포머의 각 블록은 residual connection 구조:
```
x_{l+1} = x_l + Attention(LN(x_l))    ← residual stream 에 sublayer 결과 누적
x_{l+1} = x_l + FFN(LN(x_l))
```

LAPE 가 입력 단계에 1회 add 되면, residual stream `x_0 = embedding + LAPE` 에서 시작. 이후 모든 layer 에서:
```
x_l = x_0 + Σ (Attention/FFN sublayers)
    = embedding + LAPE + Σ (sublayers)
                ↑
        영원히 살아남아 마지막 layer 까지 전달
```

→ LAPE 가 가산 형태로 residual stream 에 새겨지므로 매 layer 마다 다시 더할 필요 없음. 이것이 ViT/BERT 가 1회 add 만 채택하는 수학적 이유.

### 3.5. LAPE 적용 위치 — Stem 직후 1회 add

```
Data input: (B, 20, 5, 5, 21)              ← 데이터셋 quarter
       │
       ▼
[halo_expand(sym)]
       │
       ▼ (B, 20, 6, 6, 21)
[CellEmbedder] Conv3D(1,1,1), 21 → 128       ← Stem (no spatial mixing)
       │
       ▼ (B, 20, 6, 6, 128)
[LearnedAbsolutePE3D] +(20, 6, 6, 128) embedding   ← LAPE 1회 add (여기!)
       │                                      (이후 모든 layer 는 residual stream 으로 전달)
       ▼ (B, 20, 6, 6, 128)
[reshape] flatten spatial → (B, 720, 128)
       │
       ▼
[Attention Stage 1, 2, 3 ...]                     ← LAPE 더 이상 안 더함
```

코드 위치: `src/models/spatial/layers3d.py:150` (LearnedAbsolutePE3D 클래스), `:215-218` (call 메서드)

```python
# LearnedAbsolutePE3D.call()
def call(self, x: tf.Tensor) -> tf.Tensor:
    # x: (B, Z, qH, qW, D)
    # self.embedding: (Z, qH, qW, D) trainable
    return x + self.embedding[tf.newaxis, ...]   # broadcast add
```

### 3.6. 하이퍼파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| **격자 크기** | (20, 6, 6) | halo expand 후 형상. 05_symmetry_mode 결정 |
| **D (embedding 차원)** | 128 (encoder D 와 동일) | feature 와 직접 add 하므로 동일 차원 필수 |
| **초기화** | RandomNormal(stddev=0.02) | ViT/BERT 표준. Pre-LN 안정성 위해 작은 분산 |
| **trainable** | True | 학습으로 위치별 identity 학습 |
| **변수 수** | 20 × 6 × 6 × 128 = 92,160 | 인코더 ~0.66M 의 14% (별도 LAPE 변수) |

코드 위치: `src/models/spatial/layers3d.py:150-244`

---

## 4. 두 PE 의 동시 사용 정당화 — Dufter et al. 2022

### 4.1. 상보성 비교 표

| 항목 | LAPE (절대) | STRING (상대) |
|---|---|---|
| 인코딩 대상 | 각 위치의 **고유 정체성** | 두 위치 사이의 **상대 거리** |
| 적용 방식 | feature add | Q/K rotation (matrix multiplication) |
| 적용 위치 | Stem 직후 (입력 단계) 1회 | Attention 내부 Q/K projection 직후 (매 layer) |
| 적용 빈도 | **1회만** | **매 attention layer** |
| 번역 불변성 | ✗ 깨뜨림 (위치별 다른 add) | ✓ 유지 (R^T·R = R(차)) |
| 학습 파라미터 | (Z·H·W·D) 위치별 변수 | 회전 주파수만 (위치별 변수 없음) |
| Residual stream 전달 | ✓ 가산이라 자연 전달 | ✗ 매 layer 새로 적용 필요 (회전이라 전달 안 됨) |
| 본 프로젝트 역할 | **boundary cell 식별** | **상대 거리 기반 attention** |

### 4.2. 상보성의 메커니즘적 이유

LAPE 와 STRING 은 **서로 다른 자리 + 서로 다른 수학적 연산** 으로 들어가므로 간섭 없음:

- LAPE 는 **embedding 단계** 에서 **feature add** → token feature 에 위치 identity 새김
- STRING 은 **attention 내부** 에서 **Q/K rotation** → attention score 에 상대 거리 정보 주입

두 작용이 직교 (orthogonal): LAPE 가 새기는 absolute identity 와 STRING 이 만드는 relative score 가 서로 정보 영역이 다름. Dufter et al. 2022 의 실험 결과는 두 PE 의 결합이 둘 중 하나만 사용하는 경우보다 항상 우수함을 입증.

> **원문 인용** (Dufter et al. 2022, Computational Linguistics 48(3):733):
> "Their absolute and relative position embeddings are complementary. Indeed, in their experiments the combination outperforms either alone."

---

## 5. 본 프로젝트 코드 정합성 검증

| 항목 | 학계 표준 | 본 프로젝트 코드 | 정합 |
|---|---|---|:---:|
| LAPE 적용 시점 | patch/token embedding 직후 (Vaswani 2017, BERT, ViT) | CellEmbedder 직후 (`layers3d.py:215-218`) | ✓ |
| LAPE 적용 빈도 | 1회 (입력 단계) | 1회 | ✓ |
| LAPE 가산 방식 | broadcast add | broadcast add (`x + self.embedding[None, ...]`) | ✓ |
| STRING 적용 위치 | Q/K projection 직후, dot product 직전 (RoFormer §3.4) | `attention3d.py:285-294` 의 정확히 그 위치 | ✓ |
| STRING 적용 빈도 | 매 attention layer (RoPE 표준) | 3 stages × 1 attention = 3회 | ✓ |
| STRING 회전 방식 | R(r) · Q matrix multiplication | block-diagonal 2×2 matmul (`STRINGRelativePE3D.call`) | ✓ |
| V 회전 적용 여부 | ✗ 제외 (RoPE/STRING 만장일치 표준) | ✗ (Q, K 만 회전, V 는 그대로. `string_pe(Q, K, coords)` 호출에서 V 가 인자 아님) | ✓ |
| 상대 위치 보장 | R^T · R = R(차) | 교환 조건 + 직교행렬로 자동 | ✓ |
| LAPE + 상대 PE 동시 사용 | Dufter et al. 2022 권장 | LAPE + STRING 동시 도입 | ✓ |

**결론**: 9개 항목 모두 학계 표준에 정확히 부합. 본 프로젝트의 LAPE/STRING 적용 위치는 ViT (절대 PE) + RoPE (상대 PE) + Dufter et al. (조합 정당화) 의 학계 표준을 그대로 따름.

---

## 6. 결정 일자 + 결정 과정 요약

### 6.1. Attention Backbone (이전 03)
- **2026-04-07**: 사용자 의문 "Block+Grid 검증된 패턴 vs Full Attention 단순성, 어느 쪽?"
- 결정: **Full Attention, D=128, 3-stage** 채택. 핵심 이유는 N=720 에서 연산 trivial
- 상세: `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md` §2.6

### 6.2. STRING (이전 04)
- **2026-04-04**: 위치 인코딩 5종 (RPE / 3D RoPE / STRING / LieRE / GeoPE) 비교 검토
- **2026-04-08**: 사용자 결정 "STRING + LAPE 동시 구현, RPE 단계 거치지 않음"
- 결정: STRING block-diag 2×2 + 축별 독립 freq, 물리 좌표 직접 입력
- 상세: `2026-04-04 3D 위치 인코딩 기법 검토.md` §3.3, §3.6.2

### 6.3. LAPE (이전 05)
- **2026-04-08**: 사용자 의문 "Quarter symmetry 로 인한 boundary 정보 손실. 어떻게 식별?"
- 결정: **LAPE 도입, 인코더에만, BC mask 채널 미사용**
- 상세: `2026-04-04 3D 위치 인코딩 기법 검토.md` §3.6.1, §3.6.3

### 6.4. 통합 (본 문서, 2026-04-09)
- 03/04/05 결정을 "공간 정보 학습 backbone" 이라는 하나의 일관된 설계 의도로 통합
- 04/05 의 적용 위치가 학계 표준 (ViT/BERT/RoFormer/Dufter) 에 정확히 부합함을 §5 정합성 검증 표로 확인

---

## 7. 참고 문헌

### 학계 원문 (필수 인용)
1. **Vaswani et al. 2017**. "Attention Is All You Need". NeurIPS. (sinusoidal absolute PE, input 단계 1회 add)
2. **Devlin et al. 2018**. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL. (learnable absolute PE, input 단계 1회 add)
3. **Dosovitskiy et al. 2020**. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale". ICLR 2021. (ViT, learnable absolute PE, patch embedding 직후 1회 add)
4. **Su et al. 2021**. "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv 2104.09864. (RoPE, Q/K rotation, 매 attention layer)
5. **Dufter, Schmitt & Schütze. 2022**. "Position Information in Transformers: An Overview". Computational Linguistics 48(3):733. (절대 + 상대 PE 조합 정당화)
6. **Reid et al. 2025**. "STRING: Separable Translationally Invariant Position Encodings". arXiv 2502.02562. ICML 2025 Spotlight. (RoPE 일반화, dense Lie 생성자, 교환 조건)
7. **Tu et al. 2022**. "MaxViT: Multi-Axis Vision Transformer". ECCV. (Block + Grid Attention 비교 baseline)

### 본 프로젝트 검토 문서
- `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md` §2.1~2.6 (Attention Backbone 결정 상세)
- `2026-04-04 3D 위치 인코딩 기법 검토.md` §1, §3.1~3.5 (5종 PE 비교), §3.6 (LAPE/STRING 적용 위치 학계 근거)
- `2026-04-04 정밀도 및 경량화 검토.md` §4 (Attention FLOPs 비교)

### 본 프로젝트 코드
- `src/models/spatial/layers3d.py` line 150-244 (LearnedAbsolutePE3D 클래스)
- `src/models/spatial/layers3d.py` line 215-218 (LAPE call: `x + self.embedding[None, ...]`)
- `src/models/spatial/attention3d.py` line 45-200 (STRINGRelativePE3D 클래스)
- `src/models/spatial/attention3d.py` line 202-330 (FullAttention3D 클래스)
- `src/models/spatial/attention3d.py` line 285-294 (Q/K/V projection → string_pe(Q, K, coords) → softmax(QK^T/√d)·V)

### 본 폴더 cross-reference
- `00_README.md` 의 아키텍처 다이어그램 + LAPE/STRING 위치 표
- `02_cell_embedder.md` (CellEmbedder = Stem, LAPE 직전)
- `05_symmetry_mode.md` (halo expand 결정, LAPE/STRING 좌표가 (6,6) 격자임을 명시)
- 검토 과정의 역사: 본 폴더 외부의 검토 문서 (`2026-03-30~2026-04-07`)
