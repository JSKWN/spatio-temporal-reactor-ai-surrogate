# Conditional LAPE 적용 검토

> **결정**: 기존 `LearnedAbsolutePE3D` (LAPE 테이블 1개) → `ConditionalLAPE3D` (테이블 2개, sym_type별 분기) 확장
> **결정 일자**: 2026-04-14
> **근거 문서**: `05a_symmetry_distinguishability_proof.md` §8.5 방안 C 채택
> **본 문서의 위치**: LAPE가 무엇이고, 대칭 조건이 어떻게 LAPE에 반영되며, 인코더 내부에서 신호가 어떻게 흐르는지 정리

---

## 0. 결정 요약

| 항목 | 결정 | 핵심 이유 |
|---|---|---|
| **적용 방안** | 방안 C: Conditional LAPE | 인코더 설계 원칙 무위반 + 물리적 해석 자연스러움 |
| **메커니즘** | LAPE 테이블 2개 (mirror / rotation), sym_type으로 선택 | BERT segment embedding과 동일 원리 |
| **적용 위치** | CellEmbedder 직후, Attention 진입 전 (기존 LAPE와 동일) | Residual stream으로 전 layer에 전달 |
| **sym_type 소스** | HDF5 metadata `symmetry_type` 필드 | 데이터 생산 → 전처리 → HDF5 → 모델 일관 |
| **파라미터 증감** | +92,160 (LAPE 2배), 인코더 대비 +14% | 미미한 증가 |

---

## 1. LAPE 동작 원리

### 1.1. LAPE가 해결하는 문제

Transformer는 입력 토큰을 **집합(set)** 으로 처리한다. 위치 정보 없이는 cell (z=0, y=0, x=0)과 cell (z=19, y=4, x=4)을 구분할 수 없다. LAPE는 격자의 각 공간 위치에 고유한 **학습 가능 벡터 (정체성 벡터)** 를 할당하여 이 문제를 해결한다.

### 1.2. 현재 구현 (`layers3d.py:150-244`)

```
[초기화] (20, 6, 6, 128) 크기의 trainable 텐서 생성
         RandomNormal(mean=0, stddev=0.02)
         → 720개 위치 × 128차원 = 92,160개 학습 가능 스칼라

[순전파] x + embedding[None, ...]              (broadcast 합산)
         입력 (B, 20, 6, 6, 128) 의 각 위치에 해당 위치의 128차원 벡터를 더함

[학습]   L_data, L_data_halo, L_diff_rel 손실의 gradient가 각 위치의 임베딩을 갱신
         → boundary cell은 interior cell과 다른 방향으로 진화 (BC 식별 학습)
```

코드:
```python
# LearnedAbsolutePE3D.call()  —  layers3d.py:215-218
def call(self, x: tf.Tensor) -> tf.Tensor:
    # x: (B, Z, qH, qW, D)
    # self.embedding: (Z, qH, qW, D) trainable
    return x + self.embedding[tf.newaxis, ...]    # broadcast add
```

### 1.3. 1회 적용으로 충분한 이유 — Residual Stream

Pre-LN Transformer의 각 블록은 residual connection 구조이다:

```
x_{l+1} = x_l + Attention(LN(x_l))
x_{l+1} = x_l + FFN(LN(x_l))
```

LAPE가 입력 단계에서 1회 add되면:

```
x_0 = CellEmbedder(input) + LAPE         ← 여기서 LAPE 신호가 새겨짐
x_1 = x_0 + Attention(LN(x_0))           ← x_0의 LAPE 성분이 살아있음
x_2 = x_1 + FFN(LN(x_1))
...
x_final = x_0 + Σ(sublayers)
        = CellEmbedder + LAPE + Σ(sublayers)
                         ↑ LAPE는 끝까지 전달됨
```

**LAPE는 가산(add)이라 residual stream에 영원히 살아남는다.** 이것이 ViT/BERT가 입력 단계에서 1회 add만 채택하는 수학적 이유이다. 매 layer마다 다시 더할 필요가 없다.

> 상세: `03_attention_and_position.md` §3.4

### 1.4. LAPE와 STRING의 역할 분담

본 프로젝트는 **절대 위치 인코딩 (LAPE)** 과 **상대 위치 인코딩 (STRING)** 을 동시에 사용한다. 두 PE는 서로 다른 자리 + 서로 다른 수학적 연산으로 작용하므로 **간섭 없음** (Dufter et al. 2022 상보성 정당화).

| | LAPE (절대) | STRING (상대) |
|---|---|---|
| **인코딩 대상** | "나는 위치 (z,y,x)이다" — 고유 정체성 | "A와 B 사이 거리는 (dz,dy,dx)" — 쌍별 관계 |
| **적용 방식** | token feature에 **가산** | Q/K에 **회전** (matrix multiplication) |
| **적용 시점** | CellEmbedder 직후 **1회** | 매 attention layer Q/K projection 직후 |
| **적용 대상** | value stream (feature 자체) | score stream (attention 점수) |
| **본 프로젝트 역할** | boundary cell 식별, **대칭 유형 정체성** | 상대 거리 기반 attention |
| **코드 위치** | `layers3d.py:215-218` | `attention3d.py:285-294` |

> 상세: `03_attention_and_position.md` §4

---

## 2. Conditional LAPE — LAPE를 대칭 유형별로 분기

> **용어**: 본 문서에서 LAPE "테이블"은 **(20, 6, 6, 128) 형상의 trainable 텐서** 를 의미한다. 720개 cell 각각에 128차원 벡터가 할당된, 총 92,160개의 학습 가능한 부동소수점 숫자이다.

### 2.1. 동기: 왜 LAPE 테이블 1개로는 부족한가

LAPE 테이블이 1개이면 → mirror LP든 rotation LP든 동일한 위치 임베딩을 받음 → 모델은 halo cell 값 패턴에서만 대칭을 추론해야 함.

`05a` 문서 §7.4에서 밝힌 구조적 한계:
- halo cell은 inner 2행/2열만 제약 → 4개 연료 cell 쌍은 halo 값만으로 **미제약**
- halo 값 패턴만으로 대칭을 추론하는 것은 가능하나 (Theorem 1), attention에 **암묵적 학습 부담**을 지움

Conditional LAPE는 이 한계를 보완: **위치 임베딩 자체에 대칭 정보를 내재**시킴.

### 2.2. BERT Segment Embedding과의 유사성

| | BERT | 본 프로젝트 |
|---|---|---|
| **구분 대상** | 문장 A / 문장 B | mirror / rotation |
| **메커니즘** | segment별 다른 임베딩 벡터를 token에 add | sym_type별 다른 LAPE 테이블을 cell에 add |
| **class 수** | 2 | 2 |
| **원리** | 이산 조건에 따른 위치 임베딩 분기 | 동일 |

BERT에서 segment embedding은 모델이 두 문장의 경계를 명시적으로 인식하게 한다. Conditional LAPE도 동일하게 — 모델이 mirror 모드인지 rotation 모드인지를 위치 임베딩 수준에서 명시적으로 인식하게 한다.

### 2.3. 구체적 동작

**현재 (LearnedAbsolutePE3D, 테이블 1개)**:

```python
self.embedding: (20, 6, 6, 128)              # 92,160 파라미터

def call(self, x):
    return x + self.embedding[None, ...]      # 모든 샘플에 동일 임베딩
```

**변경 후 (ConditionalLAPE3D, 테이블 2개)**:

```python
self.lape_mirror:   (20, 6, 6, 128)          # 92,160 파라미터
self.lape_rotation: (20, 6, 6, 128)          # 92,160 파라미터
                                               # 합계: 184,320 파라미터

def call(self, x, sym_type):
    # sym_type: (B,) int32, 값 {0=mirror, 1=rotation}

    cond = sym_type[:, None, None, None, None] == 0   # (B,1,1,1,1) bool
    lape = tf.where(cond,
                    self.lape_mirror[None, ...],       # (1,20,6,6,128)
                    self.lape_rotation[None, ...])     # (1,20,6,6,128)
    return x + lape                                    # (B,20,6,6,128)
```

### 2.4. `tf.where` 의 역할과 gradient 흐름

`tf.where`는 sym_type 값에 따라 두 LAPE 테이블 중 하나를 선택하는 분기문이다:

```
sym_type == 0 (mirror)   → lape = lape_mirror    → x + lape_mirror
sym_type == 1 (rotation) → lape = lape_rotation  → x + lape_rotation
```

**Gradient 흐름**: `tf.where`에 의해 선택된 테이블만 계산 그래프에 포함된다. 선택되지 않은 테이블은 계산 그래프 밖이므로 gradient가 0이다. 선택된 테이블에는 **모든 물리량 loss** (L_data, L_data_halo, L_diff_rel, L_Bateman 등) 의 gradient가 흐른다. "대칭 정보에 관한 loss만 흐르는 것"이 아니다.

```
[sym_type=0 (mirror) 일 때의 계산 그래프]

입력 x → x + lape_mirror → Attention → ... → 디코더 출력
                                                    ↓
                                    L_data + L_data_halo + L_diff_rel + ...
                                                    ↓ backprop
                                    lape_mirror:   모든 loss의 gradient 수신 → 갱신됨
                                    lape_rotation: 계산 그래프 밖 → gradient 0 → 갱신 안 됨
```

- `lape_mirror`는 mirror 데이터의 모든 물리량 loss로부터 학습 → **mirror 배치에 최적화된 위치 임베딩**으로 진화
- `lape_rotation`은 rotation 데이터의 모든 물리량 loss로부터 학습 → **rotation 배치에 최적화된 위치 임베딩**으로 진화
- 두 테이블이 서로 다른 값으로 수렴하는 이유: 같은 물리량이라도 대칭 유형에 따라 공간 분포가 다르기 때문
- 현재 학습 (100LP, 전부 mirror) 에서는 `lape_mirror`만 갱신됨. 향후 rotation LP 추가 시 `lape_rotation`이 학습 시작

### 2.5. 왜 Keras Embedding 레이어 대신 직접 weight를 생성하는가

`lape_mirror`와 `lape_rotation`은 둘 다 **(20, 6, 6, 128) 형상의 trainable 텐서**이다. Keras의 `Embedding` 레이어도 내부적으로 동일한 trainable 텐서를 사용한다. 차이는 수학이 아닌 **Keras API 선택**이다:

| 방식 | 저장 형태 | 판정 | 이유 |
|---|---|:---:|---|
| `Embedding(2, 92160)` | (2, 92160) 2D 행렬 → (20,6,6,128)로 reshape 필요 | X | reshape 불필요. Embedding은 어휘 수천~수만개용 추상화 |
| **`add_weight()` × 2** | (20, 6, 6, 128) 자연 4D 형상 그대로 | O | 형상 유지로 코드 가독성 확보. `get_norm_map()` 분석 직관적. 2-class에 적합 |

둘 다 본질은 동일한 trainable 텐서이다.

### 2.6. sym_type 인코딩 규약

| sym_type 값 | 대칭 유형 | MASTER 변수 | 비고 |
|:---:|---|---|---|
| `0` | mirror | nsym=0 (GEN_DIM) | 현재 100LP 데이터 |
| `1` | rotation | nsym=1 (GEN_DIM) | 향후 신규 LP |

MASTER 매뉴얼 GEN_DIM의 nsym 값과 일치시킴.

---

## 3. 신호 경로 — sym_type이 시스템을 흐르는 전체 경로

### 3.1. 전체 데이터 흐름

```
[HDF5 파일]
  metadata.attrs['symmetry_type'] = 'mirror'       # 문자열
  metadata.attrs['lp_geometry'] = 'quarter_core'
       │
       │  (1) 데이터셋 초기화 시 1회 읽기
       ▼
[Dataloader]
  sym_str = 'mirror'
  sym_int = 0                                       # 정수 인코딩
       │
       │  (2) 매 배치와 함께 전달
       ▼
[Model 입력]
  quarter_data: (B, 20, 5, 5, 21)     ← HDF5 샘플 데이터
  sym_type:     (B,) int32            ← metadata에서 읽은 상수
       │
       │  (3) halo_expand: sym_type으로 매핑 모드 결정
       ▼
[halo_expand(quarter_data, sym_type)]
  halo_data: (B, 20, 6, 6, 21)
       │
       │  (4) CellEmbedder: 채널 projection (sym_type 무관)
       ▼
[CellEmbedder Conv3D(1,1,1)]  21 → 128
  embedded: (B, 20, 6, 6, 128)
       │
       │  (5) ★ ConditionalLAPE3D: sym_type으로 테이블 선택 ★
       ▼
[ConditionalLAPE3D(embedded, sym_type)]
  positioned: (B, 20, 6, 6, 128)      ← 위치 정체성 + 대칭 정보 내재
       │
       │  (6) flatten
       ▼
[reshape → (B, 720, 128)]
       │
       │  (7) 3 stages × [Pre-LN + FullAttention3D(STRING) + FFN3D]
       │      ★ sym_type 전달하지 않음 — 신호는 residual stream에 있음 ★
       ▼
  encoded: (B, 720, 128) → reshape → (B, 20, 6, 6, 128)
       │
       ▼
  Mamba SSM → Decoder → Output
```

### 3.2. sym_type이 닿는 곳 / 닿지 않는 곳

| 컴포넌트 | sym_type 사용 | 이유 |
|---|:---:|---|
| **halo_expand()** | O | mirror/rotation에 따라 halo cell 매핑이 다름 |
| **ConditionalLAPE3D** | O | 대칭 유형별 다른 위치 임베딩 선택 |
| CellEmbedder | X | 채널 projection은 대칭 무관 |
| LayerNorm (Pre-LN) | X | LN의 γ, β에 외부 조건을 주입하는 FiLM/AdaLN 변조는 디코더 전용 (`04` §1.4) |
| FullAttention3D / STRING | X | 상대 위치 인코딩은 대칭 무관 |
| FFN3D | X | token-wise 비선형 변환은 대칭 무관 |

sym_type은 인코더에서 **2곳** (halo_expand + Conditional LAPE) 에서만 사용되며, 이후 신호는 residual stream을 통해 자동 전달된다.

### 3.3. 기존 인코더 결정 사항과의 정합성

인코더의 LayerNorm에는 FiLM/AdaLN 같은 조건부 affine 변조를 적용하지 않고, 디코더에서만 p_load 조건부 변조를 수행한다는 결정이 있다 (`04_normalization_omitted_options.md` §1.4 "인코더 vs 디코더 분리").

| 기존 결정 사항 | Conditional LAPE 적합성 |
|---|---|
| 인코더 LN은 일반 Pre-LN. LN의 γ, β에 외부 조건 미주입 (`04` §1.4) | **적합** — Conditional LAPE는 LN 이전 단계에서 feature에 add하는 것이지, LN 파라미터를 변조하지 않음 |
| FiLM/AdaLN 변조는 디코더 전용 (`04` §1.4) | **적합** — LAPE 테이블 선택은 affine 변조 (γ·x + β) 가 아님 |
| 인코더에 p_load 미주입 (`04` §1.4) | **적합** — sym_type은 p_load와 별개. 데이터셋 수준의 기하 속성 |
| CellEmbedder 입력 21ch 유지 (`02_cell_embedder.md`) | **적합** — 입력 텐서 구조 변경 없음 |

→ 기존 인코더 결정 사항과 **충돌 없음** (`05a` §8.5 판정과 일치)

### 3.4. 디코더로의 대칭 정보 전달 — 한계 및 향후 과제

Conditional LAPE 신호는 인코더 내부에서는 residual stream을 통해 전 layer에 보존된다 (§1.3). 그러나 인코더 이후의 경로에서는 보존이 보장되지 않는다:

```
인코더 출력 (B, 20, 6, 6, 128)    ← LAPE 신호가 residual stream에 포함
       │
       ▼
[Mamba SSM] — S6 selective scan (선형 재귀)
       │       인코더와 달리 residual connection 구조가 다름
       │       → LAPE 신호가 Mamba의 상태 전이 과정에서 희석될 수 있음
       ▼
[Decoder] — 인코더에서 전달받은 feature를 기반으로 출력 생성
              LAPE 신호가 약화된 상태로 수신할 가능성
```

**우려**: Mamba의 selective scan은 입력 의존적으로 B, C, Δ 행렬을 갱신하며, 인코더의 residual stream처럼 원본 신호를 가산 형태로 그대로 보존하는 구조가 아니다. 따라서 인코더에서 add된 대칭 유형 정보가 Mamba를 거치며 희석될 수 있다.

**향후 과제**: 디코더의 AdaLN-Zero에 sym_type을 추가로 주입하는 방안을 별도 검토할 필요가 있다. 구체적으로는 기존 p_load 조건 경로에 sym_type 임베딩을 합산하는 방식이다 (`05a` §8.5 방안 A/B: "디코더 AdaLN-Zero 입력에 합산: `p_load_emb + sym_emb → MLP → γ, β`"). 이는 디코더 설계 시 함께 결정한다.

### 3.5. 배치 처리

현재 100LP 데이터는 전부 mirror이므로 sym_type은 배치 내 상수이다. 그러나 구현은 **샘플별 선택이 가능한 구조** (`tf.where` + per-sample condition) 로 작성한다:

- 현재: 전 배치 동일 → `tf.where`가 사실상 단일 테이블 선택과 동치
- 향후: mirror + rotation LP 혼합 학습 시 → 샘플별 독립 선택 가능
- 비용: 0 (broadcasting이므로 per-batch나 per-sample이나 연산량 동일)

---

## 4. 구현 방향

### 4.1. ConditionalLAPE3D 클래스 설계

**파일**: `src/models/spatial/layers3d.py` (기존 `LearnedAbsolutePE3D` 직후에 추가)

**기존 클래스 유지**: `LearnedAbsolutePE3D`는 삭제하지 않음 (ablation 비교용)

```python
class ConditionalLAPE3D(layers.Layer):
    """대칭 유형별 조건부 3D 절대 위치 임베딩.

    mirror/rotation 두 LAPE 테이블을 보유하고, sym_type에 따라 선택.
    BERT의 segment embedding과 동일 원리 (2-class 조건부 위치 임베딩 분기).

    Args:
        z_dim:      Z 격자 크기 (예: 20)
        qh_dim:     quarter H 격자 크기 (예: 6, halo 포함)
        qw_dim:     quarter W 격자 크기 (예: 6, halo 포함)
        d_latent:   임베딩 차원 D (예: 128)
        init_scale: RandomNormal stddev (기본 0.02, ViT 표준)
    """

    def __init__(self, z_dim, qh_dim, qw_dim, d_latent, init_scale=0.02, ...):
        ...

    def build(self, input_shape):
        self.lape_mirror = self.add_weight(
            name="lape_mirror",
            shape=(self.z_dim, self.qh_dim, self.qw_dim, self.d_latent),
            initializer=RandomNormal(stddev=self.init_scale))
        self.lape_rotation = self.add_weight(
            name="lape_rotation",
            shape=(self.z_dim, self.qh_dim, self.qw_dim, self.d_latent),
            initializer=RandomNormal(stddev=self.init_scale))

    def call(self, x, sym_type):
        # x: (B, Z, qH, qW, D),  sym_type: (B,) int32
        cond = tf.equal(sym_type[:, None, None, None, None], 0)
        lape = tf.where(cond,
                        self.lape_mirror[None, ...],
                        self.lape_rotation[None, ...])
        return x + lape

    def get_norm_map(self):
        """학습 후 분석용. 대칭 유형별 위치 L2 노름 맵."""
        return {
            'mirror':   tf.norm(self.lape_mirror, axis=-1),    # (Z, qH, qW)
            'rotation': tf.norm(self.lape_rotation, axis=-1),  # (Z, qH, qW)
        }
```

### 4.2. encoder3d.py에서의 호출 (향후)

```python
class SpatialEncoder3D(tf.keras.layers.Layer):
    def call(self, x, sym_type, coords, training=False):
        # x: (B, 20, 6, 6, 21)  — halo 확장 완료
        # sym_type: (B,) int32  — 0=mirror, 1=rotation

        h = self.cell_embedder(x)                   # (B, 20, 6, 6, 128)
        h = self.cond_lape(h, sym_type)             # ★ sym_type 사용 지점
        h = tf.reshape(h, (-1, 720, self.d_latent)) # (B, 720, 128)

        for stage in self.stages:
            h = stage(h, coords, training=training) # sym_type 전달 안 함
        return tf.reshape(h, (-1, 20, 6, 6, self.d_latent))
```

### 4.3. Dataloader에서의 sym_type 공급

```python
# 데이터셋 초기화 시 1회 읽기
with h5py.File(h5_path, 'r') as f:
    sym_str = f['metadata'].attrs['symmetry_type']   # 'mirror' or 'rotation'
sym_int = 0 if sym_str == 'mirror' else 1

# config와 HDF5 metadata 불일치 검증
config_sym = model_config['data']['symmetry_type']
assert config_sym == sym_str, f"불일치: config={config_sym}, HDF5={sym_str}"

# 배치 생성 시
batch_sym = tf.fill([batch_size], sym_int)            # (B,) int32
```

### 4.4. model.yaml config (향후)

```yaml
encoder:
  lape_type: 'conditional'    # 'single' | 'conditional'
  # 'single':      기존 LearnedAbsolutePE3D (ablation용)
  # 'conditional':  ConditionalLAPE3D (기본값)
```

### 4.5. 파라미터 영향

| 항목 | 현재 | 변경 후 | 증감 |
|---|---:|---:|---:|
| LAPE 테이블 | 92,160 | 184,320 | +92,160 (+100%) |
| 인코더 전체 (~0.66M) | ~660,000 | ~752,160 | **+14%** |

인코더 전체 파라미터 대비 14% 증가. LAPE 자체가 인코더에서 차지하는 비중이 작으므로, 2배가 되어도 전체 모델 규모에 미치는 영향은 미미하다.

---

## 5. 학계 선례

Conditional LAPE에 **직접 대응하는 단일 논문**은 없으나, 대칭 유형에 따라 위치 인코딩 (또는 잠재 표현) 이 달라져야 한다는 관점을 지지하는 다수의 선행 연구가 존재한다:

| 연구 | 핵심 기여 | 본 프로젝트와의 유사점 |
|---|---|---|
| **BERT** (Devlin 2019) | segment embedding으로 문장 A/B 구분 | 2-class 조건부 위치 임베딩 분기 원리 동일 |
| **SEN** (Park et al., ICML 2022) | 대칭 변환군별 다른 임베딩 공간 학습 | 대칭군에 따라 다른 표현 → Conditional LAPE와 동일 철학 |
| **GE-ViT** (Xu et al., 2023) | ViT 위치 인코딩이 등변성 학습 방해 문제 지적 → 군 등변 PE로 대체 | 대칭군 선택에 따라 PE가 달라져야 한다는 근거 |
| **Platonic Transformer** (Niazoys et al., 2025) | RoPE를 복수 reference frame에 대해 병렬 적용 | 대칭 유형에 따라 다른 reference frame 선택 → 구조적 유사 |

> 상세 레퍼런스: `05a_symmetry_distinguishability_proof.md` §8.5

---

## 6. 대안 기법 비교 검토

Conditional LAPE 외에 대칭 정보를 모델에 반영하는 6가지 대안 기법을 논문 원본 기준으로 조사하고, 본 프로젝트 아키텍처와의 적합성을 검토한다.

### 6.1. 궤도(Orbit) 기반 PE

**논문**: Fu, J., Xie, Q., Meng, D., & Xu, Z. (2026). *Vanilla Group Equivariant Vision Transformer: Simple and Effective.* arXiv: 2602.08047.

> 참고: 원 출처로 언급된 ICLR 2025 poster 31458은 "Equivariant Symmetry Breaking Sets" (Xie & Smidt, arXiv 2402.02681) 로 궤도 기반 PE와 다른 논문이다. 설명된 개념은 상기 논문에 해당.

**핵심 메커니즘**: 대칭군 G 하에서 동치인 위치들의 집합 (궤도) 을 구하고, 궤도의 **정규 대표원** (사전순 최소 원소) 으로 매핑한 뒤 PE를 부여한다.
- C4 (90° 회전): 위치 (i,j)의 궤도 = {(i,j), (j,N-1-i), (N-1-i,N-1-j), (N-1-j,i)}
- D4 (회전+반사): 궤도 크기 최대 8
- 같은 궤도의 모든 위치가 **동일한 PE 값**을 받음

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 본 프로젝트와의 관련성 | **낮음** | 본 프로젝트는 **등변 모델이 아님**. 표준 ViT + LAPE + STRING 구조 |
| 해결하는 문제 | — | 등변 ViT에서 PE가 등변성을 깨뜨리는 문제 해결 |
| Conditional LAPE와의 차이 | — | 궤도 PE는 동치 셀에 같은 PE 부여 (등변성 보존). Conditional LAPE는 대칭 유형별 **다른** PE 부여 (대칭 식별) |

**부적합 사유**: 본 프로젝트는 등변 모델을 사용하지 않으며, 목적이 다르다. 궤도 PE는 "동치 셀이 같은 표현을 갖도록" 강제하는 반면, Conditional LAPE는 "mirror와 rotation에서 각 셀이 다른 정체성을 갖도록" 분기한다. 방향이 정반대.

### 6.2. FiLM 조건부 대칭 주입

**논문**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. arXiv: 1709.07871.

**핵심 메커니즘**: 외부 조건 신호 z로부터 γ(z), β(z)를 생성하여 네트워크 활성화에 affine 변조 적용.
- 수식: `FiLM(F) = γ(z) · F + β(z)`
- 원 논문: 자연어 질문을 GRU로 인코딩 → 각 ResNet 블록의 γ, β 생성
- 대칭 적용: sym_type (0/1)을 Embedding으로 인코딩 → 각 Transformer 블록의 LN γ, β 변조

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 메커니즘 효과 | **강함** | 매 layer에서 재적용되므로 신호 희석 없음 |
| 인코더 적합성 | **부적합** | 인코더 LN에 조건 주입 안 함 결정 (`04` §1.4) 에 위배 |
| 디코더 적합성 | **적합** | 기존 AdaLN-Zero p_load 경로에 sym_emb 합산 가능 |
| 파라미터 | 미미 | Embedding(2, D) + per-layer linear ≈ 3K |

**결론**: 인코더에는 부적합 (설계 원칙 위배). 디코더의 AdaLN-Zero에 sym_type을 합산하는 경로로 활용 가능 — 이는 §3.4에서 언급한 "디코더 대칭 정보 재주입" 방안과 일치하며, 디코더 설계 시 별도 검토한다.

### 6.3. SymPE — 확률적 대칭 파괴

**논문**: Lawrence, H., Portilheiro, V., Zhang, Y., & Kaba, S.-O. (2025). *Improving Equivariant Networks with Probabilistic Symmetry Breaking.* ICLR 2025. arXiv: 2503.21985.

**핵심 메커니즘**: 등변 모델이 Curie 원리에 의해 자기 대칭을 파괴할 수 없는 한계를 해결. 입력의 대칭군에서 군 원소를 확률적으로 샘플링하여, 대칭 파괴 위치 인코딩으로 사용.
- Curie 원리: 등변 함수 f에 대해, 입력 x가 대칭 G_x를 갖으면 출력 f(x)도 G_x를 가져야 함 → 비대칭 출력 표현 불가
- SymPE: 역변환 커널에서 g̃를 샘플 → 학습 가능 벡터 v에 g̃를 적용 → g̃·v를 위치 인코딩으로 concat
- 결과: 등변 모델의 일반화 이점 유지 + 비대칭 출력 표현 가능

**Ising 격자 실험 (가장 관련성 높은 결과)**:
- p4m 대칭 (D4 포함) 격자에서 강자성/반강자성/줄무늬 위상의 자발적 대칭 파괴
- SymPE 없이: 모든 위상이 무질서 상태로 붕괴 (대칭 파괴 불가)
- SymPE 적용: 3가지 위상 모두 정확 복원

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 본 프로젝트 관련성 | **제한적** | 등변 모델 전용. 본 프로젝트는 등변 모델이 아님 |
| Xe 진동 대칭 파괴 | **참고 가치** | 대칭 입력에서 비대칭 출력 (Xe tilt) 을 예측해야 하는 경우와 유사 |
| 구현 복잡도 | 높음 | 정규화 네트워크 + 추론 시 샘플링 필요 |

**결론**: 본 프로젝트는 등변 모델을 사용하지 않으므로 직접 적용 대상은 아니다. 다만, 향후 대칭 입력 상태에서 Xe 진동 onset을 예측해야 하는 상황에서 "등변 모델 + SymPE" 조합이 참고 가치가 있다. 현재 아키텍처 (비등변 ViT + Conditional LAPE) 에서는 비대칭 출력 표현에 구조적 제약이 없으므로 SymPE가 불필요하다.

### 6.4. ASEN — 부분군 등변성 동적 달성

**논문**: Goel, A., Lim, D., Lawrence, H., Jegelka, S., & Huang, N. (2026). *Any-Subgroup Equivariant Networks via Symmetry Breaking.* ICLR 2026. arXiv: 2603.19486.

> 참고: 원 출처는 NeurIPS 2025로 언급되었으나, 실제 학회는 ICLR 2026. 또한 FiLM 방식이 아니라 **보조 입력 concat** 방식이다.

**핵심 메커니즘**: 완전 치환 등변(S_n-equivariant) 기저 모델에 궤도 구조를 인코딩한 보조 입력을 concat하여, 대상 부분군 G에 대한 등변성을 달성.
- 보조 입력: 노드별 궤도 인덱스 A^(1) + 쌍별 궤도 인덱스 A^(2)
- 같은 궤도의 노드/쌍에 동일 인덱스 부여 → 학습 가능 임베딩으로 변환
- 결과: 동일 backbone으로 다른 대칭군 (S_n의 다양한 부분군) 에 대응 가능

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 본 프로젝트 관련성 | **낮음** | S_n 치환군 대상. D4 (회전+반사) 를 직접 다루지 않음 |
| 아키텍처 호환성 | 낮음 | GNN 기반. 본 프로젝트 ViT 기반과 상이 |
| "동적 부분군 선택" | 과장 | 학습 중 보조 입력 고정. 태스크 간 전환이지 입력별 전환이 아님 |

**결론**: 본 프로젝트와 아키텍처 (GNN vs ViT), 대칭군 (S_n vs D4), 목적 (등변성 달성 vs 대칭 유형 식별) 이 모두 다르므로 직접 적용 대상이 아니다. 다만 "보조 입력으로 대칭 정보를 주입한다"는 일반 원리는 Conditional LAPE의 설계 철학과 공유한다.

### 6.5. APE Locality-Symmetry 이중 성질

**논문**: Chen, L., Varoquaux, G., & Suchanek, F. M. (2023). *The Locality and Symmetry of Positional Encodings.* Findings of EMNLP 2023, pp.14313-14331. ACL Anthology: 2023.findings-emnlp.955.

**핵심 내용**: 위치 인코딩의 두 가지 바람직한 성질을 정량적 지표로 정의하고, 수동/학습 PE 모두가 이 성질로 수렴함을 실증.
- **Locality**: attention 가중치가 가까운 위치에 집중하는 정도
- **Symmetry**: 위치 i 기준 ±k 거리의 위치가 동일한 가중치를 받는 정도
- Sinusoidal, RoPE, ALiBi, BERT 학습 PE 모두 강한 locality + symmetry를 보임
- Attenuated Encoding 제안: `exp(-w·d²)` 형태의 거리 감쇠로 PE 초기화 → BERT보다 유의미한 성능 향상

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 본 프로젝트 관련성 | **간접적** | 1D NLP 전용. 2D/3D 격자에 대한 실험 없음 |
| 이론적 시사점 | 있음 | "등거리 위치 쌍이 동일 PE 내적을 가져야 한다"는 원리가 LAPE 설계 정당화 |
| Conditional LAPE와의 관계 | 보완적 | LAPE가 학습 시 locality + symmetry를 자연 획득한다는 간접 근거 |

**결론**: 직접 적용 기법은 아니지만, 본 프로젝트의 LAPE가 학습을 통해 locality (가까운 셀에 집중) + symmetry (등거리 셀에 유사 PE) 를 자연적으로 획득한다는 이론적 배경을 제공한다. Conditional LAPE에서 mirror/rotation 테이블이 각각 해당 대칭 하에서의 locality-symmetry 구조를 학습할 것으로 기대된다.

### 6.6. 학습 가능한 Fourier Feature PE

**논문**: Li, Y., Si, S., Li, G., Hsieh, C.-J., & Bengio, S. (2021). *Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding.* NeurIPS 2021. arXiv: 2106.02795.

**핵심 메커니즘**: M차원 좌표 x를 학습 가능한 주파수 행렬 W_r로 Fourier 변환한 뒤 MLP를 통과시켜 PE 생성.
- 수식: `r_x = (1/√D) · [cos(x·W_r^T) ; sin(x·W_r^T)]`
- 핵심 성질: `r_x · r_y`가 상대 거리 (x-y)에만 의존 (이동 불변)
- 초기화: W_r ~ N(0, σ^-2) → Gaussian 커널 `exp(-‖x-y‖²/(2σ²))` 근사 (L2 거리 편향)
- 학습 후: 데이터에 최적화된 거리 함수를 자동 학습

> 참고: D4 불변 좌표 특징 (i²+j², |ij| 등) 은 논문 원본에 없다. 그러한 좌표 전처리는 논문 위에 구축 가능한 확장 아이디어이다.

**적용성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| PE 생성 방식 | 다름 | 본 프로젝트 LAPE는 위치별 학습 변수. Fourier PE는 좌표→함수 매핑 |
| 일반화 능력 | 높음 | 미관측 좌표에 대해 보간 가능 (LAPE는 불가) |
| 대칭 정보 반영 | **없음** | 원 논문에 대칭군 개념 없음. L2 등방성만 제공 |
| 본 프로젝트 적합성 | **낮음** | 격자가 고정 (20×6×6) 이므로 보간 이점 없음. 대칭 정보 반영 불가 |

**결론**: Learnable Fourier PE는 가변 격자나 연속 좌표에서 장점이 있으나, 본 프로젝트는 격자 크기가 고정 (20×6×6) 이고 대칭 유형 구분이 핵심 요구사항이므로, 위치별 학습 변수 방식 (LAPE) 이 더 적합하다. D4 불변 좌표를 Fourier PE 입력으로 사용하는 확장은 흥미로우나, Conditional LAPE의 단순성 대비 추가 이점이 불명확하다.

### 6.7. 종합 비교

| 기법 | 핵심 원리 | 등변 모델 필요 | 대칭 유형 식별 | 인코더 적합 | 디코더 적합 | 본 프로젝트 채택 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Conditional LAPE** (채택) | 대칭 유형별 LAPE 테이블 선택 | X | **O** | **O** | — | **O** |
| 궤도 기반 PE | 동치 셀에 동일 PE | O (등변 ViT) | X | X | X | X |
| FiLM 조건부 | γ(sym)·LN(x) + β(sym) | X | O | **X** (원칙 위배) | **O** (AdaLN-Zero 합산) | △ (디코더만) |
| SymPE | 확률적 대칭 파괴 | O (등변 모델) | — | X | X | X |
| ASEN | 보조 궤도 입력 concat | O (S_n 등변) | — | X | X | X |
| APE Locality-Symmetry | PE의 locality/symmetry 이론 | X | X (이론만) | — | — | 참고 |
| Learnable Fourier PE | 좌표→Fourier→MLP | X | X | X | X | X |

**결론**:

1. **인코더**: Conditional LAPE가 유일하게 적합. 설계 원칙 무위반 + 물리적 해석 자연스러움 + 구현 단순
2. **디코더**: FiLM 방식 (AdaLN-Zero에 sym_type 합산) 이 유력한 후보. §3.4의 향후 과제와 일치
3. 등변 모델 기반 기법 (궤도 PE, SymPE, ASEN) 은 본 프로젝트 아키텍처와 전제가 다르므로 직접 적용 대상이 아님
4. APE Locality-Symmetry 이론은 LAPE 학습 행동의 이론적 배경으로 참고 가치 있음

---

## 7. 구현 순서와 의존성

| 순서 | 작업 | 선행 조건 | 대상 파일 |
|:---:|---|---|---|
| **1** | ConditionalLAPE3D 클래스 추가 | 없음 | `layers3d.py` |
| **2** | `__init__.py` export 추가 | 순서 1 | `spatial/__init__.py` |
| **3** | 단위 테스트 | 순서 1 | 테스트 파일 (신규) |
| 4 | halo_expand.py 구현 | 별도 계획 (B1) | `spatial/halo_expand.py` |
| 5 | encoder3d.py 조립 | 순서 1, 4 | `spatial/encoder3d.py` |
| 6 | 통합 테스트 | 순서 5 | 테스트 파일 |
| 7 | dataloader + HDF5 읽기 | 순서 5, HDF5 보정 | `src/data/` |

순서 1~3은 다른 컴포넌트에 대한 의존성이 없으므로 즉시 구현 가능하다.

---

## 8. 관련 문서

| 문서 | 내용 | 위치 |
|---|---|---|
| **05a_symmetry_distinguishability_proof.md** §8.5 | 방안 C 설계 근거, pseudocode, 학계 선례, 5개 방안 비교표 | 인코더 컴포넌트별 채용 이유/ |
| **03_attention_and_position.md** §3 | LAPE 설계 결정, residual stream 수학, 1회 add 정당화 | 인코더 컴포넌트별 채용 이유/ |
| **05_symmetry_mode.md** §6 | symmetry 정보 흐름 (halo expand → encode → mamba → decode → crop) | 인코더 컴포넌트별 채용 이유/ |
| **2026-04-14 대칭 정보 보존 체크리스트.md** | 전체 파이프라인 sym_type 경로 (HDF5 → dataloader → model) | implementation_plans/ |
| **04_normalization_omitted_options.md** §1.2, §1.4 | 인코더 설계 원칙 (Pre-LN, 조건 변조 없음) | 인코더 컴포넌트별 채용 이유/ |
