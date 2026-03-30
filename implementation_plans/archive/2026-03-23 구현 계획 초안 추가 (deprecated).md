> [deprecated 2026-03-30] D=24 패딩 → D=20 해결됨. ViT3D 문제 → Conv3D 전환으로 해당없음. 아키텍처 → `2026-03-29 모델 구현 계획.md` §3.

# Spatio-Temporal Reactor AI Surrogate — 수정·상세 설계 계획서 (TF2 기반)

> 수정 반영 사항: TensorFlow 2 / Keras 유지, `SuperpositioningNeighborCells`·`BoundaryPad` 제거, ViT3D 로컬 정보 손실 문제 분석, Mamba hidden state 공간 해상도 논의

***

## 0. 현행 모델의 실제 성능 및 핵심 문제 목록

노트북 훈련 결과에서 확인된 지표:[^1]

| 지표 | 값 |
|------|----|
| ROI Global MAPE | 0.694% |
| ROI Global MaxAPE | 20.196% |
| ROI Peak Global MAPE | 0.461% |
| ROI Peak Global MaxAPE | 3.913% |

평균 오차는 0.7% 수준이지만 **최대 오차가 20%** 에 달한다. 이는 특정 노드(외곽·저출력)에 집중적인 예측 실패가 있음을 의미한다. 원인을 아래에서 항목별로 분석한다.

***

## 1. ViT3D 구조적 문제 — 로컬 공간 정보 손실

### 1.1 `SuperpositioningNeighborCells` 제거 타당성

제거 결정은 타당하다. MaxViT 구조 자체가 이미 두 가지 경로로 공간 정보를 교환하기 때문이다:[^1]
- **BlockAttention3D**: 국소 블록(`4×2×2`) 내 노드들 간의 상대 위치 기반 attention → 인접 노드 상호작용
- **GridAttention3D**: 희소 그리드 패턴(`4×2×2`)으로 전역 노드들 간의 sparse attention → 원거리 상호작용
- **MBConv3D (DepthwiseConv3D)**: 커널 `3×3×3`으로 이미 6-connectivity 이웃 가중합 수행

`SuperpositioningNeighborCells`가 하던 "6방향 이웃 XS 채널 합성" 역할은 MBConv의 DWConv3D가 학습적으로 수행하므로 **하드코딩된 수작업 전처리보다 학습된 필터가 우월**하다.

### 1.2 `BoundaryPad` 제거와 대체 방법

`BoundaryPad`는 원래 두 가지 목적이 있었다:
1. **경계 조건 처리**: Y/X 축의 반사 패딩(reflect), Z축 제로 패딩
2. **SuperpositioningNeighborCells**의 입력 크기 확보

SuperpositioningNeighborCells 제거 후 BoundaryPad의 경계 조건 처리 역할도 함께 불필요해진다. MaxViT의 BlockAttention/GridAttention은 내부적으로 `tf.pad(mode='CONSTANT')` zero-padding을 이미 수행하므로 별도 레이어가 필요 없다. 단, **연료 영역과 반사체 경계**의 XS 불연속성을 고려한다면, Stem3D의 첫 Conv에 `padding='same'` 대신 **수동 reflect padding**을 선택적으로 사용할 수 있다. 이는 필수가 아닌 선택 사항이다.

### 1.3 🔴 핵심 문제: Global Pooling에 의한 공간 정보 파괴

현재 모델의 최종 출력 경로:[^1]
```
(B, 26, 7, 7, 128) — s3stage 출력
    → finalconv: Conv3D(filters=1, kernel=3×3×3, padding='valid')
    → (B, 24, 5, 5, 1)
    → squeeze → (B, 24, 5, 5)   # 출력 분포
```

`finalconv`가 `padding='valid'`이므로 공간 크기가 `(D-2, H-2, W-2)` = `(24, 5, 5)`로 맞춰진다. **이 방식은 Global Pooling은 없으나, 모든 공간 스케일의 특징이 마지막 단일 레이어에서만 통합**된다. 다음과 같은 심각한 문제가 발생한다.

**문제 1 — 다운샘플링 없는 단일 스케일 문제**

현재 3개의 MaxViTBlock3D 스테이지가 모두 `strides=(1,1,1)`을 사용한다. 즉 `(B, 26, 7, 7, C)` 크기가 끝까지 유지된다. 이는 다음을 의미한다:[^1]
- 모든 attention이 **고정된 블록/그리드 크기**에서만 작동
- 중간 스케일(e.g., `20×5×5` 전체를 한번에 보는) 특징 추출 불가
- 충분한 수용 영역(receptive field)을 확보하려면 블록·그리드 크기를 키워야 하지만, 이러면 `(20,5,5)` 노심에서 블록 수가 너무 줄어든다

**문제 2 — AxialSE의 공간 평균화 (당신이 지적한 핵심)**

`AxialSE`(Squeeze-Excitation)의 Squeeze 단계:[^1]
```python
squeezetensorw = tf.reduce_mean(x, axis=[1,2], keepdims=True)  # D,H 평균
squeezetensorh = tf.reduce_mean(x, axis=[1,3], keepdims=True)  # D,W 평균
squeezetensord = tf.reduce_mean(x, axis=[2,3], keepdims=True)  # H,W 평균
final = squeezetensorw + squeezetensorh + squeezetensord
```

이 **축 방향 mean pooling**이 바로 외곽 저출력 뉴클의 예측 오차를 키우는 핵심 원인이다. 노심 중앙 고출력 노드(`~1.2×avg`)와 외곽 저출력 노드(`~0.3×avg`)를 평균하면 SE가 생성하는 채널 가중치가 **중앙값에 편향**된다. 외곽 노드의 낮은 activation은 평균에 희석되어 SE가 이를 무시하는 방향으로 학습된다.

추가로, 정규화 미적용으로 인해 XS 값의 범위가 채널마다 크게 다르면(예: `f1` vs `c1`), SE의 채널 스케일링이 부적절하게 동작한다.

**문제 3 — 정규화 미적용 + 출력 스케일 불균형**

노트북에서 `inputxsvoxel4d`는 `.npy`에서 직접 로드하며 별도 정규화 없이 사용된다. HDF5 전처리 패키지에서는 `normstats`(mean, std)가 구현되어 있으나, 기존 모델은 이를 활용하지 않았다. 결과적으로:[^2][^1]
- XS 채널 10개의 값 범위가 서로 다름(`f1` ~ `1e-3`, `c1` ~ `1e-2`)
- 출력 Qabs도 정규화 없이 절대값으로 예측 → 외곽 저출력 노드는 Loss에서 작은 숫자이므로 gradient 기여가 낮음
- `maskedmspe_loss`(상대오차 손실)로 일부 보완했으나, SE의 공간 편향과 결합해 외곽 오차 집중 발생

**문제 4 — `relativebiastable` gradient 단절 (이전 분석 확인됨)**

실제 훈련 로그에서 다음이 모든 epoch에서 반복 출력된다:[^1]
```
WARNING: Gradients do not exist for variables
  custommaxvit3d1/s1stage/blockattention3d/relativeattention3d/relativebiastable:0
  custommaxvit3d1/s1stage/gridattention3d/relativeattention3d/relativebiastable:0
  ... (s2stage, s3stage 포함 6개)
```
모든 Relative Attention 레이어의 position bias가 학습되지 않고 있다. 즉 지금까지 훈련된 모델은 **상대 위치 편향 없는 Attention**으로 동작했다. 이를 수정하면 단독으로도 상당한 성능 향상이 가능하다.

***

## 2. `relativebiastable` Gradient 단절 — 근본 원인 및 TF2 수정

### 2.1 근본 원인 분석

`RelativeAttention3DLayer.build()`:[^1]
```python
def build(self, input_shape):
    B, D, H, W, C = ...
    self.reindexed_bias = self.get_reindexed_bias_3d(D, H, W)
    # shape: (numheads, N, N) — 상수처럼 저장됨
    super().build(input_shape)
```

`get_reindexed_bias_3d()` 내부에서 `self.add_weight()`로 `relativebiastable`을 생성하지만, 최종 반환값 `reindexed_bias`는 `tf.gather_nd(self.relativebiastable, lookup_indices)`의 결과물이다. 이 값이 `self.reindexed_bias`에 **build 시점에 한 번만 저장**되는데, TF2의 eager mode에서 이 시점의 tensor는 `self.relativebiastable`과의 계산 그래프 연결이 `call()` 호출 전에 끊어진다.

`call()` 에서는:
```python
scaled_attention_logits += self.reindexed_bias  # 이미 끊어진 상수 tensor 더함
```

결과적으로 backprop 시 `self.relativebiastable`로 가는 경로가 없다.

### 2.2 TF2 수정 방법

```python
def call(self, x, mask=None):
    shape = tf.shape(x)
    B, D, H, W, C = shape, shape[^1], shape[^2], shape[^3], shape[^4]
    N = D * H * W

    x_reshaped = tf.reshape(x, (B, N, C))
    q, k, v = self.wq(x_reshaped), self.wk(x_reshaped), self.wv(x_reshaped)
    q = self.split_heads(q, B)
    k = self.split_heads(k, B)
    v = self.split_heads(v, B)

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)

    # ✅ call() 안에서 매번 gather_nd 수행 → relativebiastable과 그래프 연결 유지
    reindexed_bias = tf.gather_nd(self.relativebiastable, self.lookup_indices)
    # self.lookup_indices는 build()에서 tf.constant로 저장 (학습 불필요)
    scaled += reindexed_bias  # 여기서 gradient 흐름 연결됨

    if mask is not None:
        scaled += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(attention_weights, v)
    ...
```

`lookup_indices`는 D, H, W에 종속되는 정수 인덱스 배열로 학습 파라미터가 아니므로, `build()`에서 `tf.constant`로 저장하면 된다. 반면 `relativebiastable` 자체는 `self.add_weight(trainable=True)`로 유지하고, **매 call마다 `tf.gather_nd`를 통해 편향을 재계산**함으로써 autograd 연결을 보장한다.

***

## 3. 개선된 Spatial Encoder 설계 (TF2/Keras 유지)

### 3.1 U-Net 스타일 Skip Connection 도입

현재 ViT3D에 없는 가장 큰 결함은 **고해상도 공간 세부 정보를 디코더로 전달하는 경로의 부재**다. 출력 `(B, 24, 5, 5)` 분포를 직접 예측하는 dense task에서 단순 encoder-only 구조는 공간 해상도 손실이 치명적이다.[^3][^4]

**권장 구조: ViT3D Encoder + U-Net 스타일 Decoder**

```
입력: (B, 20, 5, 5, C_in)
│
├─ Stem3D: Conv3D(64, 5×5×5, strides=1) → (B, 20, 5, 5, 64)    ← skip_0
│
├─ Stage 1: MaxViTBlock3D(hidden=64, strides=1)  → (B, 20, 5, 5, 64)  ← skip_1
│
├─ Stage 2: MaxViTBlock3D(hidden=128, strides=(2,1,1)) → (B, 10, 5, 5, 128) ← skip_2
│
└─ Stage 3: MaxViTBlock3D(hidden=256, strides=(2,1,1)) → (B, 5, 5, 5, 256)
                                [Bottleneck: latent z_t ← GlobalAvgPool + Dense]
│
├─ Decoder Block 2: UpSampling3D(2,1,1) + concat(skip_2) → Conv3D(128)
│
├─ Decoder Block 1: UpSampling3D(2,1,1) + concat(skip_1) → Conv3D(64)
│
└─ Output Head:
   ├─ head_3d: Conv3D(C_out, 1×1×1) → (B, 20, 5, 5, C_out)   # Qabs, Tcool, Xe...
   └─ head_scalar: GlobalAvgPool3D → Dense(2)                  # keff, AO
```

**스킵 연결의 효과**: 저레벨(Stem) 특징(XS 채널의 공간 분포)이 디코더로 직접 전달되어, **외곽 저출력 노드의 XS 패턴이 출력에 영향을 줄 수 있는 경로**를 보장한다.[^5][^3]

### 3.2 AxialSE 수정 — 공간 불균형 해소

AxialSE의 공간 편향 문제를 해결하는 두 가지 방안:

**방안 A: AxialSE의 Squeeze를 위치별 Max로 교체**
```python
# 기존: 공간 평균 → 중앙값 편향
squeezed = tf.reduce_mean(x, axis=[1,2], keepdims=True)

# 개선: 공간 최대 → 극값 포함
squeezed_max = tf.reduce_max(x, axis=[1,2], keepdims=True)
squeezed_mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
squeezed = squeezed_max + squeezed_mean  # 또는 concat → 더 많은 정보
```

**방안 B: AxialSE 완전 제거 → LayerNorm으로 대체**

실제로 원본 MaxViT 논문에서도 SE는 경량 MBConv를 위한 보조 모듈이며, 3D 노심처럼 **공간적으로 불균일한 값 분포**에서는 오히려 해롭다. MBConv의 채널-공간 표현력은 DWConv3D만으로도 충분하므로, SE 제거 후 LayerNorm만 유지하는 것이 안전한 선택이다.

### 3.3 정규화 전략

**입력 정규화 (z-score, per-channel)**:
```python
# normstats에서 로드: mean shape (10,), std shape (10,)
x_norm = (x - mean) / (std + 1e-8)  # (B, 20, 5, 5, 10)
```

전처리 패키지의 `normstats`(mean, std, min, max)를 HDF5에서 로드하여 적용한다. 채널별 z-score 정규화는 `f1`, `c1` 등 XS 값 범위 불균형을 해소한다.[^2]

**출력 정규화 (Qabs)**:
- Qabs = `p_load × P_design / N_fuel` 단위의 `[MWnode]`
- 정규화: `q_norm = (Qabs - Qabs_mean) / Qabs_std` 로 학습, 추론 시 역변환
- 대안: 0~1로 min-max normalization(`Qabs / max_Qabs`)

**손실 함수 유지**: 현행 `maskedmspe_loss`(masked MSPE)는 상대 오차 기반이므로 외곽 저출력 노드도 동등한 가중치를 받는다는 장점이 있다. 단, 정규화 적용 후에는 **절대값 MSE + MSPE 혼합 손실**도 검토할 가치가 있다. 외곽 노드 정밀도가 중요하다면 inverse-weight(출력 역수를 가중치로 사용) 전략도 가능하다.

***

## 4. Mamba h(t)의 공간 해상도 — 노드별 맥락 저장 가능 여부

### 4.1 핵심 질문: h(t)에 3D 공간 전체를 저장해야 하는가?

결론부터 말하면: **"굳이 그럴 필요 없으며, 오히려 그렇게 하면 비효율적이다."**

Mamba의 hidden state \(h_t\)는 시간적 "메모리" 역할로 설계된다. 공간 정보의 노드별 맥락은 **ViT3D Spatial Encoder가 `z_t` 속에 압축**하여 Mamba에 넘기는 구조가 올바른 역할 분리다.[^6][^7]

### 4.2 두 가지 아키텍처 선택지

**선택지 A: Scalar latent per timestep (권장 출발점)**
```
ViT3D → z_t: (B, D_latent)   # 노심 전체를 D_latent 차원으로 압축
Mamba: 시퀀스 (z_1, ..., z_T) → h_t: (B, D_model, N_state)
출력: y_t → 3D decoder로 복원
```
- `h_t`는 **시간 이력의 압축**을 담당, 공간 패턴은 decoder가 복원
- Mamba의 hidden state 크기: `D_model × N_state`, 예: `128 × 64 = 8,192`개 파라미터
- 훈련/추론 효율성 최고

**선택지 B: Spatial latent per timestep (정밀도 우선)**
```
ViT3D → z_t: (B, 20, 5, 5, C_latent)   # 공간 해상도 유지
Mamba: 노드별 독립 시퀀스 처리 → h_t: (B, 20, 5, 5, N_state)
```
- `h_t`가 `(B, 20, 5, 5, N_state)` 형태로 **노드별 시간 이력을 개별 저장**
- Xe-135 분포의 공간적 불균일성(외곽 vs. 중앙)을 위치별로 다르게 추적 가능
- 메모리: `20×5×5×N_state = 500×64 = 32,000`개 → 선택지 A의 4배
- 연산 복잡도도 비례하여 증가

**물리적 관점에서의 권장**: 제논 독화는 국소적이지 않다(노심 전체에 걸친 현상). 따라서 ViT3D가 이미 글로벌 context를 포함한 `z_t`를 만들면 Mamba가 그것을 시간적으로 추적하는 **선택지 A**로 충분히 처리 가능하다. 

단, 만약 부하추종 운전 중 **공간적으로 비대칭인 제논 분포**(AO 진동)를 정밀하게 추적해야 한다면, 선택지 B의 **일부 변형**으로서 `h_t`를 `(B, 20, C)` (Z축 슬라이스별)로만 유지하는 중간 방안도 있다. Z축(20개 평면)은 제논 oscillation의 핵심 공간 방향이므로 H×W(`5×5`)는 압축하고 Z만 유지하는 전략이다.

### 4.3 TF2에서 Mamba 구현 현황

TF2/Keras 기반의 Mamba 공식 구현은 PyTorch `mamba-ssm` 라이브러리에 비해 제한적이다. 현실적인 선택지:[^8][^7]

| 방법 | 장점 | 단점 |
|------|------|------|
| **tf.keras.layers.GRU/LSTM** (Baseline) | TF2 완벽 지원, HDF5 데이터 파이프라인 연동 용이 | Mamba의 선택적 메커니즘 없음 |
| **Custom SSM Layer in TF2** | Mamba 핵심 수식 직접 구현, gradient 제어 가능 | 구현 난이도 높음, CUDA 커스텀 커널 미지원 |
| **TF/JAX Mamba 포팅** (e.g., `keras-mamba`) | 원리 충실 | 안정성 검증 필요 |

**단계적 접근 권장**:
1. **Phase M1**: `tf.keras.layers.GRU(stateful=True)` 로 Temporal Model 프로토타입 완성 → 전체 파이프라인 먼저 검증
2. **Phase M2**: Custom SSM Layer (S4/Mamba 수식 직접 구현) 로 교체 → 성능 비교
3. **Phase M3**: 필요 시 PyTorch `mamba-ssm` 모델과 TF2 모델의 앙상블 또는 ONNX 변환 검토

***

## 5. 잠재적 모델 구조 문제점 — 전체 탐지 목록

### 5.1 입력 / 정규화 레벨

| # | 문제 | 영향 | 해결책 |
|---|------|------|--------|
| I1 | XS 입력 정규화 없음 | 채널 간 값 범위 불균일 → optimizer 비효율 | per-channel z-score, normstats 적용 |
| I2 | rodmap3d 이진값(0/1) 그대로 concat | 다른 연속값 채널과 스케일 불일치 | 그대로 사용 또는 별도 Embedding 처리 |
| I3 | Xe, Sm 초기값(0 또는 equilibrium) 구분 없음 | 시퀀스 첫 스텝에서 Mamba에 잘못된 신호 | warm-up 스텝 마스킹 또는 init flag 채널 추가 |

### 5.2 Spatial Encoder 레벨

| # | 문제 | 영향 | 해결책 |
|---|------|------|--------|
| S1 | `relativebiastable` gradient 단절 | Position bias 미학습, 상대 위치 정보 전무 | call()에서 `tf.gather_nd` 재계산 |
| S2 | 모든 스테이지 `strides=1` (하드코딩) | 수용 영역 부족, 다중 스케일 특징 없음 | Stage 2, 3에 `strides=(2,1,1)` 적용 |
| S3 | AxialSE 공간 평균 → 외곽 저출력 편향 | MaxRelErr 20% 의 주요 원인 | SE 제거 또는 max+mean 혼합 squeeze |
| S4 | 스킵 연결 없는 encoder-only → dense 출력 불리 | 공간 해상도 복원 불가 | U-Net 스타일 Decoder 추가 |
| S5 | `finalconv: padding='valid'` 단 1레이어 | 모든 공간 정보 단일 레이어에서만 통합 | Multi-scale decoder head로 교체 |
| S6 | 출력 단일 채널(Qabs만) | 다변수 예측 불가 | multi-output head 추가 |
| S7 | MBConv의 `expandnorm`, `depthwisenorm`가 변경 이력에서 BatchNorm → LayerNorm으로 수정되었으나 일부 위치에서 혼재 | 정규화 방식 불일치 | LayerNorm으로 통일 (ViT 계열 표준) |
| S8 | `D=24` 하드코딩 (실 데이터 D=20) | 4개 평면 오정렬 가능성 | D=20으로 수정, 가변 지원 |

### 5.3 Temporal Model 레벨

| # | 문제 | 영향 | 해결책 |
|---|------|------|--------|
| T1 | Temporal Model 미존재 | Xe/Sm 이력 반영 불가 | Mamba 또는 GRU Temporal Model 추가 |
| T2 | 31-way Branch를 time-batch로 처리 시 메모리 폭발 | `(B, 576, 31, 20, 5, 5)` → 수십 GB | h_t 공유 + branch sequential 처리 |
| T3 | Auto-regressive 오차 누적 | 장시간 예측 시 오차 복리 | Teacher forcing(훈련) + scheduled sampling(전환) |
| T4 | 시퀀스 경계(스텝 0 초기화) 처리 미정의 | XeSm equilibrium 초기화 오류 | `ixeeq`, `ismeq` 초기 상태 구분 처리 |

### 5.4 학습 / 손실 레벨

| # | 문제 | 영향 | 해결책 |
|---|------|------|--------|
| L1 | 외곽 void 노드 제외 마스크 단순 고정 | 정의된 외 ROI 경계 오차 무시 | 마스크 정확성 재검증 |
| L2 | Peak 노드 위치 예측 손실 없음 | GlobalMaxAPE 20% 허용 | peak location loss 또는 focal weight 추가 |
| L3 | keff, AO, Tcool, Tfuel, rhocool 손실 미정의 | multi-output 학습 불균형 | 물리 단위별 정규화 후 합산, 가중치 튜닝 |
| L4 | `TruncatedNormal` initializer에 seed 미설정 | 재현성 없음, 다중 GPU 환경에서 UserWarning | `seed=42` 등 고정 |

### 5.5 데이터 파이프라인 레벨

| # | 문제 | 영향 | 해결책 |
|---|------|------|--------|
| D1 | 현재 `.npy` 로드 → 대용량 메모리 점유 | LP 50개×576스텝×31-way → ~3.6GB raw | HDF5 + `tf.data` 스트리밍으로 전환 |
| D2 | `tf.data.Dataset.from_tensor_slices` 방식 | TFRecord 전환 시 재설계 필요 | HDF5 → `tf.data.Dataset` 래핑 또는 TFRecord 직렬화 |
| D3 | 시계열 배치: 전 길이 576스텝 일괄 처리 | VRAM 폭발 | sliding window batch (window=32, stride=8) |

***

## 6. 수정된 구현 로드맵

### Phase S — Spatial Encoder 재작성 (TF2 유지)

| 태스크 | 내용 | 완료 기준 |
|--------|------|-----------|
| S1 | `RelativeAttention3D.call()`에 `tf.gather_nd` 이동 → gradient 연결 | `relativebiastable` 경고 메시지 소멸, 학습 중 gradient norm ≠ 0 |
| S2 | `SuperpositioningNeighborCells`, `BoundaryPad` 제거, 코드 정리 | 입력 `(B, 20, 5, 5, C_in)` 직접 처리 |
| S3 | `AxialSE` 제거 또는 max+mean squeeze 수정 | 외곽 노드 예측 오차 MaxRelErr 기준 |
| S4 | Strides 설정: Stage2 `(2,1,1)`, Stage3 `(2,1,1)` | `(B,5,5,5,256)` bottleneck 확인 |
| S5 | U-Net Decoder 추가: UpSampling3D + skip concat | `(B, 20, 5, 5, C_out)` 출력 |
| S6 | 멀티 출력 헤드: `head_3d`(C_out 채널), `head_scalar`(keff, AO) | 전체 shape 검증 |
| S7 | per-channel z-score 입력 정규화 적용 | normstats 로드 → 적용 확인 |
| S8 | 단위 테스트: BOC static XS만으로 Qabs 예측 | MAPE < 2%, MaxAPE < 10% (기준값) |

### Phase M — Temporal Model (GRU Baseline → Custom SSM)

| 태스크 | 내용 | 완료 기준 |
|--------|------|-----------|
| M1 | `tf.keras.layers.GRU(stateful=True)` Temporal Model 구현 | 576 스텝 forward pass 메모리 확인 |
| M2 | ViT3D `z_t` 추출 → GRU 시퀀스 학습 | val loss 감소 확인 |
| M3 | `sliding_window_dataset`: window=32, stride=8로 배치 구성 | 메모리 < 32GB 확인 |
| M4 | Custom SSM Layer (S4 이산화, HiPPO-A 초기화) TF2 구현 | S4 단위 테스트: sin wave 기억 |
| M5 | GRU vs SSM 성능 비교: Xe-135 단순 시계열 | 수렴 속도, 장기 예측 MAPE |
| M6 | Teacher Forcing / Scheduled Sampling 구현 | 576 스텝 auto-regressive 오차 프로파일 |

### Phase T — 통합 및 End-to-End

| 태스크 | 내용 | 완료 기준 |
|--------|------|-----------|
| T1 | `SpatioTemporalSurrogate` 통합 모델: ViT3D + Temporal + Decoder | `(B, T, 20, 5, 5, C_in)` 입력 → `(B, T, 20, 5, 5, C_out)` 출력 |
| T2 | Critical path (index 0) + 30 Branch 병렬화 | 31-way forward pass 시간 측정 |
| T3 | HDF5 DataLoader: multi-LP, sliding window, normstats | `tf.data.Dataset` e2e 검증 |
| T4 | 전체 학습: Phase S (spatial only) → Phase T (joint) | val MAPE < 1%, MaxAPE < 5% |
| T5 | TFRecord 변환 (선택적): HDF5 → TFRecord | 데이터 로드 속도 비교 |

***

## 7. 폴더 구조 (TF2 기반, 수정)

```
spatio-temporal-reactor-ai-surrogate/
│
├── configs/
│   ├── model_spatial_v2.yaml    # ViT3D v2 설정 (strides, skip, no AxialSE)
│   ├── model_temporal.yaml      # GRU/SSM 설정
│   └── train.yaml
│
├── src/
│   ├── models/
│   │   ├── spatial/
│   │   │   ├── relative_attn3d.py   # ⚡ gradient 수정 버전
│   │   │   ├── mbconv3d.py          # AxialSE 제거, LayerNorm 통일
│   │   │   ├── block_attn3d.py
│   │   │   ├── grid_attn3d.py
│   │   │   ├── maxvit_block3d.py
│   │   │   ├── decoder3d.py         # 🆕 U-Net decoder
│   │   │   └── vit3d_encoder_v2.py  # 🆕 통합 (encoder + decoder)
│   │   │
│   │   ├── temporal/
│   │   │   ├── gru_temporal.py      # Baseline: tf.keras.layers.GRU
│   │   │   ├── ssm_layer.py         # 🆕 Custom S4/Mamba in TF2
│   │   │   └── temporal_model.py
│   │   │
│   │   └── surrogate.py
│   │
│   ├── data/
│   │   ├── hdf5_dataset.py          # 기존 HDF5 로더 연동
│   │   ├── sliding_window.py        # 🆕 시계열 슬라이딩 윈도우
│   │   ├── tfrecord_writer.py       # 🆕 (선택) TFRecord 변환
│   │   └── normalizer.py            # normstats 로드/적용
│   │
│   └── training/
│       ├── losses.py                # masked_mspe + multi-output loss
│       └── trainer.py
│
└── notebooks/
    ├── 01_S1_gradient_fix_test.ipynb     # relativebiastable gradient 검증
    ├── 02_S3_axialse_ablation.ipynb      # SE 제거 전후 외곽 오차 비교
    ├── 03_S4_unet_decoder_test.ipynb     # 디코더 공간 복원 검증
    └── 04_M1_gru_temporal_baseline.ipynb
```

***

## 8. 우선순위 빠른 참조

**즉시 착수 (이번 주)**:
1. `relative_attn3d.py` — `call()`에서 `tf.gather_nd` 재계산 (1~2시간 작업, 단독으로 큰 성능 향상 가능)
2. `SuperpositioningNeighborCells`, `BoundaryPad` 제거 + `D=20` 수정
3. 입력 z-score 정규화 적용 (`normstats` 연동)

**1~2주 내 (핵심 아키텍처 변경)**:
4. `AxialSE` 제거 또는 max+mean 수정
5. `strides=(2,1,1)` 도입 + U-Net Decoder
6. multi-output 헤드 (keff, AO 포함)

**3~4주 (Temporal 통합)**:
7. GRU Temporal Baseline 구현
8. 통합 end-to-end 학습

---

## References

1. [custom_voxel_attention_alpha_remove_preproc_layers_250808.ipynb](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/160091878/1cab3853-e4a9-4243-b8b0-9e05ce0e27be/custom_voxel_attention_alpha_remove_preproc_layers_250808.ipynb?AWSAccessKeyId=ASIA2F3EMEYE2COWORZS&Signature=Up03oJuUmaVcjwVTMaHFD0Kpfpo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDqy%2F7y44lHHRWhY0H5gidBj3NcW3VMbjhZaRrft6KacAIhANAqOlc1DaDVBUAxzM7BQqhCZoTLf31F7cfpZ0ZPXGyVKvMECHcQARoMNjk5NzUzMzA5NzA1Igxd371B%2FA0jF211Gjgq0AQyRDAr8uTdeU1b9juC%2F4iO%2For7I%2F%2FJ53EQ6YyWfsHv9YEz3PMxAFPwWfKdApfgl9Tz9ywBKgMb5LX4wYQg5t97DB8xV5xUIYMK6XthiE9OqfTZqGHFDWIgonUB%2BnAVj%2FndA0%2FIyaUzBg4mzVrZYJb7zQ6Ac2lwsdKZ4LhsMSLIiCu8yaEEwBt2qLmz%2FShL%2F93hMNYqXuxsz6GaKXVgkV0%2BHDUsjaSxSvtF0PhJgBBQlCtQkR%2F4Y5FltMZ0n6awgZd%2Fj0nodFh49rtdLW%2FsBKVR8dv4ADSnqW9xAjnfB1DPLZUw3wtOclKVD7qOY3Wt4g2ExT6n0iXyBuh7ug1YsuYQPc6KWQ%2F7%2BdCJwJwz6ougHWbiQsKngiUKHJtYQglot4cZHN60buqejaz20cOg94ChlPYuFF1RU3JNcBc53223mp4YVXUEJV%2F6vhq%2BtlHhotoMPPAg%2F98K8W48UtYsX0c2Likhc%2FJzNvsKlcxMxvvEpG2RA%2F9IpYrrxbCTHJc7fvDd41ytVX0F6EPGwb7KOkkFoyCYR7pWsH4VaUx1EcwTJAPlA6WWXrLoKFoYB2JObqP5p6QLixmw53eNE%2FQ32yl896tA1I9Y0TKJOD60l3oInNMBlKPThP9o8ta8XomCkXtzGSnpeHd22%2BvNEjBB4WjejHWJE0peqm9NQ4noFgNK42x1fyErKTZPcryLEjlXSKylmhCAnr3yDUe9VBbg8rg6sMmEt9W3599jZ7wE38tY4blxMNT3MaMRAdn5SZOm%2BZ21HaaUkNk1NDWZeKWh4wnPMI6lg84GOpcByAjlYIxLoqhoJtQ0vjEnWQK5eZBqrsZE0m0GDn0qDGUhwr7gkJGx%2Bt0%2BEKMITWfm95sWLU7ayCP8410xhFchDtEnB%2FtoK2ApnOHEsM1PLJykgIoLLUeFqsTEOaE6UWw7RWXBOdR4eMpTbIeQfJcPX9fvqHBPPTvt%2BIJoRq0FugkhevaIkW%2FKov4bjThAg691NyvJ6Xbq2Q%3D%3D&Expires=1774248033) - cells celltype code, executioncount 32, id 6f9afdc1, metadata , outputs , source import os, import t...

2. [2026-03-17_data_preproc_package_implementation_plan-3.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/160091878/0805f98a-733c-4dee-9bd5-1b794db2fe1e/2026-03-17_data_preproc_package_implementation_plan-3.md?AWSAccessKeyId=ASIA2F3EMEYE2COWORZS&Signature=bSSD%2BU1g4qMZ7Q9sMf8pI%2BFBmJg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDqy%2F7y44lHHRWhY0H5gidBj3NcW3VMbjhZaRrft6KacAIhANAqOlc1DaDVBUAxzM7BQqhCZoTLf31F7cfpZ0ZPXGyVKvMECHcQARoMNjk5NzUzMzA5NzA1Igxd371B%2FA0jF211Gjgq0AQyRDAr8uTdeU1b9juC%2F4iO%2For7I%2F%2FJ53EQ6YyWfsHv9YEz3PMxAFPwWfKdApfgl9Tz9ywBKgMb5LX4wYQg5t97DB8xV5xUIYMK6XthiE9OqfTZqGHFDWIgonUB%2BnAVj%2FndA0%2FIyaUzBg4mzVrZYJb7zQ6Ac2lwsdKZ4LhsMSLIiCu8yaEEwBt2qLmz%2FShL%2F93hMNYqXuxsz6GaKXVgkV0%2BHDUsjaSxSvtF0PhJgBBQlCtQkR%2F4Y5FltMZ0n6awgZd%2Fj0nodFh49rtdLW%2FsBKVR8dv4ADSnqW9xAjnfB1DPLZUw3wtOclKVD7qOY3Wt4g2ExT6n0iXyBuh7ug1YsuYQPc6KWQ%2F7%2BdCJwJwz6ougHWbiQsKngiUKHJtYQglot4cZHN60buqejaz20cOg94ChlPYuFF1RU3JNcBc53223mp4YVXUEJV%2F6vhq%2BtlHhotoMPPAg%2F98K8W48UtYsX0c2Likhc%2FJzNvsKlcxMxvvEpG2RA%2F9IpYrrxbCTHJc7fvDd41ytVX0F6EPGwb7KOkkFoyCYR7pWsH4VaUx1EcwTJAPlA6WWXrLoKFoYB2JObqP5p6QLixmw53eNE%2FQ32yl896tA1I9Y0TKJOD60l3oInNMBlKPThP9o8ta8XomCkXtzGSnpeHd22%2BvNEjBB4WjejHWJE0peqm9NQ4noFgNK42x1fyErKTZPcryLEjlXSKylmhCAnr3yDUe9VBbg8rg6sMmEt9W3599jZ7wE38tY4blxMNT3MaMRAdn5SZOm%2BZ21HaaUkNk1NDWZeKWh4wnPMI6lg84GOpcByAjlYIxLoqhoJtQ0vjEnWQK5eZBqrsZE0m0GDn0qDGUhwr7gkJGx%2Bt0%2BEKMITWfm95sWLU7ayCP8410xhFchDtEnB%2FtoK2ApnOHEsM1PLJykgIoLLUeFqsTEOaE6UWw7RWXBOdR4eMpTbIeQfJcPX9fvqHBPPTvt%2BIJoRq0FugkhevaIkW%2FKov4bjThAg691NyvJ6Xbq2Q%3D%3D&Expires=1774248033) - 2026-03-17 2026-03-23 D1-D2, E1-E2 LP lfpreprocess - ViT3D Mamba HDF5 --- TITLE

3. [How U-net works? | ArcGIS API for Python](https://developers.arcgis.com/python/latest/guide/how-unet-works/)

4. [3D U-Net: Volumetric Segmentation - Emergent Mind](https://www.emergentmind.com/topics/3d-u-net) - 3D U-Net extends the U-Net paradigm using 3D convolutions and skip connections for robust and precis...

5. [U-Net Architecture Explained](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/) - Your All-in-One Learning Portal: GeeksforGeeks is a comprehensive educational platform that empowers...

6. [A Visual Guide to Mamba and State Space Models](https://maartengrootendorst.com/blog/mamba/) - This architecture is often referred to as a selective SSM or S6 model since it is essentially an S4 ...

7. [Mamba: SSM, Theory, and Implementation in Keras and TensorFlow](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546/) - In SSMs, the hidden state is carried over to when the next input is received. This is similar to how...

8. [Mamba: SSM, Theory, and Implementation in Keras and TensorFlow | daily.dev](https://app.daily.dev/posts/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-hwrokitjz) - The post discusses the Mamba model, which utilizes selective state space models (SSM) for sequence m...

