# 기존 코드(Custom Voxel Attention) 정보

> **출처**: `2026-03-30 모델 구현 계획(공간 인코더).md` §1, §2에서 이관 (2026-04-02)
> **참고 코드**: `참고 파일/custom_voxel_attention_alpha_remove_preproc_layers_250808.ipynb`

---

## 1. 기존 코드 아키텍처

### 1.1 전체 데이터 흐름 (shape 추적)

```
MAIN INPUT (B, 24, 5, 5, 10)           ← xs_fuel 10ch
  │
  ├─ SuperpositioningNeighborCells
  │    6방향 이웃(center + up/down/back/front/left/right) 스택
  │    → (B, 24, 5, 5, 70)             ← 10ch × 7 = 70ch
  │
  ├─ BoundaryPad
  │    Z: zero [1,1], Y: reflect+zero, X: reflect+zero
  │    → (B, 26, 7, 7, 70)             ← +2 per spatial dim
  │
AUX INPUT (B, 24, 5, 5)                ← rod_position
  │
  ├─ ExpandDims → (B, 24, 5, 5, 1)
  ├─ BoundaryPad → (B, 26, 7, 7, 1)
  │
MERGE: Concatenate
  → (B, 26, 7, 7, 71)                  ← 70 + 1
  │
Stem3D
  Conv3D(128, kernel=(15,5,5), same) + GELU
  Conv3D(128, kernel=(15,3,3), same) + GELU
  → (B, 26, 7, 7, 128)
  │
MaxViTBlock3D × 3 stages
  각 stage: hidden_size=64, head_size=32, num_heads=2
            block_size=(4,2,2), grid_size=(4,2,2), strides=(1,1,1)
  │
  ├─ [MBConv3D]
  │    Conv1×1×1 expand (128→256) + LN + GELU
  │    DepthwiseConv3D(3,3,3) groups + LN + GELU
  │    AxialSE (axial squeeze → bottleneck → sigmoid scale)
  │    Conv1×1×1 project (256→64) + LN
  │    → (B, 26, 7, 7, 64)
  │
  ├─ [BlockAttention3D] + residual
  │    LN → pad to divisible → block partition
  │    (B,26,7,7,64) → pad → (B,28,8,8,64)
  │    → reshape (B×7×4×4, 4, 2, 2, 64)  ← 112B개 블록
  │    → RelativeAttention3D (N=4×2×2=16 토큰)
  │    → reconstruct → crop → + shortcut
  │    → (B, 26, 7, 7, 64)
  │
  ├─ [FFN_3D] + residual
  │    Dense expand (64→256) + GELU + Dense shrink (256→64)
  │    → (B, 26, 7, 7, 64)
  │
  ├─ [GridAttention3D] + residual
  │    LN → pad → grid partition (strided sampling)
  │    → reshape (B×13×4×4, 2, 2, 2, 64)  ← 208B개 그리드
  │    → RelativeAttention3D (N=2×2×2=8 토큰)
  │    → reconstruct → crop → + shortcut
  │    → (B, 26, 7, 7, 64)
  │
  └─ [FFN_3D] + residual → (B, 26, 7, 7, 64)
  │
  (× 3 stages 반복, 동일 구조)
  │
Final Conv3D(1, kernel=(3,3,3), valid)
  → (B, 24, 5, 5, 1)                   ← padding='valid'로 +2 제거
  │
Squeeze → (B, 24, 5, 5)

OUTPUT: power map (B, 24, 5, 5)
```

### 1.2 클래스 계층

```
CustomMaxViT3D (tf.keras.Model)
├── SuperpositioningNeighborCells     ← 파라미터 0개, 이웃 복사
├── BoundaryPad                      ← 파라미터 0개, 경계 패딩
├── Stem3D                           ← Conv3D ×2
│   ├── Conv3D(128, (15,5,5))
│   └── Conv3D(128, (15,3,3))
├── MaxViTBlock3D ×3
│   ├── MBConv3D
│   │   ├── Conv3D 1×1×1 (expand)
│   │   ├── DepthwiseConv3D 3×3×3
│   │   ├── AxialSE
│   │   │   ├── Conv3D 1×1×1 (reduce)
│   │   │   └── Conv3D 1×1×1 (expand)
│   │   └── Conv3D 1×1×1 (project)
│   ├── BlockAttention3D
│   │   ├── LayerNorm
│   │   └── RelativeAttention3D
│   │       ├── Dense (Q, K, V)
│   │       ├── relative_bias_table  ← 학습 파라미터 (gradient 문제)
│   │       └── Dense (output proj)
│   ├── FFN_3D
│   │   ├── Dense (expand)
│   │   └── Dense (shrink)
│   ├── GridAttention3D
│   │   ├── LayerNorm
│   │   └── RelativeAttention3D (위와 동일 구조)
│   └── FFN_3D
└── Conv3D(1, (3,3,3), valid)        ← 최종 출력
```

### 1.3 레이어별 상세

#### BoundaryPad

| | 설명 |
|---|---|
| **목적** | 입력이 quarter symmetry(1/4 대칭)임을 가정, 경계 반사 처리 |
| **입력** | (B, D, H, W, C) |
| **출력** | (B, D+2, H+2, W+2, C) |
| **Z축** | 상하 zero padding [1,1] |
| **Y축** | 내측(대칭면): reflect [1,0], 외측(경계): zero [0,1] |
| **X축** | Y와 동일 패턴 |
| **파라미터** | 없음 |

#### SuperpositioningNeighborCells

| | 설명 |
|---|---|
| **목적** | 각 셀에 6방향 이웃 정보를 채널로 결합 |
| **입력** | (B, D, H, W, C) |
| **출력** | (B, D, H, W, 7C) |
| **연산** | center + 6방향 이웃(zero/reflect pad 후 슬라이싱) → stack → reshape |
| **파라미터** | 없음 |

#### AxialSE (Axial Squeeze-and-Excitation)

| | 설명 |
|---|---|
| **목적** | 축별 공간 정보를 채널 가중치로 변환 |
| **입력/출력** | (B, D, H, W, C) → (B, D, H, W, C) |
| **Squeeze** | mean over [D,H] → (B,1,1,W,C), mean over [D,W] → (B,1,H,1,C), mean over [H,W] → (B,D,1,1,C), 합산 |
| **Excite** | Conv1×1(C→C/r) + GELU + Conv1×1(C/r→C) + Sigmoid |
| **Scale** | input × excitation |
| **파라미터** | Conv3D ×2 |

> **공간 정보 보존**: reduce_mean은 컨텍스트 통계 계산용이며, 최종 output = input × excitation (element-wise)으로 **공간 정보 완전 보존**. GlobalAveragePooling(차원 축소)과 달리 SE gating은 채널별 재가중치이며 출력 shape = 입력 shape.

#### MBConv3D (Mobile Inverted Bottleneck 3D)

| | 설명 |
|---|---|
| **입력** | (B, D, H, W, C_in) |
| **출력** | (B, D', H', W', hidden_size) |
| **1. Expand** | Conv1×1(C→C×expansion_rate) + LayerNorm |
| **2. Depthwise** | Conv3D(kernel, groups=C×exp) + LayerNorm + GELU |
| **3. SE** | AxialSE(reduction_ratio) |
| **4. Project** | Conv1×1(C×exp→hidden_size) + LayerNorm |
| **파라미터** | Conv3D ×3 + AxialSE + LN ×3 |
| **config** | hidden_size, expansion_rate(4), kernel_size(3,3,3), strides(1,1,1), se_reduction_ratio(4) |

#### RelativeAttention3D

| | 설명 |
|---|---|
| **입력/출력** | (B, D, H, W, C) → (B, D, H, W, hidden_size) |
| **1. Flatten** | (B, D, H, W, C) → (B, N, C) where N=D×H×W |
| **2. QKV** | Dense ×3 → (B, N, hidden_size) 각각 |
| **3. Split heads** | → (B, num_heads, N, head_size) |
| **4. Attention** | QK^T/√d_k + **relative_bias** → softmax → ×V |
| **5. Merge** | → (B, N, hidden_size) → Dense → (B, D, H, W, hidden_size) |
| **relative_bias** | `relative_bias_table`: shape (num_heads, 2D-1, 2H-1, 2W-1), `tf.gather_nd`로 (num_heads, N, N)에 인덱싱 |
| **파라미터** | Dense(Q,K,V,out) + relative_bias_table |

#### BlockAttention3D (Local Windowed Attention)

| | 설명 |
|---|---|
| **목적** | 공간을 인접 블록으로 분할 → 각 블록 내 로컬 attention |
| **Partition** | reshape: (B,D,H,W,C) → (B, D//d_b, d_b, H//h_b, h_b, W//w_b, w_b, C) → transpose [0,1,3,5,2,4,6,7] → reshape (-1, d_b, h_b, w_b, C) |
| **Attention** | 각 블록(d_b×h_b×w_b 토큰)에 RelativeAttention3D 적용 |
| **Reconstruct** | 역 transpose + reshape → 원래 공간 |
| **패딩** | 나눠떨어지지 않으면 zero pad 후 crop |
| **Residual** | shortcut + output |

#### GridAttention3D (Global Strided Attention)

| | 설명 |
|---|---|
| **목적** | 공간을 균등 간격으로 샘플링 → 전역적 sparse attention |
| **Partition** | reshape 순서 다름: (B, d_g, D//d_g, h_g, H//h_g, ...) → transpose [0,**2,4,6**,1,3,5,7] |
| **효과** | Block이 [0,1,2,3\|4,5,6,7]이면 Grid는 [0,2,4,6\|1,3,5,7] — **strided sampling** |
| **나머지** | BlockAttention3D와 동일 (attention, reconstruct, residual) |

#### FFN_3D

| | 설명 |
|---|---|
| **입력/출력** | (B, D, H, W, C) → (B, D, H, W, hidden_size) |
| **연산** | Dense(C→C×exp) + GELU + Dropout + Dense(C×exp→hidden_size) |
| **파라미터** | Dense ×2 |

#### Stem3D

| | 설명 |
|---|---|
| **입력** | (B, D, H, W, C_in) |
| **출력** | (B, D, H, W, stem_channels[1]) |
| **연산** | Conv3D(stem_ch[0], kernel[0], same) + GELU + Conv3D(stem_ch[1], kernel[1], same) + GELU |
| **현재 config** | stem_channels=(128,128), kernel_sizes=((15,5,5),(15,3,3)) |
| **참고** | BatchNorm 주석 처리됨 |

---

## 2. 문제점 / 제거 대상

### 2.0 클래스별 처리 방침

| 클래스 | 방침 | 관련 | 비고 |
|--------|:----:|:----:|------|
| **CustomMaxViT3D** | 🔧 수정 | — | 전체 리팩토링 |
| -- SuperpositioningNeighborCells | 🚫 **제거** | Prob-2 | Conv3D와 기능 중복 |
| -- BoundaryPad | ⏸️ 보류 | Prob-3 | 추후 대칭 모드별 조건부 적용 |
| -- Stem3D | 🔧 재검토 | Prob-4 | 커널 과대. CNN vs 대안 |
| ---- Conv3D(128, (15,5,5)) | 🔧 재검토 | Prob-4 | Z커널 15 축소 필요 |
| ---- Conv3D(128, (15,3,3)) | 🔧 재검토 | Prob-4 | 동상 |
| -- MaxViTBlock3D x3 | ✅ 유지 | — | 내부 수정 |
| ---- MBConv3D | ⏸️ 조건부 | §2.3 | L_diffusion 검증 통과 시 제거, 실패 시 유지 (BN→LN 통일) |
| ------ AxialSE | 🚫 **제거** | §2.3 | Attention과 기능 중복, 범용 아키텍처에서 미사용 |
| ---- BlockAttention3D | 🔧 수정 | Prob-7 | block_size 재설계 |
| ------ RelativeAttention3D | 🔧 수정 | Prob-6 | gather_nd → call() 이동 |
| -------- relative_bias_table | 🔧 수정 | Prob-6 | gradient 수정. RoPE 검토(Enh-1) |
| ---- FFN_3D | ✅ 유지 | — | |
| ---- GridAttention3D | 🔧 수정 | Prob-7 | grid_size 재설계 |
| ------ RelativeAttention3D | 🔧 수정 | Prob-6 | BlockAttention과 동일 |
| ---- FFN_3D | ✅ 유지 | — | |
| -- Conv3D(1, (3,3,3), valid) | 🔧 수정 | Prob-9 | C_out 파라미터화 + latent 출력 |

### 2.1 문제점 상세

| # | 문제 | 상세 | 조치 |
|---|------|------|------|
| Prob-1 | **D=24 반사체 포함** | 기존 입력 D=24(반사체 포함). 현재 전처리 데이터는 반사체 제외 D=20 | D=20으로 수정. 추후 반사체 포함 옵션 추가 가능 |
| Prob-2 | **SuperpositioningNeighborCells 불필요** | 6방향 이웃 복사로 10ch→70ch. Conv3D가 이미 이웃 상호작용 처리하므로 기능 중복 + 7배 메모리 낭비 | **제거 확정** |
| Prob-3 | **BoundaryPad** | 입력이 quarter symmetry(1/4 대칭)임을 가정하여 경계 반사 처리. Y/X 내측은 reflect, 외측은 zero | **우선 미사용**. 추후 대칭 모드(quarter/octant)에 따라 조건부 적용 설계 |
| Prob-4 | **Stem3D 구조 재검토** | Conv(15,5,5)+Conv(15,3,3). Z축 커널 15는 D=24 기준 과대. **CNN이 이 역할에 최적인지 재고 필요** | 축소(3×3×3 등) 또는 대안 구조(Linear projection 등) 검토 |
| Prob-5 | **BatchNorm 주석 처리** | BN 전부 주석 처리됨 (학습 불안정). LayerNorm 대체가 일관성 없음 | LayerNorm으로 통일 |
| Prob-6 | **relativebiastable gradient 누락** | `build()`에서 `tf.gather_nd` 실행 → `call()` 그래프 밖 → gradient 단절 (§2.2 상세) | 단기: `tf.gather_nd`를 `call()`로 이동. 중기: 3D RoPE 대체 검토 |
| Prob-7 | **block_size/grid_size 비호환** | block_size=(4,2,2)는 D=26 기준. D=20에서 나눗셈 호환 재설계 필요 | Block/Grid Attention 분할 전략 재검토 (D=20, H=W=5) |
| Prob-8 | **단일 스텝 예측 구조** | (B,D,H,W,C) 단일 시점 입력. Mamba 및 디코더 연계 필요 | 인코더는 per-timestep. 시간축은 T-Phase/SD-Phase에서 담당 |
| Prob-9 | **출력 형태** | 기존: power map 1ch 직접 출력 | 출력을 (B,20,5,5,D_latent) embedded vector로 변경. 물리량 복원은 디코더 담당 |
| Prob-10 | **초기화 경고** | TruncatedNormal 동일 인스턴스 재사용 → 동일 값 | seed 또는 매번 새 인스턴스 |
| **Prob-11** | **스칼라 입출력 (중대)** | 입력: p_load(스칼라). 출력: keff, AO(스칼라). 3D 텐서와 결합/분리 방법 미정 | FiLM conditioning, spatial broadcast, 별도 헤드 등 검토 |

### 2.2 Prob-6 상세: relativebiastable gradient 단절 원인

**근본 원인**: `build()` 안에서 `tf.gather_nd`가 실행되어, `call()` 시점에는 frozen tensor로 취급.

```
build() [eager, 1회 실행]
  relative_bias_table (Variable)
       ↓ tf.gather_nd  ← tf.function 그래프 밖에서 실행
  reindexed_bias (frozen EagerTensor)

call() [tf.function 트레이싱]
  ... += self.reindexed_bias  ← 상수 취급
       ↓
  loss

∴ GradientTape: loss → reindexed_bias (보임)
  reindexed_bias → relative_bias_table (불투명)
  → "Gradients do not exist for relative_bias_table"
```

**단기 수정** (5줄): `lookup_indices`(상수)는 `build()`에 두고, `tf.gather_nd`만 `call()`로 이동:

```python
def build(self, input_shape):
    self.relative_bias_table = self.add_weight(...)
    self.lookup_indices = self._compute_lookup_indices(D, H, W)  # 상수

def call(self, x):
    reindexed_bias = tf.gather_nd(self.relative_bias_table, self.lookup_indices)
    scaled_attention_logits += reindexed_bias
```

**중기**: 3D RoPE 대체 여부 — §3 Enh-1에서 논의. 기존 훈련 결과(Global MAPE 0.694%)가 위치 인코딩 없이 달성된 것이므로 gradient 복원 시 추가 개선 가능.

### 2.3 MASTER Predictor-Corrector 상세

> 매뉴얼 확인: **항상 Full Predictor-Corrector 사용** (idepl 파라미터 Not Working, 항상 Full PC 강제).
> 이전 Physical Loss 레퍼런스에서 SWPC로 기술했으나, **Full PC로 정정**.

Full Predictor-Corrector는 **제논만이 아닌 전체 물리량 커플링 루프**:

| 단계 | 수행 내용 | 갱신 물리량 |
|:----:|----------|-----------|
| **1. Predictor** | 시점 t의 φ(t), σ(t)로 Bateman ODE 해석해 적분 | N_Xe(t+1), N_I(t+1) 예측값 |
| **2. 수송 재계산** | N_Xe(t+1) 반영하여 중성자 확산방정식 풀이 | **φ(t+1)** |
| **3. 열수력 재계산** | φ(t+1) 반영하여 에너지/열전달 방정식 풀이 | **T_f(t+1), ρ_m(t+1)** |
| **4. 단면적 재계산** | T_f, ρ_m 반영하여 Taylor 전개 | **σ_a^Xe(t+1), Σ_f(t+1)** |
| **5. Corrector** | 시점 t와 t+1의 계수 가중평균으로 Bateman 재적분 | N_Xe, N_I 최종값 |

**우리 모델과의 차이**: 시점 t의 GT 값만 사용하여 1회 적분 → 단계 2~4의 커플링 루프 없음.
이것이 ~0.9% 바닥 오차의 원인 중 하나 (열수력 피드백 미반영 등 복합 요인).

**Block Attention과의 연결**: Full PC의 "인접 노드 φ 재분배"(단계 2)가 Attention의 상태 의존적 상호작용과 대응. 단 **L_diffusion이 이 학습을 유도하는 gradient 신호로 필요**.

> **결론**: MASTER가 항상 Full PC를 사용하므로, ~0.9% 바닥 오차는 Physical Loss 적분법 개선으로 해소 불가. **L_data(data loss)가 반드시 담당**해야 하며, L_diffusion은 공간 커플링 학습을 **보조**하는 역할.

### 2.4 Conv vs Attention 고찰: MBConv3D + AxialSE 유지 여부

#### 제거 근거

1. **AxialSE와 Block/Grid Attention의 기능 중복**
   - AxialSE: 축별 `reduce_mean` → bottleneck → sigmoid → 채널 재가중치 (공간적 attention의 일종)
   - Block/Grid Attention: Q·K·V로 공간 상호작용을 직접 학습
   - 두 메커니즘이 모두 "어떤 공간 위치의 어떤 채널이 중요한가"를 판단 → 중복 가능성

2. **Conv3D의 local 이웃 처리와 L_diffusion의 중복**
   - Conv3D(3,3,3) 커널이 하는 일: 인접 6~26 이웃의 가중 합산 ≈ 유한차분 스텐실의 학습 버전
   - L_diffusion이 하는 일: ∇²φ를 7점 스텐실로 **명시적 계산** → gradient로 강제
   - L_diffusion을 도입하면 Conv3D가 학습해야 할 "local 이웃 상호작용"을 Loss가 직접 강제 → Conv의 역할이 L_diffusion에 의해 대체됨

3. **Conv3D의 경계 처리 한계**
   - `padding='same'`은 내부적으로 zero padding → 경계 cell에서 0이 이웃으로 주입
   - 물리적으로: Z 상하단은 반사체(albedo 조건), XY 대칭면은 reflect 조건 → zero padding과 불일치
   - Attention은 경계 cell이 실제 존재하는 이웃만 참조 가능 (mask 처리)

4. **Conv3D 고정 커널의 한계**
   - 동일 커널이 모든 위치에 적용 → 노심 중앙과 경계에서 동일 필터
   - 확산방정식의 ∇²φ 자체는 이동 불변이므로 이론적으로 양립 가능
   - 그러나 실제로는 경계 조건, 비균일 메시(ZMESH), XS 공간 분포가 위치 의존

#### 유지 근거 (재검토)

1. **~~Inductive bias~~ — 데이터 충분성으로 약화**
   - ~~Conv는 "인접 cell이 중요하다"는 prior를 내장 → 적은 데이터에서 학습 유리~~
   - 실제 데이터: **100 LP × 576 스텝 × 31 제어봉 ≈ 178만 데이터포인트**, 각 10종+ 물리량 × (20,5,5)
   - 이 규모에서 순수 Attention도 충분히 학습 가능 → inductive bias 근거 **무효**

2. **AxialSE의 범용성 부재**
   - AxialSE가 공간 정보를 파괴하지 않는 것은 확인됨 (SE gating)
   - 그러나 Attention 기반 아키텍처(Swin3D, Video ViT, DiT 등)에서 AxialSE는 **표준 컴포넌트가 아님**
   - Attention 자체가 채널 간 상호작용을 처리하므로, SE 계열은 Conv 전용 보조 모듈에 가까움

3. **L_diffusion 도입 시 Conv 완전 대체 가능성**
   - L_diffusion이 ∇²φ를 명시적으로 강제 → Conv가 학습해야 할 "local 이웃 상호작용"을 Loss가 직접 담당
   - **단, L_diffusion의 실제 도입 가능성은 검증 필요** (아래 참조)

#### L_diffusion 도입 가능성 — 검증 완료 (2026-03-30)

> CMFD 체적 적분 잔차 사전 검증 결과: g1 median 3.50%, g2 median 6.99%

MASTER의 NNEM = CMFD + NEM 보정. 우리는 CMFD만 재현 가능하며, NEM D̂ 보정 미포함으로 5~20% 잔차 발생.
이 잔차는 코드 오류가 아닌 **방법론 구조적 차이** → ACCEPTABLE 판정.

**→ L_diffusion을 공간 커플링 보조 gradient 신호로 도입 가능.**

#### 현재 결론 (수정)

> **AxialSE: 제거. MBConv3D: 조건부.**
>
> - **AxialSE 제거 확정** — Attention과 기능 중복, 범용 아키텍처에서도 미사용, 데이터 충분
> - **MBConv3D**: L_diffusion 도입 가능성에 따라 결정
>   - L_diffusion 검증 통과 시 → MBConv3D 제거 (Conv의 역할이 Loss로 대체)
>   - L_diffusion 검증 실패 시 → MBConv3D 유지 (Conv가 유일한 local 처리)
> - **L_diffusion 사전 검증**: piecewise-test 필수 (MASTER GT 확산잔차 크기 확인)
>
> 실험 계획 (L_diffusion 검증 결과에 따라):
> 1. L_diffusion 사전 검증 (piecewise-test)
> 2. 검증 통과 시: Attention + FFN only (MBConv3D + AxialSE 전부 제거)
> 3. 검증 실패 시: MBConv3D + Attention (AxialSE만 제거)
