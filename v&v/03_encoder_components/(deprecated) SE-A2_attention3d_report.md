# SE-A2 컴포넌트 기능 테스트 보고서 — `attention3d.py`

> **작성일**: 2026-04-08
> **대상 파일**: `src/models/spatial/attention3d.py`
> **검증 단계**: SE-A2 (Spatial Encoder, Phase A, Step 2)
> **계획 문서**: `C:\Users\Administrator\.claude\plans\breezy-herding-meadow.md`

---

## 1. 목적

공간 인코더의 attention 모듈 — `STRINGRelativePE3D`, `FullAttention3D` — 의 단독 동작 검증.

특히 다음 두 가지 핵심 성질을 수치적으로 검증:
1. **회전 보존성** (rotation preserves L2 norm)
2. **이동 불변성** (translation invariance) — STRING의 핵심 수학적 보장

---

## 2. 검증 대상 컴포넌트

### 2.1. `STRINGRelativePE3D` — Block-diagonal STRING with axis-independent frequencies

| 속성 | 값 |
|---|---|
| 역할 | Q, K 벡터에 상대 위치 인코딩 적용 (attention score 계산용) |
| 구조 | Block-diagonal 2×2 회전 (RoPE 표준 패턴) |
| Frequency 자유도 | **축별 독립** — 각 head, 각 블록, 각 축 (Z/Y/X) 독립 |
| 교환 조건 | 자동 보장 (block-diagonal은 본질적으로 commute) |
| 좌표 입력 | 정규화 물리 cm (0~1 범위) |
| Frequency 초기화 | RoPE 표준 log-spaced base frequency, base=10000 |

**회전 식** (block i, 축별 독립):
```
angle_i(r_z, r_y, r_x) = θ_z[i]·r_z + θ_y[i]·r_y + θ_x[i]·r_x

[Q'_2i  ]   [ cos(angle)  -sin(angle) ] [Q_2i  ]
[Q'_2i+1] = [ sin(angle)   cos(angle) ] [Q_2i+1]
```

### 2.2. `FullAttention3D` — Multi-head full attention with STRING

| 속성 | 값 |
|---|---|
| 역할 | 500개 cell 토큰 간 full attention 계산 |
| 입력 형식 | flat sequence `(B, N, D)` (호출자가 사전에 5D → 3D reshape) |
| 구조 | Q/K/V projection → STRING(Q, K) → softmax(QK^T/√d) → out projection |
| Multi-head | num_heads × head_dim = D |
| Default num_heads | 4 (head_dim = 32) |
| QK-Norm | **미적용** (사용자 결정 2026-04-08, 학습 안정성 이슈 시 재검토) |
| Register tokens | **미적용** (사용자 결정 2026-04-08, attention sink 발생 시 재검토) |

---

## 3. 테스트 환경

| 항목 | 값 |
|---|---|
| Framework | TensorFlow 2.14 |
| Batch size | B = 2 |
| 격자 | (Z=20, qH=5, qW=5), 총 N = 500 |
| Latent 차원 | D = 128 |
| Num heads | 4 |
| Head dim | 32 (= 16개 2×2 블록) |
| 좌표 정규화 | `z/19, qy/4, qx/4` (모두 0~1 범위) |

---

## 4. 검증 항목 및 결과

### 4.1. 좌표 빌드

| 항목 | 값 | 결과 |
|---|---|:---:|
| coords 형상 | `(500, 3)` | ✅ |
| coords min | `0.0` | ✅ |
| coords max | `1.0` | ✅ |

### 4.2. `STRINGRelativePE3D` 형상 및 파라미터

| 항목 | 측정값 | 수식 / 기대값 | 결과 |
|---|---|---|:---:|
| `q_rot` 형상 | `(2, 4, 500, 32)` | input과 동일 | ✅ |
| `k_rot` 형상 | `(2, 4, 500, 32)` | input과 동일 | ✅ |
| 학습 가능 파라미터 | **192** | `num_heads × n_blocks × 3` = `4 × 16 × 3` | ✅ |

파라미터 192개 = (`theta_z` 64) + (`theta_y` 64) + (`theta_x` 64). 각 head에 16개 frequency × 3 축 = head별 48 freq.

### 4.3. **회전 보존성** — L2 norm preservation

회전 행렬은 직교 행렬이므로 L2 norm을 보존해야 함. 2×2 block-diagonal 회전 후에도 같은 성질.

| 항목 | 측정값 | 기대값 | 결과 |
|---|---|---|:---:|
| `max(|‖q‖ - ‖q_rot‖|)` | `9.5e-7` | `~0` (수치 오차 수준) | ✅ |

→ 수치 오차 수준의 차이만 발생. block-diagonal 2×2 회전이 정확히 isometry임을 확인.

### 4.4. `FullAttention3D` 형상 및 파라미터

| 항목 | 측정값 | 수식 | 결과 |
|---|---:|---|:---:|
| 출력 형상 | `(2, 500, 128)` | input과 동일 | ✅ |
| 학습 가능 파라미터 | **66,240** | (Q+K+V proj 각 128·128+128) + (out proj 128·128+128) + STRING 192 = 49,536 + 16,512 + 192 ≈ 66,240 | ✅ |

세부 분해:
- Q proj: `128·128 + 128` = 16,512
- K proj: 16,512
- V proj: 16,512
- Out proj: `128·128 + 128` = 16,512
- STRING freqs: 192
- **합**: 16,512 × 4 + 192 = **66,240** ✓

### 4.5. **이동 불변성** — Translation invariance (STRING의 핵심)

STRING의 수학적 보장: `R(rᵢ)ᵀ R(rⱼ) = R(rⱼ - rᵢ)` → 모든 좌표를 같은 양만큼 평행이동해도 attention 출력 동일.

테스트 절차:
1. 정상 좌표로 forward → 출력 `y`
2. 모든 좌표를 `+0.1` 평행이동 → forward → 출력 `y_shifted`
3. `max(|y - y_shifted|)` 측정

| 항목 | 측정값 | 기대값 | 결과 |
|---|---|---|:---:|
| `max(|y - y_shifted|)` | `5.1e-7` | `~0` (수치 오차 수준) | ✅ |

→ 수치 오차 수준 차이. **block-diagonal STRING의 commutativity 자동 보장이 수치적으로 확인됨.** 이것이 dense STRING + soft penalty 방식 대비 block-diagonal 방식의 결정적 이점.

---

## 5. 비교 박스 — 설계 의도 vs 측정

```
┌──────────────────────────────────────────────────────────────────┐
│  STRINGRelativePE3D (block-diag 2×2, axis-independent freq)     │
├──────────────────────────────────────────────────────────────────┤
│  설계 의도: 상대 위치 인코딩, 이동 불변성, 축 종류 구별            │
│  측정 결과:                                                       │
│    - shape (B,H,N,d) → (B,H,N,d)                ✓                │
│    - params 192 = num_heads × n_blocks × 3       ✓                │
│    - L2 norm 보존 (회전 isometry)                 ✓ (~9.5e-7)     │
│    - 이동 불변성 (commutativity 자동)             ✓ (~5.1e-7)     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  FullAttention3D (multi-head + STRING + Q/K/V/out projection)   │
├──────────────────────────────────────────────────────────────────┤
│  설계 의도: 500 cell 간 full attention with relative position    │
│  측정 결과:                                                       │
│    - shape (B,N,D) → (B,N,D)                    ✓                │
│    - params 66,240 = 4×(128²+128) + 192          ✓                │
│    - 4 head × 32 head_dim = 128 latent           ✓                │
│    - QK-Norm/register tokens 없음                ✓ (의도)         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. STRING 구조 결정 근거 — 논문 작성용

### 6.1. 4개 핵심 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 회전 생성 | Block-diagonal 2×2 | RoPE 표준, 안정성, 단순성 |
| Frequency 자유도 | 축별 독립 (옵션 A) | Z vs Y/X 축 종류 구별 가능 |
| 교환 조건 | 자동 (block-diagonal commute) | 수치적 검증 ~5e-7 |
| 좌표 입력 | 정규화 물리 cm (0~1) | 비균일 메시 확장 호환 |

### 6.2. 옵션 (B) 축 공유 frequency를 채택하지 않은 이유

| 옵션 | 식 | freq 파라미터 수 | 표현력 |
|---|---|:---:|---|
| **(A) 축별 독립 (채택)** | `angle_i = θ_z[i]·r_z + θ_y[i]·r_y + θ_x[i]·r_x` | 192 | 모델이 Z/Y/X 축 종류 구별 |
| (B) 축 공유 | `angle_i = θ[i]·(r_z+r_y+r_x)` | 64 | 좌표 합으로 환원, 축 구별 못 함 |

**(B)가 부적합한 4가지 이유**:
1. **물리적 부적합**: 우리 노심에서 Z(축방향, 제어봉)와 Y/X(반경, peaking factor)는 *완전히 다른 물리*. (B)는 거리 벡터 (Δz=2,0,0)과 (0,Δy=2,0)을 구별 못 함
2. **표현력 손실**: 1D RoPE 수준으로 환원
3. **파라미터 절감 무의미**: 절감되는 freq는 192-64=128개 → 인코더 0.66M의 0.02%
4. **선례 부재**: M-RoPE (Qwen-VL), RoPE-3D ViT 등 modern 3D RoPE의 표준은 모두 (A)

### 6.3. (A)를 채택한 4가지 명시적 근거

1. **물리 종류 구별**: Z 거리에 민감한 head/block과 XY 거리에 민감한 head/block을 자연스럽게 분화 가능
2. **이동 불변성 보장**: block-diagonal 자동 commutativity → STRING의 핵심 성질 R(rᵢ)ᵀR(rⱼ)=R(rⱼ-rᵢ) 성립. **본 보고서 §4.5에서 수치 확인 (~5e-7)**
3. **RoPE 표준 패턴**: M-RoPE (Qwen2-VL, Wang et al. 2024), RoPE-3D ViT 변종 등 사실상 표준
4. **수학적 동치성**: dense STRING(commutativity-strict)은 unitary 변환에 의해 block-diagonal로 환원됨 (모든 commuting skew-symmetric matrices는 동시에 block-diagonalize 가능). 즉 (A)는 STRING의 commutativity-strict 구현 중 가장 안정적이고 단순한 형태

### 6.4. 미적용 옵션의 명시적 근거 (논문용)

| 옵션 | 미적용 결정 일자 | 근거 |
|---|---|---|
| **QK-Norm** | 2026-04-08 | 우리 모델 크기(0.66M)에서 LayerNorm만으로 통상 충분. 학습 중 attention logit 폭주 관찰 시 재검토 |
| **Register tokens** | 2026-04-08 | 시퀀스(500)가 짧고 layer(3 stage)가 얕아 attention sink 발생 가능성 낮음. 학습 후 attention pattern 분석에서 sink 발견 시 추가 |
| **Dense STRING (Cayley)** | 2026-04-08 | commutativity-strict 시 block-diagonal과 수학적 동치. 추가 표현력 없이 구현 복잡도 증가 |
| **LieRE 스타일 (commutativity 미강제)** | 2026-04-08 | 이동 불변성 포기 → 우리 격자(상대 거리 = 물리적 결합 강도) 물리에 부적합 |

---

## 7. 결론

| 항목 | 결과 |
|---|:---:|
| 형상 변환 정확성 | ✅ |
| 파라미터 수 정확성 | ✅ |
| **회전 L2 norm 보존** | ✅ (수치 오차 ~9.5e-7) |
| **이동 불변성 (STRING 핵심)** | ✅ (수치 오차 ~5.1e-7) |
| Forward 오류 없음 | ✅ |

**판정**: SE-A2 모든 항목 통과. 특히 STRING의 핵심 수학적 성질(이동 불변성)이 수치적으로 확인됨. SE-B1 (encoder3d.py 전체 조립) 진입 가능.

---

## 8. 후속 검증 (SE-B 단계)

본 보고서는 attention 모듈 *단독* 검증. 인코더 전체 조립 후 추가 검증 항목:

- [ ] Multi-block attention (3 stage 누적) 후 출력 안정성 (NaN/Inf 없음)
- [ ] Attention pattern 시각화 — 학습 전/후 head별 attention map. local vs global 분화 관찰
- [ ] Gradient flow: STRING freq 변수에 grad 흐름
- [ ] 학습 후 STRING freq 분석: 어느 축이 어느 head에서 dominant frequency를 가졌는가
- [ ] Attention entropy 추적 (학습 중): logit 폭주 시 QK-Norm escalation 트리거

→ SE-B3 단계에서 `tests/encoder/` 하위에 unit test 및 학습 후 분석 hook으로 구현 예정.
