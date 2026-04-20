# ConditionalLAPE 정보 단절 우려 및 정보(위치, 대칭, 출력조건) 주입 방안 검토

> **작성일**: 2026-04-17
> **상태**: 작성 중 — §1~§4 (문제 정의 + 자문 요청서) 1차 작성. §5~§6 (자문 결과·종합) 은 외부 LLM 응답 도착 후 추가
> **검토 범위**: 정보 3종 — ① p_load (출력조건, 스칼라), ② sym_type (대칭 메타데이터), ③ 절대위치 (ConditionalLAPE 좌표)
> **검토 메커니즘**: FiLM, AdaLN(-Zero), Equation-Aware Conditioning, concat, cross-attention 등
> **주의**: p_load 의 기존 잠정 결정 (Mamba concat + 디코더 AdaLN-Zero) 도 본 검토에서 **재검토 대상**으로 다룬다

---

## 1. 배경 및 작성 동기

### 1.1 현재 결정/잠정 상태 요약

| 정보 | 인코더 | 시공간 프로세서 (Mamba+Attn) | 디코더 (Conv3D) | 비고 |
|------|--------|---------------------|------------|------|
| **p_load** | 미주입 (결정) | Mamba 입력 concat (잠정) | 미주입 (Conv3D 변경 후 잠정) | 이전: 디코더 AdaLN-Zero |
| **sym_type** | halo_expand + ConditionalLAPE (결정) | 미주입 | 미주입 | 향후 과제로만 언급 |
| **절대위치** | ConditionalLAPE 1회 add (결정) | 미주입 | 미주입 | 향후 과제 미언급 |

- 인코더는 ConditionalLAPE 결정 (`2026-04-14 Conditional LAPE 적용 검토.md`) 으로 위치/대칭이 **입력 단계 1회** add 됨
- 시공간 프로세서는 p_load만 Mamba concat (`2026-04-15 모델 구현 계획(시계열).md` §5.5)
- 디코더는 모든 conditioning 미주입 (Conv3D 1×1×1 채택 후 AdaLN 보류 상태)

### 1.2 작성 계기

- ConditionalLAPE 결정 문서 §3.4 가 이미 우려를 명시: "Mamba selective scan은 인코더의 residual stream처럼 원본 신호를 가산 형태로 그대로 보존하는 구조가 아니다. 인코더에서 add된 대칭 유형 정보가 Mamba를 거치며 희석될 수 있다"
- 디코더가 Conv3D 1×1×1로 단순화되면서 **공간 결합/조건 정보 회복 메커니즘 부재**가 새 문제로 등장
- p_load의 기존 잠정 안 (Mamba concat + 디코더 AdaLN-Zero) 도 sym/위치와 함께 통합적으로 재검토 필요
- 결론: **인코더 + 시공간 + 디코더 전부에 conditioning 을 반영하는 게 옳은가, 어느 메커니즘이 옳은가** 를 사례 탐색 + 외부 교차 확인으로 결정

### 1.3 본 문서의 위치

- `2026-04-15 p_load 주입 전략 재검토.md`: p_load 에 한정한 이전 검토 (DreamerV3/SHRED 패턴 참조)
- `2026-04-14 Conditional LAPE 적용 검토.md` §3.4: sym_type 흐름 단절 우려를 처음 언급, "디코더 설계 시 별도 검토" 로 보류
- `2026-04-17 시계열 아키텍처 학습방법론 재검증.md` §2.4: Equation-Aware Conditioning 을 "후순위 추가" 로 언급
- 본 문서: 위 3건의 우려/제안을 **통합 재검토**, 정보 3종 × 주입 위치 3곳 × 메커니즘 5종을 한 자리에서 비교

---

## 2. 문제 정의

### 2.1 정보 3종의 성격 비교

| 정보 | 차원 | 시간 변동 | 공간 분포 | 도입 목적 / 물리적 역할 | 시퀀스 내 변화 |
|------|------|---------|---------|----------------------|------------|
| **p_load** | 스칼라 (B,) | 매 시점 변동 | 없음 (전역) | 출력 수준 조건. 시간 동역학 속도 (온도 상승률, Xe 생성률) 에 직접 영향 | 575 step 내 자유롭게 변동 |
| **sym_type** | 정수 (B,), {0=mirror, 1=rotation} | 시퀀스 전체 일정 | 없음 (메타데이터) | 격자 기하 유형. halo_expand 매핑 모드 결정 + ConditionalLAPE 테이블 분기 | 시퀀스 단위 상수 |
| **절대위치** | (z, y, x) 좌표 → 임베딩 (Z, qH, qW, D) | 시퀀스 전체 일정 | 격자 위치별 다름 | **모델이 대칭 유형(mirror/rotation)을 위치 임베딩 수준에서 인식**하게 만들기 위한 도구. halo cell 의 인식 (halo cell 은 L_diff 계산 시 inner cell 과 연산되어야 하므로 도입됨) | 시퀀스 단위 상수 |

**핵심 차이**:
- p_load 는 **동적 제어 변수** (시간에 따라 변동, 시간 전이에 직접 영향)
- sym_type 과 절대위치는 **정적 메타데이터** (시퀀스 단위 상수, 공간/대칭 prior 제공)
- 절대위치와 sym_type 은 **상호 연결됨**: ConditionalLAPE 는 sym_type 별로 다른 위치 테이블을 보유하여 "위치 + 대칭"을 함께 인코딩

### 2.2 현재 파이프라인에서 정보가 흐르는 경로

```
[인코더 입력]
  state(10ch) + xs_fuel(10ch) + rod_map(1ch) → 21ch        ← p_load 미주입
  + halo_expand(sym_type)                                   ← sym_type 1회 사용 (halo cell 매핑)
       ↓
  CellEmbedder (Conv3D 1×1×1, 21→128)
       ↓
  ConditionalLAPE(sym_type) → x + lape[sym_type]            ← sym_type, 위치 1회 add
       ↓
  3 stages × [Pre-LN + FullAttention3D(STRING) + FFN3D]
       ↓
  z(t) ∈ (B, 720, 128)    ← residual stream에 sym_type/위치 신호 보존됨
       ↓
─────────────────────────────────────────────────────────────
[시공간 프로세서]
  z(t) + p_load(t) concat                                   ← p_load 첫 등장
       ↓
  Mamba-3 #1, #2  (cell별 독립 시간 전이)                     ← sym_type/위치 미주입
       ↓
  Self-Attention + 소형 MLP (시점별 720 cell 결합)            ← 모든 conditioning 미주입
       ↓
  Block 2 반복
       ↓
  d(t) ∈ (B, 720, 128)
       ↓
─────────────────────────────────────────────────────────────
[디코더]
  Conv3D 1×1×1 (128 → 10), cell별 독립                       ← 모든 conditioning 미주입
       ↓
  delta_pred + state(t) → X_next_pred
```

### 2.3 핵심 우려 — 정보 단절 시점 3곳

#### 우려 A: 인코더 → 시공간 프로세서 (sym_type, 절대위치 희석)

- 인코더 내부에서는 ConditionalLAPE가 입력에 1회 add 되어 residual stream에 영원히 보존 (`2026-04-14` §1.3)
- 그러나 Mamba 의 selective scan 은 **가산 residual 구조가 아님**:
  - SSM 재귀: `h_t = A_t · h_{t-1} + B_t · x_t`,  `y_t = C_t · h_t + D · x_t`
  - h_t 는 행렬 변환 + 입력 의존적 게이팅을 받으며 누적 → 입력 신호의 정체성이 단순 add 처럼 보존되지 않음
- 결과: 인코더 출력 z(t) 에 담긴 "대칭 유형 + 위치" 신호가 Mamba 통과 후 d(t) 에서는 희석/변형 가능

**물리적 비유**: 인코더가 매 cell 에 "여기는 (z=0, y=0, x=0) 위치이고 이 격자는 mirror 대칭이야" 라는 명찰을 붙여 줘도, Mamba가 시간 누적하면서 명찰이 흐려질 수 있음

#### 우려 B: 시공간 → 디코더 (공간 결합/조건 정보 부재)

- 디코더가 Conv3D 1×1×1 → cell 별 독립 변환
- sym_type, 절대위치, p_load 모두 디코더에 직접 주입 안 됨
- 인코더에서의 add 신호가 시공간 프로세서를 통과하면서 살아남았다는 보장이 없음
- 결과: 디코더는 d(t)의 128차원 표현이 자체적으로 "위치 (halo or inner) + 대칭 유형 + 출력조건" 정보를 충분히 담고 있다고 가정 → 가정 검증 필요

**물리적 비유**: 디코더가 "각 cell에 어떤 절대값을 그릴지" 결정하는데, 이 cell이 halo 인지 inner 인지, mirror 격자인지 rotation 격자인지, 현재 50% 출력인지 100% 출력인지 모르는 상태

#### 우려 C: p_load 의 동적 변동 추적 (기존 결정 재검토)

- p_load 는 매 시점 변동 → Mamba 입력 concat 으로 매 step 새 값 반영 가능
- 그러나 디코더에서 p_load 미주입 → 디코더는 d(t) 표현만으로 출력 수준 차이를 복원해야 함
- 출력 절대값 (예: 100% vs 50% 의 절대 출력 분포) 이 디코더 출력에 직접 반영되어야 하는데, d(t) 에 그 정보가 충분한지 불확실

### 2.4 핵심 질문 (자문 요청 핵심 후보)

본 절의 질문은 §4 자문 요청서에서 외부 LLM에 그대로 전달.

- **Q1.** ConditionalLAPE 같이 인코더 입력 단계에서만 add 된 정적 위치/대칭 정보가, Mamba+Attention 하이브리드 시공간 프로세서를 통과한 뒤 디코더에서 사용 가능한 형태로 살아남는가? 정량적 분석 사례가 있는가?

- **Q2.** 정적 메타데이터 (대칭, 절대위치) 를 인코더 + 시공간 + 디코더 **매 stage** 에 재주입하는 것이 3D 공간 대리모델/PDE 대리모델/기상 ML 의 표준 패턴인가? 일부 stage 에만 주입하는 것과의 비교 사례는?

- **Q3.** 동적 제어 변수 (p_load) 와 정적 메타데이터 (sym, 위치) 는 다른 메커니즘으로 처리해야 하는가, 같은 메커니즘으로 통합 가능한가?

- **Q4.** p_load 의 기존 잠정 안 (Mamba concat + 디코더 AdaLN-Zero) 을 유지해야 하는가, 변경해야 하는가? 디코더가 Conv3D 1×1×1로 단순화된 경우 AdaLN-Zero 적용이 여전히 자연스러운가?

- **Q5.** FiLM, AdaLN(-Zero), Equation-Aware Conditioning, concat, cross-attention 중 본 프로젝트 (3D 원자로 노심 시공간 대리모델, 720 cell token, Mamba+Attention 2:1, Conv3D 1×1×1 디코더) 에 가장 적합한 메커니즘은?

---

## 3. 검토 대상 메커니즘 후보

### 3.1 후보 메커니즘 5종 개관

| 메커니즘 | 작동 방식 | 신호 강도 | 매 layer 재적용 | 구현 복잡도 |
|---------|---------|---------|------------|----------|
| **Concat** | 조건을 입력 텐서에 채널 결합 | 중 (입력에서 1회) | X (단발) | 매우 낮음 |
| **FiLM** | γ(c)·x + β(c). 조건으로 affine 변조 | 강 | O (블록마다 적용 가능) | 낮음 |
| **AdaLN-Zero** | LayerNorm의 γ, β를 조건으로 생성. 0 초기화로 안정 | 강 | O (LN 마다 적용) | 중 (DiT 패턴) |
| **Equation-Aware** | 메타데이터를 작은 벡터로 인코딩 → 모든 블록 conditioning | 강 | O (전 블록) | 중 |
| **Cross-Attention** | 조건을 KV, 본 신호를 Q | 매우 강 | O (블록마다) | 높음 (계산량 증가) |

### 3.2 메커니즘별 상세

#### 3.2.1 Concat (현 p_load 방식)

- 작동: `[z, p_load_emb] → Linear → 새 표현`
- 장점: 가장 단순, DreamerV3/SHRED 등 시계열 물리 모델의 표준 패턴
- 단점: **입력 1회만** 주입 → 깊은 네트워크에서 신호 감쇠 가능
- 적합 정보: **동적 제어 변수** (p_load 처럼 매 시점 변동) — Mamba 입력에 매 step concat 하면 시간 축으로는 매 step 재주입되므로 감쇠 우려 적음
- 부적합 정보: 정적 메타데이터 (sym, 위치) — concat 의 위치별/시점별 영향이 약함

#### 3.2.2 FiLM (Feature-wise Linear Modulation)

- 작동: `FiLM(F) = γ(c)·F + β(c)`,  γ, β 는 조건 c 의 함수 (작은 MLP 출력)
- 출처: Perez et al. 2018 (AAAI), 시각 추론용 ResNet 변조
- 장점: 매 블록 재적용 → 신호 감쇠 없음. 채널 단위 변조로 표현력 우수
- 단점: 인코더 LN 에 조건 주입 안 함 결정 (`04_normalization_omitted_options.md` §1.4) 과 충돌 가능
- 적합 정보: 정적/동적 모두 가능. sym, 위치, p_load 모두 적용 가능
- 본 프로젝트 적용: 인코더 외부 (시공간 프로세서, 디코더) 에 한정하면 기존 결정과 충돌 없음

#### 3.2.3 AdaLN / AdaLN-Zero

- 작동: LN 의 γ, β 를 조건 c 의 MLP 출력으로 대체. AdaLN-Zero는 초기 γ=0, β=0 로 학습 시작 (residual gate 0)
- 출처: DiT (Peebles & Xie 2023), 디퓨전 트랜스포머의 클래스 조건 주입
- 장점: FiLM과 유사하나 LN 과 결합 → 정규화-변조 일관성. Zero 초기화로 학습 안정
- 단점: LN 이 있는 곳에만 적용 가능. Mamba 내부 LN 위치 의존
- 적합 정보: 정적/동적 모두 가능
- 본 프로젝트 적용: 디코더 (Conv3D 1×1×1) 에는 LN 이 없으므로 AdaLN 적용 자체가 부자연스러움 → 디코더 구조 변경 (Conv3D + LN + AdaLN-Zero 추가) 이 동반되어야 함

#### 3.2.4 Equation-Aware Conditioning

- 작동: 방정식 파라미터/메타데이터를 작은 벡터로 인코딩 → 모든 Mamba/Attention 블록의 conditioning (보통 AdaLN 으로 modulation)
- 출처: Equation-Aware Neural Operator (arXiv 2511.09729, 2025년 11월)
- 장점: 정적 메타데이터를 매 블록 명시적으로 인식. Branch 예측에서 조건 변경의 전파 강화
- 단점: AdaLN 과 동일한 구조 의존성. 메타데이터 인코딩 설계 필요
- 적합 정보: **정적 메타데이터에 특화** (방정식 파라미터, 기하 조건). 본 프로젝트의 sym_type + 절대위치에 가장 부합하는 패러다임

#### 3.2.5 Cross-Attention (조건을 KV로)

- 작동: `Q = WQ·z`, `K = WK·c`, `V = WV·c`, → `softmax(QK^T/√d)·V`
- 출처: Perceiver IO, Stable Diffusion (텍스트 조건)
- 장점: 매우 강한 신호. 조건이 복잡한 구조 (시퀀스, 토큰 집합) 인 경우 자연스러움
- 단점: 계산량 증가 (O(N·M·d), N=720, M=조건 토큰 수). 본 프로젝트 조건은 스칼라/저차원 → 과도한 복잡성
- 적합 정보: 조건 자체가 풍부한 구조를 가진 경우 (예: 텍스트, 다른 시퀀스). 본 프로젝트에는 부적합

### 3.3 정보 × 메커니즘 적합도 (1차 잠정 평가, 자문 후 갱신 예정)

| 정보 | Concat | FiLM | AdaLN-Zero | Equation-Aware | Cross-Attn |
|------|:----:|:----:|:--------:|:------------:|:--------:|
| p_load (동적 스칼라) | ◎ | ○ | ○ | △ | X |
| sym_type (정적 이산) | △ | ○ | ◎ | ◎ | X |
| 절대위치 (정적 격자) | X | △ | ○ | ◎ | △ |

(◎ 우수, ○ 적합, △ 가능, X 부적합 — 자문 결과로 검증 필요)

### 3.4 주입 위치 × 메커니즘 조합 가능성

| 위치 | Concat | FiLM | AdaLN-Zero | Equation-Aware |
|------|:----:|:----:|:--------:|:------------:|
| 인코더 (이미 LAPE add) | △ (중복 가능성) | X (기존 결정 위배) | X (기존 결정 위배) | X (기존 결정 위배) |
| Mamba 입력 | ◎ | ○ | △ (LN 위치 의존) | ○ |
| Attention 블록 | ○ | ◎ | ◎ | ◎ |
| 디코더 Conv3D 1×1×1 | △ | ○ | △ (LN 추가 필요) | ○ |

---

## 4. 자문 요청서 (외부 LLM 교차 확인용)

> 아래 §4.1~§4.4 는 외부 LLM 에 컨텍스트 없이 그대로 전달 가능하도록 자기완결적으로 작성

### 4.1 배경

본 프로젝트는 3D 원자로 노심 (가압경수로 SMR) 의 시공간 동역학을 예측하는 신경망 대리모델이다.

**파이프라인**:
```
[입력] state(10ch, Xe/I/flux/온도) + xs_fuel(10ch) + rod_map(1ch) = 21ch
       격자: (B, 20, 6, 6, 21)  — 20 축방향 × 6×6 quarter core (halo 포함)
       메타데이터: sym_type ∈ {0=mirror, 1=rotation}, p_load ∈ [0, 1] 스칼라
       ↓
[공간 인코더] (구현 완료, 변경 불가)
   - CellEmbedder: Conv3D 1×1×1 (21→128)
   - ConditionalLAPE3D: sym_type 별 LAPE 테이블 2개 보유, x + lape[sym_type] 1회 add
     (목적: 모델이 대칭 유형을 위치 임베딩 수준에서 인식. halo cell 은 L_diff 계산을 위해 도입됨)
   - 3 stages × [Pre-LN + FullAttention3D(STRING) + FFN3D]
   - 출력: z(t) ∈ (B, 720, 128)
       ↓
[시공간 프로세서] (설계 완료)
   - Block 1, 2 (각 독립 가중치)
   - 각 Block: [Mamba-3 #1, Mamba-3 #2, Self-Attention + 소형 MLP]
   - Mamba-3: hidden=128, state_dim=16, head_dim=64, num_heads=2, 복소 상태 ON, Trapezoidal 이산화 ON
   - 학습: parallel scan (575 step 한번에). 추론: 1-step recurrence
   - 출력: d(t) ∈ (B, 720, 128)
       ↓
[디코더] (설계 완료)
   - Conv3D 1×1×1 (128→10), cell별 독립 변환
   - delta_pred 출력 → state(t)[10ch] + delta_pred = X_next_pred (정규화 절대값)
```

**현재 conditioning 주입 상태**:

| 정보 | 인코더 | 시공간 프로세서 | 디코더 |
|------|--------|-------------|------|
| p_load (동적 스칼라) | 미주입 | Mamba 입력 concat (잠정) | 미주입 |
| sym_type (정적 이산) | halo_expand + ConditionalLAPE 1회 add | 미주입 | 미주입 |
| 절대위치 (정적 격자) | ConditionalLAPE 1회 add | 미주입 | 미주입 |

**핵심 우려**:

(1) ConditionalLAPE는 인코더 입력 단계에서 텐서에 1회 add 되어, 인코더 내부의 Pre-LN residual stream을 통해 모든 attention layer로 전달된다. 그러나 시공간 프로세서의 Mamba-3 SSM 은 selective scan 구조 (`h_t = A_t · h_{t-1} + B_t · x_t`, A_t/B_t/C_t 는 입력 의존적 게이팅) 이며, 인코더의 Pre-LN residual 처럼 원본 신호를 단순 add 형태로 보존하지 않는다. 인코더에서 add 된 위치/대칭 정보가 Mamba 통과 후 d(t)에서 살아남는가?

(2) 디코더가 Conv3D 1×1×1 (cell별 독립) 로 단순화되어, 디코더는 d(t) 의 128차원만으로 모든 정보를 복원해야 한다. sym_type, 절대위치, p_load 가 모두 디코더에 직접 주입되지 않는데, 이 정보가 d(t) 에 충분히 담겨 있다는 가정이 타당한가?

(3) p_load 의 기존 잠정 결정 (Mamba concat + 디코더 AdaLN-Zero) 은 디코더가 Conv3D 1×1×1로 단순화되기 전 (Full Attention + AdaLN 디코더 가정) 에 내려졌다. Conv3D 디코더로 변경된 현 시점에 이 결정을 유지할지, 변경할지 재검토 필요하다.

**검토 대상 메커니즘**: FiLM, AdaLN(-Zero), Equation-Aware Conditioning, Concat, Cross-Attention

### 4.2 핵심 질문

- **Q1.** ConditionalLAPE 같이 인코더 입력 단계에서만 add 된 정적 위치/대칭 정보가, Mamba-3 + Attention 하이브리드 시공간 프로세서를 통과한 뒤 디코더에서 사용 가능한 형태로 살아남는가? SSM 통과 시 정적 조건 정보의 감쇠를 정량 분석한 사례가 있는가?

- **Q2.** 정적 메타데이터 (대칭 유형, 절대위치) 를 인코더 + 시공간 + 디코더의 매 stage 에 재주입하는 것이 3D 공간 대리모델, PDE 대리모델, 기상 ML 의 표준 패턴인가? 일부 stage 에만 주입하는 것과의 정량 비교 사례는?

- **Q3.** 동적 제어 변수 (p_load 같은 매 시점 변동 스칼라) 와 정적 메타데이터 (시퀀스 단위 상수) 는 다른 메커니즘으로 처리해야 하는가? 통합 가능한가?

- **Q4.** 본 프로젝트의 잠정 결정 (p_load → Mamba concat + 디코더 AdaLN-Zero) 을 유지해야 하는가, 변경해야 하는가? 디코더가 Conv3D 1×1×1로 단순화되어 LN 자체가 없는 경우 AdaLN-Zero 적용이 자연스러운가?

- **Q5.** FiLM, AdaLN(-Zero), Equation-Aware Conditioning, concat, cross-attention 중 본 프로젝트의 정보 3종 (p_load, sym_type, 절대위치) × 주입 위치 3곳 (인코더 외부 / 시공간 프로세서 / 디코더) 에 각각 어느 메커니즘이 가장 적합한가?

### 4.3 사례 탐색 요청 항목

다음 도메인에서 conditioning 주입 메커니즘 사례를 찾아 정리해 주십시오:

1. **3D 공간 대리모델/PDE 대리모델**:
   - 정적 메타데이터 (격자 기하, 경계 조건, 방정식 파라미터) 를 다단계로 주입한 사례
   - Equation-Aware Neural Operator (arXiv 2511.09729) 외에 유사 패턴이 있는가?

2. **SSM/Mamba 기반 모델**:
   - SSM 통과 시 위치 정보/조건 정보의 감쇠를 측정/분석한 사례
   - Mamba 의 selective scan 에 외부 조건을 주입하는 표준 패턴

3. **기상/기후 ML** (GraphCast, FourCastNet, GenCast, Pangu-Weather, ClimaX 등):
   - 위경도 좌표, 시각, 계절 등 정적/시간 메타데이터의 주입 위치와 메커니즘
   - "인코더 + 프로세서 + 디코더" 구조에서 conditioning 흐름

4. **세계 모델/RL** (DreamerV3, V-JEPA 2, Ctrl-World 등):
   - action / 정적 task 메타데이터의 주입 위치 분리
   - DreamerV3 의 action → GRU 입력 concat 패턴이 본 프로젝트의 p_load → Mamba concat 과 동일한가?

5. **DiT 계열의 AdaLN-Zero** (Diffusion Transformers):
   - 스칼라/이산 조건 주입의 표준 형태
   - Conv3D 1×1×1 같은 LN 없는 디코더에 대한 변형/대안

### 4.4 답변 형식 요청

- 각 사례마다 출처 (논문 제목, arXiv ID 또는 DOI, 저자, 연도) 명시
- "구조적 권고" 와 "실험적 검증된 권고" 를 구분
- 본 프로젝트의 주입 위치 × 정보 × 메커니즘 조합표 (§3.3, §3.4) 에 대한 평가/수정 제안
- 본 프로젝트의 핵심 우려 (특히 Q1: SSM 통과 시 정적 조건 감쇠) 가 실제 문제인지 또는 과한 우려인지 판단

---

## 5. 자문 결과 정리 및 심화 검토

### 5.1 답변 A — Equation-Aware Encoder + 정보 종류별 메커니즘 분리 전략

#### 핵심 주장

**Equation-Aware Encoder는 FiLM/AdaLN과 다른 구조**라는 주장이 핵심:
- 1단계 (encoder stage): 메타데이터 (sym_type, 절대위치, rod_map 통계) → 조건 벡터 c (작은 MLP)
- 2단계 (injection stage): c → 네트워크 modulation (FiLM 또는 AdaLN) in 시공간 프로세서/디코더

이는 단순 "메커니즘 선택" 문제가 아니라, 메타데이터 **인코딩**과 **주입**을 분리하는 설계 철학.

#### 정보 종류별 메커니즘 다변화 주장의 타당성

**A의 핵심 주장**: sym_type (전역) ≠ 절대위치 (token-wise) → 다른 메커니즘 필요
- sym_type: scalar 이산값 (0=mirror, 1=rotation) → FiLM/AdaLN 적합 (전역 channel modulation)
- 절대위치: 720개 cell × (z,y,x) → per-token 다름 → FiLM/AdaLN으로 압축하면 공간 식별성 손실 → "위치 re-add (residual)"가 더 자연스러움

**우리 프로젝트 맥락 분석**:
- ConditionalLAPE가 이미 절대위치를 residual add 형태로 처리 중 (§1.1 표, "input 단계 1회 add")
- 그러나 Mamba 통과 후 재주입이 필요한 상황 → 재주입 시 "다시 add할 것인가 vs FiLM으로 변조할 것인가"라는 새 의사결정 필요
- A의 주장대로면: Attention 블록에서 "위치 정보 re-add" 형태로 반복 → Attention Q/K에 RoPE(Rope Position Embedding) 적용 가능성 열림

**한계**: 절대위치를 "2단계 인코딩" (LAPE 수치 → c_pos 벡터로 재인코딩)할 때 정보 병목 발생. LAPE의 (Z, qH, qW, D) 고차원 embedding을 작은 c_pos로 압축하면 cell-wise 차이가 사라질 위험 → token-wise 해상도 보존 전략이 별도 필요.

#### c_rod (Rod Map 통계 인코더) 제안의 실용성

**A가 제안**: rod_map (제어봉 상태) → 별도 통계 벡터 (삽입 깊이 mean, 축방향 centroid, 반경 1차/2차 모멘트, bank occupancy, halo 인접성, symmetry-aware imbalance) → small MLP → c_rod

**타당성 평가**:
- 장점: 제어봉은 외부 제어 입력인데, 단순 21ch 채널 하나(rod_map)보다 **통계적 특징**(distribution/centrality)을 명시적으로 encode하면 모델이 "제어봉 전체의 영향"을 직관적으로 학습 가능 → Branch 예측 정확도 향상 가능
- 단점: 우리 프로젝트에서 rod_map은 이미 인코더 입력의 21ch 중 1ch로 포함됨. 추가 통계 인코딩이 **보완(새로운 정보)**인지 **중복(벌써 인코더가 학습함)**인지 실험적으로 확인 필요
- 제안: Phase 2 에서 선택적 추가 가능 (우선순위: 낮음)

#### 4개 판별 지표 (Discrimination Metrics) 활용

A 제안:
1. **(a) Halo cell 오차**: 경계 cell에서의 예측 오차 (L_diff 계산에 필수이므로 중요)
2. **(b) Mirror vs Rotation 오차 분리**: sym_type별 예측 정확도 차이 → 대칭 정보가 제대로 주입되었나 진단
3. **(c) p_load 구간별 보정 오차**: 50%, 75%, 100% 출력 등 부하 수준별 성능 확인
4. **(d) 정상(Steady) vs 과도(Transient) 오차 분해**: 정상상태 계산 vs 동적 변화 추종 능력 분리 평가

**우리 프로젝트 도입 권고**: 이 4개 지표는 단순 RMSE보다 **조건화 주입이 실제로 작동하는지 진단**하는 데 핵심적. 특히 (a), (b)는 위치/대칭 정보 주입을 직접 검증. (c), (d)는 p_load 동적 조건화 검증.
- 실행 시점: Phase 1 기본선 모델 (현재 A안 수준) 완성 후 즉시 도입 필요

---

### 5.2 답변 B — B_t 게이팅 수학적 분석 + "Heavy Processor, Light Decoder" 물리 손실 논거

#### 핵심 주장 1: Mamba B_t 게이팅의 정적 신호 필터링

**수학적 메커니즘**:

Mamba SSM 재귀:
```
h_t = A_t · h_{t-1} + B_t · x_t
y_t = C_t · h_t + D · x_t
```

x_t가 시간 불변 상수 (예: ConditionalLAPE로부터 add된 위치 신호)라면:
- B_t는 입력 x_t에 대한 게이팅 (input-dependent) → 신호가 state h_t로 전달되는 정도를 동적으로 조절
- 시간이 지나면서 h_t의 evolution은 A_t · h_{t-1} (과거 dynamics) 이 지배적 → 상수 x_t의 영향은 **누적되지 않고 희석**
- 결과: 정적 편향(static bias)은 B_t에 의해 "동역학 없는 노이즈"로 처리 → 명시적 재주입 없이는 사라짐

**평가**: Q1 우려 (정적 신호 희석) **100% 유효**한 수학적 증명 제시. 이는 §2.2 우려 A의 물리적 비유를 수학 수준에서 엄밀화.

#### 핵심 주장 2: PDE 도메인의 표준 — ALL stages 재주입

**기상/물리 대리모델 사례**:
- GraphCast, GenCast: Processor 내 매 블록의 Attention 전후에 조건(위경도, 시각, 계절 메타데이터) 재주입
- Pangu-Weather, FourCastNet: 위위도 positional embedding + 시간 조건 을 Transformer/Conv 각 층에 반복 주입

**논거**: PDE의 타원형(Elliptic) 특성 때문에 "위치"는 시간 불변이 아니라 **매 스텝 물리 계산의 essential input**. 시뮬레이터가 매 시간스텝마다 ∇²φ(공간 라플라시안)을 계산하듯이, 신경망도 매 Processor 블록마다 공간 정보를 반영해야 수학적으로 일관성 있음.

**우리 프로젝트 적용**: 정적 메타데이터 (sym_type, 절대위치) 를 Processor 매 블록 + 디코더 직전에 재주입해야 함. 기존 결정 (encoder 1회 add만) 은 **미흡**.

#### 핵심 주장 3: "Heavy Processor, Light Decoder" 와 Physical Loss Gradient 투명성

**수학적 직관**:
```
L_physics = Loss(d(t), ground_truth_physics)
∇L / ∂ Processor ∝ ∇L / ∂ d(t) × ∂d/∂Processor

깊은 디코더 (FullAttn 5층):
  d(t) = Dec_deep(z_t) — 많은 비선형성, feature transformation
  → ∇d/∂Processor 가 여러 층의 chain rule을 거침
  → gradient 신호가 "변형" 및 "감쇠" 가능 (vanishing/exploding)

얕은 디코더 (Conv3D 1×1×1, MLP 1~2층):
  d(t) = MLP_shallow(z_t) — 단순 선형/ReLU 변환
  → ∇d/∂Processor 가 "거의 항등 근처" (nearly identity)
  → Physical Loss gradient가 "거울처럼 반사" → Processor로 직달
```

**논거**: 얕은 디코더는 "투명한 유리" — 물리 손실의 신호가 Processor의 Attention 층(공간 결합을 학습하는 층)에 **왜곡 없이** 도달. 따라서 Attention이 정말로 공간 미분(∇²) 을 모사하도록 강제.

**우리 프로젝트 맥락**: 이미 Conv3D 1×1×1 디코더로 "얕은" 결정을 했으므로, B의 논거는 **현재 설계 방향을 강력히 정당화**. 다만 이는 "디코더 조건화 불필요"로 이어지지 않음 — 반대로 "조건화를 디코더에 먹이면 더 효과적" (gradient가 깨끗하므로 조건 신호도 정확히 전파).

#### 핵심 주장 4: 통합 c_t 벡터 방식 (구현 단순화)

**제안**: 
```python
c_t = MLP([p_load(t), sym_type_emb, pos_emb])
→ 모든 Processor 블록 + 디코더에 주입 (FiLM 또는 AdaLN)
```

**장점**: 
- 구현 단순 (1개 MLP + 1개 injection 메커니즘만 설계)
- 정보 간 상호작용 허용 (c_t 내부에서 p_load ↔ sym_type ↔ pos 결합 가능)

**단점**:
- pos_emb는 token-wise (720개 서로 다른 값) 인데, MLP 하나로 압축하면 **공간 식별성 손실** → cell (0,0,0) 과 cell (19,5,5) 의 위치 차이가 c_t에서 구분 안 될 가능성
- A의 주장 (절대위치는 token-wise → re-add 형태가 나음) 과 충돌

**우리 프로젝트 적용**: 통합 c_t는 p_load + sym_type까지만 적용하고, 절대위치는 **별도 token-wise 재주입** 필요. 즉, 혼합 방식.

#### 평가 종합

B의 강점:
- Q1 우려를 수학적으로 증명 (설득력 높음)
- "Heavy Processor, Light Decoder" 철학이 우리 현재 설계와 정렬 (Conv3D 1×1×1 선택 정당화)

B의 약점:
- 통합 c_t 벡터가 token-wise 정보 (절대위치) 에는 부적합 → A와 다른 트레이드오프

---

### 5.3 추가 자문 (Skip Connection 설계 + LAPE 가중치 관리 전략)

> **자문 질문**: 시공간 Attention에서 인코더의 ConditionalLAPE를 skip connection 등으로 병렬 활용 가능한가? 잠재적 위협은? LAPE 가중치를 인코더에서만 업데이트하는 방식이 유효한가?

#### 5.3.1 Skip Connection 방향 타당성 (구조 판단)

**두 답변 모두 "Attention 직전 fresh positional injection" 방향이 타당하다고 판단.**

근거:
- **ClimaX** (Nguyen et al., 2023, ICML 2023): 위치 임베딩 + lead time 임베딩을 ViT backbone에 더한 뒤 전달. Backbone이 위치 정보를 직접 참조하는 패턴. [[proceedings.mlr.press]](https://proceedings.mlr.press/v202/nguyen23a/nguyen23a.pdf)
- **Pangu-Weather** (Bi et al., 2023, Nature): Earth-specific positional bias를 Transformer attention 연산 안에 내장. 단순 입력 add보다 backbone이 공간 정보를 직접 다루는 설계. [[nature.com]](https://www.nature.com/articles/s41586-023-06185-3) — **논문 출처**: Bi, K., Xie, L., Zhang, H., et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619(7970), 533–538. doi:10.1038/s41586-023-06185-3
- **Mamba 위치 정보 처리**: Mamba 계열은 원 저자 구현에서 "positional encoding을 기본 미사용"이며, scan order가 성능에 영향을 준다는 보고 존재 → **Mamba 경로에 위치 정보 강하게 주입하기보다, Attention 경로에서 위치를 보강하는 것이 더 자연스러운** 분업.

**결론**: 조합표의 "절대위치 = Encoder + Attention 재주입, Mamba = 미주입" 구조는 기상 대형 시공간 모델의 설계 패턴과 정합.

#### 5.3.2 위협 A — 신호 스케일 불일치: ReZero 패턴으로 원천 해결

**두 답변 모두 α · lape 형태의 학습 가능 gate 사용을 권고.**

**답변 B의 더 구체적인 해결책**: **ReZero 패턴** (Bachlechner et al., 2021, UAI 2021)
- 구현 (TF2/Keras):
  ```python
  alpha = self.add_weight(name='alpha', shape=(1,),
                          initializer='zeros', trainable=True)
  d_attn_input = d_mamba + alpha * lape   # broadcast
  ```
- **핵심**: `alpha` = **0으로 초기화된 학습 가능 단일 스칼라**
- 효과:
  - 학습 초기: `alpha ≈ 0` → Mamba 동역학만 Attention으로 흘러 안정적 수렴
  - 학습 진행: 모델이 "위치 정보가 필요하다"고 판단할 때 alpha를 스스로 키움
  - 스케일 불일치 문제 원천 해결 (LAPE norm vs Mamba output norm 비율을 α가 자동 조절)

> **논문 출처 (추후 참고문헌)**: Bachlechner, T., Majumdar, B. P., Mao, H. H., Cottrell, G. W., & McAuley, J. (2021). ReZero is All You Need: Fast Convergence at Large Depth. *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 161, 1352–1361.

답변 A의 표현("α·lape 형태의 학습 가능 scalar/vector gate")도 동일한 방향이며, B가 "0 초기화"를 더 명시적으로 정의한 것.

#### 5.3.3 위협 B — Gradient 경로 충돌: detach() 필수 명시

**답변 B의 핵심 추가 지적**: 인코더 LAPE 캐시를 Attention에 주입할 때 **반드시 gradient를 끊어야(detach)** 함.

```python
# 잘못된 방식 (gradient가 인코더로 역전파됨)
lape_cache = encoder_lape(sym_type)

# 올바른 방식 (TF2): 프로세서 loss가 인코더 LAPE를 오염시키지 않음
lape_cache = tf.stop_gradient(encoder_lape(sym_type))
```

**우리 프로젝트 2-phase 전략과의 관계**:
- Phase 2에서 인코더 전체를 freeze하면 → gradient 자동 차단 → detach() 불필요
- 그러나 end-to-end 학습 또는 인코더 일부만 freeze하는 경우 → detach() 명시 필요
- 안전을 위해 **항상 `.detach()` 적용하는 것이 권장** (freeze 여부에 무관)

답변 A는 "frozen prior처럼 취급"이라는 표현으로 같은 개념을 기술했으나, 실제 구현 코드 수준의 명시는 B가 더 구체적.

#### 5.3.4 위협 C — 과도한 위치 정보 주입: 경로 단일화 원칙

두 답변 모두 **동일한 결론**: LAPE skip connection과 RoPE를 동시에 사용하지 말 것.

- 답변 A: "일관된 한 종류의 명시적 위치 경로 유지" 권고. Pangu-Weather, ClimaX 모두 단일 위치 메커니즘 사용.
- 답변 B: "RoPE 또는 LAPE 중 하나 선택. Phase 2에서 LAPE 먼저 시도, 부족하면 RoPE 교체"

**우리 프로젝트 결정**: Phase 2에서 **LAPE skip connection (방안 iii 또는 iv)** 먼저 적용. RoPE는 Phase 4의 선택적 옵션으로 이동.

#### 5.3.5 가중치 관리: 방안 (iv) Decoupled LAPE 추가 제안

**답변 B의 핵심 신규 제안**: 기존 방안 (i)~(iii)에 더해 **방안 (iv) Decoupled LAPE** 제안.

```python
# 방안 (iv) 구현 (TF2/Keras)
encoder_lape  = ConditionalLAPE(dim=128, name='encoder_lape')   # 인코더 전용
processor_lape = ConditionalLAPE(dim=128, name='processor_lape') # 프로세서 전용

# 프로세서 Attention 직전 (block 내부)
self.alpha = self.add_weight(name='alpha', shape=(1,),
                             initializer='zeros', trainable=True)
d_attn_input = d_mamba + self.alpha * processor_lape(sym_type)
```

**방안 (iv)의 이론적 근거**:
- Gradient 충돌 완전 해소 (detach 불필요, 각자 loss에 최적화)
- 프로세서 LAPE가 "Mamba 통과 후 깊은 잠재 공간에서의 Attention 라우팅"에 최적화 가능
- 메모리 낭비 없음: 720 × 128 ≈ 92K 파라미터 추가 (전체 모델 대비 무시 가능)

**비판적 검토 (우리 프로젝트 특수성)**:
- **반론**: ConditionalLAPE가 인코딩하는 정보(격자 물리 좌표 + 대칭 유형)는 "물리적으로 고정된 사실"로, 인코더 입력 단계에서나 깊은 잠재 공간에서나 의미가 변하지 않음. B의 "잠재 공간 최적화" 논거가 물리 도메인에서는 일반 NLP/CV보다 설득력이 약함.
- **장점 유지**: 그럼에도 gradient 완전 독립이라는 구현 안전성 측면은 설득력 있음.
- **2-phase 전략과의 관계**: Phase 2에서 인코더 freeze → encoder.lape 자동 freeze → 방안 (iii)과 방안 (iv)의 gradient 차이가 사실상 사라짐. 차이는 "1st Phase prior를 활용하느냐(iii) vs 새로 학습하느냐(iv)"로 귀결.

**최종 판단**: 방안 (iii) 또는 (iv) 모두 유효. 차이 요약:

| 방안 | Prior 활용 | Cold Start | Gradient 독립 | 구현 복잡도 |
|------|-----------|-----------|-------------|-----------|
| **(iii) Freeze + cache** | O (1st Phase prior 직접 활용) | X | △ (detach 필요) | 낮음 |
| **(iv) Decoupled** | X (새로 학습) | O | O | 낮음 (동일) |

- Phase 2에서 인코더 freeze 가정 시: (iii)이 더 자연스럽고 1st Phase 학습 결과 활용 가능
- 만약 end-to-end 학습 시: (iv)가 더 안전

#### 5.3.6 sym_type FiLM과 processor.lape의 중복 가능성 (B의 핵심 지적)

**B의 주장**: processor.lape(sym_type)이 이미 sym_type 조건부 → Attention에 별도 sym_type FiLM 추가는 중복일 수 있음.

**심화 검토**:
- `processor.lape(sym_type)`: sym_type에 따라 다른 LAPE 테이블 선택 → **위치 임베딩 자체**가 대칭 정보를 인코딩 → 공간적 위치 + 대칭 동시 주입
- `sym_type FiLM(x)`: γ(sym_type) · x + β(sym_type) → **feature 채널 전체를 sym_type에 따라 조절** (multiplicative + additive)

→ 두 연산은 수학적으로 다름 (LAPE add는 bias 주입, FiLM은 feature scaling)

→ **중복이 아니라 보완 관계일 가능성** 있음. 그러나 둘 다 처음부터 적용하면 "효과 측정"이 어려움.

**결론**: Phase 2에서 processor.lape 주입(방안 iii 또는 iv)만 먼저 적용 → 효과 확인 후 Phase 3에서 sym_type FiLM 추가 여부 결정. 이는 Phase 1 계획(Attention-only FiLM)과도 조합 가능.

#### 5.3.7 p_load 디코더 conditioning 메커니즘 수정

**두 답변 모두**: Conv3D 1×1×1 디코더에 LN이 없으므로 AdaLN-Zero 부적합 → FiLM 또는 Concat 권장.

- 답변 A: "FiLM 1순위, AdaLN-Zero 2순위"
- 답변 B: `[d_t, p_load, is_halo]` Concat 후 Conv3D 통과 권장

**비교**:
- FiLM: γ(p_load)·d_t + β(p_load) — 조건에 따른 채널 변조. 채널별 p_load 민감도 학습 가능.
- Concat: [d_t(128ch), p_load(1ch)] → 단순 채널 결합 후 1×1 conv로 투영. 구현 단순.

**권고**: Concat이 구현 단순하고 DiT/DreamerV3 등 선례 많음. Phase 3에서 Concat 먼저 시도, 부족 시 FiLM으로 교체. `is_halo` flag 추가 여부는 §6.5 항목 (1) 참조.

#### 5.3.8 참고 문헌 (논문 출처 기록)

| 구분 | 인용 | 논문 정보 |
|------|------|---------|
| **Pangu-Weather** | Nature 2023 | Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619(7970), 533–538. doi:10.1038/s41586-023-06185-3 |
| **ClimaX** | ICML 2023 | Nguyen, T., Brandstetter, J., Kapoor, A., Gupta, J. K., & Grover, A. (2023). ClimaX: A Foundation Model for Weather and Climate. *Proceedings of the 40th International Conference on Machine Learning (ICML)*. PMLR 202, 25904–25938. |
| **ReZero** | UAI 2021 | Bachlechner, T., Majumdar, B. P., Mao, H. H., Cottrell, G. W., & McAuley, J. (2021). ReZero is All You Need: Fast Convergence at Large Depth. *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 161, 1352–1361. |
| **SkipInit** | NeurIPS 2020 | De, S., & Smith, S. L. (2020). Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 19964–19975. |
| **FiLM** | AAAI 2018 | Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI)*, 3942–3951. |
| **Mamba PE 미사용** | GitHub Discussion | state-spaces/mamba GitHub Issues #51: positional encoding과 scan order 관계 논의 |

---

### 5.4 ConditionalLAPE 시공간 Attention 주입 방안 — Skip connection vs Decoupled LAPE 심화 비교

> **사용자 미결 ①**: 시공간 Attention 직전 위치 정보 주입을 어떤 방식으로 할 것인가?

#### 5.4.1 두 방안 정의 재확인

**방안 (iii) Skip connection (encoder_lape 캐시 + stop_gradient)**:
```python
# TF2/Keras
# 인코더에서 학습된 ConditionalLAPE를 그대로 가져와 재사용
lape_cache = tf.stop_gradient(encoder_lape(sym_type))  # gradient 차단
self.alpha = self.add_weight(name='alpha', shape=(1,),
                             initializer='zeros', trainable=True)
d_attn_input = d_mamba + self.alpha * lape_cache       # 매 attention block에서
```
- encoder_lape 가중치는 1st Phase에서만 학습
- 시공간 프로세서는 이 가중치의 **출력값만 빌려옴** (값은 빌리되, gradient는 차단)
- 추가 파라미터: alpha 스칼라 1개 (per block)

**방안 (iv) Decoupled LAPE (별도 인스턴스 새로 학습)**:
```python
# TF2/Keras
processor_lape = ConditionalLAPE(dim=128, name='processor_lape')  # 신규 인스턴스
self.alpha = self.add_weight(name='alpha', shape=(1,),
                             initializer='zeros', trainable=True)
d_attn_input = d_mamba + self.alpha * processor_lape(sym_type)
```
- encoder_lape와 processor_lape는 **완전 독립** 가중치
- processor_lape는 시공간 학습 단계에서 처음부터 학습 (cold start)
- 추가 파라미터: ConditionalLAPE 약 92K + alpha (per block)

#### 5.4.2 비교 차원 1 — 잠재 공간 의미 변화 가정의 강도

**일반 NLP/CV 도메인 가정**:
- 신경망의 layer가 깊어질수록 feature가 **의미적으로 변환**됨 (low-level edge → high-level concept)
- 따라서 입력 단계의 위치 임베딩과 깊은 layer의 attention이 필요로 하는 위치 정보가 다를 수 있음
- → Decoupled (별도 학습)이 유리한 근거

**우리 프로젝트 (물리 도메인)**:
- ConditionalLAPE가 인코딩하는 정보 = "격자 좌표 (z, y, x) + 대칭 유형 (mirror/rotation)"
- 이 정보는 **물리적으로 고정된 사실** — layer 깊이와 무관하게 의미가 변하지 않음
- cell (3, 2, 1)의 위치는 인코더 입력에서나 Mamba 통과 후에나 동일
- → 일반 NLP/CV의 "잠재 공간 의미 변화" 가정이 우리 도메인에는 약하게 적용

**판단**: 물리 도메인에서는 Decoupled의 잠재 공간 최적화 논거가 일반 도메인보다 약함. **Skip connection이 상대적으로 유리**한 차원.

#### 5.4.3 비교 차원 2 — 2-Phase 학습 전략과의 상호작용

**우리 프로젝트의 2-Phase 학습 전략**:
- Phase 1 (공간 학습): Encoder + Decoder만으로 공간 표현 학습 → ConditionalLAPE도 함께 학습
- Phase 2 (시공간 학습): 인코더 freeze, Mamba+Attention+CRS+Branch 학습

**Skip connection (방안 iii)**:
- Phase 1에서 학습된 ConditionalLAPE의 위치/대칭 prior를 **Phase 2에서 그대로 활용**
- 인코더 freeze 시 자동으로 gradient 차단 (detach() 보험으로만 추가)
- "Phase 1 결과의 직접적 연속성" 확보

**Decoupled LAPE (방안 iv)**:
- Phase 2에서 processor.lape를 **새로 학습 (cold start)**
- Phase 1의 위치/대칭 학습 결과 활용 불가 (다른 가중치)
- 시공간 학습 부담 증가 (위치 임베딩까지 새로 익혀야 함)

**판단**: 2-Phase 전략과 자연스럽게 어울리는 것은 **Skip connection**. Decoupled는 Phase 1 prior를 버리는 비효율.

#### 5.4.4 비교 차원 3 — 수렴 안정성

**Skip connection**:
- alpha=0 초기화 (ReZero) + 정확한 위치 정보 (1st Phase 학습 완료) → 학습 매우 안정
- alpha가 0에서 시작해 점차 위치 정보 비중 증가 → 점진적 적응

**Decoupled LAPE**:
- alpha=0 초기화 + **미학습 LAPE** → 초기에는 무의미한 위치 신호
- alpha가 커져도 LAPE가 학습되지 않은 상태라면 noise 주입 효과
- LAPE와 alpha를 동시에 학습 → 학습 동역학 더 복잡 (chicken-and-egg 문제)

**판단**: **Skip connection이 수렴 안정성 측면에서 명확히 유리**.

#### 5.4.5 비교 차원 4 — 메모리 / 계산 비용

| 항목 | Skip connection | Decoupled LAPE |
|------|----------------|---------------|
| 추가 파라미터 | alpha 스칼라 (per block) | ConditionalLAPE ~92K + alpha |
| 추가 forward 비용 | encoder.lape lookup 1회 (캐시 가능) | processor.lape forward 매 block |
| 메모리 | 무시 가능 | 92K (전체 모델 대비 작음) |

**판단**: Skip connection이 약간 더 효율적이지만, 차이는 실질적으로 무시 가능.

#### 5.4.6 비교 차원 5 — 물리적 해석

**Skip connection**:
- "1st Phase에서 학습한 격자 좌표계를 시공간 학습에서도 동일 사용"
- → "공간 표현 학습 단계에서 익힌 좌표 의미를 시간 모델이 그대로 참조"
- 직관적, 일관성 있음

**Decoupled LAPE**:
- "시공간 처리에 특화된 별도 좌표계를 새로 학습"
- → "같은 격자에 대해 두 가지 좌표 표현이 존재하게 됨"
- 의미 불명확: 격자 (3,2,1)의 위치가 인코더에서와 시공간 모델에서 왜 다르게 표현되어야 하는가?

**판단**: 물리적 해석 측면에서 **Skip connection이 더 자연스럽고 설명 가능**.

#### 5.4.7 비교 차원 6 — 실험 검증 / 디버깅 용이성

**Skip connection**:
- encoder.lape 변경 시 영향이 시공간 모델까지 전파 → 통합적 분석
- "위치 정보가 충분히 활용되는가"의 진단을 alpha gate 학습 추세로 직접 확인 가능

**Decoupled LAPE**:
- encoder.lape와 processor.lape를 독립적으로 분석 가능
- 그러나 두 LAPE가 서로 어떻게 다른지 비교 분석 필요 → 추가 진단 작업

**판단**: Skip connection이 진단/디버깅에서 약간 더 단순.

#### 5.4.8 종합 비교표

| 비교 차원 | Skip connection (iii) | Decoupled LAPE (iv) |
|----------|---------------------|--------------------|
| **잠재 공간 의미 변화 가정** | 물리 도메인에 적합 ✓ | NLP/CV 도메인에 적합 |
| **2-Phase 학습 전략** | 자연스러운 연속 ✓ | Cold start, prior 손실 |
| **수렴 안정성** | 매우 안정 ✓ | 복잡한 학습 동역학 |
| **메모리/계산 비용** | 약간 유리 ✓ | 92K 추가 (무시 가능) |
| **물리적 해석** | 일관성 있음 ✓ | 의미 불명확 |
| **실험 검증 용이성** | 단순 ✓ | 추가 분석 필요 |
| **end-to-end 학습 시 안전성** | detach() 필수 (보험으로 적용) | Gradient 자동 분리 ✓ |
| **유연성** | encoder.lape에 의존 | 시공간 전용 최적화 가능 |

**대다수 차원에서 Skip connection이 우세하지만**, end-to-end 학습 안전성과 시공간 전용 최적화 관점에서는 Decoupled가 유리.

#### 5.4.9 Hybrid 가능성

**전략**: 초기엔 Skip connection으로 시작 → 성능 plateau 시 Decoupled로 전환 또는 추가
```python
# TF2/Keras

# Stage 1: Skip connection만
lape_cache = tf.stop_gradient(encoder_lape(sym_type))
d_attn = d_mamba + self.alpha * lape_cache

# Stage 2 (plateau 후): Skip + 학습 가능 보조 projection
self.proj = tf.keras.layers.Dense(D)  # 작은 추가 모듈
d_attn = d_mamba + self.alpha * self.proj(lape_cache)

# Stage 3 (필요 시): Decoupled로 완전 전환
d_attn = d_mamba + self.alpha * processor_lape(sym_type)
```

**Hybrid 권장 시점**:
- Phase 2에서 (b) mirror/rotation 분리 오차가 plateau를 보이고
- (a) halo cell error도 정체될 때
- → 위치 정보 표현이 부족하다는 신호 → Decoupled 추가 검토

#### 5.4.10 최종 결정 ✅ — Skip connection (방안 iii) + Raw scalar α (2026-04-20)

**결정 사항**:
- **위치 주입 방식**: Skip connection (방안 iii) — encoder 캐시 + stop_gradient
- **Gate 형태**: **Raw scalar α (단일 스칼라)** + Weight Decay (§5.4.12 비교 검토 결과)
- **차원**: 단일 스칼라 (per attention block) — 720 cell 전체에 공평하게 적용

```python
# TF2/Keras — 시공간 프로세서의 매 Attention 블록 직전
class STBlock(tf.keras.layers.Layer):
    def build(self, input_shape):
        # 단일 스칼라 alpha, 0 초기화, 학습 가능
        self.alpha = self.add_weight(
            name='lape_alpha', shape=(1,),
            initializer='zeros', trainable=True
        )

    def call(self, d_mamba, sym_type, encoder_lape):
        lape_cache = tf.stop_gradient(encoder_lape(sym_type))  # gradient 차단
        d_attn_input = d_mamba + self.alpha * lape_cache       # broadcast
        return self.self_attention(d_attn_input)

# Optimizer: weight_decay=1e-4 (alpha 폭주 제어)
```

**결정 근거**:
1. 우리 프로젝트는 **2-Phase 학습 전략** 채택 → Phase 1에서 학습된 LAPE prior를 Phase 2에서 직접 활용 가능
2. **물리 도메인 격자 좌표는 layer 깊이와 무관하게 의미 불변** → Decoupled의 잠재 공간 최적화 논거 약함
3. 수렴 안정성·구현 단순성·물리적 해석 모두 우세
4. ReZero alpha gate가 학습 가능 → 시공간 적응 여지 충분
5. 외부 자문도 (iii) 권고 (§5.4.11 참조)

**2-Phase 학습과의 관계**:
- Phase 1: encoder + decoder 학습 → ConditionalLAPE도 함께 학습
- Phase 2: 인코더 freeze (`requires_grad=False`) + Mamba/Attention/CRS/Branch 학습
  - encoder.lape 자동 freeze → gradient 자동 차단
  - **detach()는 이중 보험** (freeze 여부와 무관한 안전장치)

**조건부 후속 검토 (Phase 2 평가 후 재검토)**:
- (b) mirror/rotation 분리 오차 또는 (a) halo cell error가 plateau 시
- → Hybrid (alpha 외에 학습 가능 작은 projection 추가) 또는 (iv) Decoupled 전환 고려
- → 5.4.9 Hybrid 전략 단계적 적용

#### 5.4.11 외부 자문 의견 정리 (2026-04-20)

> 본 결정 과정에서 받은 외부 자문 의견 요약 — 우리 §5.4 분석과 정합

**방안 (iii) 지지 논거**:
- **Skip connection 일반 이점**: long skip이 저수준 정보를 직접 전달하여 feature reuse + 학습 안정화. LAPE는 "저수준 spatial prior"에 가까우므로 Attention 직전 직접 전달이 **U-Net의 long skip이 공간 정보 복원하는 직관과 동일**.
- **역할 분리 명확성**: detach된 cache 사용 시 Phase 2의 CRS/Branch loss가 인코더 LAPE 흔들지 못함 → 인코더와 프로세서의 학습 책임 분리.
- **Frozen prior 모듈성**: 큰 본체의 prior를 유지한 채 새 모듈만 학습하는 접근이 모듈성·안정성 측면 유리 (transfer learning 일반 패턴).
- **해석**: "인코더가 배운 위치표는 유지하고, 프로세서는 그 위치표를 활용하는 법만 배운다"

**방안 (iv) 지지 문헌 (참고용 — 채택 안 함)**:

| 논문 | 핵심 주장 | 우리 도메인 적용 한계 |
|------|----------|------------------|
| **LaPE** (Layer-adaptive Position Embedding) | ViT의 각 layer가 동일 PE를 쓰는 방식의 한계 지적, layer-adaptive PE 제안 | NLP/CV에서는 layer별 의미 변화 강함. 물리 도메인은 격자 좌표 의미 불변 → 적용 의의 약함 |
| **TAPE** (Transformer Adaptive Position Encoding) | PE를 입력 1회성 신호가 아니라 층을 거치며 문맥화되는 신호로 다룸 | 우리 ConditionalLAPE는 alpha gate 학습으로 어느 정도 적응성 확보 → 굳이 PE 자체 학습 불필요 |

**한계 요약 (방안 iv)**:
- Processor용 positional module을 처음부터 다시 학습 → Phase 1 prior 활용 불가
- Encoder와 processor가 유사한 위치 개념을 이중 학습할 가능성
- "더 맞춤형이지만, 더 많은 자유도와 더 많은 불확실성"

**결론**: 외부 자문도 본 단계에서는 (iii) 권고. (iv)는 추후 plateau 시 검토 옵션으로 유지.

#### 5.4.12 Gate 형태 비교 및 결정 (2026-04-20)

> 사용자 명시 제약 + 2026 최신 논문 검토 결과 종합

##### 사용자 명시 제약 (3가지)

1. **Input-independent**: Query/입력에 의존하지 않음
2. **공간 공평**: (20, 6, 6) = 720 cell 전체에 단일 스칼라 동일 적용
3. **공간별 스케일 차등 금지**: cell별 차등 시 위치 좌표계 일관성 깨짐

→ 자동 탈락:
- Channel-wise α (D-dim vector) — 채널 차등
- Token-wise gate — cell별 차등
- Gated Attention (Qwen, NeurIPS 2025) — Query 의존 + element-wise

##### 제약 만족 후보 4가지 비교

| 후보 | 형태 | 시작값 0 | Bounded | 표현력 | 단순성 | 우리 적합도 |
|------|------|--------|---------|-------|-------|----------|
| **ⓐ Raw α + WD** | `α` (학습 + weight decay) | ✓ | ✗ (자유) | unbounded | ★★★★ | **★★★★★** |
| ⓑ Tanh(α) | `tanh(α)` (Flamingo 패턴) | ✓ | [-1, 1] | bounded | ★★★ | ★★★ |
| ⓒ Scaled Tanh | `s · tanh(α/s)` | ✓ | [-s, s] | bounded 확장 | ★★ | ★★★ |
| ⓓ LayerNorm + α | `α · LN(lape)` | ✓ | LN 정규화 | unbounded | ★★ | ★★★★ |

##### 결정: **방안 ⓐ Raw α + Weight Decay (ReZero 원형)**

```python
# TF2/Keras
class STBlock(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='lape_alpha', shape=(1,),
            initializer='zeros', trainable=True
        )

    def call(self, d_mamba, lape_cache):
        return d_mamba + self.alpha * lape_cache

# Optimizer 설정 (weight_decay=1e-4 표준값)
optimizer = tf.keras.optimizers.AdamW(learning_rate=..., weight_decay=1e-4)
```

##### 결정 근거

1. **사용자 제약 모두 충족** ✓
   - 단일 스칼라, input-independent, 공간 공평
2. **Tanh/sigmoid의 saturation 회피**
   - bounded 함수의 saturation 영역 (vanishing gradient) 자체를 사용 안 함
   - LAPE norm이 작을 경우 큰 alpha가 필요할 수 있음 → bounded 부적합
3. **Mamba 출력 norm 변동에 대응**
   - Mamba 출력은 시점/layer별 norm 변동 → bounded gate로는 충분히 보상 못 할 가능성
   - Raw α는 그 변동을 자유롭게 흡수
4. **2026 최신 트렌드 정합**
   - DeepSeek mHC (2026): bounded gate가 아닌 manifold 제약으로 안정성 확보
   - TRANSPONDER (2025): zero-initialized learnable scalar (ReZero 패턴 회귀)
   - 핵심 트렌드: "bounded gate보다 다른 안정화 메커니즘 (WD, manifold 제약 등) 활용"
5. **Weight Decay로 폭주 제어**
   - α에 대한 L2 regularization (1e-4 표준)
   - α가 너무 커지지 않도록 자연스러운 제어
6. **가장 단순, 가장 검증된 패턴**
   - ReZero (Bachlechner et al., 2021, UAI) 원형
   - 검증된 채택: GPT-J, ResNet 변형, 깊은 Transformer

##### 학습 모니터링 항목 (Phase 2)

학습 과정에서 다음을 확인하여 변형 필요성 판단:

| 관찰 | 의미 | 대응 |
|------|------|------|
| α가 0 근처에 머무름 | LAPE 정보 활용도 낮음 | 정상 (잠재 공간이 위치 정보 자체적 학습 중일 수 있음) |
| α가 양수로 안정 수렴 | LAPE 정보 활용 정상 | 유지 |
| α가 음수로 학습 | LAPE를 noise로 처리 (이상 신호) | 점검: LAPE 정합성 또는 sigmoid 양수 강제 검토 |
| α 값 폭주 (`> 10`) | WD 부족 또는 LAPE norm 너무 작음 | WD 강화 (1e-3) 또는 ⓓ LayerNorm 적용 |
| α의 layer별 변동 매우 큼 | layer별 LAPE 활용 차등 큼 | 정상 (Adaptive Skip 2024 발견) |

##### 보조/후속 옵션

- **Phase 2 plateau 시**: 방안 ⓓ LayerNorm(LAPE) + α 적용 (스케일 안정화)
- **만약 α가 음수 학습 시**: sigmoid(α - bias) 양수 강제로 변경
- **End-to-end 학습 전환 시**: 방안 (iv) Decoupled LAPE 검토

##### 참고문헌 (gate 메커니즘)

| 논문 | 출처 | 우리에 시사점 |
|------|------|-------------|
| **ReZero** | Bachlechner et al., 2021, UAI | Raw scalar α (0 init) 원형 |
| **Flamingo gated cross-attn** | Alayrac et al., 2022, NeurIPS | Tanh(α) bounded gate (cross-modal) |
| **ControlNet zero conv** | Zhang et al., 2023, ICCV | 외부 정보 zero-init 주입 |
| **Adaptive Skip Connections** | Ryan, 2024 | Layer별 alpha 차등 학습 발견 |
| **Gated Attention (Qwen)** | NeurIPS 2025 Best Paper | Input-dependent gate (우리 제약 위배, 참고만) |
| **GateSkip** | arXiv:2510.13876, 2025 | Sigmoid-linear gate (대안적 접근) |
| **TRANSPONDER** | preprint, 2025 | ReZero 변형 + per-block scalar |
| **DeepSeek mHC** | arXiv:2512.24880, 2026 | Manifold 제약 (gate 대신) |
| **Deep Delta Learning** | preprint, 2025 | Scalar β로 geometric transform 제어 |

---

### 5.5 p_load 주입 전략 — 이중 경로 (Mamba concat + Attention AdaLN-Zero, 공유 임베더) 결정 (2026-04-20)

> **상세 검토**: [23_p_load_Mamba_주입_3D_Spatiotemporal_사례조사(2026-04-20).md](적용%20가능%20기술%20검토/추가%20조사/23_p_load_Mamba_주입_3D_Spatiotemporal_사례조사(2026-04-20).md)

#### 5.5.1 결정 개요

| 항목 | 결정 |
|------|------|
| **주입 방식** | 이중 경로 (Mamba concat + Attention AdaLN-Zero) |
| **임베더 구조** | 공유 임베더 + 양방향 gradient (detach 없음) |
| **공간 적용** | 720 cell 전체 균일 broadcast (전역 동일) |
| **차선책** | embed_dim 확장 → AdaLN 제거 → AdaLN detach → 별도 임베더 |

#### 5.5.2 결정 근거 — p_load의 단일 물리 의미

**물리적 정정 (사용자 통찰)**:
- 도플러 피드백, peaking 억제는 **국소(Local) 현상** — cell별 T_fuel, xs_fuel 등에 의존, 중앙 제어 p_load와 직접 관련 없음
- p_load의 진정한 의미: **단일 의미** (노심 전체 출력 스케일 + 시간 가속 페달)
- 시간/공간 의미 분리되지 않음

**결과**:
- 별도 임베더 (방안 3) 근거 (시간/공간 다른 의미) **무효**
- 공유 임베더 + 양방향 gradient = **시너지** (충돌 아님)
- 두 경로 모두 동일 목표: "robust한 p_load 표현 학습"

#### 5.5.3 ConditionalLAPE 다층 주입 패턴과의 비교

| 정보 | 주입 위치 1 | 주입 위치 2 | 임베더 | 학습 시점 | 공간 적용 |
|------|-----------|-----------|------|---------|---------|
| **위치/대칭** (ConditionalLAPE) | 인코더 add | Attention skip (ReZero α) | 단일 prior | Phase 1 → freeze | cell별 다름 (token-wise) |
| **p_load** | Mamba concat | Attention AdaLN-Zero | 공유 + 양방향 gradient | Phase 2 학습 | **모든 cell 동일** (전역 균일) |

**공통**: 정보 다중 주입 (단절 방지)
**차이**: 학습 시점 (freeze vs 학습 중), 공간 적용 (cell별 vs 균일)

#### 5.5.4 구현 (TF2/Keras)

```python
class STBlock(tf.keras.layers.Layer):
    def __init__(self, dim=128, p_load_embed_dim=8, num_cells=720):
        super().__init__()
        self.num_cells = num_cells
        # 공유 임베더 (양방향 gradient 수렴)
        self.p_load_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(p_load_embed_dim, activation='gelu'),
            tf.keras.layers.Dense(p_load_embed_dim),
        ], name='p_load_encoder')
        # AdaLN-Zero (γ, β 0 초기화)
        self.adaln_proj = tf.keras.layers.Dense(
            2 * dim, kernel_initializer='zeros', name='adaln_proj'
        )
        self.mamba = MambaLayer(...)
        self.attention = Attention(...)
        self.ln_no_affine = tf.keras.layers.LayerNormalization(
            center=False, scale=False
        )

    def call(self, z_t, p_load):
        # 공유 임베딩
        p_load_emb = self.p_load_encoder(p_load)               # (B, T, embed_dim)

        # Mamba 측 — 720 cell 균일 broadcast 후 concat
        p_load_b = tf.tile(
            p_load_emb[:, :, None, :], [1, 1, self.num_cells, 1]
        )                                                       # (B, T, 720, embed_dim)
        mamba_in = tf.concat([z_t, p_load_b], axis=-1)          # 모든 cell 동일 p_load
        d_mamba = self.mamba(mamba_in)

        # Attention 측 — AdaLN-Zero (detach 없음, 양방향 gradient)
        gamma_beta = self.adaln_proj(p_load_emb)                # (B, T, 2*dim)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)
        gamma = gamma[:, :, None, :]                             # cell 차원 broadcast
        beta = beta[:, :, None, :]

        x_norm = self.ln_no_affine(d_mamba)
        x_modulated = x_norm * (1 + gamma) + beta               # 모든 cell 동일 변조

        return d_mamba + self.attention(x_modulated)
```

**핵심 설계 포인트**:
- 공유 `p_load_encoder`: 단일 모듈, 양방향 gradient
- `tf.tile(...)`: 공간 균일 broadcast 명시 (720 cell 동일 값)
- `adaln_proj` 0 초기화: AdaLN 효과 점진 도입
- `tf.stop_gradient` 미사용: 시너지 양방향 gradient

#### 5.5.5 단계적 도입 (Phase 2 → 3)

- **Phase 2**: Mamba concat 단독 (AdaLN-Zero로 자연스럽게 비활성)
- **Phase 3**: AdaLN 활성화 (γ, β 학습 진행)

#### 5.5.6 차선책 우선순위

| 우선순위 | 증상 | 차선책 |
|--------|------|------|
| 1 | (c) 오차 plateau, 표현 용량 부족 | embed_dim 확장 (8→16→32) |
| 2 | AdaLN 효과 없음 (γ,β ≈ 0) | AdaLN 제거 (Mamba concat 단독) |
| 3 | 학습 발산, gradient 불안정 | 방안 2 (AdaLN 측 detach) |
| 4 | 양 경로 plateau (표현 충돌) | 방안 3 (별도 임베더) |

#### 5.5.7 모니터링 항목

1. p_load_encoder gradient norm (사망 여부)
2. adaln_proj 출력 (γ, β) 분포 (활성화 정도)
3. p_load 구간별 예측 오차 (정/감/저/상승)
4. 양방향 gradient 균형 (Mamba 측 vs AdaLN 측 비율)

---

## 6. 종합 검토 및 본 프로젝트 적용 권고

### 6.1 두 답변의 공통점

#### (1) Q1 우려 유효성에 대한 동의

| 항목 | 답변 A | 답변 B |
|------|--------|--------|
| Mamba B_t 게이팅이 정적 신호를 희석하는가? | "의견 암시" (sym/위치 메커니즘 분리 필요 → 희석 인정) | 수학적으로 100% 증명 |
| 현재 encoder 1회 add 만으로는 충분한가? | 미흡 | 미흡 |
| 매 stage 재주입이 필요한가? | 필요 (메커니즘은 정보별로 다르게) | 필요 (PDE 물리 표준) |

**결론**: 두 답변 모두 기존 결정 (encoder ConditionalLAPE 1회 add, 이후 미주입) 을 **재평가 필요**로 판단.

#### (2) "정적 메타데이터 (sym_type, 절대위치) 의 다단계 주입" 이 피할 수 없는 설계 선택

- A: "메커니즘 설계의 세밀함" 강조 (정보 종류별로 다르게)
- B: "PDE 물리 표준" 강조 (모든 stage)
- 결과: 추상적으로는 다르지만, **구체적 실행에서는 합치**

### 6.2 두 답변의 주요 차이점

#### (1) 절대위치 (token-wise 좌표) 처리 전략

| 측면 | 답변 A | 답변 B |
|------|--------|--------|
| **메커니즘** | Token-wise re-add (또는 Attention RoPE) | Token-wise를 포함한 통합 c_t 벡터 (MLP로 압축) |
| **이유** | token-wise 다양성 보존 필요 | 구현 단순화 우선 |
| **위험** | 구현 복잡도 ↑ | 공간 해상도 손실 ↑ |

**평가**: A의 주장이 더 물리적으로 정밀함. Token-wise 좌표를 단일 c_t로 압축하면 (19,5,5) 위치의 특이성이 사라질 수 있음. 다만 실험적으로 검증 필요.

#### (2) p_load 주입 위치 (Mamba vs Decoder)

| 측면 | 답변 A | 답변 B |
|------|--------|--------|
| **기존 잠정 결정 평가** | (재검토 제안) | 유지 + Decoder도 추가 (필수) |
| **Decoder 조건화** | 선택적 / 필요시만 | 필수 (conv3d 이전 concat/FiLM) |
| **근거** | 정적/동적 메커니즘 분리 관점 | Gradient 투명성 논거 |

**평가**: B의 "decorator도 조건화하라" 주장이 설득력 있음. 이미 얕은 디코더를 선택했으므로, gradient가 깨끗하게 흐를 것 — 조건화 신호도 정확히 전파될 것이라는 논거.

#### (3) 구현 철학: "세밀한 분화" vs "통합 단순화"

| 측면 | 답변 A | 답변 B |
|------|--------|--------|
| **철학** | 정보 종류별 최적 메커니즘 선택 | 통합 설계로 구현 단순화 |
| **예시** | c_sym (FiLM) + c_pos (re-add) + c_rod (별도) + p_load (concat) | c_t = MLP([p_load, sym_type_emb, pos_emb]) → 모든 곳에 일관 주입 |
| **장점** | 각 정보의 특성에 최적화 | 코드 통일, 유지보수 쉬움 |
| **단점** | 구현 복잡, 상호작용 예측 어려움 | 정보 종류별 정밀도 손실 가능 |

### 6.3 본 프로젝트 최종 권고 (Hybrid 접근)

두 답변의 차이를 모두 수용하되, **정보 특성에 따라 적응적으로 선택** 하는 혼합 방식을 권고:

#### 전략: "B의 통합 c_t (p_load + sym_type) + A의 token-wise 절대위치 re-add"

```
정보 3종별 주입 전략:

1. p_load (동적 스칼라, 시간 변동)
   - 위치: Mamba 입력 + Attention 입력 (매 시점 반복)
   - 메커니즘: Concat (기존 잠정 결정 유지)
   - 근거: 동적 변수는 매 시점 변동이 자동 재주입 역할

2. sym_type (정적 이산, 시퀀스 단위 상수)
   - 위치: Processor 매 블록 입력 + Decoder 입력
   - 메커니즘: FiLM (또는 AdaLN)
   - 상세: sym_type_emb = Embedding(sym_type, dim=32) → MLP → γ(sym), β(sym) → FiLM modulation
   - 근거: 전역 스칼라 메타데이터 → FiLM/AdaLN이 설계적으로 최적

3. 절대위치 (정적 격자, token-wise 다름)
   - 위치: Attention 블록 입력 직전 (매 Processor 블록마다) — **Skip connection으로 원본 LAPE 신호 주입**
   - 메커니즘: Token-wise Skip Connection (인코더 LAPE 캐시 활용) + 선택적 FiLM
   - 상세:
     ```
     [인코더 단계]
     x_input (21ch) + halo_expand
     → CellEmbedder (21→128)
     → LAPE 계산: lape_emb = LAPE(sym_type)[cell_indices]
     → z = cnn_features + lape_emb
     → 3개 FullAttn stage
     → z(t) ∈ (B, 720, 128) 출력
     
     [캐시 저장]
     lape_cache[sym_type] = lape_emb  # (B, 720, D) 저장
     
     [시공간 프로세서]
     Mamba: z(t) → d_mamba(t)  # 희석된 위치 정보 포함
     
     Attention 직전:
     d_attn_input = d_mamba(t) + lape_cache[sym_type]  # Skip connection (신선한 위치 신호)
     Attention(d_attn_input) → d(t)
     ```
   - 근거: 
     * Mamba에서 희석된 위치 정보를 보완하면서도
     * Mamba 처리 결과 (공간 동역학 상호작용)도 함께 활용
     * Token-wise 차이 완벽히 보존 (원본 LAPE는 왜곡 없음)
     * DreamerV3, Video transformer 등에서 검증된 패턴

4. Rod Map 통계 (선택, Phase 2+)
   - 위치: Processor 초기 블록 또는 Mamba concat
   - 메커니즘: c_rod = MLP(rod_statistics) → concat to Mamba input
   - 우선순위: 낮음 (Phase 1에서는 기본 rod_map 1ch 사용)
```

#### 정보 × 위치 × 메커니즘 최종 조합표

| 정보 | 인코더 | 시공간 Mamba | 시공간 Attention | 디코더 | 메커니즘 |
|------|--------|-------------|----------------|--------|---------|
| **p_load** | - | **Mamba concat ✓** ② | **AdaLN-Zero ✓** ② | (미주입 ✓) ③ | 공유 임베더 + 양방향 gradient + 720 cell 균일 broadcast |
| **sym_type** | ConditionalLAPE halo_expand ✓ | (별도 미주입 ✓) ④ | (별도 미주입 ✓) ④ | (별도 미주입) | ConditionalLAPE에 위임 (Phase 2 후 재검토) |
| **절대위치** | ConditionalLAPE add (1회) ✓ | - | **Skip connection ✓** ① | - | tf.stop_gradient(encoder_lape) + Raw α (단일 스칼라) + WD |
| **rod_map** | 입력 21ch ✓ | (Phase 4 검토) | - | - | 통계 인코딩 (선택) |

**범례**: ✓ = 결정 완료 (모든 항목 결정 완료, 2026-04-20)

**결정 항목 상세**:
- **① 절대위치 시공간 Attention 주입 방식 — 결정 완료 ✅**: Skip connection (방안 iii) + Raw α + WD (§5.4.10, §5.4.12)
- **② p_load Mamba/Attention 주입 형태 — 결정 완료 ✅**: 이중 경로 (방안 1)
  - **Mamba**: 공유 임베더 (MLP 1→8) → 720 cell 균일 broadcast → concat
  - **Attention**: 같은 공유 임베더 → AdaLN-Zero (γ, β 0 초기화) → 720 cell 균일 변조
  - **공유 임베더**: 양방향 gradient (detach 없음 — p_load 단일 의미라 시너지)
  - 차선책: embed_dim 확장 → AdaLN 제거 → AdaLN detach → 별도 임베더
  - 상세: §5.5
- **③ p_load 디코더 주입 — 결정 완료 ✅**: 미주입 (Mamba에서 이미 반영, 이중 책임 + Shortcut Learning 회피)
- **④ sym_type 별도 주입 — 결정 완료 ✅ (잠정)**: 미주입 (ConditionalLAPE가 sym_type 조건부 처리). Phase 2의 (b) mirror/rotation 분리 오차 측정 결과에 따라 재검토.

### 6.4 실행 단계별 Plan

#### Phase 1 (현재 기본선)
- 확정: 인코더 ConditionalLAPE (encoder 입력 단계 1회 add)
- 미추가 (Phase 2에서 추가): p_load 주입, 시공간 Attention LAPE 재주입
- 목표: 기본 인코더-Mamba 학습 안정성 확인 (validation/test curve 평탄화)

#### Phase 2 (시공간 학습 본격화 — LAPE skip + p_load Mamba concat)
**(a) 절대위치 skip connection (LAPE)** (§5.4.10):
- 추가: ReZero gated re-add (Attention 블록 직전)
  ```python
  lape_cache = tf.stop_gradient(encoder_lape(sym_type))
  d_attn = d_mamba + self.alpha * lape_cache  # alpha: 단일 스칼라, 0 초기화
  ```
- Gate 형태: **Raw scalar α + Weight Decay** (§5.4.12)
  - Optimizer: `tf.keras.optimizers.AdamW(weight_decay=1e-4)`

**(b) p_load Mamba 주입** (§5.5):
- 추가: 공유 임베더 + 720 cell 균일 broadcast + concat
  ```python
  p_load_emb = self.p_load_encoder(p_load)  # (B, T, 8)
  p_load_b = tf.tile(p_load_emb[:, :, None, :], [1, 1, 720, 1])
  mamba_in = tf.concat([z_t, p_load_b], axis=-1)
  ```
- AdaLN-Zero는 자연스럽게 비활성 (γ=β=0 시작)

**평가**: 6개 지표
- 기본: (a) halo cell error, (b) mirror/rotation 분리 오차, (c) p_load 구간 오차, (d) steady/transient 분해
- 추가: (e) α gate 학습 추세 (LAPE), (f) ‖α·lape‖ / ‖d_mamba‖ 비율

**목표**:
- LAPE skip 실제 위치 정보 기여 정량화
- p_load concat 단독 효과 측정 (AdaLN 없이)

#### Phase 3 (Attention AdaLN-Zero 활성화)
- 추가: p_load Attention AdaLN-Zero (§5.5.4) — 공유 임베더 출력을 adaln_proj에 통과
  ```python
  gamma_beta = self.adaln_proj(p_load_emb)  # 0 초기화 → 점진 활성화
  ```
- 비교: Phase 2 (concat only, AdaLN γ≈β≈0) vs Phase 3 (concat + AdaLN active)
- (선택) sym_type 추가 주입 검토: Phase 2 (b) mirror/rotation 분리 오차 결과에 따라 결정
- 평가: AdaLN 활성화가 (c) 구간별 오차에 기여하는지 정량화
- **결정 분기** (§5.5.5, §5.5.6 차선책 매트릭스):
  - AdaLN 효과 명확 → 이중 경로 채택
  - AdaLN 효과 미미 → AdaLN 제거 (concat 단독 유지)
  - 학습 발산 → AdaLN detach (방안 2)
  - 표현 충돌 → 별도 임베더 (방안 3)

#### Phase 4 (선택적: Rod Map, RoPE)
- 평가 기준: Phase 1~3 이후 성능 plateau 도달 시
- 우선순위: 낮음 (Branch 예측 정확도 주요 지표 아닐 경우)

### 6.5 미결 사항 & 후속 검토

> **2026-04-20 사용자 명시 미결 항목** (조합표 기준)

#### (사용자 미결 ①) 절대위치 시공간 Attention 주입 방식 — 결정 완료 ✅ (2026-04-20)
- **결정**: Skip connection (방안 iii) + **Raw scalar α (단일 스칼라) + Weight Decay**
- **구현 (TF2)**:
  ```python
  # Attention 블록 직전 (block 내부)
  self.alpha = self.add_weight(
      name='lape_alpha', shape=(1,),
      initializer='zeros', trainable=True
  )
  lape_cache = tf.stop_gradient(encoder_lape(sym_type))
  d_attn = d_mamba + self.alpha * lape_cache
  
  # Optimizer
  optimizer = tf.keras.optimizers.AdamW(learning_rate=..., weight_decay=1e-4)
  ```
- **세부 결정 근거**:
  - 주입 방식: §5.4 (Skip vs Decoupled 6차원 비교) — Skip 채택 (2-Phase 전략 + 물리 도메인)
  - Gate 형태: §5.4.12 — Raw α + WD 채택 (사용자 제약 + Tanh saturation 회피)
- **Tanh/sigmoid 비채택**: bounded saturation 문제, Mamba 출력 norm 변동 흡수 한계
- **WD 역할**: alpha 외 모든 학습 가중치에 적용되는 일반 정규화. 폭주 자연 제어 (alpha만을 위한 추가 비용 아님)
- **보조 옵션 (Phase 2 plateau 시 검토)**:
  - α 폭주 시 (>10): WD 1e-3으로 강화 또는 LayerNorm(LAPE) + α 적용
  - α가 음수 학습 시: sigmoid(α - bias) 양수 강제로 변경
  - End-to-end 학습 전환 시: 방안 (iv) Decoupled LAPE 검토

#### (사용자 미결 ②) p_load Mamba/Attention 주입 형태 — 결정 완료 ✅ (2026-04-20)
- **결정**: **이중 경로 (방안 1) — 공유 임베더 + Mamba concat + Attention AdaLN-Zero**
  - **Mamba**: 공유 임베더 (MLP 1→8) → 720 cell 균일 broadcast → concat
  - **Attention**: 같은 공유 임베더 → AdaLN-Zero (γ, β 0 초기화) → 720 cell 균일 변조
  - **공유 임베더**: 양방향 gradient (detach 없음)
- **결정 근거**: p_load는 단일 물리 의미 (전역 출력 스케일 + 시간 가속 페달)
  - 도플러/peaking 억제는 국소 현상 → p_load와 직접 관련 없음 → 시간/공간 의미 분리 안 됨
  - 두 경로 모두 동일 목표 (robust한 p_load 표현) → gradient 시너지
- **상세**: §5.5 (요약), [23번 조사 문서](적용%20가능%20기술%20검토/추가%20조사/23_p_load_Mamba_주입_3D_Spatiotemporal_사례조사(2026-04-20).md) §6.5 (전체)
- **차선책 (1순위 미작동 시)**:
  1. 표현 용량 부족 → embed_dim 확장 (8→16→32)
  2. AdaLN 효과 없음 → AdaLN 제거 (concat 단독)
  3. 학습 발산 → 방안 2 (AdaLN detach)
  4. 표현 충돌 → 방안 3 (별도 임베더)
- **모니터링**: p_load_encoder gradient norm, γ/β 분포, 구간별 오차, 양방향 gradient 균형

#### (사용자 미결 ③) p_load 디코더 주입 여부 — 결정 완료 ✅ (2026-04-20)
- **결정**: **미주입** (확정)
- **근거**: Mamba에서 이미 p_load 반영 (concat) → d(t)에 정보 내포 → 디코더 추가 주입 시 이중 책임 + Shortcut Learning 위험
- **재검토 트리거 (예외 시)**: Phase 2의 (c) p_load 구간별 보정 오차 측정 결과 d(t)에 p_load 정보 손실 확인 시 (가능성 낮음)

#### (사용자 미결 ④) sym_type 별도 주입 여부 — 결정 완료 ✅ (잠정, 2026-04-20)
- **결정**: **미주입** (잠정 확정)
- **근거**: ConditionalLAPE가 sym_type 조건부로 위치 임베딩을 분기 → sym_type 정보가 위치 신호 안에 내포
- **재검토 트리거**: Phase 2의 (b) mirror/rotation 분리 오차 측정 결과 sym 정보 부족 확인 시 → 별도 FiLM 등 추가 검토

---

> 아래는 본 문서 작성 과정에서 누적된 일반 미결 사항 (자문 검증 결과 일부 해결됨)

#### (1) 절대위치 Skip Connection 방식의 잠재적 위협 및 해결책 ✅

> §5.3 추가 자문에서 세 위협 모두 해결책이 도출됨.

| 위협 | 원인 | 해결책 | 근거 |
|------|------|--------|------|
| **(A) 스케일 불일치** | Mamba 출력 norm ≠ LAPE embedding norm | **ReZero**: `alpha * lape`, alpha=0 초기화 학습 가능 스칼라 (Bachlechner et al., 2021, UAI) | 학습이 alpha를 스스로 조절 → 스케일 자동 보정 |
| **(B) Gradient 충돌** | skip connection이 인코더 LAPE 가중치를 프로세서 loss로 오염 | **detach()**: `lape_cache = lape(sym_type).detach()` | 프로세서 loss가 LAPE 파라미터로 역전파 차단 |
| **(C) 위치 정보 중복** | LAPE skip + RoPE 동시 적용 시 간섭 | **경로 단일화**: LAPE skip 또는 RoPE 중 하나만 (Phase 2에서 LAPE 먼저 시도) | ClimaX, Pangu-Weather 모두 단일 위치 메커니즘 |

**확정된 구현 패턴 (TF2/Keras)**:
```python
# Attention 블록 직전 (각 Processor Block)
class STBlock(tf.keras.layers.Layer):
    def build(self, input_shape):
        # 단일 스칼라 alpha, 0 초기화 학습 가능
        self.alpha = self.add_weight(
            name='lape_alpha', shape=(1,),
            initializer='zeros', trainable=True
        )

    def call(self, d_mamba, lape_value):
        lape_cache = tf.stop_gradient(lape_value)        # gradient 차단
        d_attn_input = d_mamba + self.alpha * lape_cache  # broadcast
        return self.self_attention(d_attn_input)
```

**추가 실험 지표** (§5.3.2에서 추가 권고):
- Attention 입력 norm 대비 주입된 LAPE norm 비율 (α·‖lape‖ / ‖d_mamba‖)
- α gate의 학습 추세 (초반 폭주 여부 확인)

#### (2) ConditionalLAPE 레이어의 가중치 관리 전략 ✅

> §5.3 추가 자문에서 방안 (iv)가 추가됨. 총 4가지 방안 비교.

**4가지 방안 비교표**:

| 방안 | 개요 | 1st Phase prior 활용 | Gradient 독립 | 잠재 공간 최적화 | 결정 |
|------|------|------------------|------------|--------------|------|
| **(i) 가중치 공유** | encoder.lape = processor.lape (동일 인스턴스) | O | X (상충 심각) | X | ❌ 금지 |
| **(ii) 가중치 분리** | 프로세서 전용 LAPE 선언, but 사용 안 함 | X | O | X (미학습) | ❌ 낭비 |
| **(iii) Freeze cache** | encoder.lape 캐시, detach() 적용 | O | O (detach) | X (고정) | ✅ 권고 (2-phase 전략) |
| **(iv) Decoupled** | encoder.lape + processor.lape 완전 독립 학습 | X (cold start) | O | O (별도 최적화) | ✅ 고려 가능 |

**방안 (iii) 권고 근거**:
- 2-phase 전략에서 Phase 1에서 학습된 "격자 물리 좌표 + 대칭 유형" prior가 Phase 2에서도 물리적으로 유효
- ConditionalLAPE가 인코딩하는 정보(위치 + 대칭)는 레이어 깊이와 무관하게 의미 불변 → B의 "잠재 공간 최적화" 논거의 물리 도메인 유효성 제한적
- Phase 2에서 인코더 freeze → 자동 gradient 차단 → detach() 실질적 보험

**방안 (iv) 권고 근거**:
- end-to-end 학습 또는 인코더 일부만 freeze하는 경우 gradient 안전
- 메모리 추가: 720 × 128 ≈ 92K 파라미터 (모델 전체 대비 무시 가능)
- 구현 구체적 코드 (답변 B 제공):
```python
encoder.lape = ConditionalLAPE(dim=128)    # 인코더 전용
processor.lape = ConditionalLAPE(dim=128)  # 프로세서 전용 (독립 학습)
alpha = nn.Parameter(torch.zeros(1))       # ReZero

d_attn_input = d_mamba_t + alpha * processor.lape(sym_type)
```

**최종 권고**:
- **Phase 1~2 (2-phase 전략, 인코더 freeze)**: 방안 (iii) 권장 — 1st Phase prior 활용, 구현 단순
- **만약 end-to-end 학습 계획 변경 시**: 방안 (iv) 전환 (gradient 안전성)
- **공통 적용**: ReZero (alpha=0 초기화) 반드시 사용
- **점검 지표**: α gate 학습 추세, gradient magnitude (LAPE 레이어), halo cell error, mirror/rotation 분리 오차

#### (3) Token-wise 벡터를 MLP로 압축할 때의 정보 손실 정량화
- A와 B의 충돌점: 절대위치를 c_t에 포함시키면 공간 해상도 손실
- Skip connection 방식 택할 경우 (iii) 이 문제는 회피, 하지만 대안 방식도 Phase 2~3에서 비교 검증 추천
- 실험 설계: 
  * Baseline: 절대위치 미주입 (현재)
  * 방법1: Skip connection + freeze (제안 방식)
  * 방법2: Token-wise re-add 반복 계산 (A의 순수 방식)
  * 방법3: c_t 통합 (B의 단순화 방식)
  * 지표: Halo cell error, 공간 패턴 정확도 (polar/cartesian moment 비교)

#### (4) 디코더 Conv3D 1×1×1에 FiLM/AdaLN 추가 시 구조 변경 필요성
- 현재: Conv3D 1×1×1 (128→10), 정규화 층 없음
- FiLM 추가 시: `Conv3D(128→128) → FiLM(c) → Conv3D(128→10)`? 아니면 다른 구조?
- 권고: 별도 설계 문서 필요 (본 문서 범위 외)

#### (5) "p_load AdaLN-Zero" 와 "p_load Concat" 의 상충 검토
- 기존 잠정 결정: "Mamba concat + 디코더 AdaLN-Zero"
- B의 주장: "디코더에도 조건화 필수, concat or FiLM 둘 다 가능"
- 질문: 같은 p_load를 Mamba concat과 디코더 AdaLN 양쪽에 주입하면 중복/보강?
- 권고: Phase 3 에서 실험으로 결정

### 6.6 후속 문서 갱신 항목

본 검토 완료 후 아래 문서들을 순차적으로 갱신 필요 (본 세션 범위 외):

| 문서 | 갱신 항목 | 우선순위 |
|------|-----------|---------|
| `2026-04-15 모델 구현 계획(시계열).md` §5.5 | p_load → Mamba concat 확정 + Attention FiLM 추가 | 높음 |
| `2026-04-15 모델 구현 계획(시계열).md` §6 | 디코더 conditioning 설계 (Phase 3) | 높음 |
| `2026-04-15 p_load 주입 전략 재검토.md` | 기존 내용 유지, Phase 3 결과 추가 | 중간 |
| `2026-04-14 Conditional LAPE 적용 검토.md` | §3.4 "Mamba 통과 시 희석" 우려 → 확정, 재주입 전략 추가 | 중간 |
| `04_normalization_omitted_options.md` | §1.4 Mamba/Decoder conditioning 갱신 | 중간 |

### 6.7 한계 & 신뢰도

#### 본 검토의 한계
- **답변 원본 미확인**: 세션 압축으로 인해 두 외부 LLM 답변의 원문이 부분적으로만 복원됨. 뉘앙스/세부 사항이 누락되었을 가능성
- **사례 실증 부족**: A/B의 reference (논문 출처, 구체 사례)가 완전 수집되지 않았을 수 있음
- **물리 검증 부재**: 원자로 시뮬레이션의 특수성 (타원형 PDE, 제어봉 개입의 즉각성) 을 기상 모델 사례와 직접 비교하는 것의 타당성 재확인 필요

#### 권고사항의 신뢰도
- **높음** (Phase 1~2): p_load concat (기존 검증됨), sym_type FiLM, 절대위치 token-wise 재주입 — 이론적 근거 명확
- **중간** (Phase 3): 디코더 조건화, gradient 투명성 — B의 수학 논거는 설득력 있으나 원자로 도메인 특수성 미검증
- **낮음** (Phase 4): c_rod 통계 인코더 — 기본 구현 검증 후 선택적 추가

### 6.8 최종 권고 요약

```
즉시 실행:
✓ Phase 1 기본선: 인코더 ConditionalLAPE 단독 (1차 add)
  → 인코더-Mamba 학습 안정성 확인

우선 순위 높음:
✓ Phase 2: 시공간 학습 본격화 (LAPE skip + p_load Mamba concat)
  → 절대위치: ReZero gated re-add (Raw α + WD) (§5.4.10)
  → p_load: 공유 임베더 + 720 cell broadcast + concat (AdaLN-Zero 비활성) (§5.5)
  → 평가: 6개 지표 (4개 판별 + α gate + norm 비율)

우선 순위 중간:
✓ Phase 3: AdaLN-Zero 활성화 (이중 경로 완성)
  → p_load AdaLN: 공유 임베더 → adaln_proj (0 초기화) → γ, β 변조
  → 비교: concat only vs concat + AdaLN
  → 결정 분기: AdaLN 효과 / 차선책 (§5.5.6)

우선 순위 낮음:
○ Phase 4: Rod map 통계 인코더 (c_rod) + RoPE 비교

핵심 원칙 (§5.3, §5.4, §5.5 자문 + 결정으로 확정):
1. **절대위치 (LAPE)**: Skip connection (Raw α + WD), 단일 스칼라 (§5.4.10, §5.4.12)
   - 근거: 사용자 제약 (단일 스칼라, input-independent, 공간 공평) + Tanh saturation 회피
2. **p_load**: 이중 경로 (Mamba concat + Attention AdaLN-Zero), 공유 임베더 + 양방향 gradient (§5.5)
   - 근거: p_load는 단일 물리 의미 → gradient 시너지 (충돌 아님)
3. **공간 적용**: 720 cell 균일 broadcast (p_load 전역 특성과 정합)
4. **Gradient 안전**:
   - LAPE: tf.stop_gradient(encoder_lape) — Phase 1 prior 보호
   - p_load: detach 없음 — 공유 임베더 학습 진행 중 모듈
5. **폭주 제어**: Weight Decay (1e-4 표준) — 모든 가중치 일반 정규화
6. **LAPE와 RoPE 동시 사용 금지** (§5.3.4)
7. **p_load 디코더 미주입** (§6.5 ③): Mamba에서 이미 반영, 이중 책임 회피
8. **sym_type 별도 미주입 (잠정)** (§6.5 ④): ConditionalLAPE에 위임, Phase 2 후 재검토

참고 논문 (gate 메커니즘 + 위치 정보 + p_load 주입):
  - Pangu-Weather (Bi et al., 2023, Nature): 공간 backbone의 위치 prior 활용 패턴
  - ClimaX (Nguyen et al., 2023, ICML): 시간 조건 + 위치 임베딩 backbone 주입 패턴
  - ReZero (Bachlechner et al., 2021, UAI): 0 초기화 학습 가능 스칼라 gate (LAPE α 채택)
  - ControlNet (Zhang et al., 2023, ICCV): frozen prior + 외부 정보 zero-init 주입
  - Flamingo (Alayrac et al., 2022, NeurIPS): Tanh gated cross-attn (참고, 비채택)
  - DiT AdaLN-Zero (Peebles & Xie, 2023, ICCV): 0 초기화 scale/shift (p_load AdaLN 채택)
  - DeepSeek mHC (2026, arXiv:2512.24880): manifold 제약 안정성
  - MetMamba (2024, arXiv:2408.06400): 3D weather AdaLN-Mamba (참고, 우리 빠른 입력에 부분 적용)
  - TimeMachine (2024, ECAI): 다변량 forecasting concat 표준
  - Block-Biased Mamba (NeurIPS 2025): channel-specific bias 효과
```

---

## 7. 관련 문서

| 문서 | 관련 내용 |
|------|----------|
| `공간인코더 구현 계획/2026-04-14 Conditional LAPE 적용 검토.md` §3.4 | sym_type 흐름 단절 우려 최초 언급, "디코더 설계 시 별도 검토" 보류 |
| `공간인코더 구현 계획/2026-04-14 Conditional LAPE 적용 검토.md` §6 | FiLM/AdaLN/궤도 PE/SymPE 등 6개 메커니즘 비교 (인코더 관점) |
| `공간인코더 구현 계획/인코더 컴포넌트별 채용 이유/04_normalization_omitted_options.md` §1.4 | 인코더 LN에 조건 주입 안 함 결정 + p_load Mamba/디코더 분리 결정 |
| `시계열모델(SSM) 구현 계획/2026-04-15 p_load 주입 전략 재검토.md` | p_load → Mamba concat + 디코더 AdaLN-Zero 결정 (본 문서에서 재검토) |
| `시계열모델(SSM) 구현 계획/2026-04-15 모델 구현 계획(시계열).md` §5.5 | p_load Mamba 입력 concat 잠정 |
| `시계열모델(SSM) 구현 계획/2026-04-15 모델 구현 계획(시계열).md` §6 | Conv3D 1×1×1 디코더 (조건 주입 미정) |
| `시계열모델(SSM) 구현 계획/2026-04-17 시계열 아키텍처 학습방법론 재검증.md` §2.4 | Equation-Aware Conditioning 후순위 추가 언급 |
