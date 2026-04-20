# p_load Mamba 주입 방식 + 3D Spatiotemporal Mamba 사례 조사

> **작성일**: 2026-04-20
> **목적**: p_load (외부 제어 입력, 시간 변동 스칼라)을 Mamba 시공간 프로세서에 주입하는 방식 결정에 필요한 외부 사례 검증
> **연관 문서**: [2026-04-17 ConditionalLAPE 정보 단절 우려 및 정보(위치, 대칭, 출력조건) 주입 방안 검토.md](../../2026-04-17 ConditionalLAPE 정보 단절 우려 및 정보(위치, 대칭, 출력조건) 주입 방안 검토.md) §6.5 미결 ②

## 1. 조사 배경

### 1.1 본 프로젝트 상황
- 시공간 프로세서: Mamba + Self-Attention 하이브리드
- 격자: 3D (20축방향 × 6×6 quarter core) = **720 cell tokens**
- 시간: 575 step
- 외부 제어 입력: p_load (출력 수준 스칼라, 매 시점 자유 변동)

### 1.2 주입 방식 후보 (3가지)
- **(α) D_latent 채널 concat**: `(B, T, 720, D)` → `(B, T, 720, D+1)` — 가장 단순
- **(β) 토큰 추가**: `(B, T, 720, D)` → `(B, T, 720+n, D)` — global token
- **(γ) FiLM / AdaLN-Mamba**: γ(p_load)·x + β(p_load) — modulation 방식

### 1.3 1차 자문 사례 (사용자 제공) — 검증 필요
사용자가 1차 자문으로 7개 사례를 제시:
1. CoLoRSMamba (2026)
2. RiverMamba (NeurIPS 2025)
3. TimeMachine (2025로 명시)
4. CMDMamba (Frontiers 2025)
5. FSMamba (2026)
6. DyG-Mamba (NeurIPS 2025)
7. Block-Biased Mamba (NeurIPS 2025)

본 문서는 위 사례 검증 + 추가로 우리에 더 가까운 3D Spatiotemporal Mamba 사례 조사 결과 정리.

## 2. 1차 자문 사례 검증 결과

| 사례 | 검증 | 정확성 | 우리와 차원 일치 | 우리에 시사점 |
|------|------|-------|--------------|------------|
| **RiverMamba** ([arXiv:2505.22535](https://arxiv.org/abs/2505.22535)) | ✓ NeurIPS 2025 | abstract 일치, 세부는 PDF 필요 | ✗ (**2D 격자 + space-filling curve로 1D 시퀀스화**) | 외부 입력 concat 패턴 |
| **TimeMachine** ([arXiv:2403.09898](https://arxiv.org/abs/2403.09898)) | ⚠️ **ECAI 2024** (사용자 "2025" 오류) | 4-Mamba 구조 일치 | ✗ (시계열 1D) | 다변량 concat 표준 |
| **CoLoRSMamba (2026)** | ❌ 검증 불가 | 정확한 매칭 못 찾음 | - | 인용 정보 재확인 필요 |
| **CMDMamba** ([Frontiers 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599799/full)) | ✓ 실재 | 다변량 concat 채택 | ✗ (금융 시계열 1D) | concat 패턴 |
| **FSMamba (2026)** | △ 검증 미진행 | - | - | - |
| **DyG-Mamba** ([arXiv:2408.06966](https://arxiv.org/abs/2408.06966)) | ✓ NeurIPS 2025 | 인용 정확 (B, C input-dependent) | ✗ (동적 그래프) | input-dependent SSM param |
| **Block-Biased Mamba** ([arXiv:2505.09022](https://arxiv.org/abs/2505.09022)) | ✓ NeurIPS 2025 | 인용 정확, 핵심 발견 | ✗ (1D sequence) | **channel-specific bias** |

### 2.1 정정 사항
- **TimeMachine**: 2025 → **ECAI 2024**로 정정
- **CoLoRSMamba (2026)**: 검색 결과 미확인 → 인용 정보 재확인 필요
- **FSMamba (2026)**: 검증 미진행

### 2.2 핵심 발견 — Block-Biased Mamba (NeurIPS 2025)

**문제 진단**:
> "Mamba shares parameters across all channels → effective width 감소, capacity 손실"
> "S4D는 channel별 독립 recurrent unit, Mamba는 공유"

**제안 (B²S6 unit)**:
- 입력을 작은 block으로 partition
- **channel-specific, input-independent bias term** 추가
- Long-Range Arena에서 S4 / S4D 능가

**우리 케이스 시사점**:
- p_load를 **"channel-specific bias"** 형태로 Mamba에 주입하면 inductive bias 개선 가능
- 단순 concat 외 **bias 형태 추가 주입**이 검증된 효과

### 2.3 핵심 발견 — DyG-Mamba (NeurIPS 2025)

**핵심**:
- "B와 C parameter를 **input-dependent**로 정의"
- 시간 간격(외부 조건)을 control signal로 활용
- 외부 조건이 SSM dynamics를 직접 조절 (CoLoRSMamba가 주장한 것과 같은 방향)

**우리 케이스 시사점**:
- p_load(t)를 SSM의 B, C parameter 생성에 직접 활용 가능
- 단순 concat보다 강한 조건화, 단 구현 복잡도 증가

## 3. 1차 자문의 한계 — 차원 불일치

### 3.1 차원 비교

| 사례 | 격자 차원 | 토큰 구성 | 우리와 일치 |
|------|---------|---------|----------|
| RiverMamba | **2D** (위경도) | space-filling curve로 **1D 시퀀스화** | ✗ |
| TimeMachine | 1D (시계열) | 시간 축 시퀀스 | ✗ |
| CMDMamba | 1D | 시간 축 시퀀스 | ✗ |
| DyG-Mamba | 그래프 (구조 비격자) | 노드/엣지 토큰 | ✗ |
| Block-Biased Mamba | 1D | 평탄화 시퀀스 | ✗ |
| **우리 프로젝트** | **3D** (20×6×6) | **격자 그대로 720 token** | - |

→ **1차 자문 사례 모두 우리와 차원 다름.** 원리는 차원 무관 적용 가능하나, 직접 비교 어려움.

## 4. 추가 조사 — 3D Spatiotemporal Mamba 사례

### 4.1 MetMamba (2024) ⭐ — 우리에 가장 직접 시사

**출처**: [arXiv:2408.06400](https://arxiv.org/abs/2408.06400)

**구조**:
- Regional weather forecasting
- **AdaLN-Mamba3D 블록**: depth-wise 3D conv + Mamba selective scan + AdaLN
- 외부 입력 (elapsed year time, seasonal variation) → **AdaLN으로 주입**
- "Spatial-temporal 입력 native 처리 (3D conv encoder 불필요)"

**중요 Ablation 결과** ⚠️:
> "Performance regression when **elapsed year time information supplied as expanded channel**" (Figure 7 ablation)
- **외부 입력을 channel concat했을 때 성능 저하**
- AdaLN이 더 효과적

**우리 케이스 시사점**:
- 우리 잠정 결정 (p_load Mamba **concat**)에 대한 **반론 근거**
- 단, 외부 입력 동역학 차이 고려 필요 (다음 §4.2 참조)

### 4.2 외부 입력 동역학 차이 분석

| 측면 | MetMamba | 우리 프로젝트 |
|------|----------|------------|
| **외부 입력** | elapsed year time (연도 시간) | p_load (출력 수준) |
| **변동 빈도** | 매우 느림 (계절 단위) | 빠름 (매 575 step 자유) |
| **시간 동역학 영향** | 모델 전체 정규화 (계절 적응) | 즉각적 (Xe 생성률, 온도 상승률) |
| **물리적 의미** | "지금이 어느 계절인지" 모델에 알림 | "현재 출력 수준에 맞춰 동역학 조절" |

→ MetMamba ablation 결과 (concat 부정)을 **우리에 그대로 적용하면 위험**:
- MetMamba: **느린 외부 입력** → AdaLN modulation 적합 (전역 변조)
- 우리: **빠른 외부 입력** → concat이 매 시점 직접 반영 (다를 수 있음)

→ **결론**: 우리 케이스에 대해서는 별도 실험 필요.

### 4.3 Mamba4D (CVPR 2025)

**출처**: [arXiv:2405.14338](https://arxiv.org/abs/2405.14338), [GitHub](https://github.com/IRMVLab/Mamba4D)

**구조**:
- 4D point cloud video (3D + 시간)
- **Intra-frame Spatial Mamba** + **Inter-frame Temporal Mamba** 분리
- Disentangled spatial-temporal SSM

**우리 케이스 시사점**:
- 우리 구조 (Mamba 시간 + Attention 공간)와 유사 철학
- 공간/시간 처리 분리 패턴 검증 사례
- 단, point cloud는 비격자 → 우리 격자와 차이

### 4.4 FH-Mamba (Arctic Sea Ice, 2026)

**출처**: [arXiv:2602.13522](https://arxiv.org/html/2602.13522)

**구조**:
- **3D Hilbert scan**: 3D 시공간 grid를 locality-preserving path로 1D 변환
- "외부 forcing factors (wind, ocean currents, atmospheric conditions)" 통합
- Frequency-enhanced approach

**우리 케이스 시사점**:
- 3D + 외부 forcing 통합 사례
- 단, Hilbert scan으로 1D 변환 → 우리 token 유지와 차이

### 4.5 Geo-Mamba (2025)

**출처**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843225005011)

**구조**:
- "Multi-source geographic factor integration"
- Spatiotemporal modeling
- 다중 source 외부 조건 통합

**우리 케이스 시사점**:
- 외부 입력 통합 패턴 사례 (구체 메커니즘 추가 확인 필요)

### 4.6 NeuroMamba (NeurIPS 2025) — 검증 미완료

**출처**: [OpenReview](https://openreview.net/pdf/f9ba5851ac21b47de0e08b90bdb43de92fa460e5.pdf) (접근 실패 403)

**구조 (간접 확인)**:
- 전체 뇌 voxel (3D) → spatiotemporal sequence
- 직접-sequence Mamba 아키텍처

**향후 추가 검증 필요**.

## 5. 종합 — p_load 주입 방식 후보 정리

### 5.1 3가지 주요 후보 + 검증 사례

| 방식 | 검증 사례 | 우리 적합성 | 비고 |
|------|---------|----------|------|
| **(α) Channel concat** | TimeMachine, CMDMamba, FSMamba | ✓ (빠른 외부 입력) | **MetMamba ablation에서 부정 결과** ⚠️ (느린 조건 한정) |
| **(β) Token 추가 (global)** | (검증 사례 적음) | △ | Mamba는 토큰 독립이라 한계 |
| **(γ) AdaLN-Mamba** | **MetMamba (3D weather)** | ★ (검증 우수) | 우리 빠른 조건에 효과적인지 미검증 |
| **(δ) Channel-specific bias** | Block-Biased Mamba | ★★ (Mamba 안정성 개선) | 정적 bias라 빠른 변동 직접 반영 어려움 |
| **Input-dependent SSM param** | DyG-Mamba | △ | 구현 복잡 |

### 5.2 우리 권고 (잠정)

**Phase 2 시작 — concat (단순) 또는 AdaLN-Mamba (MetMamba 패턴) 비교 실험**

#### 1순위 권고: 방안 (α) Channel Concat (단순, 검증된 표준)
```python
# TF2/Keras
# p_load shape: (B, T) → broadcast → (B, T, 720, 1)
p_load_broadcast = tf.tile(p_load[:, :, None, None], [1, 1, 720, 1])
mamba_input = tf.concat([z_t, p_load_broadcast], axis=-1)  # (B, T, 720, D+1)
```

**근거**:
- 다변량 forecasting 표준 패턴 (TimeMachine, CMDMamba 등)
- 가장 단순 (가중치 추가 미미)
- 매 시점 변동 자연 반영 (우리 빠른 조건에 적합)

**우려**:
- MetMamba ablation에서 channel concat 부정 결과 → Phase 2 학습 시 모니터링 필수

#### 2순위 검토: 방안 (γ) AdaLN-Mamba (MetMamba 패턴)
- Mamba 내부 LN을 AdaLN으로 변형 (Mamba block은 LN 포함)
- 외부 입력 → MLP → γ, β 생성 → LN 변조

**근거**:
- 3D weather forecasting (MetMamba) 검증
- channel concat 대비 우수 결과 (단, 느린 조건 한정)

**우려**:
- 우리 빠른 조건에서도 효과적인지 미검증
- 구현 복잡도 증가

#### 비채택: 방안 (β) Token 추가
- Mamba가 토큰 독립 처리 → global token 효과 제한
- Self-Attention에서만 효과 있음 → 부분적 활용

#### 비채택 (Phase 2): 방안 (δ) Channel-specific bias
- 정적 bias 형태 → 빠른 변동 직접 반영 어려움
- Phase 4 검토 옵션 (concat과 결합 가능)

#### 비채택: Input-dependent SSM param
- DyG-Mamba 패턴은 우리에 과한 복잡도
- p_load는 단순 스칼라이므로 SSM param 동적 생성까지는 불필요

### 5.3 Phase 2 실험 설계 권고

**비교 실험**:
1. **Baseline (α)**: Channel concat
2. **Comparison (γ)**: AdaLN-Mamba (Mamba 내부 LN 변형)

**평가 지표**:
- (c) p_load 구간별 보정 오차 (50%, 75%, 100% 출력 수준별)
- (d) 정상 vs 과도 오차 분해 (transient 동역학 추종 능력)
- 학습 수렴 속도, 최종 성능

**결정 시점**: Phase 2 비교 실험 결과 후

## 6. 한계 및 추가 검증 필요 사항

### 6.1 본 조사의 한계
- **차원 일치 사례 부족**: 우리와 정확히 같은 "3D 격자 + cell-wise token + 외부 제어 입력 (시간 변동 스칼라)" 사례 적음
- **MetMamba**가 가장 가깝지만 외부 입력 동역학이 다름 (느림 vs 빠름)
- **NeuroMamba** 접근 실패 → 추가 자료 필요
- **CoLoRSMamba (2026)**: 검색 미확인 → 사용자 인용 정보 재확인 필요
- **FSMamba (2026)**: 검증 미진행

### 6.2 추가 검증 권장 사항
- NeuroMamba (NeurIPS 2025) 다른 출처 시도
- CoLoRSMamba 정확한 인용 정보 확인
- FSMamba 검증
- "3D 격자 + 외부 제어 입력 (시간 변동 스칼라)" 정확히 일치하는 사례 추가 탐색

## 6.5 [최종 결정 — 2026-04-20] 이중 경로 전략 (공유 임베더 + Mamba concat + Attention AdaLN-Zero)

### 6.5.1 사용자 통찰 및 최종 결정

> 정보 단절 방지를 위해 **p_load도 ConditionalLAPE처럼 다층 주입** 필요.
> - Mamba = embedded concat (시간 동역학 가속)
> - Attention = AdaLN-Zero (전역 컨텍스트 제공)

**최종 결정**: **공유 임베더 + 양방향 gradient 허용 (방안 1)** 채택

### 6.5.2 결정 근거 — p_load의 단일 물리 의미 (정정)

#### 의견 검토 과정 (3단계)
1. **초기**: 옵션 A (공유 임베더 + 양방향 gradient)
2. **중간 검토**: 방안 2 (공유 + AdaLN detach) → 방안 3 (별도 임베더) 검토
3. **최종 (물리 정정 후)**: 방안 1 (공유 임베더 + 양방향 gradient) 회귀

#### p_load 물리적 의미 정정 (핵심)

**잘못된 가정 (방안 3 근거였음)**:
- "Mamba 측 의미: 시간 미분 (Xe 축적률, 온도 상승률)"
- "AdaLN 측 의미: 공간 결합 강도 조절 (도플러 피드백, peaking 억제)"

**정정된 사실**:
- **도플러 피드백, peaking 억제는 국소(Local) 현상**:
  - cell별 T_fuel, xs_fuel(U-238 공명 흡수), flux 분포에 의존
  - 중앙 제어 p_load와 직접 관련 없음
  - 이미 cell별 잠재 벡터(flux, T, xs_fuel 등)에 의해 결정됨
- **p_load의 진정한 의미**: **단일 의미**
  - 노심 전체 핵분열 스케일 (Target Total Power)
  - 시간 동역학 가속 페달 (Xe 축적, 온도 변화 속도)
  - 시간/공간 의미 분리 안 됨

#### Gradient 충돌 vs 시너지 재평가

p_load가 단일 물리 의미를 가지므로:
- Mamba 요구: "시간 변화율 맞추기 위해 p_load 표현 업데이트"
- AdaLN 요구: "공간 활성화 스케일 맞추기 위해 p_load 표현 업데이트"
- 두 요구 모두 **동일한 목표**: "현재 출력 100% vs 50%를 가장 잘 나타내는 robust 임베딩"
- → **양방향 gradient = 시너지** (충돌 아님)
- → 공유 임베더가 두 경로의 평균이 아닌 **둘 다 만족하는 robust 표현** 학습

### 6.5.3 ConditionalLAPE 패턴과의 비교

| 정보 | 주입 위치 1 | 주입 위치 2 | 임베더 구조 | 학습 시점 | 공간 적용 방식 |
|------|-----------|-----------|----------|---------|------------|
| **위치/대칭** (ConditionalLAPE) | 인코더 add (1차) | Attention skip (ReZero α) | 단일 prior | Phase 1 학습 → freeze | cell별 다른 위치 임베딩 (token-wise) |
| **p_load** | Mamba concat | Attention AdaLN-Zero | **공유 임베더 + 양방향 gradient** | Phase 2 학습 (계속) | **모든 720 cell에 동일 broadcast** (전역 균일) |

**공통 철학**: 정보 다중 주입 (단절 방지)
**차이**:
- ConditionalLAPE는 freeze된 prior, p_load는 학습 중 모듈 (detach 불필요)
- ConditionalLAPE는 cell별 다름 (위치 좌표), p_load는 cell별 동일 (전역 출력 수준)

### 6.5.4 p_load의 공간 균일 Broadcast 특성

**핵심 사실**:
- p_load는 **시간 변동 (T축) + 공간 균일 (모든 cell 동일)** 입력
- shape: `(B, T)` → 모든 720 cell에 같은 값 복제 (broadcast)
- 즉, 매 시점 t에서 cell (0,0,0)도, cell (19,5,5)도 **동일한 p_load(t)** 값 받음

**물리적 의미와 정합**:
- p_load = 노심 전체 출력 수준 (Total Power Target)
- 운전자가 중앙 제어 → 전 노심 동일 명령
- → "공간 균일"이 물리적 본질에 정합

**ConditionalLAPE α gate와의 일관성**:
- LAPE skip: 단일 스칼라 α (모든 cell 공평 적용)
- p_load Mamba concat: 동일 p_load_emb (모든 cell 공평 broadcast)
- p_load AdaLN: 동일 γ, β (모든 cell 공평 변조)
- → **모두 "공간 균일" 적용** — 일관된 패턴

**구현 시 명시 필수**: tile/broadcast로 720 cell 차원에 복제.

### 6.5.5 최종 구현 (방안 1, TF2/Keras)

```python
class STBlock(tf.keras.layers.Layer):
    def __init__(self, dim=128, p_load_embed_dim=8, num_cells=720):
        super().__init__()
        self.num_cells = num_cells
        # 공유 p_load 임베더 (양방향 gradient 수렴 지점)
        self.p_load_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(p_load_embed_dim, activation='gelu'),
            tf.keras.layers.Dense(p_load_embed_dim),
        ], name='p_load_encoder')

        # AdaLN-Zero projection (γ, β 0 초기화)
        self.adaln_proj = tf.keras.layers.Dense(
            2 * dim, kernel_initializer='zeros', name='adaln_proj'
        )

        self.mamba = MambaLayer(...)
        self.attention = Attention(...)
        self.ln_no_affine = tf.keras.layers.LayerNormalization(
            center=False, scale=False
        )

    def call(self, z_t, p_load):
        # p_load: (B, T, 1) — 시간별 스칼라

        # [1] 공유 임베딩 (양방향 gradient 수렴)
        # (B, T, 1) → (B, T, embed_dim)
        p_load_emb = self.p_load_encoder(p_load)

        # [2] Mamba 측 — 공간 균일 broadcast 후 concat
        # (B, T, 1, embed_dim) → (B, T, 720, embed_dim) [모든 cell 동일 값]
        p_load_b = tf.tile(
            p_load_emb[:, :, None, :], [1, 1, self.num_cells, 1]
        )
        # z_t: (B, T, 720, D), p_load_b: (B, T, 720, embed_dim)
        # 모든 cell이 동일한 p_load_emb를 받음 (공간 균일)
        mamba_in = tf.concat([z_t, p_load_b], axis=-1)
        d_mamba = self.mamba(mamba_in)

        # [3] Attention 측 — AdaLN-Zero, 공간 균일 변조
        # detach 없음! 양방향 gradient 허용
        gamma_beta = self.adaln_proj(p_load_emb)  # (B, T, 2*dim)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)
        # cell 차원 broadcast: (B, T, 1, dim) [모든 cell에 동일 γ, β]
        gamma = gamma[:, :, None, :]
        beta = beta[:, :, None, :]

        # AdaLN: LN의 normalized output에 γ, β 변조 (공간 균일)
        x_norm = self.ln_no_affine(d_mamba)
        x_modulated = x_norm * (1 + gamma) + beta

        return d_mamba + self.attention(x_modulated)
```

**핵심 설계 포인트**:
- `p_load_encoder`: 단일 모듈, 양방향 gradient로 학습
- `tf.tile(...)`: **명시적 공간 broadcast** — 720 cell에 동일 값 복제
- `gamma[:, :, None, :]`: cell 차원에 broadcast (자동 확장)
- `adaln_proj`: **0 초기화 (AdaLN-Zero)** — 학습 초기 AdaLN 효과 없음 → Mamba 먼저 안정 학습
- `tf.stop_gradient` **사용 안 함** — p_load는 단일 의미이므로 시너지

### 6.5.6 MetMamba ablation 결과 재해석

**MetMamba 부정 결과의 한정 조건**:
- 외부 입력: "elapsed year time" — **느린 전역 조건** (계절 단위)
- Channel concat 시 매 시점 동일한 값 → noise처럼 처리될 가능성
- → AdaLN modulation이 더 효과적

**우리 p_load와의 차이**:
- 우리 p_load: **빠른 동적 제어 변수** (매 575 step 자유 변동)
- 매 시점 다른 값 → channel concat이 시간 변동 자연 반영
- 단, 공간 축으로는 동일 broadcast (이는 양쪽 동일)
- → MetMamba 결과는 **우리에 직접 적용 안 됨** (시간 동역학 차이)

### 6.5.7 단계적 도입 (Phase 2 → Phase 3)

#### Phase 2: Mamba concat 단독 (AdaLN 비활성)
- `p_load_encoder` + Mamba concat만 적용
- AdaLN-Zero 초기화로 자연스럽게 비활성 (γ=0, β=0)
- 평가: (c) p_load 구간별 보정 오차로 concat 단독 효과 측정

#### Phase 3: AdaLN-Zero 활성 (이중 경로 완성)
- `adaln_proj` 학습 진행되며 AdaLN 효과 활성화
- 비교: Phase 2 (concat only, γ≈β≈0) vs Phase 3 (concat + AdaLN active)
- 효과 정량화

#### 결정 분기
| Phase 2 결과 | 다음 단계 |
|------------|---------|
| (c) 오차 충분히 낮음 | concat only 유지, AdaLN 생략 가능 |
| (c) 오차 plateau | Phase 3 AdaLN 활성 → 효과 확인 |
| AdaLN 효과 미미 (γ, β ≈ 0 유지) | concat only 유지, 복잡도 회피 |
| AdaLN 효과 명확 | 이중 경로 채택 |

### 6.5.8 모니터링 항목 (Phase 2/3 학습 중 추적)

학습 안정성과 효과 검증을 위한 필수 모니터링:

1. **p_load_encoder gradient norm** (각 block별)
   - 0 근처 사망 여부 — Mamba/AdaLN이 p_load를 활용하지 않는 신호
   - 정상: 0보다 큰 값으로 학습 진행
2. **adaln_proj 출력 (γ, β) 분포**
   - 0에서 얼마나 벗어나는지 추적
   - 폭주 여부 (큰 양수/음수 극값)
   - AdaLN 효과 활성화 정도 진단
3. **p_load 구간별 예측 오차**
   - 정출력 / 감출력 / 저출력 / 상승 구간 분리 측정
   - AdaLN 추가 전후 비교
4. **양방향 gradient 균형**
   - p_load_encoder의 gradient 크기 비율 (Mamba 측 vs AdaLN 측)
   - 한쪽이 압도적이면 표현 편향 신호

### 6.5.9 차선책 (1순위 방안 1 미작동 시)

**1순위 (방안 1, 공유 임베더 + 양방향 gradient)** 가 학습에서 문제를 보일 경우 단계적 차선책 적용.

#### 차선 시나리오 1: 표현 용량 부족
**증상**: Mamba concat과 AdaLN 모두 작동하지만 (c) p_load 구간별 오차 plateau, 구간 차이 학습 부족
**차선책 1**: **embed_dim 확장** (8 → 16 → 32)
- 공유 임베더 차원만 키움
- 가장 단순, 우선 시도
- 코드 변경 최소

#### 차선 시나리오 2: AdaLN 효과 없음
**증상**: γ, β가 학습 종료 후에도 0 근처 (AdaLN-Zero 시작 상태에서 안 벗어남), AdaLN 추가가 성능에 무영향
**차선책 2**: **AdaLN 제거, Mamba concat만 유지**
- 공유 임베더 + Mamba concat 단독 (단순화)
- 모델 복잡도 감소
- 의미: p_load의 공간 변조 효과가 본질적으로 미미

#### 차선 시나리오 3: Gradient 충돌 발견
**증상**: p_load_encoder의 gradient norm이 불안정, 학습 발산, 두 경로 loss가 반대 방향
**차선책 3**: **방안 2 (공유 임베더 + AdaLN detach)** 전환
```python
# AdaLN 측만 stop_gradient 적용 (Mamba 우선)
gamma_beta = self.adaln_proj(tf.stop_gradient(p_load_emb))
```
- p_load_encoder는 Mamba loss로만 학습
- AdaLN은 받은 표현을 변조에만 활용
- 위험: AdaLN 효과 제한, "moving target" 문제

#### 차선 시나리오 4: 표현 충돌 발견 (드문 경우)
**증상**: p_load_encoder가 어느 한쪽도 만족 못 하는 타협 표현으로 수렴, 양 경로 모두 plateau, 표현 분석 시 불일치 명확
**차선책 4**: **방안 3 (별도 임베더)** 전환
```python
self.p_load_mamba_mlp  = MLP(1 → 8)  # Mamba 전용
self.p_load_adaln_proj = MLP(1 → 2*dim, kernel_initializer='zeros')  # AdaLN 전용
```
- Gradient 완전 분리
- 파라미터 추가 ~88개 (무시 수준)
- 단점: 두 표현 일관성 없음 (단, 충돌 시에는 분리가 더 나음)

### 6.5.10 차선책 우선순위 매트릭스

| 우선순위 | 증상 | 차선책 | 변경 규모 | 위험도 |
|--------|------|------|--------|------|
| 1 | (c) 오차 plateau, 표현 용량 부족 | embed_dim 확장 | 매우 작음 | 매우 낮음 |
| 2 | AdaLN 효과 없음 (γ,β ≈ 0 유지) | AdaLN 제거 (concat 단독) | 작음 | 낮음 |
| 3 | 학습 발산, gradient 불안정 | 방안 2 (AdaLN detach) | 작음 | 중간 |
| 4 | 양 경로 plateau (표현 충돌) | 방안 3 (별도 임베더) | 중간 | 낮음 |

→ **순서**: 표현 용량 확장 → AdaLN 제거 → AdaLN detach → 별도 임베더

---

## 7. 결론 및 다음 작업

### 7.1 결론 (2026-04-20 최종)

**p_load 주입 전략 — 이중 경로 + 공유 임베더 + 양방향 gradient (방안 1) 채택**:
- **Mamba**: embedded concat (공유 임베더 출력 + 720 cell 균일 broadcast)
- **Attention**: AdaLN-Zero (공유 임베더 출력 + adaln_proj 0 초기화 + 720 cell 균일 변조)
- **공유 임베더**: 양방향 gradient 허용 (detach 없음)

**근거**:
- p_load는 단일 물리 의미 (전역 출력 스케일 + 시간 가속 페달)
- 시간/공간 의미 분리되지 않음 → 공유 임베더가 robust 표현 학습
- ConditionalLAPE의 다층 주입 패턴과 일관 (정보 단절 방지)
- 720 cell 공간 균일 broadcast = p_load의 전역 특성과 정합

**MetMamba ablation 결과 재해석**:
- "느린 전역 조건 (elapsed year)" channel concat 부정 → 우리 빠른 동적 변수에는 직접 적용 안 됨
- 우리 p_load는 매 시점 변동 → concat이 시간 변동 자연 반영

**차선책 (1순위 미작동 시)** — §6.5.9, §6.5.10:
1. 표현 용량 부족 → embed_dim 확장 (8→16→32)
2. AdaLN 효과 없음 → AdaLN 제거 (concat 단독)
3. Gradient 충돌 → 방안 2 (AdaLN detach)
4. 표현 충돌 → 방안 3 (별도 임베더)

### 7.2 다음 작업
1. 본 조사 결과를 [2026-04-17 ConditionalLAPE...md](../../2026-04-17 ConditionalLAPE 정보 단절 우려 및 정보(위치, 대칭, 출력조건) 주입 방안 검토.md) §5.5 신규 섹션으로 정리 (요약본)
2. §6.5 미결 ② 결정 완료 표기 (방안 1: 공유 임베더 + Mamba concat + Attention AdaLN-Zero)
3. §6.5 미결 ③ → 결정 완료 (p_load 디코더 미주입)
4. §6.5 미결 ④ → 결정 완료 (sym_type Phase 2 후 검토)

## 참고 문헌

| 논문 | 출처 | 주요 시사점 |
|------|------|---------|
| **MetMamba** | arXiv:2408.06400 | 3D weather + AdaLN-Mamba3D, channel concat 부정 ablation |
| **Mamba4D** | CVPR 2025, arXiv:2405.14338 | 4D point cloud, disentangled spatial/temporal |
| **RiverMamba** | NeurIPS 2025, arXiv:2505.22535 | 2D weather, space-filling curve, 외부 입력 concat |
| **DyG-Mamba** | NeurIPS 2025, arXiv:2408.06966 | input-dependent SSM param |
| **Block-Biased Mamba** | NeurIPS 2025, arXiv:2505.09022 | channel-specific bias 효과 입증 |
| **TimeMachine** | ECAI 2024, arXiv:2403.09898 | 다변량 forecasting concat 표준 |
| **CMDMamba** | Frontiers 2025 | 금융 시계열 다변량 concat |
| **FH-Mamba** | arXiv:2602.13522 | 3D Hilbert scan + 외부 forcing |
| **Geo-Mamba** | ScienceDirect 2025 | Multi-source factor integration |
