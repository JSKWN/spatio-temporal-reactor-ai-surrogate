# π₀ (pi-zero) — Flow Matching + Dual Expert 로봇 정책

> **출처**: Physical Intelligence (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control.* [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
> **공개 수준**: 부분공개 (논문 + HuggingFace 모델 공개)
> **우리 아키텍처 맥락**: **Dual Expert 구조**(물리 백본 + 행동 전문가)와 **Blockwise Causal Attention** 패턴. 우리 인코더-디코더 비대칭 설계와 구조적 유사성

---

## 1. 시스템 개요

### 직관

π₀는 "하나의 모델이 여러 로봇을 제어하는 범용 정책"을 목표로 한다. 핵심 설계: VLM(시각-언어 이해)과 Action Expert(행동 생성)를 **Attention으로 연결하되, 파라미터는 분리**한다. 이를 통해 사전학습된 VLM 지식을 보존하면서 행동 생성을 독립적으로 학습한다.

### 아키텍처 (3.3B params)

```
[이미지 256 tokens + 언어 20 tokens]     [관절 상태 q_t + 행동 A_t^τ]
           ↓                                        ↓
    [Expert 1: PaliGemma 3B]              [Expert 2: Action Expert 315M]
    width=2048, depth=18                   width=1024, mlp_dim=4096
           ↓                                        ↓
    ┌──────────────────── Shared Self-Attention ──────────────────┐
    │  Blockwise Causal: [이미지+언어] → [상태] → [행동]          │
    │  행동 토큰이 모든 블록에 양방향 attend                        │
    └─────────────────────────────────────────────────────────────┘
           ↓
    [Action Decoder MLP] → 예측 행동 (H=50 future steps)
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 — Flow Matching + Action Chunking

```
Flow matching: V_θ(A_t^τ, o_t) 학습
  노이즈 혼합: A_t^τ = τ·A_t + (1-τ)·ε
  추론: 10-step Forward Euler (τ: 0→1, step=0.1)
  
Action chunking: H=50 steps per chunk
  50Hz 제어 → 1초분 행동을 한번에 예측
  Open-loop 실행 (0.5-0.8초마다 재추론)
```

**우리 문제에서의 대응**: Action chunking은 "여러 미래 시점을 한번에 예측"하는 것. 우리는 1-step 자기회귀이므로 직접 적용하지 않지만, **multi-step loss** (V-JEPA 2와 유사하게 2-4 step 동시 예측)로 학습 안정성을 높이는 아이디어와 관련.

### 2.2 상태 표현 — Dual Expert (MoE-유사 구조)

**이것이 π₀에서 가장 구조적으로 관련 깊은 기법이다.**

```
Expert 1 (VLM, 2.3B): 이미지/언어 토큰 전용
Expert 2 (Action, 0.3B): 상태/행동 토큰 전용
→ 두 Expert가 공유 Self-Attention 레이어에서 상호작용
→ 파라미터는 분리 (Expert 1의 사전학습 지식 보존)
```

**핵심 설계**:
- **토큰 유형별 고정 라우팅**: MoE의 학습 라우터 대신, 토큰 유형(이미지/언어 vs 상태/행동)에 따라 Expert를 고정 할당
- **Shared Attention**: 두 Expert의 토큰이 하나의 Attention에서 만남 → Expert 간 정보 교환
- **비대칭 크기**: Expert 1(2.3B) >> Expert 2(0.3B). 이해 능력 > 행동 생성 능력에 더 많은 파라미터 배분

**우리 아키텍처와의 구조적 유사성**:

| π₀ | 우리 A안 |
|---|---|
| Expert 1 (VLM, 큰 모델): 관측 이해 | **인코더 (FullAttn ×3)**: 노심 공간 이해 |
| Expert 2 (Action, 작은 모델): 행동 생성 | **Mamba**: 시간 전이 (인코더보다 작음) |
| Shared Attention: Expert 간 정보 교환 | **디코더 (FullAttn)**: 인코더 공간 정보 + Mamba 시간 정보 결합 |

이 대응은 "공간 이해에 많은 파라미터, 시간 전이에 적은 파라미터, 결합에 Attention"이라는 설계 원칙이 로보틱스와 원자로 대리모델에서 공통적으로 나타남을 시사한다.

### 2.3 제어/조건 주입 — Blockwise Causal Attention

```
Block 1: [이미지 + 언어]  → 양방향 Self-Attention
Block 2: [관절 상태]      → Block 1에 attend 가능, 자기 내부 양방향
Block 3: [행동]           → Block 1, 2 모두에 attend, 자기 내부 양방향
```

- **인과적 정보 흐름**: 관측 → 상태 → 행동 순서로만 정보 전달
- 상태 블록은 행동 블록에 attend하지 않음 → 상태 표현 캐싱 가능

**우리 문제 대응**: 우리 파이프라인도 유사한 인과적 흐름:
```
관측 [state(t)] → 인코더 [공간 이해] → Mamba [시간 전이] → 디코더 [물리 복원]
```
각 단계가 이전 단계의 출력에만 의존하는 인과적 구조.

### 2.4 학습 전략 — 대규모 사전학습 + 과제별 미세조정

```
사전학습: 10,000+ 시간, 903M timesteps, 7개 로봇 플랫폼, 68개 과제
미세조정: 과제당 5-100시간의 curated 데이터
```

가중치: 로봇-과제 조합별 n^0.43으로 균형 조절

**우리 문제 대응**: 우리 2-Phase 학습과 유사:
- 사전학습 (Phase 1): Branch 데이터로 인코더/디코더 pretrain (범용 공간 이해)
- 미세조정 (Phase 2): CRS 데이터로 Mamba 학습 (특화 시간 동역학)

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| Flow matching | X | 결정론적 예측 문제. 확률적 생성 불필요 | - |
| Action chunking (H=50) | △ | Multi-step loss(2-4 step)로 변형 적용 가능 | 하 |
| **Dual Expert 구조** | O (이미 유사) | 인코더(큰 모델)+Mamba(작은 모델)+디코더(결합). π₀와 구조적 동치 | - |
| **Blockwise Causal 정보 흐름** | O (이미 유사) | 인코더→Mamba→디코더의 인과적 흐름 | - |
| **비대칭 파라미터 배분** | O | 공간 이해(인코더)에 더 많은 파라미터, 시간 전이(Mamba)는 가볍게 | - |
| 토큰 유형별 Expert 분리 | △ | 인코더/Mamba/디코더가 이미 암묵적으로 "Expert 분리" | - |

---

## 4. 핵심 차용 후보

### 이미 반영된 설계 패턴
- **Dual Expert 원칙**: 큰 모델(공간 이해) + 작은 모델(시간 전이) + Attention(결합). 우리 A안과 구조적 동치
- **인과적 정보 흐름**: 관측→이해→전이→출력의 단방향 흐름

### 수정 후 적용 가능
- **Multi-step loss**: action chunking의 변형. 1-step teacher forcing + 2-4 step unroll loss 혼합
- **인코더→디코더 Cross-Attention**: π₀에서 Expert 1의 토큰이 Shared Attention으로 Expert 2에 전달되듯, 인코더 중간 feature를 디코더에 Cross-Attention으로 전달하는 skip connection 검토

### 부적합
- **Flow matching / Action chunking**: 결정론적 1-step 자기회귀에 불필요
- **다중 로봇 통합**: 단일 원자로 노심 대상이므로 다중 실체 통합 불필요

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| π₀ | Physical Intelligence. 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
