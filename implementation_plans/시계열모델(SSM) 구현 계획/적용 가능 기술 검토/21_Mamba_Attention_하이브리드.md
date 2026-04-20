# Mamba + Attention 하이브리드 아키텍처 — 도메인 횡단 조사 (2024-2026)

> **작성일**: 2026-04-15
> **목적**: SSM(Mamba)과 Attention을 결합하는 하이브리드 패턴의 전체 현황 조사. 우리 아키텍처에서의 적용 가능성 평가
> **범위**: LLM, Vision, 시계열, PDE, 로보틱스, 의료, 음성 등 전 도메인
> **핵심 발견**: SSM:Attention = 3:1~10:1 비율이 합의. 7-8% Attention만으로 충분. 이론적 증명 존재(2026)
> **우리 아키텍처 맥락**: A안(인코더FullAttn → Mamba cell-wise → 디코더FullAttn + Physical Loss) 채택. Mamba 자체에 Attention 추가는 보류하되, 근거 자료로 보존

---

## 1. 왜 하이브리드인가? — 이론적 배경

### 직관

SSM(Mamba)과 Attention은 각각 **대체 불가능한 강점**을 가진다:

| | SSM (Mamba) | Attention (Transformer) |
|---|---|---|
| **강점** | 선형 복잡도 O(T), 고정 메모리, 장기 시퀀스 효율 | 전역 문맥 파악, 연관 기억(associative recall), 정밀 검색 |
| **약점** | 고정 크기 h(t)에 과거 압축 → 정보 손실 | O(T²) 복잡도, KV 캐시 메모리 증가 |
| **물리적 비유** | 노트에 요약을 적어가며 이동하는 학생 (요약 과정에서 세부사항 손실) | 모든 교과서를 펼쳐놓고 필요한 부분을 찾는 학생 (정확하지만 책상이 가득 참) |

**하이브리드의 아이디어**: SSM으로 대부분의 시퀀스를 효율적으로 처리하되, 주기적으로 Attention을 삽입하여 전역 동기화를 수행. "요약 노트를 쓰다가, 가끔 전체 교과서를 펼쳐 확인하는" 방식.

### 이론적 증명 (2026년 3월)

> *Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models.* [arXiv:2603.08859](https://arxiv.org/abs/2603.08859)

- 선택 복사(selective copying), 연관 기억(associative recall) 등의 과제에서, 순수 Transformer나 순수 SSM은 **큰 파라미터 또는 큰 메모리**가 필요
- **소규모 하이브리드 모델이 이 과제들을 증명 가능하게(provably) 해결**
- 학습된 하이브리드가 6배 큰 비하이브리드 모델보다 우수

### 우리 문제에서의 의미

현재 A안에서 Mamba는 인코더/디코더의 FullAttention 사이에 위치한다. Mamba 자체에 Attention을 추가하지 않더라도, 파이프라인 전체로 보면 "Attention → SSM → Attention" 하이브리드 구조이다. 만약 A안에서 공간 결합이 부족하다면, Mamba 내에 Attention을 삽입하는 것이 이론적으로 정당화된 대안이다.

---

## 2. 도메인별 주요 사례

### 2.1 대규모 언어 모델 (LLM)

| 모델 | 기관 (게재) | 규모 | SSM:Attn 비율 | 결합 방식 | 핵심 성과 |
|------|-----------|------|-------------|----------|----------|
| **Jamba** | AI21 (ICLR 2025) | 52B | 7:1 | 교차 배치 + MoE | 최초 대규모 하이브리드. 256K 컨텍스트 |
| **Zamba/Zamba2** | Zyphra (2024) | 7B | Mamba backbone + 공유 Attn 1~2개 | 공유 가중치 Attn | Mistral-7B, Gemma-7B 상회 |
| **Samba** | Microsoft (ICLR 2025) | 3.8B | ~1:1 | Mamba + Sliding Window Attn | 4K 학습 → 256K 추론 외삽 |
| **Nemotron-H** | NVIDIA (2025) | 8B/56B | ~6:1 | 균등 분산 (24 Mamba-2, 4 Attn) | 순수 Transformer 8B 전 과제 상회 |
| **Hymba** | NVIDIA (ICLR 2025) | 1.5B | 5:1 | **병렬 융합** (동일 레이어 내) | Llama 3.2 상회, 캐시 메모리 10분의 1 |
| **Bamba** | IBM (2025) | 9B | 9:1 | 교차 배치 | LLaMA-3.1-8B 대등, 7배 적은 데이터 |
| **Qwen3-Next** | Alibaba (2025) | 80B/3B active | 3:1 | **Gated DeltaNet** + Attn | Mamba-2 대신 GDN 채택. 90% 비용 절감 |
| **OLMo Hybrid** | AI2 (2026) | 7B | 3:1 | GDN + Attn | **통제 실험: GDN hybrid > pure GDN > Transformer > Mamba2 hybrid** |
| **Nemotron 3** | NVIDIA (2025) | 30B~500B | 하이브리드+MoE | 교차 배치 | 1M 컨텍스트, 4배 효율 |
| **Granite 4.0** | IBM (2025) | - | 9:1 | Mamba-2/Transformer | 70%+ RAM 절감. ISO 42001 인증 |
| **Falcon H1** | TII (2025-2026) | 7B | 하이브리드 | Mamba2 + Transformer | AIME-24 88.1% (47B 모델 상회) |

**LLM 도메인의 합의**:
- **7-8% Attention이면 충분** (NVIDIA Nemotron-H 실증)
- **Gated DeltaNet이 Mamba-2를 대체하는 추세** (Qwen3-Next, OLMo Hybrid, Qwen3.5)
- AI2의 통제 실험 결과: GDN 하이브리드 > 순수 GDN > 순수 Transformer > Mamba-2 하이브리드

### 2.2 비전 (Vision)

| 모델 | 기관 (게재) | 결합 방식 | 핵심 |
|------|-----------|----------|------|
| **MambaVision** | NVIDIA (CVPR 2025) | 계층적: 초기 Mamba, **후기 Self-Attention** | ImageNet 88.1% SOTA. Swin/ConvNeXt 상회 |
| **VMamba** | NeurIPS 2024 Spotlight | 순수 SSM (2D Selective Scan) | 하이브리드 아님, 비교 기준 |

MambaVision의 **계층적 배치**가 주목할 만하다: 초기 단계(저수준 특징)는 Mamba, 후기 단계(고수준 의미)는 Attention. 이는 "효율적 처리 → 전역 통합" 흐름.

### 2.3 비디오 이해

| 모델 | 기관 (게재) | 결합 방식 |
|------|-----------|----------|
| **VAMBA** | ICCV 2025 | Mamba-2 비디오 인코딩 + Cross-Attention 텍스트 갱신. 1024+ 프레임, 50% 메모리 절감 |

### 2.4 시계열 예측

| 모델 | 기관 (게재) | 결합 방식 |
|------|-----------|----------|
| **SST** | CIKM 2024 | MoE 라우팅: 입력 스케일에 따라 Mamba 또는 Transformer 선택 |
| **MLA** | 2026 | Mamba(전역) → LSTM(지역) → Attention(집약). 3단계 파이프라인 |

### 2.5 PDE / 과학 계산

| 모델 | 기관 (게재) | 결합 방식 |
|------|-----------|----------|
| **MNO** | JCP 2025 | Mamba 단독 (Attention 미사용). PDE 4개 벤치마크에서 Transformer 대비 ~90% 오차 감소 |

**주목**: PDE 도메인에서 Mamba+Attention 하이브리드는 **아직 주요 사례가 없다**. MNO는 Mamba 단독으로 우수한 성능을 보였으나, 공간 결합이 필요한 3D 다중물리 문제에서의 검증은 부재.

### 2.6 기타 도메인

| 도메인 | 모델 | 결합 방식 |
|--------|------|----------|
| 의료 영상 | CDA-Mamba, AttmNet, HybridMamba | 3D 분할에서 고의미 레이어에 Attention 선택 삽입 |
| 음성 | TRAMBA, HELIX, MambAttention | 시간/주파수 Mamba + Attention. +7.3% PESQ |
| 강화학습 | Decision Mamba (NeurIPS 2024) | Mamba(장기 기억) → Transformer(예측) 순차 파이프라인 |
| 확산 모델 | DiMSUM (NeurIPS 2024) | Mamba-Transformer 하이브리드 확산. FID 2.11 |

---

## 3. 결합 전략 분류

조사된 모델들의 결합 방식을 3가지로 분류할 수 있다:

### 3.1 교차 배치 (Interleaved) — 가장 보편적

```
[Mamba] → [Mamba] → ... → [Attention] → [Mamba] → ... → [Attention] → ...
```
- Jamba (7:1), Nemotron-H (6:1), Bamba (9:1), Qwen3-Next (3:1)
- SSM 레이어 사이에 **주기적으로 Attention 삽입**
- 장점: 구현 단순, 비율 조절 용이
- 단점: Attention 위치가 고정 (모든 시점에서 동일한 패턴)

### 3.2 병렬 융합 (Parallel) — 신흥

```
           ┌─ [Attention heads] ─┐
입력 ─────┤                      ├──→ 합산 → 출력
           └─ [SSM heads]       ─┘
```
- Hymba (NVIDIA ICLR 2025)
- 동일 레이어 내에서 Attention heads와 SSM heads가 **병렬로** 작동
- 장점: 매 레이어에서 두 가지 관점을 동시 활용
- 단점: 구현 복잡도 높음

### 3.3 계층적 배치 (Hierarchical)

```
[Mamba stages (저수준)] → [Attention stages (고수준)]
```
- MambaVision: 초기 Mamba, 후기 Attention
- 장점: 각 레이어 유형의 강점에 맞는 역할 분담
- 단점: 유연성 부족

### 우리 아키텍처(A안)와의 관계

A안은 사실상 **계층적 배치의 변형**이다:
```
[Encoder Attention (공간)] → [Mamba (시간)] → [Decoder Attention (공간+물리)]
```
Attention은 공간 처리에, Mamba는 시간 처리에 특화. 만약 Mamba 내부에도 Attention을 넣는다면 교차 배치가 된다.

---

## 4. Gated DeltaNet — Mamba-2의 후계자?

### 직관

OLMo Hybrid(AI2, 2026)의 통제 실험이 충격적이었다: **Mamba-2 하이브리드가 순수 Transformer보다 낮은 성능**. 반면 Gated DeltaNet 하이브리드는 모든 조합 중 최고 성능.

| 순위 | 조합 | 성능 |
|------|------|------|
| 1 | **GDN 하이브리드** (GDN + Attention) | 최고 |
| 2 | 순수 GDN | |
| 3 | 순수 Transformer | |
| 4 | Mamba-2 하이브리드 | |
| 5 | 순수 Mamba-2 | 최저 |

Gated DeltaNet은 Mamba-2의 게이팅에 **delta rule**(연관 기억 갱신 규칙)을 결합한 것으로, 상태 추적 능력이 향상된다.

### 우리 문제에서의 시사점

현재 Mamba-2(SSD)를 계획하고 있으나, Gated DeltaNet이 더 나은 선택일 수 있다. 단:
- GDN의 TensorFlow 2 구현은 미확인
- Mamba-2가 Drama에서 MBRL 검증 완료인 반면, GDN은 LLM에서만 검증
- **Mamba-2로 시작하고, 성능 부족 시 GDN 전환 검토**가 현실적

---

## 5. 우리 아키텍처 적용성 평가

### A안 맥락에서의 평가

A안은 이미 "Attention → SSM → Attention" 구조이므로, Mamba 자체에 Attention 추가는 현 시점에서 불필요. 그러나 관련 기법들은 향후 개선 시 참고 가능.

| 기법 | 적용 가능 | 근거 | 우선순위 |
|------|----------|------|---------|
| 교차 배치 (Mamba 내 Attention) | 보류 | A안 디코더 FullAttn이 이미 공간 결합 수행. 실험 결과 부족 시 전환 | 낮음 |
| Gated DeltaNet | 보류 | Mamba-2 대비 우위 보고. TF2 구현 확인 후 검토 | 중간 |
| 병렬 융합 (Hymba 패턴) | X | 구현 복잡도 대비 이점 불확실 | - |
| 계층적 배치 | O (이미 적용) | A안 자체가 Attn→SSM→Attn 계층 | - |
| Global token (TokenWM 유래) | △ | Mamba에 4~8 global token 추가로 전역 상태 유지. A안 보강 가능 | 중간 |

---

## 6. 핵심 차용 후보

### 즉시 적용 (A안 내에서)
- **A안 자체가 하이브리드**: Encoder(Attn) → Mamba → Decoder(Attn). 이론적으로 정당화됨

### A안 보강 시 적용 가능
- **Global token 추가**: Mamba에 4~8개 시스템 토큰을 추가하여 전역 상태(keff 추세, 총 Xe) 유지. Mamba 내 Attention 없이도 전역 정보 중개 가능
- **잠재 예측 loss (CDP)**: Mamba에 직접 gradient 제공. 디코더 FullAttn 경유 gradient의 보조 수단

### A안 실패 시 전환 대안
- **교차 배치 (C안)**: Mamba N blocks + Attention 1 block. 비율 4:1~7:1에서 시작
- **Gated DeltaNet**: Mamba-2 대체. LLM에서 우위 확인. PDE 도메인 검증 필요

### 부적합
- **병렬 융합 (Hymba)**: 구현 복잡도 과도. 우리 규모(~수 M params)에서 이점 불확실
- **MoE 라우팅 (SST)**: 데이터 스케일 부족. MoE는 대규모 데이터셋에서 효과적

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| Jamba | AI21. ICLR 2025. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| Zamba | Zyphra. 2024. [arXiv:2405.16712](https://arxiv.org/abs/2405.16712) |
| Samba | Microsoft. ICLR 2025. [arXiv:2406.07522](https://arxiv.org/abs/2406.07522) |
| Nemotron-H | NVIDIA. 2025. [arXiv:2504.03624](https://arxiv.org/abs/2504.03624) |
| Hymba | NVIDIA. ICLR 2025. [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) |
| Qwen3-Next | Alibaba. 2025. [vLLM blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html) |
| OLMo Hybrid | AI2. 2026. [AI2 Blog](https://allenai.org/blog/olmohybrid) |
| Bamba | IBM. 2025. [IBM Research](https://research.ibm.com/blog/bamba-ssm-transformer-model) |
| Nemotron 3 | NVIDIA. 2025. [NVIDIA Blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/) |
| Falcon H1 | TII. 2025-2026. [TII](https://falconllm.tii.ae/) |
| MambaVision | NVIDIA. CVPR 2025. [arXiv:2407.08083](https://arxiv.org/abs/2407.08083) |
| MNO | Cheng et al. JCP 2025. [arXiv:2410.02113](https://arxiv.org/abs/2410.02113) |
| Gated DeltaNet | ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| Expressivity-Efficiency | 2026. [arXiv:2603.08859](https://arxiv.org/abs/2603.08859) |
| TransMamba | AAAI 2026. [arXiv:2503.24067](https://arxiv.org/abs/2503.24067) |
| Mamba-3 | ICLR 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) |
