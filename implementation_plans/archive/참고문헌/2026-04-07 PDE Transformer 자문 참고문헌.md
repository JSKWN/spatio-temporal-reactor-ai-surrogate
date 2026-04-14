# PDE Transformer 자문 참고문헌

> **작성일**: 2026-04-07
> **배경**: 공간 인코더 설계 재개 시점에 외부 자문(2025~2026 PDE Transformer 동향)을 받음. 본 문서는 자문에서 거론된 논문들을 정리하고, 우리 모델에 차용 가능한 부분과 그 외 차이점을 기록.
> **도입 여부**: **모두 미정**. 본 문서는 아이디어 풀(pool)이며, 실제 채택은 SE-Phase 진행 중 별도 결정.
> **관련 작업 문서**: `implementation_plans/공간인코더 구현 계획/2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md`

---

## 1. 차용 후보 핵심 정리

| 논문 | 분야 | 우리에게 유용한 부분 | 도입 여부 |
|------|------|----------------------|:---:|
| DiT (2023) | 이미지 생성 | **AdaLN-Zero 블록 디자인** | 미정 (유력) |
| PDE-Transformer (ICML 2025) | PDE 풀이 | "DiT 블록이 PDE 도메인에서도 잘 됨"의 사후 검증 (사례 인용) | 인용만 |
| STRING (ICML 2025) | 위치 인코딩 | **Lie 군 + 교환조건, Z-XY 결합 + 비균일 메시 + 이동 불변성 엄밀** | **2026-04-04 검토에서 확정** (RPE→STRING 단계적 교체) |
| LieRE (2024) | 위치 인코딩 | STRING의 모태 (교환조건 없는 Lie 군 회전 PE) | 비교 대안 (선정 안 됨) |
| IFactFormer-m (2025) | PDE/유체 | 병렬 축 분해(parallel factorization) | 미정 (옵션) |
| SpiderSolver (NeurIPS 2025) | 비균일 메시 | 거미줄 토크나이제이션 — 추후 반사체 포함 시 참고 | 보류 |
| PINTO (2025) | Physics-informed | Cross-attention 커널 적분 아이디어 — physical loss 설계 참고 | 보류 |
| Point-wise DiT (arXiv:2508.01230, 2025) | 물리 시스템 | AdaLN-Zero 2회 적용 (좌표 + 파라미터 분리 주입) 패턴 | 미정 |

---

## 2. 논문별 상세

### 2.1 DiT — AdaLN-Zero 블록의 원본

- **출처**: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
- **arXiv**: 2212.09748

#### 핵심 기여
ViT 블록의 LayerNorm을 **조건부 변조**로 교체:
```
표준 ViT:    x = x + Attention(LN(x))
             x = x + FFN(LN(x))

DiT(AdaLN-Zero):  γ1,β1,α1, γ2,β2,α2 = MLP(condition)
                  x = x + α1·Attention(γ1·LN(x) + β1)
                  x = x + α2·FFN(γ2·LN(x) + β2)
                  # α 초기값 = 0 → 학습 시작 시 identity
```

#### Diffusion과의 관계 — **블록 자체는 무관**
- DiT의 "D"는 처음 적용된 분야가 diffusion이었기 때문
- 블록 입장에서 condition은 그냥 스칼라/벡터 (timestep, class, viscosity, p_load 등 무엇이든 가능)
- **노이즈 복원 학습 절차와 블록 디자인은 직교**
- → **diffusion 모델이 아니어도 사용 가능**

#### FiLM과의 차이
| | 기존 FiLM | AdaLN-Zero |
|---|---|---|
| 변조 위치 | raw feature | **LayerNorm 후** (정규화 분포) |
| 잔차 게이팅 | 없음 | **α 학습 가능, 0으로 초기화** |
| 학습 안정성 | 조건 신호 강도에 민감 | 둔감 (분포가 unit variance에서 시작) |
| 블록당 파라미터 | ~2D | ~6D² (조건 MLP) |

→ 사용자 인식대로 **"ViT + FiLM 개선판"**이 정확한 한 줄 요약.

#### 우리에게 유용한 부분 — **AdaLN-Zero 블록 디자인 자체**
- p_load(스칼라) 주입에 직접 적용 가능
- 향후 time embedding, burnup, boron 등 추가 조건도 동일 메커니즘
- 기존 Pre-LN Transformer 블록에 5~10줄 drop-in
- **차용 대상: 블록 디자인 1개. Diffusion 학습 절차는 채택 안 함.**

#### 우리와 다른 점 (차용하지 않는 부분)
- Diffusion forward/reverse process — 우리는 결정론적 회귀 (다음 시점 직접 예측)
- Class label embedding — 우리는 연속 스칼라(p_load)
- 픽셀-노이즈 학습 — 우리는 물리량-잔차 학습

---

### 2.2 PDE-Transformer — DiT를 PDE에 갖다 쓴 사례

- **출처**: ICML 2025 (PMLR v267)
- **arXiv**: 2505.24717
- **저장소**: `tum-pbs/pde-transformer`
- **제목**: PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations

#### 핵심 기여
1. **DiT 블록을 그대로 가져옴** (AdaLN-Zero 포함)
2. **Diffusion training은 안 함** — MSE 회귀 학습으로 다음 시점 직접 예측
3. 조건 입력 매핑: t(diffusion 타임스텝) → 물리 시간/파라미터(viscosity, Reynolds 등)
4. PDE 특화 wrapper 추가:
   - Multi-scale 처리 (계층적 패치)
   - Non-square 도메인 지원
   - 주기 경계 조건
5. 16종 PDE 데이터셋에서 FactFormer·scOT 대비 우월

#### 우리에게 유용한 부분 — **사후 검증 사례 인용만**
- "DiT 블록(AdaLN-Zero)이 PDE 도메인에서도 잘 작동한다"는 ICML 2025 검증을 §3 근거로 인용
- 본질적으로 **DiT 블록 차용 = AdaLN-Zero 차용**과 동일하므로 추가 차용 항목 없음

#### 우리와 다른 점 — **백본 자체 차용은 부적합**
| | PDE-Transformer | 우리 모델 |
|---|---|---|
| 시간 처리 | 각 시점 독립 + roll-out | **Mamba SSM (시간 hidden 누적)** |
| 입력 단위 | (B, C, D, H, W) 전체 시점 | per-timestep 인코더 → Mamba → per-timestep 디코더 |
| Multi-scale | 큰 격자(256×256 등)에 유효 | (20,5,5)에 무가치 |
| Non-square 도메인 | 핵심 기능 | 불필요 (정육면체) |
| Periodic BC | 핵심 기능 | 불필요 (반사 BC) |
| Mamba와의 인터페이스 | 검증 안 됨 | 우리가 직접 설계 |

→ **백본 전체 차용 ✗**, **DiT 블록(AdaLN-Zero) 차용만 의미 있음** — 그리고 그건 PDE-Transformer 없이 DiT 원본만 봐도 충분

---

### 2.3 STRING — Translation-Invariant 위치 인코딩 ★ 4월 4일 확정

- **출처**: ICML 2025 Spotlight
- **arXiv**: 2502.02562
- **제목**: STRING: Separable Translationally Invariant Position Encodings
- **선정 근거 문서**: `공간인코더 구현 계획/2026-04-04 3D 위치 인코딩 기법 검토.md` §1, §4

#### 핵심 기여
- **R(r) = exp(L₁·r₁ + L₂·r₂ + L₃·r₃)** — 좌표 r을 d×d 직교 회전 행렬로 인코딩
- 학습 가능 generator {L_k}가 dense d×d → 축 간 결합 자동 표현
- **교환 조건(commutativity)** 부과로 R(rᵢ)ᵀR(rⱼ) = R(rⱼ - rᵢ) **수학적 엄밀 보장**
- "행렬 곱셈을 사용하는 모든 translation-invariant PE 중 가장 일반적" (논문 self-claim)
- 연속 실수 좌표 입력 → 비균일 메시 자연 처리
- gradient 안전 (call() 내부 forward 연산, RPE의 Prob-6 회피)

#### 우리에게 유용한 부분
- **위치 인코딩 1순위 (4월 4일 확정)**
- 추후 반사체 포함 시 비균일 Z 메시(30+10·n+30 cm) 직접 입력
- 제어봉 Z축 변화의 XY 평면 영향(Z-XY cross coupling) 자동 인코딩
- **이동 불변성 엄밀 보장** = 동일 상대 거리 노드 쌍이 위치 무관하게 동일 attention 가중치 → 확산방정식의 본질(translation invariance)에 정확히 부합

#### 도입 단계 (4월 4일 결정)
| 단계 | PE | 이유 |
|---|---|---|
| 초기 (SE-A, SE-B) | RPE (gradient 수정) | 5줄 수정으로 즉시 사용, 이미 검증 |
| 성능 실험 후 | **STRING 교체** | RPE 대비 개선 확인 후 교체 |
| Fallback | 3D RoPE | STRING 구현 지연 시 |

---

### 2.4 LieRE — STRING의 모태 (선정 안 됨)

- **출처**: Stanford 2024
- **arXiv**: 2406.10322
- **제목**: LieRE: Generalizing Rotary Position Encodings

#### 핵심 기여
- RoPE(1D 회전 PE)의 일반화 — 임의 차원/연속 좌표로 확장
- dense generator matrix L_k → 축 간 결합 표현
- 연속 좌표 입력, gradient 안전

#### STRING과의 차이 — STRING이 채택된 이유
| | LieRE | STRING |
|---|:---:|:---:|
| 이동 불변성 | △ 근사 | **✅ 엄밀** (교환 조건) |
| 자유도 | 더 자유로운 회전 학습 가능 | 교환 조건으로 자유도 일부 제약 |
| 비균일 메시 | ✅ | ✅ |
| 검증 | Stanford 2024 | ICML 2025 Spotlight |

→ 확산방정식의 translation invariance가 우리 핵심 물리이므로, **수학적 엄밀 보장이 있는 STRING을 택함**. LieRE는 비교 대안으로만 기록.

---

### 2.5 IFactFormer-m — 병렬 축 분해

- **출처**: 2025
- **arXiv**: 2412.18840
- **제목**: Implicit Factorized Transformer

#### 핵심 기여
- 기존 FactFormer의 **chained axis factorization**(축 순차 처리)을 **parallel factorization**으로 교체
- Z-XY 상호작용 강화 (체인 방식의 정보 손실 회피)
- 3D 난류 채널 유동 장기 예측에서 FNO·IFNO 능가

#### 우리에게 유용한 부분
- **Full Attention 외 옵션**으로 검토 가치
- N≈500에서 매우 저비용
- LieRE PE와 결합 가능

#### 우리와 다른 점
- 우리는 N=500이라 Full Attention 자체가 트리비얼 → factorization 이득 제한적
- 일반 PDE/유체 검증, 노심 격자 검증 부재
- → **1순위 Full Attention, 성능 부족 시 fallback 옵션**

---

### 2.6 SpiderSolver — 비균일 메시/복잡 기하

- **출처**: NeurIPS 2025
- **제목**: SpiderSolver

#### 핵심 기여
- **거미줄(spiderweb) 토크나이제이션** — 복잡한 기하 + 불규칙 이산화 처리
- 도메인 경계 기하로 가이드하는 패치 분할

#### 우리에게 유용한 부분 — **현 단계 보류, 추후 참고**
- 추후 반사체(K=1, K=22, 30cm) 포함 시 Z축 비균일 메시 처리 참고
- 현재 quarter-core 균일 메시에는 과잉

#### 우리와 다른 점
- 우리는 정규 격자 (Z=20, qH=5, qW=5) — 거미줄 토크나이제이션 불필요
- 반사체 포함 확장 시점에 재검토

---

### 2.7 PINTO — Physics-Informed Cross-Attention

- **출처**: 2025, Computer Physics Communications
- **제목**: PINTO: Physics-informed Transformer Neural Operator

#### 핵심 기여
- **Cross-attention 기반 반복 커널 적분 연산자**
- PDE 솔루션 도메인 좌표 → 경계 조건 인식 벡터로 변환
- **시뮬레이션 데이터 없이 물리 손실만으로 학습 가능**
- 초기·경계 조건 일반화

#### 우리에게 유용한 부분 — **physical loss 설계 참고, 백본 차용 아님**
- Cross-attention을 커널 적분으로 해석하는 시각 — 우리 L_diffusion/L_Bateman의 공간 결합 표현에 아이디어 차용 가능
- 단, **우리는 GT 데이터 기반 학습이 주된 학습 방식**이고 physical loss는 보조

#### 우리와 다른 점
- PINTO: data-free, physics-only 학습
- 우리: data-driven + physics-loss 보조 (역할 반대)
- 백본 구조 차용은 부적합

---

### 2.8 Point-wise Diffusion Models for Physical Systems

- **출처**: 2025
- **arXiv**: 2508.01230

#### 핵심 기여
- **AdaLN-Zero를 두 번 적용**하는 point-wise DiT 블록
- 좌표 조건 + 물리 파라미터를 각각 별도 2-layer MLP로 임베딩
- 두 조건을 분리해서 adaptive layer normalization으로 주입

#### 우리에게 유용한 부분 — **이중 AdaLN-Zero 패턴**
- 우리도 좌표(LieRE PE)와 p_load(스칼라)를 분리해서 주입할 수 있음
- 두 신호의 물리적 의미가 다르므로(공간 위치 vs 운전 조건) 분리가 자연스러움

#### 우리와 다른 점
- 본 논문도 diffusion 학습 절차 사용 (우리는 미사용)
- point-wise 처리 — 토큰 간 attention 미사용. 우리는 attention 핵심
- → **이중 조건 주입 아이디어만** 차용

---

## 3. 차용 정책 정리

### 3.1 차용 (블록/패턴 수준)
- **AdaLN-Zero 블록** (DiT 2023) — p_load 주입용
- **이중 조건 주입 패턴** (Point-wise DiT 2025) — 좌표 + p_load 분리 주입
- **STRING PE** (ICML 2025) — 위치 인코딩 (4월 4일 검토에서 확정, RPE→STRING 단계적 교체)

### 3.2 사후 검증 인용
- **PDE-Transformer (ICML 2025)** — "AdaLN-Zero가 PDE 도메인에서도 SOTA" 근거로 인용

### 3.3 옵션/Fallback
- **IFactFormer-m** — Full Attention 성능 부족 시 fallback

### 3.4 아이디어 참고
- **PINTO** — Physical loss 공간 결합 표현에 cross-attention 시각 차용 가능

### 3.5 보류
- **SpiderSolver** — 추후 반사체 포함 시 재검토

---

## 4. 명시적 비채택 사항

| 기술 | 비채택 이유 |
|------|------------|
| Diffusion forward/reverse process | 우리는 결정론적 회귀 (현재 → 다음 시점 직접 예측). 노이즈 복원 절차 불필요 |
| PDE-Transformer 백본 전체 | Mamba와 인터페이스 검증 부재, multi-scale·non-square·periodic BC 모두 우리 (20,5,5)에 무가치 |
| PINTO data-free 학습 | 우리는 데이터(MASTER GT) 풍부 + physical loss 보조. 학습 패러다임 반대 |
| Class label embedding (DiT) | 우리 조건은 연속 스칼라 (p_load). 카테고리 임베딩 불필요 |

---

## 5. 정리

> 외부 자문에서 거론된 5+1개 논문(DiT 포함) 중, 우리에게 본질적으로 유용한 차용 대상은 **(1) AdaLN-Zero 블록 디자인** (DiT 2023, PDE-Transformer 검증), **(2) STRING 위치 인코딩** (4월 4일 자체 검토에서 이미 확정), **(3) 이중 조건 주입 패턴** (Point-wise DiT) 3가지. 나머지는 백본 구조 차이, 학습 패러다임 차이, 격자 크기 차이로 부적합하거나 보류.
>
> **AdaLN-Zero, 이중 조건 주입 도입 여부는 SE-Phase 진행 중 별도 결정.** STRING은 4월 4일 검토에서 확정. 본 문서는 의사결정의 근거 풀(pool).
