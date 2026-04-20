# UltraReview 요청서 — 원자로 시공간 AI 대리모델 아키텍처 검증

> **작성일**: 2026-04-20
> **목적**: Mamba-3 + Attention 하이브리드 기반 원자로 시공간 AI 대리모델의 **최종 확정 설계** 를 외부 검토자에게 전달하여 구조적 타당성 / 치명적 허점 / 대안 제안을 받기 위한 요청서.
> **활용 범위**: (1) 외부 LLM 수동 제출 (GPT / Gemini / Grok), (2) Zen MCP 등 자동화 도구 (설치 후), (3) 사내 시니어 연구자 피어 리뷰.
> **선행 외부 자문 기록**: 2026-04-16 (Q1~Q6 기반 시계열 아키텍처 자문) → Phase 2a/2b 분할, Phase 3 신설의 근거가 됨. 본 요청서는 후속 라운드.

---

## 0. 본 요청서 읽는 법 (검토자용)

### 0.1 프로젝트 한 줄 요약

**운영 중 원자로 노심의 5분 단위 상태 진화 (Xe/I 동역학, 제어봉 이동, 출력 변화) 를 물리 시뮬레이터 (MASTER) 수준의 정확도로 재현하는 AI 대리모델.** 출력: 노심 격자 (quarter 5×5, 500 cell) 의 10채널 물리 상태 (중성자속, 핵종 농도, 온도 등) 를 최대 575 step (약 48시간) 자기회귀 전개.

### 0.2 검토자에게 필요한 사전 지식

- **필수**: 딥러닝 시공간 대리모델 (Neural Operator, Mamba / SSM, Transformer) 기본. Physical Loss / PINN 개념
- **유용**: 원자로 노심 물리 (2군 확산 방정식, Xe/I Bateman ODE, 제어봉 효과) — 단, **본 요청서에 최소 배경은 포함됨**
- **프레임워크**: TF2 / Keras (구현 확정)

### 0.3 본 요청서 구조

- §1 아키텍처 요약 (파이프라인 한눈에 보기)
- §2 검토 질문 Q1~Q6 (핵심)
- §3 원치 않는 답변 (범위 제한)
- §4 기대 산출물 형식
- §5 첨부 문서 (원한다면 본 세트 전체를 함께 제공)

---

## 1. 아키텍처 요약

### 1.1 전체 파이프라인

```
[데이터셋 (HDF5)]
   ├ state(t): 10ch (Xe, I, flux 2군, 온도 2종, power, AO, ...)
   ├ xs_fuel: 10ch (시나리오 고정 거시단면적)
   ├ rod_map: 1ch (제어봉 삽입 깊이 분포)
   ├ p_load(t): 스칼라 (출력 수준 0~1)
   └ sym_type: 메타데이터 (mirror or rotation)
        ↓
[halo_expand] (5,5) → (6,6), sym_type 분기
        ↓
[Spatial Encoder] (구현 완료 · SE-A1~B3 V&V 통과)
   ├ CellEmbedder Conv3D 1×1×1 (21→128)
   ├ ConditionalLAPE3D (sym_type 분기 trainable 절대 PE)
   └ FullAttention3D × 3 stages (STRING 상대 PE)
        ↓ z(t) ∈ (B, 720, 128)
[Spatiotemporal Processor] ★코드 미착수★
   ├ N=2 Blocks: [Mamba-3 × 2 + Attention + 소형 MLP]
   ├ p_load 주입: 공유 임베더 → Mamba concat + Attention AdaLN-Zero (이중 경로)
   └ 절대위치 재주입: encoder_lape skip connection (Raw α + WD, ReZero 패턴)
        ↓ d(t) ∈ (B, 720, 128)
[Decoder] ★코드 미착수★
   └ Conv3D 1×1×1 (128→10), cell-wise 독립, linear activation
        ↓ delta_pred ∈ (B, 720, 10)
[잔차 결합] state(t) + delta_pred = X_next_pred (정규화 공간)
        ↓
[Loss 계산] ★코드 미착수★
   ├ L_data (정규화 공간 MSE)
   ├ L_data_halo (λ=0.3)
   └ L_physics (역정규화 후 물리 단위)
       ├ L_Bateman (Xe/I ODE 잔차, Branch + CRS 공통)
       ├ L_sigma_a_Xe (Taylor 단면적 정합성)
       ├ L_diffusion (L_diff_rel, 상대 잔차)
       └ L_keff (후순위/검토중)
```

### 1.2 핵심 설계 결정 (확정)

| 영역 | 결정 | 대안 대비 |
|------|------|---------|
| **공간 인코더** | FullAttention × 3, ConditionalLAPE3D (mirror/rotation 2 테이블), STRING 상대 PE | Equivariant 모델, 단일 LAPE 등 6개 대안 비교 후 확정 |
| **프로세서** | Mamba-3 + Attention 하이브리드 (Jamba 패턴), N=2 블록, Mamba:Attention = 2:1 | A안 (cell-wise Mamba 단독), B안 (DreamerV3 단일 벡터 압축) 탈락 |
| **p_load 주입** | 이중 경로 — Mamba concat + Attention AdaLN-Zero | 단일 경로 (디코더만) → 물리 일관성 사유로 이중 경로 변경 |
| **LAPE skip** | Raw α + WD (ReZero 패턴) — 인코더 LAPE를 attention 직전 add, stop_gradient | 미적용, ReZero 정통 대안 |
| **디코더** | Conv3D 1×1×1 경량, **linear activation (활성화 없음)** | FullAttention+AdaLN (A안 원형) 탈락. Softplus/ReLU는 정규화 공간 부호 제약으로 금지 |
| **정규화** | z-score (일반) + log-z-score (양수 제약 채널) — 채널별 선택 | 양수성 보장은 역정규화 단계의 exp가 담당 |
| **L_Bateman** | 해석해 (행렬 지수함수), Phase 1부터 적용 (Branch 5분 진화 확인 완료) | Euler forward, RK4 탈락 |
| **학습 방법론** | 3-Phase — (1) Branch 공간 사전학습 → (2a) Mamba 단독 학습 + (2b) 상호 적응 → (3) K-step 자기회귀 | 2-Phase 단순화 탈락 (외부 자문 후 Phase 2a/2b 분할 + Phase 3 신설) |
| **h(t) 관리** | Detach 기본, Phase 2 말기 제한적 gradient 허용 | Full gradient-through (외부 자문에서 충돌) |
| **차등 학습률** | Phase 2b에서 E:1e-5, A:5e-5, M:1e-3 | 통일 LR 탈락 |

### 1.3 데이터 규모

- **LP (Loading Pattern)**: 100개 (서로 다른 연료 장전 배치)
- **p_load profile**: 20종 (출력 시나리오)
- **CRS (Control Rod Scenario)**: LP×profile 당 1개 (round-robin 매칭) → 총 100개 시나리오
- **시퀀스 길이**: 575 step (5분 간격, 약 48시간)
- **Branch**: CRS 각 시점에서 rod_map만 변경한 29개 파생 케이스
  - CRS 전체: 100 시나리오 × 575 step = 57,500 시점
  - Branch 전체: ≈ 1,667,500 시점 (5분 Xe 진화 반영 확인됨)

### 1.4 대칭 처리

- 원자로 노심은 대칭 (mirror: 4조각, rotation: 8조각) 이 존재 → quarter 5×5 만 저장
- halo_expand 로 인코더 내부만 (6,6) 확장 (대칭 조건 반영), 최종 crop
- ConditionalLAPE3D: sym_type → mirror 테이블 or rotation 테이블 선택 (add 방식, residual stream)

---

## 2. 검토 질문 (Q1~Q6)

각 질문에 대해 **(a) 이론적 타당성**, **(b) 본 문제 (원자로 물리) 특화 적합성**, **(c) 잠재 허점 / 개선안** 3가지 관점에서 답변 요청.

### Q1. 시공간 프로세서의 Mamba + Attention 하이브리드 구조

**현재 결정**:
- N=2 블록 적층, 각 블록 = [Mamba-3 × 2 → Attention → 소형 MLP]
- Mamba:Attention = 2:1 비율
- Attention은 각 시점 독립적으로 720 cell 공간 결합 (stateless)
- Mamba는 cell별 독립 시간 진화

**검토 요청**:
- 이 구조가 원자로 **시공간 결합 동역학** (공간 분포 변화 + 시간 진화가 결합된 Xe 진동, 제어봉 이동 효과) 을 충실히 표현할 수 있는가?
- Mamba:Attention 비율 2:1 이 최적인가? 본 물리 문제에서 시간 > 공간 이라면 3:1 또는 4:1이 더 나은가? 그 반대인가?
- N=2 깊이가 5분 단위 × 575 step 시퀀스에 충분한가? 깊이를 늘리면 parallel scan 비용이 어떻게 증가하는가?
- Mamba → Attention 순서의 이론적 근거가 본 프로젝트에 유효한가? 역순 (Attention → Mamba) 이 유리한 시나리오가 있는가?

### Q2. p_load 이중 경로 주입 (Mamba concat + Attention AdaLN-Zero)

**현재 결정**:
- `p_load` (출력 수준 스칼라) 를 공유 임베더로 8D 벡터 변환
- **Mamba concat 경로**: 각 cell feature 에 broadcast concat (시간 동역학에 직접 주입)
- **Attention AdaLN-Zero 경로**: Attention block의 Pre-LN에 γ, β 조건 변조 (공간 결합 변조)
- 두 경로 모두에 gradient 흐름

**검토 요청**:
- 이중 경로의 필요성 — 단일 경로 (AdaLN만 또는 concat만) 대비 실제 기여는?
- 두 경로 간 정보 중복 (redundancy) 로 학습이 혼란스러워질 위험은?
- p_load 임베더를 공유 (shared) 해도 되나, 경로별 분리해야 하나?
- AdaLN-Zero의 γ, β = 0 초기화가 학습 초기 "경로가 없는 것과 같은 상태"에서 시작 → Mamba concat 만 유효한 상황에서 학습 경로가 편향되지 않는가?

### Q3. 3-Phase 학습 방법론 (Phase 1 → 2a/2b → 3)

**현재 결정**:

| Phase | 학습 대상 | 입력 | Loss | 기간 비중 |
|:-----:|---------|------|------|:--------:|
| **Phase 1** | Encoder + Attention(Phase 1 내부) + Decoder. Mamba 1-step | Branch 데이터 (풍부, 공간 다양성) | L_data + L_physics (단일 step) | 대 |
| **Phase 2a** | **Mamba 단독** 학습 (Encoder/Attn/Decoder freeze) | CRS parallel scan + Branch sampling | L_data + L_physics + Branch 보조 | 중 |
| **Phase 2b** | 전체 미세조정 (차등 학습률 — Mamba 1e-3, 나머지 1e-5~5e-5) | CRS parallel scan | L_data + L_physics | 중 |
| **Phase 3** | K-step 자기회귀 + 커리큘럼 (K=2→4→8→16→...→575) | CRS 자기회귀 rollout (parallel scan OFF) | L_data + L_physics 강조 + Scheduled sampling | 소~중 |

**검토 요청**:
- Phase 2a → 2b 전이 비율이 30% 에폭에서 적절한가? 전이 판정 지표 (수렴 기준) 는 무엇을 써야 하나?
- Phase 3 K-step 커리큘럼의 지수 증가 (1→2→4→8→...) 가 선형 증가 (1→2→3→...) 또는 고정 K 대비 유리한 물리적 근거는?
- Phase 3에서 Teacher Forcing OFF → Scheduled Sampling 전이 필요한가? 필요하다면 어떤 스케줄?
- 차등 학습률의 비율 (Mamba가 다른 모듈의 100~20배) 이 co-adaptation을 방해하지는 않는가?
- Phase 1 에서 **Branch 1-step 학습** 이 "Mamba 학습을 쉽게 하는 바탕" 이 되는가, 아니면 "Mamba 없는 공간 인코더의 편향" 을 고착시키는가?

### Q4. h(t) 관리 — Detach vs Gradient-through

**현재 결정**: Detach 기본 (h(t) 스냅샷 후 branch forward pass 시 branch loss 가 CRS prefix 로 역전파 안 됨). Phase 2 말기에 제한적 gradient 허용 (실험적 옵션).

**검토 요청**:
- 외부 자문 답변 A (detach 권장) vs 답변 B (gradient 통과 권장) 이 충돌했음. 본 물리 문제 (Xe τ ≈ 9.2h >> 5분 step) 에서 실제로 어느 쪽이 더 정확한가?
- Detach 기본이 Branch 다양성 신호 (≈ 1.67M 시점) 를 CRS 학습에 전혀 활용하지 못하는 손실을 어떻게 보상해야 하나?
- Gradient-through 도입 시 575-step backward 그래프 + 29 branch 조합의 메모리 폭발을 어떻게 관리하나?

### Q5. Physical Loss 조합 (L_Bateman / L_sigma_a_Xe / L_diffusion / L_keff)

**현재 결정**:

| Loss | 역할 | 계산 공간 | 시간 의존 | Phase 적용 |
|------|------|---------|---------|----------|
| L_data | 데이터 fitting | 정규화 공간 MSE | 무 | 전 Phase |
| L_data_halo (λ=0.3) | halo cell 보강 | 정규화 공간 | 무 | 전 Phase |
| **L_Bateman** | Xe/I ODE 잔차 (해석해) | 역정규화 물리 단위 | 유 (단일 step) | Phase 1부터 |
| **L_sigma_a_Xe** | Taylor 단면적 정합성 | 역정규화 물리 단위 | 무 | Phase 1부터 |
| **L_diffusion** (L_diff_rel) | 상대 PDE 잔차 (CMFD 7-point) | 역정규화 물리 단위 | 무 | Phase 1부터 |
| L_keff | Rayleigh 몫 | 역정규화 물리 단위 | 무 | **후순위/검토중** |

**검토 요청**:
- L_keff 후순위 판단이 타당한가? Rayleigh 몫 계산의 수치적 불안정성 (분모에 φᵀφ) 이 본 문제 규모에서 얼마나 치명적인가?
- L_Bateman + L_sigma_a_Xe + L_diffusion 세 항이 gradient 스케일 차이로 서로를 억제하는 문제는? ReLoBRaLo 등 자동 가중치 조정 필요 시점은?
- L_diffusion (L_diff_rel 상대 잔차 형식) 이 절대 잔차 대비 학습에 유리한 이유가 본 문제에서 유효한가?
- 역정규화 후 물리 단위 계산은 log-z-score 채널의 exp 역변환이 양수성을 자동 보장. 이것이 `L_Bateman` 의 `log(N_Xe)` 같은 연산에서 안전한가?

### Q6. 외부 제어 입력 (rod_map, xs_fuel) 의 처리

**현재 결정**:
- `xs_fuel` (시나리오 고정) + `rod_map` (시간 변화) 을 state(t) 의 일부로 concat
- 모델 내부에서 별도 조건 주입 경로 없음

**검토 요청**:
- `xs_fuel` (연료 단면적) 이 "시나리오 고정 상수" 인데 state 에 포함하여 매 시점 재계산하는 것이 낭비인가? 별도 condition vector 로 분리해 효율화 가능한가?
- `rod_map` 은 제어봉 삽입 깊이 "분포" 로 시공간 급변 신호. Mamba 가 이 급변 신호를 안정적으로 처리하는가? (경험상 SSM은 급변 신호에 약점)
- rod_map 을 별도 채널로 두지 말고, FiLM 또는 cross-attention 으로 주입하는 것이 유리하지 않은가?

---

## 3. 범위 외 (이미 확정되어 되돌리지 않을 항목)

검토자는 아래 항목에 대해서는 **"그 방식 말고 X를 쓰라"는 의견을 내지 말 것**:

- **프레임워크**: TF2 / Keras (PyTorch, JAX 로 변경 불가)
- **공간 격자**: Quarter 5×5 → halo 6×6 (720 cell)
- **인코더**: FullAttention × 3, STRING + ConditionalLAPE3D (SE-A1~B3 V&V 통과 완료, 구현 완료)
- **데이터 출처**: MASTER (원자로 노심 해석 코드) — 다른 시뮬레이터로 대체 불가
- **정규화 방식**: z-score / log-z-score 채널별 선택 (이미 전처리 완료)
- **Loss 적용 시점**: L_physics는 **역정규화 후 물리 단위**, L_data는 정규화 공간 (확정)

단, 위 항목이 **Q1~Q6의 답변에 직접 영향을 준다** 면 언급 가능. 예: "L_diffusion 가중치는 STRING 상대 PE의 스케일과 결합되어..." 같은 맥락.

---

## 4. 기대 산출물 형식

검토자에게 다음 형식으로 회신 요청:

```markdown
# UltraReview 회신 — [검토자 이름 / 모델명]
## 작성일: YYYY-MM-DD

## 전체 평가 (한 줄)
[예: "구조적으로 타당하나 Phase 3 커리큘럼에 치명적 허점 존재"]

## Q1 Mamba+Attention 하이브리드 구조
### (a) 이론적 타당성
### (b) 원자로 물리 특화 적합성
### (c) 잠재 허점 / 개선안
[구체 제안 + 참고 문헌]

## Q2 p_load 이중 경로
...

## Q3 3-Phase 학습
...

(... Q6까지)

## 종합 권고 사항 (priority-ranked)
1. [가장 중요한 개선 제안]
2. ...

## 부록: 인용 문헌 / 관련 사례
```

**답변 길이 가이드**: 각 Q당 300~800단어. 종합 권고 100~300단어. 인용은 논문 제목 + 저자 + 연도 + DOI/arXiv ID (가능 시).

---

## 5. 첨부 문서 (필요 시 함께 전달)

본 요청서로 충분하지 않다면 아래 문서를 **추가로** 전달:

### A. 메인 참조
- [2026-04-20 모델 설계 최종 확정 사항 (통합 정리).md](2026-04-20 모델 설계 최종 확정 사항 (통합 정리).md) — 본 요청서의 기반 문서, 전 설계 영역 망라
- [2026-04-20 3-Phase 학습 방법론 (1차 정리, 추가 검토 예정).md](학습 방법론 설계/2026-04-20 3-Phase 학습 방법론 (1차 정리, 추가 검토 예정).md) — Q3 상세

### B. 컴포넌트별 설계
- [2026-03-30 모델 구현 계획(공간 인코더).md](공간인코더 구현 계획/2026-03-30 모델 구현 계획(공간 인코더).md) — Q1 공간 인코더 부분 (패치박스 포함)
- [2026-04-15 모델 구현 계획(시계열).md](시계열모델(SSM) 구현 계획/2026-04-15 모델 구현 계획(시계열).md) — Q1/Q2 시계열 프로세서
- [2026-04-17 ConditionalLAPE 정보 단절 우려 및 정보 주입 방안 검토.md](시계열모델(SSM) 구현 계획/2026-04-17 (최종결정 전 검토본) ConditionalLAPE 정보 단절 우려 및 정보(위치, 대칭, 출력조건) 주입 방안 검토.md) — Q2/Q6 주입 전략 근거

### C. Physical Loss
- [Physical Loss 통합 레퍼런스](디코더 구현 및 Physical Loss 적용/Physical loss 검토 및 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md) — Q5 근거
- [2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md](디코더 구현 및 Physical Loss 적용/Physical loss 검토 및 개선 계획/L_Bateman/2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md) — Q5 L_Bateman Phase 1 적용 근거

### D. 선행 외부 자문 (참고용)
- [2026-04-16 시계열 아키텍처 설계 검토 (외부 자문 답변 A/B)](시계열모델(SSM) 구현 계획/2026-04-16 (최종결정 전 검토본) 시계열 아키텍처 설계 검토.md)
- [2026-04-16 시계열 학습방법론 검토 (외부 자문 Q1~Q6)](학습 방법론 설계/2026-04-16 (최종결정 전 검토본) 시계열 학습방법론 검토.md)

### E. 검증 및 실험 방법론 (Phase별 테스트 체계)
- [2026-04-20 컴포넌트 검증 및 실험 방법론 (초안).md](학습 방법론 설계/검증 및 실험 계획/2026-04-20 컴포넌트 검증 및 실험 방법론 (초안).md) — TR-S/V/R/X/M/P 테스트 ID 체계, Top-7 토론 집중 결정, MLflow 운영

---

## 6. 검토자 제출 안내

### 6.1 수동 제출 (외부 LLM)

1. 본 §0~§4 를 먼저 제출 (본 요청서만 읽어도 답변 가능하도록 구성)
2. 깊이 있는 답변을 원하면 §5 첨부 문서를 추가 제공
3. 권장 모델: GPT-5 / Gemini Ultra / Grok / Claude (본 AI 외)

### 6.2 자동화 (Zen MCP 등 설치 후)

- Node.js 설치 필요 (https://nodejs.org/ LTS 버전)
- Zen MCP 설치 후 `codereview` 또는 `thinkdeep` 명령으로 여러 모델에 자동 제출 가능

### 6.3 회신 수신 후

- 회신은 `implementation_plans/UltraReview 결과/{날짜} {모델명} 회신.md` 형식으로 저장
- 통합 정리 문서에 §7 (신설) "UltraReview 2차 검토 반영" 으로 결과 통합 → 3-Phase 학습 방법론 및 관련 설계 2차 정리 진행
