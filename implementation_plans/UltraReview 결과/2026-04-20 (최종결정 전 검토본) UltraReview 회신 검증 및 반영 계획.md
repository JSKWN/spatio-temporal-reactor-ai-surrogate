# UltraReview 회신 검증 및 반영 계획

> **작성일**: 2026-04-20
> **목적**: `2026-04-20 Claude Opus 4.7 회신.md` 의 검토 의견을 **자체 재검토** (self-critique) 한 뒤, 놓친 잠재 위험을 추가 식별하고, 최종 결론을 기존 설계 문서에 반영하기 위한 **실행 계획** 을 제시.
> **작성 맥락**: `/ultraplan` 명령어 기반. 요청서 (`2026-04-20 UltraReview 요청서.md`) 와 회신 본문을 재확인 후 작성.
> **본 문서의 위치**: 회신 자체는 외부 검토자 관점의 **의견** 이며, 본 문서는 그 의견을 검증 + 실행 연결 하는 중간 다리.

---

## 0. 요약 (한 눈에)

### 0.1 회신의 권고 사항 (10개) 의 재검토 결론

| # | 권고 (회신 §종합) | 우선순위 (회신) | 재검토 판정 | 판정 근거 |
|---|-----------------|--------------|---------|---------|
| 1 | Phase 2b 차등 LR 100× → 20× 축소 | 🔴 P0 | **유지** (수치 완화) | 방향 타당. 다만 "ULMFiT/LoRA 표준 10~20×" 는 과장. **10~50× 범위 권고** 로 수정 |
| 2 | Phase 3 Scheduled Sampling 명시 | 🔴 P0 | **유지** | 표준 기법. 즉시 반영 가능 |
| 3 | N=2 → N=3 baseline 변경 | 🟡 P1 | **권고 약화** | "Xe oscillation 2 rounds tight" 는 직관적 논증일 뿐 물리적 엄밀성 부족. **ablation 필수** 로만 유지, baseline 은 실험 결과로 결정 |
| 4 | Phase 2a → 2b 전환 loss-plateau 자동화 | 🟡 P1 | **유지** | 방향 타당. 트리거 수치 구체화 필요 |
| 5 | Uncertainty Weighting (Kendall) 도입 | 🟡 P1 | **유지** (대안 병기) | PINN 도메인에서는 ReLoBRaLo 도 경쟁력. **둘 다 검토 대상** 으로 확장 |
| 6 | Phase 1 에 CRS 5-10% 혼합 | 🟢 P2 | **유지** | Branch 편향 완화 타당 |
| 7 | γ/β, α 모니터링 threshold 사전 정의 | 🟢 P2 | **유지** | 표준 ablation. 실행 가능 |
| 8 | rod_map 입력 rebalancing (16D 별도 embed) | 🟢 P2 | **약화** | "1/21 gradient 영향" 논증은 정적 분석. 학습 후 가중치 재분배 효과 무시함. **Phase 2 monitoring 후 필요시** 조건부 |
| 9 | log() ε-clamp 명시 | 🟢 P2 | **유지** | 방어적 코딩 필수 |
| 10 | L_keff Phase 2b 중반 조기 도입 (소 λ) | 🟢 P2 | **유지 (실험 유보)** | 이론적 가치 있으나 실험 결과로 최종 판정 |

### 0.2 회신이 놓친 추가 잠재 위험 (신규 발견, 5개)

| # | 추가 위험 | 우선순위 | 요약 |
|---|---------|--------|------|
| A | **Phase 3 h(0) 초기값 + burn-in 의 물리 정합성** | 🔴 P0 | h(0) = zeros 는 Xe 초기값 0 = 비물리. Burn-in 중 예측의 물리성 보증 부재 |
| B | **100 시나리오 데이터의 일반화 한계** | 🔴 P0 | train/val/test 분할 후 ~70 LP 학습 → 과적합 구조적 위험. Test split 전략 미결 |
| C | **Phase 2a freeze 중 Decoder Jacobian 문제** | 🟡 P1 | L_physics 가 Decoder 통과할 때, Decoder freeze 면 Jacobian 이 "Phase 1 시점" 에 고정 → Mamba 가 받는 signal 질 저하 가능 |
| D | **Mamba parallel scan vs 1-step recurrence 수치 동치성** | 🟡 P1 | 학습(parallel)과 추론(recurrence) 의 float32 round-off 차이가 long-horizon 에서 누적되면 **학습 모델 ≠ 배포 모델** |
| E | **LR scheduling / batch size / val split 전략 미결** | 🟡 P1 | 3-Phase 문서 §11 에 미결 항목으로 있으나 실험 착수 전 결정 필수 |

### 0.3 실행 계획 요약

- **Step 1 (즉시, 0.5일)**: 회신 권고 중 P0 2건 (차등 LR, Scheduled Sampling) + 추가 위험 A (Phase 3 burn-in 물리 정합성) 을 3-Phase 문서 §6 에 직접 반영
- **Step 2 (1일)**: 추가 위험 B (데이터 분할 전략) 를 신규 문서 `2026-04-21 데이터 분할 및 일반화 전략.md` 으로 설계
- **Step 3 (1일)**: 회신 권고 P1 3건 + 추가 위험 C, D 를 기존 문서 §별 각주/미결 항목으로 추가
- **Step 4 (0.5일)**: 검증 방법론 문서 (§5.E 참조) 에 TR-X (ablation) 테스트 항목 신설
- **Step 5 (0.5일)**: 통합 정리 문서 §7 에 "UltraReview 2차 검토 반영" 신설

---

## 1. 회신 권고 10건 각각에 대한 자체 재검토

각 권고를 **(a) 논증 강도**, **(b) 요청서 §3 범위 외 저촉 여부**, **(c) 수정/유지/철회 판정** 으로 분석.

### 1.1 [P0] Phase 2b 차등 LR 100× → 20× 축소

**회신 논증**: "ULMFiT/LoRA 표준은 10~20×, 100× 는 one-way adapt 위험"

**재검토**:
- **(a) 논증 강도**: 방향은 타당하나 "ULMFiT/LoRA 표준 10~20×" 는 과장
  - ULMFiT 실제: layer 간 discriminative LR 배율 2.6^depth (10 layer 이면 끝단 대비 10배 수준). 전체 layer 간 편차는 확실히 10-30× 범위
  - LoRA 실제: base 모델 freeze, adapter LR 자체가 1e-4~1e-3. freeze 는 "×0" 이므로 편차 무한대
  - DreamerV3: 모듈 간 LR 2~5× 차이
  - 따라서 "20× 표준" 은 내가 임의 채택한 수치. 정확히는 "10~50× 범위 권고" 가 적절
- **(b) 범위 외 저촉 여부**: 해당 없음 (학습 hyperparameter 는 범위 내)
- **(c) 판정**: **유지 (문구 수정)**. "100× → 20×" 를 "100× → 20~50× 범위에서 실험으로 조정" 으로 완화

**반영 내용**:
- 3-Phase 문서 §5.3.1 차등 LR 표의 "잠정" 표기 유지
- **권고 수치 추가**: Mamba 5e-4, Attn 1e-4, Enc/Dec 2e-5 (20× 비율) 또는 Mamba 5e-4, Attn 5e-5, Enc/Dec 1e-5 (50× 비율) 을 **Phase 2b 초기 ablation** 으로 비교

### 1.2 [P0] Phase 3 에 Scheduled Sampling 명시

**회신 논증**: "Hard Teacher-Forcing-OFF 전환은 distribution shift 위험. 각 K 단계 내 ε 선형 감소"

**재검토**:
- **(a) 논증 강도**: 강함. Bengio et al. 2015 가 정본, sequence prediction 에서 표준
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **유지**

**반영 내용**:
- 3-Phase 문서 §6 에 §6.9 신설: "Scheduled Sampling 프로토콜"
- 각 K 단계 내 ε (GT 사용 확률) 선형 감소 (0.9 → 0.1)
- 구현 예시 (TF2): `use_gt = tf.random.uniform(()) < eps`

### 1.3 [P1] N=2 → N=3 baseline 변경

**회신 논증**: "Xenon oscillation 2 rounds cross-coupling tight"

**재검토**:
- **(a) 논증 강도**: 직관적이나 엄밀하지 않음
  - Mamba 는 cell-wise 독립, Attention 은 시점별 공간 결합. "round" 개념이 명확하지 않음
  - Xenon oscillation 은 축방향 반대칭 모드 + 반경방향 공간 분포의 결합 현상. 이 모드를 압축된 잠재 공간에서 학습하는 데 필요한 깊이는 **이론적으로 결정 불가**, 실험만이 답
  - Transformer 일반 깊이 (10-30 layer) 와의 단순 비교도 약함 (우리는 공간 시퀀스 720 개로 매우 짧음)
- **(b) 범위 외 저촉 여부**: 해당 없음 (N=2 는 확정 사항이 아님, §11.1 에 확장 가능성 명시)
- **(c) 판정**: **약화 권고** — "N=3 baseline 변경" 이 아닌 "**N=2/N=3 ablation 필수**" 로

**반영 내용**:
- 3-Phase 문서 §10 또는 검증 문서 (§5.E) 에 TR-X1 신설: "Phase 2a 초기 N=2 vs N=3 ablation"
- 시계열 구현 계획 §11.1 "Phase 2 후반 검토" → "Phase 2a 초기 검토" 로 승격

### 1.4 [P1] Phase 2a → 2b 전환을 loss-plateau 자동화

**회신 논증**: "30% epoch 은 pre-scheduled, 실제 수렴과 무관"

**재검토**:
- **(a) 논증 강도**: 강함. 자동화 트리거는 ML training 표준 practice
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **유지** — 구체 수치 명시 필요

**반영 내용**:
- 3-Phase 문서 §5.2 에 전환 기준 추가:
  - 1차 기준: "Mamba validation loss 의 최근 5 epoch 상대 변화 평균 < 1%"
  - 2차 기준 (백업): "epoch 비율이 20%~50% 범위에 도달"
  - 두 기준 중 먼저 충족되는 시점에 전환

### 1.5 [P1] Uncertainty Weighting (Kendall) 도입

**회신 논증**: "Physical Loss 의 gradient scale mismatch. Kendall 이 구현 단순, ReLoBRaLo 대비"

**재검토**:
- **(a) 논증 강도**: 방향 타당. 그러나 "Kendall > ReLoBRaLo" 단정은 과함
  - Kendall (2018): multi-task learning 용. log-variance learnable weight. 구현 단순
  - ReLoBRaLo (2021): PINN 특화. Loss 간 비율 역사 기반 지수 이동 평균. PINN 논문에서 더 자주 사용
  - 우리 도메인은 **PINN 에 더 가까움** (Physical Loss 주도)
- **(b) 범위 외 저촉 여부**: 해당 없음 (후순위 도입 옵션으로 §12 에 ReLoBRaLo 이미 언급됨)
- **(c) 판정**: **유지 (대안 병기)**

**반영 내용**:
- 3-Phase 문서 §12 의 "ReLoBRaLo (Phase 2b 또는 Phase 3)" 항목 확장
- 병기 대안: Uncertainty Weighting (Kendall 2018). 비교 실험 항목으로 TR-X2 신설
- 둘 다 Phase 2b 중반 도입 시점으로

### 1.6 [P2] Phase 1 에 CRS 단일 step 5-10% 혼합

**회신 논증**: "Branch 는 '5min 응답' 편향. CRS 샘플로 48h 분포 signal 유지"

**재검토**:
- **(a) 논증 강도**: 타당. Branch data 는 시점 t 의 rod_map 변경 + 5min 전이로, 48h 시퀀스 전체의 구조적 분포와는 다른 sub-manifold 를 채움
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **유지** — 비율 수치는 ablation

**반영 내용**:
- 3-Phase 문서 §4.2 "학습 설정" 에 추가:
  - "데이터: Branch (~167만 시점, 단일 step) + CRS 단일 step 샘플링 (5% 비중)"
  - 5% 는 권고값, 10% 까지 ablation

### 1.7 [P2] γ/β, α 모니터링 threshold 사전 정의

**회신 논증**: "Phase 2b 말기 |γ| < 0.01 이면 AdaLN 죽음 → concat 단독 fallback"

**재검토**:
- **(a) 논증 강도**: 방어적 설계, 타당. threshold 0.01 은 임의이나 방향 정확
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **유지**

**반영 내용**:
- 통합 정리 문서 §3.4 "모니터링 항목" 에 threshold 추가:
  - γ/β: RMS < 0.01 이 5 epoch 연속 지속 시 AdaLN 경로 "죽음" 판정
  - lape_alpha: |α| > 1.0 또는 α < -0.5 시 폭주/음수 학습 경보

### 1.8 [P2] rod_map 입력 rebalancing (16D 별도 embed)

**회신 논증**: "21ch 중 rod_map 1ch = 1/21 ≈ 4.8% gradient 영향"

**재검토**:
- **(a) 논증 강도**: **부분적** — 정적 분석으로 맞지만, 학습 후 가중치 재분배로 자동 보정되는 효과 무시
  - Conv3D 1×1×1 의 21→128 가중치는 학습 후 rod_map 관련 가중치 norm 이 자동으로 커질 수 있음
  - Deep learning 에서 "입력 차원 비중" 이 곧 "gradient 영향" 은 아님
- 다만 rod_map 은 공간적으로 **희소 신호** (대부분 cell 에서 rod 미설치/상단/하단 등 동일값) 이므로 학습 초기 gradient signal 이 약한 건 사실
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **약화** — "즉시 변경" 에서 "Phase 2 성능 미달 시 조건부 검토" 로

**반영 내용**:
- 시계열 구현 계획 §11.1 "Phase 2 후반 검토" 에 추가:
  - "rod_map 입력 embedding rebalancing: Phase 2b 에서 rod_map 관련 검증 (TR-R? 참조) 부진 시 1→16 별도 Conv3D 1×1×1 embed + concat(16+112D) 으로 변경 검토"
- 즉시 baseline 변경 아님

### 1.9 [P2] log() ε-clamp 명시

**회신 논증**: "float32 에서 exp(-large) → 0 underflow → log() 발산"

**재검토**:
- **(a) 논증 강도**: 강함. 방어적 코딩 필수
- **(b) 범위 외 저촉 여부**: 해당 없음
- **(c) 판정**: **유지**

**반영 내용**:
- Physical Loss 통합 레퍼런스 (별도 폴더) 에 안전성 코드 추가:
  - `safe_log(x) = tf.math.log(tf.maximum(x, 1e-20))` helper
  - N_Xe, N_I, flux 모든 log 연산에 적용
- 구현 시점: L_Bateman 구현 직전

### 1.10 [P2] L_keff Phase 2b 중반 조기 도입 (소 λ)

**회신 논증**: "Phase 3 말기 도입 시 flux 이미 수렴 → gradient signal 약함"

**재검토**:
- **(a) 논증 강도**: 이론적 타당, 실험으로 검증 필요
- **(b) 범위 외 저촉 여부**: "L_keff 후순위/검토중" 자체가 확정 아님 → 범위 내
- **(c) 판정**: **유지 (실험 유보)**

**반영 내용**:
- Physical Loss 문서에 미결 항목으로 추가:
  - "Phase 2b 중반 L_keff 조기 도입 (λ=0.001 시작) vs Phase 3 말기 도입 비교 실험"

---

## 2. 회신이 놓친 추가 잠재 위험

### 2.1 [P0] 추가 위험 A — Phase 3 h(0) 초기값 + burn-in 의 물리 정합성

**위험 내용**:

Phase 3 K-step autoregressive 에서 첫 시점 h(0) 은 어떻게 설정되는가?

현재 문서 (3-Phase §6, 시계열 §9.1) 상태:
- 추론 시: `h(0) = zeros (초기)` (§9.2, §8.1 패턴 A)
- 학습 시 Phase 3: Prefix burn-in K step 이 "잠정" 으로 §11 미결 항목에 있음

**문제**:
1. h(0) = zeros 는 **비물리적 초기값**. Xe 초기 농도는 0 이 아니라 정상 운전 평형 근처 (~1e15 #/cm³). 0 에서 시작하면 첫 몇 step 은 "Xe 없는 상태의 확산" 을 예측 → 물리 위반
2. Prefix burn-in 의 길이 K 와 burn-in 중 예측의 물리 정합성 기준 부재
3. Training data 에서 h(0) 는 t=0 에서 zeros, inference 에서도 t=0 에서 zeros 로 시작 → "t=0 시점 특별 처리" 패턴이 학습되지만 일반 중간 시점 시작에는 부적합

**회신에서 누락한 이유**: 회신이 Q1-Q6 에 집중, 추론/초기조건은 Q 에 없었음.

**영향 범위**: Phase 3 학습의 기반 가정이 흔들릴 수 있음.

**권고**:
- **방안 1 (recommended)**: Prefix burn-in 을 학습 시간 **필수** 로 명시. K_burnin = 8-16 step 을 teacher forcing 으로 돌려 h(K_burnin) 확보, 이후 K-step autoregressive
- **방안 2**: h(0) 를 학습 가능한 initialization 으로 (learnable parameter), 단 이 경우 시나리오 전반의 평균 초기 상태만 학습 가능
- **방안 3**: Dataset 의 t=0 시점 실제 GT state 를 모델 forward 한 후 얻은 h(1) 을 저장, 이후 시점부터 시작

**반영 계획**:
- 3-Phase 문서 §6.2 (학습 흐름) 수정:
  - "첫 입력만 GT → Prefix burn-in K_burnin step teacher forcing 후 h(K_burnin) → K-step autoregressive"
- §11 미결 항목에서 Prefix burn-in K 을 **필수** 로 승격, K_burnin 구체 수치 (8/16/32) ablation

### 2.2 [P0] 추가 위험 B — 100 시나리오 데이터의 일반화 한계

**위험 내용**:

- LP: 100개, p_load profile: 20종, CRS 1:1 round-robin → 100 시나리오
- 모델은 "임의 LP × 임의 profile" 에 대한 일반화 대리모델을 목표로 함
- Train/val/test 분할 (일반적 70/15/15) → 학습 시 70 시나리오 노출

**문제**:
1. 100 시나리오는 **generalization 관점에서 매우 적음**
   - 비교: 기상 ML (GraphCast 등) 은 40년치 daily data (~15000 일) 로 학습
   - 우리 스케일은 각 시나리오가 매우 긴 (48h, 575 step) 이지만 서로 다른 LP/profile 은 100 가지 만
2. **분할 전략 미결**:
   - LP 단위 split (70 LP train / 15 val / 15 test) → test 는 완전 새 LP
   - Profile 단위 split → 학습 LP 와 동일 LP 의 새 profile
   - Step 단위 split → 같은 시나리오의 다른 시점 구간
   - 각 전략이 "일반화" 의 다른 차원을 검증
3. **과적합 구조적 위험**: 100 시나리오 × 575 step = 57,500 CRS 시점만으로 128D × ~수백K params 모델 학습 → 과적합 가능성

**회신에서 누락한 이유**: 회신이 아키텍처/학습 방법론에 집중, 데이터 규모/일반화는 별개 축.

**영향 범위**: Phase 3 성공 판정, 최종 배포 모델의 신뢰성

**권고**:
- **Split 전략 사전 확정**:
  - 1차 분할: LP 단위 (70 train / 15 val / 15 test). 완전 새 LP 일반화 능력 테스트
  - 2차 (보조): val LP 내 profile split (학습 LP 의 새 profile)
  - 3차 (보조): step 단위 split (같은 시나리오의 후반 구간 예측)
- **Data augmentation 검토**:
  - Mirror/rotation 기하 augmentation: 이미 halo_expand 에서 sym_type 분기
  - Noise injection: Physical Loss 와 충돌 가능
  - Time reversal: 물리적으로 무의미 (decay 방향성)
- **검증 커리큘럼**: LP generalization gap = test LP MSE / train LP MSE. 이 ratio 가 모델 과적합 지표.

**반영 계획**:
- 신규 문서 작성: `implementation_plans/학습 방법론 설계/2026-04-21 데이터 분할 및 일반화 전략.md`
- 통합 정리 문서 §5 학습 방법론 요약에 "데이터 분할 전략" 서브섹션 추가

### 2.3 [P1] 추가 위험 C — Phase 2a freeze 중 Decoder Jacobian 문제

**위험 내용**:

Phase 2a 는 **Mamba 만** 학습, Encoder/Attn/Decoder freeze.

그러나 Loss 계산은 `X_next_pred = state(t) + delta_pred` → `L_physics(denormalize(X_next_pred))` 경로로, L_physics gradient 는:

```
L_physics → denorm(X_next_pred) → X_next_pred → delta_pred → Decoder → Processor → Mamba
                                                              ↑
                                                         freeze (업데이트 안 됨)
```

Decoder freeze 는 Decoder **가중치** 업데이트 차단이지, **Jacobian (gradient 전달)** 차단은 아님 → OK. Mamba 는 정상적 gradient 받음.

그러나 **문제**:
1. Decoder 의 Jacobian 이 "Phase 1 시점의 표현" 에 고정됨. Mamba 가 학습되면서 Processor 출력 d(t) 의 분포가 변화 → Decoder 가 가정한 입력 분포와 불일치 → Mamba 가 받는 signal 질 저하 가능
2. 특히 Physical Loss 는 역정규화 공간이라 Decoder 출력의 작은 변화도 큰 loss 변화 유발 가능

**회신에서 누락한 이유**: Q3 에서 3-Phase 분석 시 Phase 2a 의 gradient flow 는 정상이라고 전제했음. Decoder Jacobian drift 는 더 미묘한 문제.

**영향 범위**: Phase 2a 의 학습 신호 정확성

**권고**:
- **완화 1 (간단)**: Phase 2a 에서 Decoder 도 **매우 낮은 LR** 로 함께 학습 (freeze 대신 LR 1e-6). 실질적으로 거의 불변이나 Jacobian 이 Mamba 변화에 적응
- **완화 2 (더 보수적)**: Phase 2a 기간을 짧게 유지 (10-20% epoch). Mamba 의 큰 변화 전에 2b 진입
- **모니터링**: Phase 2a 학습 중 `|Decoder 출력 분포 변화|` vs Phase 1 종료 시점. 분포 drift 가 커지면 조기 2b 전환

**반영 계획**:
- 3-Phase 문서 §5.2 에 Phase 2a 설정 상세 추가:
  - 옵션 A (baseline): Decoder 완전 freeze
  - 옵션 B (안전): Decoder LR 1e-6 (거의 freeze 이나 Jacobian 적응)
  - 두 옵션 TR-X3 ablation 으로 비교

### 2.4 [P1] 추가 위험 D — Mamba parallel scan vs 1-step recurrence 수치 동치성

**위험 내용**:

Mamba-3 의 학습은 parallel scan (575 step 한번에), 추론은 1-step recurrence. 두 모드는 **수학적으로 동치** 이지만 **수치적으로 다를 수 있음**:

1. Parallel scan: 연관성 있는 tree-reduction 연산. FP32 의 round-off 가 tree depth 에 따라 log(T) 배 누적
2. 1-step recurrence: 순차 연산. Round-off 가 T 배 누적 가능
3. 복소 상태 + Trapezoidal 이산화는 위상 민감. 575 step 후 두 방식의 복소 상태 위상 차이 가능

**문제**:
- 학습한 모델이 추론에서 다르게 동작하면 **학습 모델 ≠ 배포 모델**
- Long horizon (575 step) 에서 오차 누적 → Phase 3 K=575 학습이 실제 배포 성능과 어긋남

**회신에서 누락한 이유**: Mamba-3 파라미터 설정이 SOTA 기반이고, 일반적으로 Mamba 원논문에서 수치 동치성이 검증됨. 그러나 복소 상태 + Trapezoidal 은 Mamba-3 고유 확장이라 추가 검증 필요.

**영향 범위**: 배포 모델의 예측 정확성

**권고**:
- **검증 TR-M1 신설**: Phase 2b 초기에 random input 에 대해 parallel scan 결과 vs 1-step recurrence loop 결과 비교
  - `max |y_parallel - y_recurrence|` 가 기계 epsilon 수준 (< 1e-5) 이어야 정상
  - 차이가 크면 구현 버그 또는 precision 문제
- **학습 시 일부 recurrence 모드 혼합**: Phase 2b 말기 또는 Phase 3 에 1-step recurrence 로도 일부 학습 (TF 모드)

**반영 계획**:
- 검증 방법론 문서 (§5.E) 에 TR-M1 추가: "Parallel scan vs recurrence 수치 동치성 검증"
- 시계열 구현 계획 §9 에 "수치 동치성 검증 체크리스트" 서브섹션

### 2.5 [P1] 추가 위험 E — LR scheduling / batch size / val split 미결

**위험 내용**:

3-Phase §11 미결 항목:
- LR scheduling (warmup, cosine decay 등)
- Batch size
- Validation split 전략
- Gradient clipping

이들은 "튜닝 단계" 로 유보되어 있으나, **실험 착수 전 결정되어야 하는 필수 항목**. 결정 없이 실험 시작 시 결과 해석 불가.

**회신에서 누락한 이유**: 회신은 Q1-Q6 에 집중. 이들은 Q 에 없음.

**영향 범위**: 실험 설계 자체

**권고**:
- 즉시 결정 항목 (실험 착수 전):
  - **LR warmup**: 100-500 step linear warmup (AdamW 기본 best practice)
  - **Cosine decay** (Phase 마지막 20% 구간): 각 Phase 내 LR decay
  - **Gradient clipping**: `tf.clip_by_global_norm(grads, 1.0)` 표준
  - **Batch size**: 메모리 허용 최대 (CRS 시퀀스 batch 는 575 × 720 × 128 × 4 bytes ≈ 210 MB / 시퀀스). GPU 메모리 80GB 기준 batch 4-8
  - **Validation split**: 추가 위험 B 와 연계
- 실험 중 조정 항목:
  - λ weights (loss balancing)
  - K_burnin
  - Scheduled sampling ε schedule

**반영 계획**:
- 3-Phase 문서 §11 의 일부 항목을 §4.2, §5.4, §6.4 본문으로 이동 (잠정 → 권고 수치)
- 별도 표 "Phase 별 hyperparameter 권고값" 신설

---

## 3. 요청서 §3 "범위 외" 저촉 여부 재확인

요청서 §3 에 명시된 "되돌리지 않을 항목":

| # | 항목 | 회신/추가 위험에서 저촉? |
|---|------|-------------------|
| 1 | 프레임워크 TF2/Keras | 저촉 없음 |
| 2 | 공간 격자 Quarter 5×5 → (6,6) | 저촉 없음 |
| 3 | 인코더 FullAttention ×3, STRING + ConditionalLAPE | 저촉 없음 |
| 4 | 데이터 출처 MASTER | 저촉 없음 (추가 위험 B 는 데이터 '규모' 이슈, 출처 변경 아님) |
| 5 | 정규화 방식 (z-score / log-z-score 채널별) | 저촉 없음 |
| 6 | L_physics 역정규화 후, L_data 정규화 공간 | 저촉 없음 |

**결론**: 회신의 10개 권고 + 추가 위험 5개 모두 §3 범위 외 저촉 없음.

---

## 4. 통합 반영 계획 (실행 단계)

### 4.1 우선순위 매트릭스

| 항목 | 출처 | 우선순위 | 영향 문서 | 작업 예상 시간 |
|------|------|--------|---------|------------|
| Phase 2b 차등 LR 문구 수정 | 회신 §1.1 | P0 | 3-Phase §5.3 | 15분 |
| Phase 3 Scheduled Sampling 명시 | 회신 §1.2 | P0 | 3-Phase §6 신설 | 30분 |
| Phase 3 h(0) + Burn-in 필수화 | 추가 A | P0 | 3-Phase §6.2 | 30분 |
| 데이터 분할 전략 문서 신설 | 추가 B | P0 | 신규 문서 | 2시간 |
| N=2/N=3 ablation 항목화 | 회신 §1.3 | P1 | 검증 §TR-X1 | 30분 |
| Phase 2a→2b loss-plateau 트리거 | 회신 §1.4 | P1 | 3-Phase §5.2 | 20분 |
| Uncertainty Weighting + ReLoBRaLo 병기 | 회신 §1.5 | P1 | 3-Phase §12 | 30분 |
| Decoder Jacobian 옵션화 | 추가 C | P1 | 3-Phase §5.2 | 30분 |
| Parallel vs recurrence 수치 동치성 TR | 추가 D | P1 | 검증 §TR-M1 | 30분 |
| Hyperparameter 권고값 표 신설 | 추가 E | P1 | 3-Phase 신설 표 | 1시간 |
| Phase 1 CRS 5-10% 혼합 | 회신 §1.6 | P2 | 3-Phase §4.2 | 15분 |
| γ/β, α 모니터링 threshold | 회신 §1.7 | P2 | 통합 §3.4 | 15분 |
| rod_map rebalancing 조건부 | 회신 §1.8 | P2 | 시계열 §11.1 | 15분 |
| log() ε-clamp 명시 | 회신 §1.9 | P2 | Physical Loss 레퍼런스 | 15분 |
| L_keff 조기 도입 실험 유보 | 회신 §1.10 | P2 | Physical Loss 문서 | 15분 |
| 통합 §7 "UltraReview 2차 반영" | 종합 | - | 통합 문서 | 1시간 |

**총 예상 시간**: 약 7-9 시간 (반나절 ~ 하루)

### 4.2 실행 순서 (병렬/직렬)

```
[Step 1 — 즉시 (동일 세션)]
  ├─ Phase 3 h(0) + Burn-in 필수화 (추가 A)
  ├─ Phase 2b 차등 LR 문구 수정 (회신 §1.1)
  └─ Phase 3 Scheduled Sampling (회신 §1.2)
       → 3-Phase 문서 §5, §6 수정

[Step 2 — 1일 이내]
  ├─ 데이터 분할 전략 신규 문서 (추가 B)
  ├─ Hyperparameter 권고값 표 (추가 E)
  └─ 검증 방법론 §TR-X1, TR-X3, TR-M1 추가 (회신 §1.3, 추가 C, D)

[Step 3 — 2-3일 이내]
  ├─ Phase 2a→2b loss-plateau (회신 §1.4)
  ├─ Uncertainty Weighting / ReLoBRaLo (회신 §1.5)
  ├─ Phase 1 CRS 5-10% 혼합 (회신 §1.6)
  ├─ γ/β, α 모니터링 threshold (회신 §1.7)
  ├─ log() ε-clamp (회신 §1.9)
  └─ L_keff 조기 도입 실험 유보 (회신 §1.10)

[Step 4 — 마무리]
  ├─ rod_map rebalancing 조건부 (회신 §1.8)
  └─ 통합 정리 §7 "UltraReview 2차 반영" 신설
       → 전체 변경 사항을 통합 문서에 요약
```

### 4.3 변경 트래킹

각 Step 완료 시:
- Git commit: `Refactor: [Step N] UltraReview 반영 (항목 X, Y, Z)`
- 변경 로그를 본 문서 §5 에 추가

---

## 5. 변경 로그 (실행 시 채움)

| 일자 | Step | 변경 내용 | 커밋 |
|------|------|---------|------|
| 2026-04-20 | - | 본 계획 문서 초안 작성 | (pending) |
| 2026-04-21 | Step 1 | Phase 3 h(0) + Burn-in 필수화, 차등 LR 문구, Scheduled Sampling | (pending) |
| 2026-04-21 | Step 2 | 데이터 분할 전략 문서, HP 표, 검증 TR-X1/X3/M1 | (pending) |
| ... | ... | ... | ... |

---

## 6. 본 계획의 한계

- 회신 자체가 **실험 기반이 아닌 구조 분석** 이므로, 본 계획의 권고 수치 (예: Phase 2b LR 20~50×, K_burnin 8-16) 는 모두 ablation 으로 최종 확정 필요
- 선행 외부 자문 답변 A/B 전문 (2026-04-16) 이 회신 작성 시 미열람 상태였음. 해당 문서의 세부 논증과 본 계획이 상충할 시 재조정 필요
- "100 시나리오 일반화" 위험 (추가 B) 은 구조적 한계이며 본 계획으로 완전 해결 불가. 데이터 확보 (신규 LP 생성) 가 근본 해결책이나 MASTER 시뮬레이션 비용이 걸림돌

---

## 7. 참고

- [2026-04-20 UltraReview 요청서.md](../2026-04-20 UltraReview 요청서.md)
- [2026-04-20 Claude Opus 4.7 회신.md](2026-04-20 Claude Opus 4.7 회신.md)
- [2026-04-20 모델 설계 최종 확정 사항 (통합 정리).md](../2026-04-20 모델 설계 최종 확정 사항 (통합 정리).md)
- [2026-04-20 3-Phase 학습 방법론 (1차 정리, 추가 검토 예정).md](../학습 방법론 설계/2026-04-20 3-Phase 학습 방법론 (1차 정리, 추가 검토 예정).md)
- [2026-04-20 컴포넌트 검증 및 실험 방법론 (초안).md](../학습 방법론 설계/검증 및 실험 계획/2026-04-20 컴포넌트 검증 및 실험 방법론 (초안).md)
