# Branch Xenon Evolution 검증 결과

**실행일**: 2026-04-20
**데이터**: `D:\lf_preprocessed_100LP_mirrored_2026-04-02\보관용 2026-04-04`
- 주 분석: LP_0050.h5 / `t14_262_p50_ramp_down`
- 교차 검증: LP_0010.h5, LP_0042.h5
**테스트 코드**: `2026-04-20_branch_xenon_evolution_test.py`

---

## 1. 검증 목적

[Physical Loss 통합 레퍼런스](../implementation_plans/디코더 구현 및 Physical Loss 적용/physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md) §0.2 표기:
> "branch_xe: Frozen Xenon: CRS와 거의 동일"

이 표기가 정확한지 실제 데이터로 검증.

**핵심 질문**:
- Q1. branch_xe와 critical_xe의 차이가 부동소수점 정밀도(round-off)인가, 실제 물리 변화인가?
- Q2. Branch가 5분간 Xe 진화를 반영하는가?
- Q3. L_Bateman을 Phase 1 (Branch 단일 step) 학습에서 적용 가능한가?

---

## 2. 검증 방법

### 데이터 구조
- `critical_xe`: shape (575, 20, 5, 5), dtype float32, 단위 #/barn-cm
- `branch_xe`: shape (575, 30, 20, 5, 5) — 30개 branch, b=0은 critical 자체

### 비교 기준 3가지
1. **Float32 정밀도 (round-off)**: `value × machine_eps (1.19e-7)`
2. **측정된 Branch 차이**: `|branch_xe[t, 1:] - critical_xe[t]|`
3. **측정된 시간 변화 (5분)**: `|critical_xe[t+1] - critical_xe[t]|`

### 판정 기준
- 측정값이 round-off의 **수십 배 이내** → 정밀도 노이즈
- 측정값이 round-off의 **수백 배 이상** → 실제 물리 변화

---

## 3. 결과 요약

### 3.1 b=0 정체성 확인

| 시점 t | `|branch_xe[t, 0] - critical_xe[t]|` max |
|------|-----------------------------------|
| 0 | 0.0e+00 |
| 100 | 0.0e+00 |
| 287 | 0.0e+00 |
| 500 | 0.0e+00 |

→ **b=0은 critical 시나리오 자체** (정확히 0 차이)
→ 실제 Branch 분기는 b=1..29 (29개)

### 3.2 시점별 측정 차이 vs 정밀도 (LP_0050)

| t | p_load | critical_xe max | float32 round-off | Branch 차이 max | 시간 변화 max |
|---|------|-----------------|------------------|---------------|------------|
| 100 | 0.625 | 5.27e-09 | 6.28e-16 | 3.92e-11 (**62,341×**) | 1.85e-11 (**29,440×**) |
| 287 | 0.979 | 4.11e-09 | 4.90e-16 | 5.65e-11 (**115,413×**) | 9.32e-12 (**19,031×**) |
| 500 | 1.000 | 3.66e-09 | 4.36e-16 | 5.13e-11 (**117,619×**) | 4.10e-12 (**9,404×**) |

→ Branch 차이가 정밀도의 **수만 배** 수준 → 정밀도 노이즈 아님
→ 시간 변화가 정밀도의 **수만 배** 수준 → 실제 5분 진화

### 3.3 전 시점 통합 (LP_0050)

| 항목 | max | mean |
|------|-----|------|
| Branch 차이 (b=1..29) | 5.75e-11 (**78,766× round-off**) | 5.85e-12 (8,019×) |
| 시간 변화 (t→t+1) | 5.20e-11 (**71,305× round-off**) | 3.66e-12 (5,010×) |

→ **Branch 차이 ≈ 5분 시간 변화 + 추가 변화 (제어봉 변화 효과)**

### 3.4 다중 LP 교차 검증

| LP | 시나리오 | Branch 차이 (× round-off) | 시간 변화 (× round-off) |
|----|--------|------------------------|----------------------|
| LP_0050 | t14_262_p50_ramp_down | 78,766× | 71,305× |
| LP_0010 | t14_262_p50_ramp_down | **101,696×** | **94,667×** |
| LP_0042 | t12_363_p50_ramp_down | **126,385×** | **98,057×** |

→ **모든 LP에서 일관되게 정밀도 대비 5 자릿수 큰 차이**
→ 일관된 물리 패턴 (LP 의존성 없음)

### 3.5 물리적 일관성 (Xe-135 동역학과 비교)

| 항목 | 값 |
|------|---|
| Xe-135 half-life | 9.14 시간 (32,904 s) |
| λ_Xe | 2.11 × 10⁻⁵ /s |
| 5분간 decay만 (lower bound) | **0.63%** |
| 측정된 5분 변화 | **~1%** (decay + generation - absorption) |
| 측정된 Branch 변화 | **~1.4%** (제어봉 변화 + 5분 진화) |

→ 측정값이 **물리적 예상치와 일치**
→ critical_xe 단위 환산: 6e-9 #/barn-cm = **6e15 #/cm³** (PWR 정상 농도 범위 ✓)

---

## 4. 분석

### 4.1 Branch는 Frozen Xenon 아님

기존 [Physical Loss 통합 레퍼런스](../implementation_plans/디코더 구현 및 Physical Loss 적용/physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md) §0.2의 "Frozen Xenon" 표기는 **부정확**:

- 절대값으로는 critical과 가까움 (10⁻¹¹ 수준 차이)
- 그러나 5분 시간 진화량 (10⁻¹¹)과 **같은 크기**
- → **5분간 Xe 변화는 정확히 반영됨**

### 4.2 Branch 변화가 시간 변화보다 큼

LP_0050 전 시점 통합:
- 시간 변화 (decay + 일반 동역학): 71,305× round-off
- Branch 차이 (제어봉 변경 + 5분 진화): **78,766× round-off**

→ Branch 분기가 5분 시간 변화보다 약간 큰 변화 생성
→ 이유: 제어봉 변경 → flux 변화 → Xe 흡수율 변화 → Xe 농도 변화

### 4.3 데이터 생성 절차 추정

검증 결과로부터 추정:
1. critical 시나리오에서 시점 t의 상태 추출
2. 30개 branch (b=0=critical, b=1~29=다른 rod_offset) 적용
3. 각 branch에서 **5분간 연소 (Xe 동역학 진행)** 후 결과 저장
4. → branch_xe[t, b]는 "rod_offset 적용 + 5분 진화 후" 상태

---

## 5. 결론

### 5.1 검증 결과

| 질문 | 답 | 근거 |
|------|---|------|
| Q1. 차이가 정밀도 노이즈인가? | **아니오** | round-off 대비 **78,766× 큼** (LP_0050 전 시점 max) |
| Q2. 5분 진화 반영하는가? | **예** | 시간 변화와 동일 자릿수, decay 0.63% 예상치와 일치 |
| Q3. L_Bateman Phase 1 적용 가능? | **예** | 단일 step Bateman 잔차 학습 가능 |

### 5.2 Physical Loss 통합 레퍼런스 표기 정정 권고

```
원본 §0.2:
  "branch_xe: Frozen Xenon: CRS와 거의 동일"

정정안:
  "branch_xe: 제어봉 변경 후 5분 진화 결과.
   critical과 ~1% 차이 (실제 물리 변화, 정밀도 노이즈 아님).
   b=0은 critical 자체, b=1..29가 실제 분기."
```

---

## 6. 학습에의 함의

### 6.1 L_Bateman Phase 1 적용 가능성 확정

[2026-04-16 C안 학습 방법론 설계](../implementation_plans/학습 방법론 설계/2026-04-16 C안 학습 방법론 설계.md) §6에서 "기본 Loss (CRS + Branch 공통): L_data + L_data_halo + L_diff_rel + L_Bateman + L_σXe"로 명시되어 있음.

본 검증으로 다음 확정:
- Branch에서 단일 step Bateman 잔차 (`(N(t+5min) - N(t))/300s`) 계산 가능
- Phase 1 (Branch만 사용)에서도 시간 의존 Loss 학습 가능
- 풍부한 Branch 데이터 (~167만 시점)로 L_Bateman 더 효과적 학습

### 6.2 Loss 적용 매트릭스 (정정)

| Loss | 시간 의존 | Branch 가능? | CRS 가능? |
|------|---------|------------|---------|
| L_data, L_data_halo | 무 | ✓ | ✓ |
| L_diff_rel | 무 | ✓ | ✓ |
| L_sigma_a_Xe | 무 | ✓ | ✓ |
| **L_Bateman** | 유 (단일 step) | **✓** (5분 진화 반영) | ✓ (575 step) |
| L_keff (후순위) | 무 | ✓ | ✓ |
| L_branch (보조) | - | - | ✓ |

### 6.3 후속 작업

1. Physical Loss 통합 레퍼런스 §0.2 표기 정정
2. 모델 설계 최종 확정 사항 통합 정리 문서의 Phase 1 Loss 정정 (L_Bateman 포함)
3. L_Bateman 코드 구현 시 Branch 단일 step과 CRS 시퀀스 모두 처리 가능하도록 설계

---

## 7. 참고

- 검증 코드: `2026-04-20_branch_xenon_evolution_test.py`
- 실행 결과: `2026-04-20_branch_xenon_evolution_test_output.txt`
- 관련 문서: `../implementation_plans/디코더 구현 및 Physical Loss 적용/2026-04-20 Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성.md`
