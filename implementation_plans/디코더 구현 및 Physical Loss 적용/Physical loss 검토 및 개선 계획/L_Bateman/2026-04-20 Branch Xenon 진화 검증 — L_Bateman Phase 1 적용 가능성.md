# Branch Xenon 진화 검증 — L_Bateman Phase 1 적용 가능성

> **작성일**: 2026-04-20
> **계기**: 모델 설계 최종 확정 사항 정리 작업 중, "Branch frozen Xenon 가정"의 정확성 의문 제기
> **검증**: 실제 데이터 (`D:\lf_preprocessed_100LP_mirrored_2026-04-02\보관용 2026-04-04`)에서 LP_0050, LP_0010, LP_0042 검증
> **결론**: Branch는 Frozen Xenon 아님. 5분 시간 진화를 정확히 반영. **L_Bateman Phase 1 적용 가능**.
> **검증 자료**: [piecewise-test/2026-04-20_branch_xenon_evolution_결과.md](../../piecewise-test/2026-04-20_branch_xenon_evolution_결과.md)

---

## 1. 문제 제기 배경

### 1.1 기존 표기

[Physical Loss 통합 레퍼런스](physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md) §0.2:
> "branch_xe: Frozen Xenon: CRS와 거의 동일"

이 표기를 근거로 다음 가정 도출:
- Branch에서는 Xe가 시간 진행 안 함 (frozen)
- L_Bateman (`dN_Xe/dt = -λ·N + γ·Σ_f·φ - σ_a·N·φ`)은 Branch에서 잔차가 trivial → 의미 없음
- → **Phase 1 (Branch만 사용)에서 L_Bateman 적용 불가**

### 1.2 사용자 통찰

> "Branch가 frozen Xenon이 아닐 수 있음. 데이터 생산 시 제어봉 우선 삽입하고 5분간 연소 후 branch 결과를 내도록 한 것 같음. 실제 데이터 확인 필요."

---

## 2. 검증 방법

### 2.1 비교 기준

| 기준 | 정의 | 의미 |
|------|------|------|
| Float32 round-off | `value × machine_eps (1.19e-7)` | 부동소수점 정밀도 한계 |
| Branch 차이 | `|branch_xe[t, b≠0] - critical_xe[t]|` | 분기 효과 |
| 시간 변화 | `|critical_xe[t+1] - critical_xe[t]|` | 5분 진화 |

### 2.2 판정 로직

- 측정값 ≈ round-off → **정밀도 노이즈** (frozen)
- 측정값 >> round-off → **실제 물리 변화** (not frozen)
- 측정값 ≈ 시간 변화 자릿수 → **5분 진화 반영**

---

## 3. 검증 결과

### 3.1 정밀도 vs 실제 변화 (LP_0050 전 시점)

| 항목 | max | round-off 대비 |
|------|-----|--------------|
| Float32 round-off | 7.30e-16 | 1× (기준) |
| **Branch 차이 max** | **5.75e-11** | **78,766×** |
| **시간 변화 max** | **5.20e-11** | **71,305×** |

→ **정밀도 노이즈가 절대 아님** (5 자릿수 큼)

### 3.2 다중 LP 일관성

| LP | Branch 차이 | 시간 변화 |
|----|----------|---------|
| LP_0050 | 78,766× round-off | 71,305× round-off |
| LP_0010 | 101,696× round-off | 94,667× round-off |
| LP_0042 | 126,385× round-off | 98,057× round-off |

→ 모든 LP에서 일관 패턴

### 3.3 물리 일관성

- critical_xe 단위 환산: 6e-9 #/barn-cm = **6e15 #/cm³** (PWR 정상 범위)
- Xe-135 5분 decay만 (lower bound): **0.63%**
- 측정된 5분 변화: ~1% (decay + generation - absorption, 정상)
- 측정된 Branch 변화: ~1.4% (제어봉 변화 + 5분 진화)

→ 측정값이 물리적 예상치와 일치

### 3.4 b=0 정체성

`branch_xe[t, b=0] == critical_xe[t]` (정확히 0 차이)
→ b=0은 critical 자체. **실제 분기는 b=1~29 (29개)**

---

## 4. 데이터 생성 절차 추정

검증 결과로부터 추정되는 절차:
1. critical 시나리오에서 시점 t의 노심 상태 추출
2. 30개 branch 적용 (b=0=critical, b=1~29=다른 rod_offset)
3. 각 branch에서 **5분간 연소 (Xe 동역학 진행)** 후 결과 저장
4. → branch_xe[t, b]는 "rod_offset 적용 후 5분 진화한 결과"

---

## 5. 학습에의 함의

### 5.1 L_Bateman 적용 가능성 (정정)

| Loss | 시간 의존 | Branch 가능? | CRS 가능? | 비고 |
|------|---------|------------|---------|------|
| L_data, L_data_halo | 무 | ✓ | ✓ | |
| L_diff_rel | 무 | ✓ | ✓ | |
| L_sigma_a_Xe | 무 | ✓ | ✓ | |
| **L_Bateman** | 유 (단일 step) | **✓** | ✓ | **본 검증으로 정정** |
| L_keff (후순위) | 무 | ✓ | ✓ | |
| L_branch (보조) | - | - | ✓ | |

### 5.2 Phase 1 학습 강화

Phase 1 (Branch 단일 step) Loss 정정:
```
L_phase1 = L_data + L_data_halo + L_diff_rel + L_Bateman + L_sigma_a_Xe
                                                ↑
                                     본 검증으로 추가 가능 확인
```

**장점**:
- 풍부한 Branch 데이터 (~167만 시점)로 L_Bateman 더 효과적 학습
- Phase 2 시작 시 이미 Bateman 동역학 인지된 상태
- CRS 학습 시 Bateman 부담 감소

### 5.3 L_Bateman 구현 시 고려사항

- **Branch 단일 step**: `(N_xe(t+5min) - N_xe(t)) / 300s` ≈ ODE 우변
- **CRS 시퀀스**: 575 step 전체에 대해 동일 잔차
- → 동일 함수 사용 가능 (단일 step과 시퀀스 모두 처리)

---

## 6. 권고 조치

### 6.1 즉시 조치

| # | 대상 | 조치 |
|---|------|------|
| 1 | [Physical Loss 통합 레퍼런스](physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md) §0.2 | "Frozen Xenon" 표기 정정 (5분 진화 반영) |
| 2 | [모델 설계 최종 확정 사항 통합 정리](../2026-04-20 모델 설계 최종 확정 사항 (통합 정리).md) §5.2 | Phase 1 Loss에 L_Bateman 포함 |
| 3 | [2026-04-16 C안 학습 방법론 설계](../학습 방법론 설계/2026-04-16 C안 학습 방법론 설계.md) | 본 검증 결과 반영 (L_Bateman Phase 1 명시) |

### 6.2 후속 작업

- L_Bateman 코드 구현 시 단일 step + 시퀀스 모두 처리 가능하도록 설계
- Branch 데이터 활용 비율 학습 전략 (sampling rate, λ_branch) 결정

---

## 7. 표기 정정안

### 7.1 Physical Loss 통합 레퍼런스 §0.2 (제안)

```
원본:
| `branch_xe` | (T,31,Z,qH,qW) | float64 | N_Xe | Frozen Xenon: CRS와 거의 동일 |

정정안:
| `branch_xe` | (T,30,Z,qH,qW) | float32 | N_Xe | b=0=critical, b=1~29=rod_offset 분기 + 5분 진화 결과 (검증: 2026-04-20) |
```

> **shape 정정 추가 필요**: 실제 데이터는 (T, 30, ...) 이고 dtype은 float32. 31-way 표기와 float64 표기 모두 정정 필요.

---

## 8. 참고 자료

- 검증 코드: [piecewise-test/2026-04-20_branch_xenon_evolution_test.py](../../piecewise-test/2026-04-20_branch_xenon_evolution_test.py)
- 검증 결과: [piecewise-test/2026-04-20_branch_xenon_evolution_결과.md](../../piecewise-test/2026-04-20_branch_xenon_evolution_결과.md)
- 검증 출력: [piecewise-test/2026-04-20_branch_xenon_evolution_test_output.txt](../../piecewise-test/2026-04-20_branch_xenon_evolution_test_output.txt)
- 관련 결정: [Physical Loss 통합 레퍼런스 §1 L_Bateman](physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md)
