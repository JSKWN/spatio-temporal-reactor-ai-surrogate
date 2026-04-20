# JNET0 역전 현상 분석 계획

## 현상
반사체 인접 연료 노드에서:
- φ_fuel(5.95e13) >> φ_refl(1.26e13) (노드 평균 flux 기준)
- 그런데 JNET0 < 0 (반사체→연료 방향 유입)
- Fick's law (J ∝ -dφ/dx, 고→저 방향) 위배

## 물리적 의문
- Fick's law가 인접면에서 정확도가 떨어지는 것은 타당 (불균일 매질 경계)
- 하지만 **방향 자체가 역전**되는 것은 물리적으로 이상
- 반사체에서 되돌아오려면 **면(surface)에서의 국소 flux**가 역전되어야 함:
  ```
  노드 평균: φ_fuel >> φ_refl
  면 근처:   φ_fuel_surface < φ_refl_surface  ← 이런 상황이 필요
  ```
- 즉 연료 노드 내에서 면 방향으로 flux가 **급격히 하락**하고,
  반사체에서는 면 근처에 **작은 봉우리**가 있어야 함
- NEM의 4차 다항식 전개가 이런 형상을 만들 수 있는가?

## 분석 단계

### Step 1: 면 flux(surface flux) 추정 — 가용 데이터 확인

MAS_NXS에서 면 flux를 직/간접적으로 구할 수 있는지:
- **ADF (Assembly Discontinuity Factor)**: ADF × φ̄ = φ_surface?
  - R0에서 ADF = 1.0 확인 → φ_surface ≈ φ̄ (면 flux ≈ 노드 평균)
  - ADF=1이면 면 flux와 노드 평균이 같다는 의미 → 역전 설명 불가?
  - 단, ADF는 MASTER의 SET(Simplified Equivalence Theory) 적용 결과
- **JNET0 + Fick으로 면 flux 역산**:
  ```
  J = -D × (φ_surface - φ̄) / (h/2)   (half-node Fick)
  → φ_surface = φ̄ - J × h/(2D)
  ```
  JNET0(×2)와 φ̄, D, h로부터 각 면의 φ_surface를 추정 가능

### Step 2: 면 flux 프로파일 재구성

JNET0(×2)을 이용하여 연료 노드와 반사체 노드 양쪽의 면 flux를 계산:
```
연료측 면 flux: φ_fuel_surface = φ̄_fuel - J_fuel × h/(2D_fuel)
반사체측 면 flux: φ_refl_surface = φ̄_refl + J_refl × h/(2D_refl)
  (부호: 반사체에서는 유입 방향이 +)
```

**검증 기대**:
- JNET0 < 0인 면에서: φ_fuel_surface < φ_refl_surface → 역전 설명
- JNET0 > 0인 면에서: φ_fuel_surface > φ_refl_surface → Fick 일치

### Step 3: NEM 4차 다항식 형상 고찰

NEM에서 노드 내 flux 전개:
```
ψ(u) = a₀ + a₁ξ₁(u) + a₂ξ₂(u) + a₃ξ₃(u) + a₄ξ₄(u)
```
- a₀ = φ̄ (노드 평균)
- a₁ = (ψ_r - ψ_l)/2 (면 flux 차이)
- 고차항 a₃, a₄는 transverse leakage에 의존

면 flux: ψ(u=0) = a₀ - a₁ + a₂ (left face), ψ(u=1) = a₀ + a₁ + a₂ (right face)
→ a₂ < 0이면 **양쪽 면 flux 모두 노드 평균보다 낮음** (배럴 형태)
→ a₂가 충분히 크면 면 flux가 노드 평균 대비 크게 하락 가능

### Step 4: 반사체 flux "봉우리" 가능성

물리적으로 반사체 면 근처에 flux 봉우리가 생기는 시나리오:
- SS+H₂O 반사체에서 고속중성자가 탄성산란으로 축적
- 반사체 두께가 충분하면 "반사 풀(reflection pool)" 형성
- 반사체 면에서의 flux가 반사체 평균보다 높을 수 있음

이를 확인하는 방법:
- 반사체 노드의 면 flux (Step 2에서 계산) 확인
- 반사체 노드 내에서 면 방향 flux > 노드 평균 → 봉우리 확인

### Step 5: 종합 판단

| 시나리오 | φ_fuel_surface | φ_refl_surface | 물리적 의미 |
|----------|:-:|:-:|---|
| A: 정상 누설 | 높음 | 낮음 | Fick 일치 (JNET0 > 0) |
| B: 역전 | **낮음** | **높음** | 연료 면 하락 + 반사체 면 상승 |
| C: 수치 아티팩트 | - | - | NEM 수렴 문제 |

## 데이터 요구
- MAS_NXS: FLX, DIF, JNET0(×2) — 이미 파싱 완료
- 추가 파싱 불필요: Step 2의 면 flux 역산은 기존 데이터로 계산 가능

## 수행된 분석 결과

### Step 1-2: half-node Fick 면 flux 역산
- 선형 외삽으로는 NEM 4차 다항식 면 flux 복원 불가
- 결과 불일치: Fick 일치율 0% (누설면/역전면 모두)

### 경계면 전류 연속 검증
- **E/W**: fuel + refl = 0% (완벽 연속 ✓)
- **N/S**: fuel + refl ≠ 0 (연속 불성립)
- **내부 연료-연료 면에서도 N/S 불일치** → 반사체 문제 아님

### I/J ↔ NESW 매핑 검증
- **확정**: E = +I 방향, W = -I 방향 (`(I,J).E + (I+1,J).W = 0` ✓)
- **미확정**: N/S 방향
  - `(I,J).N + (I,J+1).S ≠ 0` (N=+J 아님)
  - `(I,J).N + (I+1,J).S ≠ 0` (N=+I 아님)
  - 어떤 조합으로도 N/S 전류 연속 미성립

### D̂ 역산
- S면(JNET0<0): D̂ ≈ -0.001~-0.019
- W면(JNET0>0): D̂ ≈ +0.084~+0.089 (D̃의 80%)

## 미해결 사항 (MASTER 매뉴얼/소스코드 확인 필요)
1. **JNET0-N/S의 물리적 방향**: E/W는 ±I 확정, N/S는 미확정
2. **면 flux 직접 재구성**: NEM 계수 없이는 불가
3. **JNET0 역전 현상의 물리적 원인**: N/S 방향 확정 후 재분석 필요
4. **밸런스 SUM × 2 = 0%는 여전히 유효** (N/S 방향 불명확과 무관하게 성립)

## 산출물
- `piecewise-test/albedo 값 캘리브레이션 테스트/2026-04-01_albedo_R3_surface_flux_analysis.py` / `_output.txt`
- `piecewise-test/albedo 값 캘리브레이션 테스트/2026-04-01_albedo_R3_current_continuity_output.txt`
