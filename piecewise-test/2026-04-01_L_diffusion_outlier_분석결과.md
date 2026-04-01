# L_diffusion Outlier 진단 분석 결과

**실행일**: 2026-04-01
**데이터**: 10 LP (LP_0000~LP_0009) x 10 CRS steps = 100 scenarios, N=38,000 nodes

---

## 1. LP별 잔차 비교

| LP | profile | g1 median | g1 max | g2 median | g2 max |
|---|---|---|---|---|---|
| LP_0000 | t12_363_p50_power_lower | 2.39% | 19.0% | 6.51% | 23.9% |
| LP_0001 | t12_363_p50_power_upper | 2.30% | 17.5% | 7.42% | 26.4% |
| LP_0002 | t12_363_p50_ramp_down | 2.42% | 18.0% | 7.18% | **34.2%** |
| LP_0003 | t12_363_p50_ramp_up | 2.20% | 19.0% | 6.28% | 27.4% |
| LP_0004 | t12_8_p50_power_lower | 2.02% | **20.5%** | 5.36% | 23.3% |
| LP_0005 | t12_8_p50_power_upper | 2.10% | 18.3% | 7.24% | 28.3% |
| LP_0006 | t12_8_p50_ramp_down | 2.26% | 17.6% | 6.49% | 32.6% |
| LP_0007 | t12_8_p50_ramp_up | 2.03% | **21.1%** | 5.97% | 30.4% |
| LP_0008 | t14_262_p50_power_lower | 2.12% | 20.6% | 5.88% | 26.6% |
| LP_0009 | t14_262_p50_power_upper | 3.29% | 17.1% | 6.38% | **35.8%** |

- **median은 모든 LP에서 안정적** (g1: 2.0~3.3%, g2: 5.4~7.4%)
- **max는 LP마다 다름** — LP 고유의 flux 분포가 특정 위치에서 CMFD 오차를 증폭
- keff는 모든 LP에서 1.00000 (CRS 결과)

---

## 2. Outlier 위치

### Quarter-core XY 맵

```
         x=0     x=1     x=2     x=3     x=4
y=0     [A2]    [A3]   [A2]*g2  [A2]    [B3]     N=sym
y=1     [A3]    [A2]    [A3]    [A2]    [A3]
y=2    [A2]*g1  [A3]    [B5]    [A3]     ·       W=sym
y=3     [A2]    [A2]    [A3]     ·       ·
y=4     [B3]    [A3]     ·       ·       ·

*g1 = g1 outlier (LP_0007, z=4)
*g2 = g2 outlier (LP_0009, z=17)
```

### Outlier 위치 상세

| | g1 outlier | g2 outlier |
|---|---|---|
| Quarter 좌표 | (qy=2, qx=0) | (qy=0, qx=2) |
| Full-core | (j=7, i=5) | (j=5, i=7) |
| Assembly | **A2** | **A2** |
| 대칭면 | **W면** (Mirror CMFD) | **N면** (Mirror CMFD) |
| 나머지 5면 | 모두 fuel (CMFD) | 모두 fuel (CMFD) |
| z 위치 | z=4 (하부) | z=17 (상부) |

**공통 특성**: 둘 다 **A2 어셈블리**, **대칭면 1면 + 연료 5면** 구조. 반사체 인접이 아님.

---

## 3. 축방향 flux 프로파일

### g1 outlier: LP_0007, (qy=2,qx=0)

- outlier 위치의 flux가 reference(1,1) 대비 **30~40% 낮음** (비대칭 위치)
- z=3→4에서 g1 gradient가 **2.51e+13** (가장 큼)
- outlier의 g1 잔차: z=0~19 전 구간에서 **13~21%** (위치 전체가 높음)
- **reference(1,1)의 g1 잔차: 0.02~3.4%** → 위치 차이가 핵심

### g2 outlier: LP_0009, (qy=0,qx=2)

- outlier g2 잔차: z=3부터 급등, z=15~17에서 **32~36%**
- reference(1,1)도 g2 잔차가 **24%** → 이 위치뿐 아니라 g2 자체가 높은 축방향 패턴
- g2 잔차가 z 증가에 따라 점진적으로 증가하는 것은 assembly-level CMFD의 구조적 한계

---

## 4. 면별 누설 분해

### g1 worst: LP_0007, z=4, (qy=2,qx=0)

| 항목 | g1 값 | Removal 대비 |
|---|---|---|
| leak_N | -1.107e+15 | -4.77% |
| leak_S | +6.191e+14 | +2.67% |
| leak_E | -8.198e+13 | -0.35% |
| **leak_W** | **0** | **0% (symmetry)** |
| leak_B | -1.653e+15 | -7.12% |
| leak_T | +1.087e+15 | +4.68% |
| **leak_total** | **-1.136e+15** | **-4.89%** |
| removal | +2.322e+16 | 100% |
| source | -2.696e+16 | |
| **RESIDUAL** | **-4.880e+15** | **21.02%** |

- 누설 총합(-4.89%)은 잔차(21%)에 비해 작음
- **잔차의 주원인**: source(2.696e+16)와 removal(2.322e+16)의 차이 = 3.74e+15
- 누설(-1.14e+15)이 이 차이를 보상하지 못함
- W면 대칭: Mirror CMFD (REFLECT)로 이웃 flux 반영 (기존 J=0에서 수정)

### g2 worst: LP_0009, z=17, (qy=0,qx=2)

| 항목 | g2 값 | Removal 대비 |
|---|---|---|
| **leak_total** | **+7.101e+12** | **+0.12%** |
| removal | +5.877e+15 | 100% |
| source (Σ_s12×φ_g1) | -7.972e+15 | |
| **RESIDUAL** | **-2.088e+15** | **35.53%** |

- g2 누설은 거의 0 (0.12%) — 누설이 문제가 아님
- **잔차 원인**: scatter source(Σ_s12×φ_g1=7.97e+15)가 removal(5.88e+15)보다 36% 큼
- g1→g2 scattering이 과대평가 또는 g2 absorption이 과소평가
- 이것은 **assembly-level CMFD가 node-average flux로 group coupling을 계산하는 본질적 한계**

---

## 5. XS 비교

| 항목 | outlier (z=4) | fuel avg | ratio |
|---|---|---|---|
| D_g1 | 1.410 | 1.416 | **1.00** |
| D_g2 | 0.493 | 0.481 | **1.02** |
| Sr_g1 | 0.0259 | 0.0261 | **0.99** |
| Sa_g2 | 0.0971 | 0.0764 | **1.27** |

- **D (확산계수)**: outlier와 평균이 거의 동일 → XS 이상치가 아님
- **Sa_g2**: outlier가 27% 높음 → A2 어셈블리의 축방향 위치(z=4)에서 열중성자 흡수가 큼
- **결론**: XS 자체가 outlier 원인이 아니라, **flux 분포의 공간적 특성**이 핵심

---

## 6. LP간 동일 위치 비교 (핵심 결과)

### (qy=2,qx=0), z=4 — g1 잔차

| LP | profile | g1 max |
|---|---|---|
| LP_0006 | t12_8_ramp_down | **0.6%** |
| LP_0002 | t12_363_ramp_down | 4.3% |
| LP_0000 | t12_363_power_lower | 5.5% |
| LP_0009 | t14_262_power_upper | 5.3% |
| LP_0003 | t12_363_ramp_up | 7.3% |
| LP_0001 | t12_363_power_upper | 13.0% |
| LP_0005 | t12_8_power_upper | 13.6% |
| LP_0004 | t12_8_power_lower | **20.5%** |
| LP_0007 | t12_8_ramp_up | **21.1%** |

**결론: position-specific이면서 LP-dependent**
- 같은 위치에서 LP에 따라 **0.6%~21.1%**까지 변동
- t12_8 그룹의 power_lower와 ramp_up이 특히 높음
- ramp_down은 0.6%로 거의 완벽 → **flux 분포 형태가 잔차를 결정**

### (qy=0,qx=2), z=17 — g2 잔차

| LP | profile | g2 max |
|---|---|---|
| LP_0008 | t14_262_power_lower | **11.1%** |
| LP_0001 | t12_363_power_upper | 18.7% |
| LP_0004 | t12_8_power_lower | 23.3% |
| LP_0000 | t12_363_power_lower | 23.9% |
| LP_0002 | t12_363_ramp_down | 26.7% |
| LP_0006 | t12_8_ramp_down | 26.9% |
| LP_0003 | t12_363_ramp_up | 27.4% |
| LP_0005 | t12_8_power_upper | 28.3% |
| LP_0007 | t12_8_ramp_up | 30.4% |
| LP_0009 | t14_262_power_upper | **35.8%** |

**결론: 구조적으로 높은 위치, LP가 크기만 결정**
- **모든 LP에서 11~36%** — 이 위치의 g2 잔차는 구조적으로 높음
- LP_0008만 11%로 상대적 양호 (power_lower 시나리오)

---

## 7. CRS 스텝간 유사성 원인

### g1 outlier: LP_0007, z=4, (qy=2,qx=0)

| metric | CV (std/mean) |
|---|---|
| phi_g1 | **11.2%** |
| rel_g1 | **0.095%** |

- flux는 스텝에 따라 11% 변동하지만, **잔차 비율은 0.1% 변동**
- **잔차는 geometry+BC의 구조적 오차** (flux 크기에 비례하여 스케일링)

### g2 outlier: LP_0009, z=17, (qy=0,qx=2)

| metric | CV (std/mean) |
|---|---|
| phi_g2 | **2.1%** |
| rel_g2 | **0.24%** |

- 동일 결론: 잔차 비율이 거의 상수 → 구조적 오차

---

## 종합 결론

### Outlier 원인 분류

| 원인 | g1 outlier (21%) | g2 outlier (36%) |
|---|---|---|
| 위치 특성 | 대칭면(W=Mirror) 인접 A2 | 대칭면(N=Mirror) 인접 A2 |
| 누설 기여 | leak=-4.9% (잔차의 일부) | leak=+0.1% (거의 무관) |
| **주 원인** | source-removal 불균형 | Σ_s12×φ_g1 과대 (group coupling 오차) |
| LP 의존성 | 0.6~21% (flux 형태 의존) | 11~36% (구조적으로 높음) |
| 스텝 의존성 | 없음 (CV=0.1%) | 없음 (CV=0.2%) |

### Physical Loss 관점에서의 의미

1. **g1 잔차 (median 2.2%)**: assembly-level CMFD로 충분히 양호. max 21%는 특정 LP+위치 조합
2. **g2 잔차 (median 6.4%)**: assembly-level에서 group coupling (Σ_s12×φ_g1)의 구조적 한계
   - NEM D-hat correction 없이는 개선 불가
   - trainable parameter로 부분 보정 가능
3. **대칭면**: Mirror CMFD (REFLECT)로 수정 완료. sym 노드 g1 median 31% 개선 (2.50→1.73%)
   - full-core 9x9로 확장하면 대칭면 문제 해소 (추후 검토)

### 생성 파일

- `outlier_lp_comparison.png`: LP별 max 잔차 bar chart
- `outlier_xy_map.png`: Quarter-core XY 맵 + outlier 위치
- `axial_flux_profile.png`: 축방향 flux 프로파일
- `cross_lp_comparison.png`: LP간 동일 위치 비교
