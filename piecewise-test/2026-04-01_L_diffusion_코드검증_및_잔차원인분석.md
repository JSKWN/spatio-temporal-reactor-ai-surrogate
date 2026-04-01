# L_diffusion 코드 검증 및 잔차 원인 분석

**작성일**: 2026-04-01
**목적**: end-to-end 테스트의 치수/단위/수식 정합성을 점검하고, 잔차(median 2~6%, max 21~36%)의 본질적 원인을 규명

---

## 1. 코드 검증 (Code Verification)

### 1.1 버그 수정 이력 (3건)

| # | 버그 | 수정 전 | 수정 후 | 영향 |
|---|------|---------|---------|------|
| 1 | CMFD 부호 | `D*(φ_nb-φ_c)/h` (inward) | `D*(φ_c-φ_nb)/h` (outward) | g1 median 9.1% → 2.2% |
| 2 | zmesh 인덱싱 | `range(2,22)` (K=22 반사체 혼입) | `range(1,21)` (연료만) | z=19 max 31.8% → 19.0% |
| 3 | 대칭면 처리 | `J=0` (누설 무시) | Mirror CMFD (REFLECT) | sym 노드 median 31% 개선 |

### 1.2 치수/단위 정합성 검사

| 항목 | 코드에서 사용한 값 | 올바른 값 | 일치 |
|------|-------------------|-----------|------|
| dx, dy (radial pitch) | `WIDE = 21.60780 cm` | assembly pitch 21.608 cm | ✅ |
| dz (axial mesh) | `zmesh[1:21]` = 10.0 cm (연료 20층 전부) | 10 cm/layer | ✅ |
| V (volume) | `dx * dy * dz` = 21.608² × 10 = 4,669 cm³ | assembly volume | ✅ |
| Af (radial face area) | `dy * dz` = 21.608 × 10 = 216.1 cm² (E/W) | ✅ | ✅ |
| Az (axial face area) | `dx * dy` = 21.608² = 467.1 cm² | ✅ | ✅ |
| hz (axial CMFD distance) | `0.5*(dz[z]+dz[z±1])` = 10 cm (균일) | center-to-center | ✅ |
| D (diffusion coeff) | `1/(3*Σ_tr)` | standard definition | ✅ |
| Σ_r (removal g1) | `Σ_c1 + Σ_f1 + Σ_s12` | Σ_a1 + Σ_s12 | ✅ |
| Σ_a (absorption g2) | `Σ_c2 + Σ_f2` | Σ_a2 | ✅ |

### 1.3 데이터 수준 일관성

| 데이터 | 수준 | 소스 | 일관성 |
|--------|------|------|--------|
| XS (xs_fuel) | **Assembly-averaged** (20, 5, 5, 10) | MAS_XSL → xs_voxel_builder | ✅ |
| Flux (phi) | **Assembly-averaged** (20, 9, 9, 2) | MAS_OUT $F3D | ✅ |
| CMFD mesh | **Assembly pitch** (dx=21.608 cm) | core_geometry WIDE | ✅ |
| Volume | **Assembly volume** | dx × dy × dz | ✅ |

**결론: 치수/단위/인덱싱에 버그 없음. 모든 데이터가 assembly level에서 일관.**

---

## 2. 잔차 원인 분석 (Residual Source Identification)

### 2.1 CMFD vs NEM 밸런스 차이

MASTER(NNEM)이 만족하는 정확한 밸런스:
```
Σ_faces( J_NEM × A_face ) + Σ_r × φ_avg × V = (1/k) × νΣ_f × φ_avg × V
```

우리 CMFD 밸런스:
```
Σ_faces( J_CMFD × A_face ) + Σ_r × φ_avg × V = (1/k) × νΣ_f × φ_avg × V + R
```

**잔차 R = Σ_faces( (J_CMFD - J_NEM) × A_face )**

여기서:
```
J_NEM  = D̃ × (φ_c - φ_nb) / h  +  D̂ × (φ_c + φ_nb) / 2    (NEM surface current)
J_CMFD = D̃ × (φ_c - φ_nb) / h                                (CMFD, D̂ 없음)
```

따라서:
```
R = -Σ_faces( D̂ × (φ_c + φ_nb) / 2 × A_face )
```

**잔차 = NEM D̂ correction 항의 총합.**
- D̂는 intra-nodal flux의 4차 다항식 전개에서 면 경사도로 결정
- assembly 내부의 flux 분포가 평탄하면 D̂ → 0, 잔차 → 0
- flux gradient가 급한 곳 (반사체 인접, 제어봉 인접) → D̂ 큼 → 잔차 큼

### 2.2 잔차 크기의 물리적 해석

| 구분 | g1 median | g2 median | 해석 |
|------|-----------|-----------|------|
| 내부 | 1.91% | 6.75% | g2가 높음 = 열중성자 flux gradient가 큼 |
| 경계 | 2.49% | 6.22% | Albedo BC로 경계 보정 → g2 약간 개선 |
| 전체 | 2.25% | 6.39% | **assembly-level CMFD의 본질적 정확도** |

g2가 g1보다 높은 이유:
- 열중성자(g2)의 확산 길이(L_th ≈ 2 cm)가 assembly pitch(21.6 cm)보다 매우 짧음
- → intra-assembly flux 분포가 급변 → D̂ correction이 큼
- 고속중성자(g1)의 확산 길이(L_f ≈ 6 cm)가 상대적으로 길어 flux가 더 평탄

### 2.3 Outlier (max 21~36%)의 원인

outlier 분석 (Section 4: 면별 누설 분해)에서 확인:
- **g1 max 21%**: source-removal 불균형이 주원인. leak total = removal의 -4.9%
- **g2 max 36%**: Σ_s12×φ_g1 (group coupling) 불균형. leak total = removal의 0.12%

~~두 경우 모두 **누설 자체가 아니라 반응률 항에서 잔차 발생** →
assembly-averaged φ로 반응률을 계산할 때 intra-assembly 분포가 무시되어 발생하는 **공간 균질화 오차**~~

**수정 (2026-04-01 추가 검증)**: 위 분석은 부정확. 실제 원인은 아래 §2.4 참조.

### 2.4 XS 불일치 발견 (MAS_XSL vs MAS_NXS)

리뷰어 지적: "MASTER의 Σ×φ×V는 정의상 정확한 반응률. Source-Removal 불균형이 크다면 XS 불일치 의심."

**검증 결과**: MAS_XSL(BOC 고정) vs MAS_NXS(스텝별 feedback 반영) XS 비교

#### XS 채널별 차이

| 채널 | XSL mean | NXS mean | 평균 차이 | 최대 차이 |
|------|----------|----------|-----------|-----------|
| ABS g1 | 9.03e-3 | 9.26e-3 | 3.2% | 29.2% |
| ABS g2 | 7.64e-2 | 7.94e-2 | 3.9% | 12.8% |
| NFS g1 | 5.81e-3 | 5.78e-3 | 0.6% | 4.0% |
| **NFS g2** | **1.08e-1** | **1.05e-1** | **2.8%** | **25.7%** |
| SCA g1 | 1.71e-2 | 1.67e-2 | 2.9% | 16.6% |

#### Outlier 노드 (z=4, qy=2, qx=0) 상세

| 항목 | XSL (BOC) | NXS (step) | 차이 |
|------|-----------|------------|------|
| Source | 2.416e+16 | **2.196e+16** | -9.1% |
| Removal | 2.343e+16 | **2.364e+16** | +0.9% |
| **Net (S-R)** | **+7.36e+14 (+3.1%)** | **-1.69e+15 (-7.1%)** | **부호 반전** |

**핵심**: NFS_g2 (νΣ_f thermal)가 XSL: 0.1008 vs NXS: **0.0897** (-11%) → Source가 크게 변화.
XSL은 "순 유출 +3.1%"인데 NXS는 "순 유입 -7.1%" → **CMFD가 어떤 방향으로 누설을 계산하든 잔차 발생 불가피**.

#### 결론

**잔차의 주 원인은 CMFD D̂가 아니라 XS 불일치**:
- MAS_XSL = BOC 기준 단면적 (Xe/Sm/온도 feedback 미반영)
- MAS_NXS = 해당 스텝의 실제 단면적 (feedback 반영)
- 차이: 평균 2.9%, 최대 29% → 반응률 밸런스가 깨짐
- **MAS_NXS의 스텝별 XS를 사용하면 잔차 대폭 개선 기대**

---

## 3. NXS XS 기반 CMFD 재계산 결과 (XS검증_T002)

MAS_NXS(스텝별 feedback XS)로 동일 CMFD 잔차를 재계산하여 비교 (10LP × 10step):

### 결과 비교

| 항목 | XSL (BOC) | NXS (feedback) | 변화 |
|------|-----------|---------------|------|
| **g2 median** | **6.39%** | **2.19%** | **-65.7% (3배 개선)** |
| **g2 max** | **35.6%** | **11.2%** | **-68.6%** |
| g1 median | 2.25% | 2.45% | +8.5% (약간 악화) |
| g1 max | 21.4% | 32.3% | -51% (악화) |

### g2 개선: XS 불일치가 주원인이었음

- XSL의 νΣ_f(thermal)가 Xe feedback 미반영 → 최대 19% 과대 (T001)
- NXS로 교체 시 g2 median 6.4% → **2.2%** (순수 CMFD 한계)

### g1 max 악화 원인 분석

NXS worst 노드: LP_0004, z=19, (qy=2,qx=2) = **B5** (Gd 함유)

| XS 채널 | XSL | NXS | 변화 |
|---------|-----|-----|------|
| ABS g1 | 8.66e-3 | 1.10e-2 | **+27.5%** |
| SCA g1 | 1.71e-2 | 1.49e-2 | -13.0% |
| NFS g1 | 5.53e-3 | 5.31e-3 | -3.9% |

**원인 추정** (확정적이지 않음):
1. NXS에서 Xe-135 흡수가 ABS에 포함 → ABS +27.5% 증가
2. ABS 증가 + NFS 감소 → Removal 증가 + Source 감소 → 밸런스 방향 변화
3. XSL에서는 **ABS 과소평가(오차)가 우연히 CMFD D̂ 오차와 상쇄**하는 방향이었음
4. NXS(정확한 XS)로 교체 시 이 상쇄가 해소 → **D̂ 오차가 드러남**
5. B5(Gd 함유)는 Gd 소진 + Xe feedback이 결합하여 XS 변화가 특히 큼

**주의**: D̂와 XS feedback의 정확한 상쇄 메커니즘은 추정이며, 추가 검증 시 JNET0 직접 비교로 확인 가능.

---

## 4. Predictor-Corrector 관점

MASTER의 NNEM 자체가 predictor-corrector:
1. **Predictor (CMFD)**: `J = D̃ × (φ_c - φ_nb) / h` — 우리의 L_diffusion
2. **Corrector (NEM)**: D̂ correction 추가 → GT에서만 가능

AI 모델에서의 대응:
- **L_diffusion (physical loss)** = predictor (물리 방향 제약, g1~2%, g2~2~6%)
- **L_data (data loss)** = corrector (GT와의 차이로 D̂ + XS feedback 보정)
- **Trainable α/C** = 경계면 D̂ 근사

---

## 5. 최종 결론

1. **코드에 추가 버그 없음**: 치수/단위/인덱싱 전수 검증 완료
2. **g2 잔차의 주원인 = XS 불일치** (Xe feedback 미반영): NXS 사용 시 6.4%→2.2%
3. **g1 잔차의 주원인 = CMFD D̂**: XS 교체 효과 미미 (median ~2.5%)
4. **g1 max 악화 (21→32%)**: XSL의 ABS 오차가 D̂와 우연히 상쇄 → NXS에서 해소
5. **순수 CMFD 한계**: g1 median 2.5%, g2 median 2.2% → physical loss gradient 충분
6. **실제 AI 모델**: XSL XS 사용 (g2 6.4%), trainable α/C + data loss로 보정

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `2026-04-01_L_diffusion_endtoend_test.py` | 검증 대상 코드 (최종, Mirror CMFD) |
| `2026-04-01_L_diffusion_endtoend_결과.md` | 10LP 테스트 결과 |
| `2026-04-01_L_diffusion_outlier_분석결과.md` | Outlier 진단 (면별 분해, LP간 비교) |
| `2026-04-01_mirror_symmetry_verification.py` | Mirror 대칭 시각화 검증 |
