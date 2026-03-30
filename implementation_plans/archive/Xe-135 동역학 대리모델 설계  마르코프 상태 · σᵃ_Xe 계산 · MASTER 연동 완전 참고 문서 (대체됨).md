> [대체됨 2026-03-30] → `2026-03-30 Physical Loss 통합 레퍼런스.md`. 오류 없음, 내용 그대로 흡수.

# Xe-135 동역학 대리모델 설계: 마르코프 상태 · σᵃ_Xe 계산 · MASTER 연동 완전 참고 문서

> **문서 목적**: SMR/PWR 부하추종 운전 해석을 위한 Spatio-Temporal AI Surrogate 모델에서 Xe-135 동역학을 정확히 예측하기 위한 수식 체계, MASTER 코드 연동 방법, 구현 주의사항, Q&A를 통합 정리한 단일 참고 문서.

***

## 1. Xe-135 동역학 핵심 수식

### 1.1 베이트만(Bateman) 연립 방정식

원자로 노심에서 I-135와 Xe-135의 시간 변화를 기술하는 ODE는 다음과 같다.[^1][^2]

**I-135 방정식:**

\[
\frac{dN_I}{dt} = \gamma_I \,\Sigma_f \,\phi - \lambda_I \,N_I \quad \text{[Eq. 1]}
\]

**Xe-135 방정식:**

\[
\frac{dN_{Xe}}{dt} = \gamma_{Xe}\,\Sigma_f\,\phi + \lambda_I\,N_I - \lambda_{Xe}\,N_{Xe} - \sigma_a^{Xe}\,\phi\,N_{Xe} \quad \text{[Eq. 2]}
\]

| 기호 | 의미 | 대표값 |
|------|------|--------|
| \(N_I,\; N_{Xe}\) | I-135, Xe-135 수밀도 [atoms/cm³] | 상태변수 |
| \(\gamma_I\) | I-135 누적 핵분열 수율 (직접 + Te-135 경유) | ≈ 0.0639 |
| \(\gamma_{Xe}\) | Xe-135 직접 핵분열 수율 | ≈ 0.0023 |
| \(\lambda_I\) | I-135 붕괴상수 | 2.87×10⁻⁵ s⁻¹ (반감기 ≈ 6.57 h) |
| \(\lambda_{Xe}\) | Xe-135 붕괴상수 | 2.10×10⁻⁵ s⁻¹ (반감기 ≈ 9.14 h) |
| \(\Sigma_f\) | 거시적 핵분열 단면적 [cm⁻¹] | 노드별 상이 |
| \(\phi\) | 열중성자속 (2-군 thermal) [n/cm²/s] | 노드별 상이 |
| \(\sigma_a^{Xe}\) | Xe-135 2-군 유효 미시 흡수 단면적 [cm²] | **상태·온도·연소도 의존** |

> **물리적 핵심**: Xe-135의 소멸 경로는 두 가지다 — 자연 붕괴(\(\lambda_{Xe} N_{Xe}\))와 중성자 흡수(\(\sigma_a^{Xe} \phi N_{Xe}\)). 고출력에서는 중성자 흡수 경로가 지배적이며, 이것이 Xe-135의 흡수 단면적을 정확히 계산해야 하는 이유다.

### 1.2 평형 Xe-135 농도 (참고)

정상 상태(\(dN_{Xe}/dt = 0,\; dN_I/dt = 0\))에서:

\[
N_{Xe}^{eq} = \frac{(\gamma_I + \gamma_{Xe})\,\Sigma_f\,\phi}{\lambda_{Xe} + \sigma_a^{Xe}\,\phi} \quad \text{[Eq. 3]}
\]

고출력 노심에서는 분모의 \(\sigma_a^{Xe}\phi\)가 \(\lambda_{Xe}\)보다 훨씬 커지므로, 평형 Xe 농도가 출력과 거의 무관해지는 포화(saturation) 특성이 나타난다.[^3]

***

## 2. 완전한 마르코프 상태 구성

### 2.1 마르코프 성질 요건

마르코프 조건: 상태 \(s_t\)만 주어지면 이전 이력 없이 \(s_{t+1}\)을 예측할 수 있어야 한다. Xe-135 동역학에서 이를 위반하는 대표적 함정은 **I-135 누락**이다 — Xe 생성의 약 95%가 I-135 붕괴를 경유하므로, \(N_I\)를 상태에서 제외하면 마르코프 성질이 깨진다.[^3]

### 2.2 필수 상태 변수 전체 목록

| 카테고리 | 변수 | Shape (노드맵 기준) | 마르코프 필요 이유 |
|----------|------|--------------------|--------------------|
| **독물질** | \(N_{Xe}(t)\) | (20, 5, 5) | Eq. 2의 직접 상태변수 |
| **독물질** | \(N_I(t)\) | (20, 5, 5) | Xe 간접 생성원; 없으면 마르코프 위반 |
| **중성자속** | \(\phi_{th}(t)\) | (20, 5, 5) | 소멸항·생성항의 직접 계수 |
| **단면적** | \(\Sigma_f(t)\) | (20, 5, 5) | 생성항 \(\gamma\Sigma_f\phi\) 계산 |
| **열수력** | \(T_{fuel}(t)\) | (20, 5, 5) | \(\sigma_a^{Xe}\) 계산 입력 (Doppler 효과) |
| **열수력** | \(T_{mod}(t)\) | (20, 5, 5) | **열중성자 스펙트럼의 주 결정 인자** → \(\sigma_a^{Xe}\) 직접 지배[^4][^5] |
| **열수력** | \(\rho_{mod}(t)\) | (20, 5, 5) | 감속능 변화 → 스펙트럼 이동 → \(\sigma_a^{Xe}\) 변화 |
| **연소도** | \(B_u\) | (20, 5, 5) | \(\sigma_a^{Xe}\) 보간 기준 축; 핵분열 생성물 축적으로 스펙트럼 변화[^6] |
| **제어봉** | `rod_map(t)` | (20, 5, 5) | 국소 중성자속 분포 변화 → Xe 공간 분포 비대칭 |

**붕괴상수 등 상수는 상태 벡터 불필요**: \(\lambda_I, \lambda_{Xe}, \gamma_I, \gamma_{Xe}\)는 물리 상수이므로 Bateman ODE 코드에 직접 하드코딩한다.

### 2.3 Python 상태 딕셔너리 예시

```python
state_t = {
    # Bateman 핵심 상태
    "N_Xe":    tensor(B, 20, 5, 5),   # Xe-135 수밀도
    "N_I":     tensor(B, 20, 5, 5),   # I-135 수밀도

    # 중성자 관련
    "phi_th":  tensor(B, 20, 5, 5),   # 2군 열중성자속
    "Sigma_f": tensor(B, 20, 5, 5),   # 거시 핵분열 단면적

    # 열수력 (모델 예측값 또는 MASTER TH 출력)
    "T_fuel":  tensor(B, 20, 5, 5),
    "T_mod":   tensor(B, 20, 5, 5),
    "rho_mod": tensor(B, 20, 5, 5),

    # 연소도 (초기값 + 누적 추적)
    "burnup":  tensor(B, 20, 5, 5),   # [MWd/kgU]

    # 제어봉
    "rod_map": tensor(B, 20, 5, 5),
}
```

***

## 3. σᵃ_Xe 계산: ENDF를 쓰면 안 되는 이유

### 3.1 2-군 유효 단면적의 개념

MASTER는 연속 에너지 핵자료가 아닌 **노드 등가 2-군 유효 단면적**을 사용한다. 이 값은 격자 물리 코드(KARMA, DeCART2D, CASMO-3 등)의 수송 계산을 통해 에너지 스펙트럼 가중 평균으로 도출된다:[^7]

\[
\sigma_{a,g}^{Xe} = \frac{\int_{E \in g} \sigma_a^{Xe}(E)\,\phi(E)\,dE}{\int_{E \in g}\phi(E)\,dE} \quad \text{[Eq. 4]}
\]

\(\phi(E)\)는 해당 노드의 에너지 스펙트럼으로, 온도·밀도·연소도에 따라 형태가 달라진다. 따라서 원시 ENDF/B 핵자료(\(\sigma_a^{Xe}(E)\), 연속 에너지)를 직접 Eq. 4에 대입하려면 격자 수송 코드를 새로 구현해야 하므로, 사실상 불가능에 가까운 작업량이 발생한다.

### 3.2 스펙트럼 경화(Spectrum Hardening)와 감속재 의존성

Xe-135는 약 0.084 eV에서 극대 흡수 공명을 가진다. 열중성자 스펙트럼 형태를 변화시키는 주요 인자별 효과:[^8][^9]

| 인자 | 스펙트럼 효과 | \(\sigma_a^{Xe}\)에 미치는 영향 |
|------|-------------|---------------------------------|
| \(T_{mod}\) 상승 | 스펙트럼 경화 (고에너지 이동) | **감소** (공명에서 멀어짐)[^10] |
| \(\rho_{mod}\) 감소 | 감속능 저하 → 경화 | **감소** |
| \(T_{fuel}\) 상승 | Doppler broadening (U-238 흡수 증가) | **간접적**, 미미한 영향[^5][^11] |
| \(B_u\) 증가 | 핵분열 생성물 축적 → 경화 | **감소 경향**[^12] |

**결론**: \(\sigma_a^{Xe}\)를 정확히 계산하려면 \(T_{mod}\)와 \(\rho_{mod}\)가 **핵심 입력**이며, \(T_{fuel}\)만으로는 불충분하다.

***

## 4. MASTER 코드 연동: 데이터 추출 방법

### 4.1 방법 A — MAS_NXS 파일 (Ground Truth 확보용)

MASTER 매뉴얼 46·240페이지에 설명된 **노드별 단면적 출력 기능**을 활성화한다.

**설정 방법** (`%EDT_OPT` 블록):

```
%EDT_OPT
 0 0 0 0 0 1   ← 6번째 변수 inxs = 1 로 설정
```

**MAS_NXS 파일에서 추출 가능한 항목** (매뉴얼 242페이지 기준):

| 출력 변수명 | 의미 | 단위 |
|-------------|------|------|
| `XSXE35(g)` | 노드별 Xe-135 **거시** 흡수 단면적 \(\Sigma_a^{Xe}\) | cm⁻¹ |
| `NDXE` | 노드별 Xe-135 수밀도 \(N_{Xe}\) | atoms/cm³ |

**미시 단면적 역산:**

\[
\sigma_a^{Xe} = \frac{\Sigma_a^{Xe}}{N_{Xe}} = \frac{\texttt{XSXE35(g)}}{\texttt{NDXE}} \quad \text{[Eq. 5]}
\]

이 값이 MASTER가 실제로 사용한 온도·연소도 피드백을 포함한 **정확한 2-군 미시 단면적**이며, AI 모델의 Ground Truth 데이터셋으로 직접 활용할 수 있다.

> **활용 시나리오**: MASTER 학습 데이터 생성 시 매 시간 스텝·매 노드에서 `XSXE35(g)`와 `NDXE`를 추출하여 Eq. 5로 \(\sigma_a^{Xe}\) 계산 → HDF5 학습 데이터에 포함.

### 4.2 방법 B — MAS_XSL 파일 파싱 (추론용 화이트박스 모듈)

MASTER의 교차단면적 라이브러리 파일 **MAS_XSL**(매뉴얼 197페이지)에는 격자 물리 코드가 사전 계산한 단면적 테일러 전개 계수가 저장되어 있다. 파일 구조 요약:[^7]

```
[핵종 블록: XE135 (CASMO 코드 54135)]
  [번업 레퍼런스 포인트 수 N_BU]
  [기준 상태값: Tf0, Tm0, rho_m0, ppm0]
  ↓ 각 번업 포인트마다 반복
  · σ_ref(Bu)         ← 기준 단면적 (2군: fast + thermal)
  · ∂σ/∂√Tf           ← 연료온도 편미분 계수
  · ∂σ/∂Tm_1~6        ← 감속재 온도 편미분 계수 (최대 6구간)
  · ∂σ/∂ρm            ← 감속재 밀도 편미분 계수
  · ∂σ/∂ppm           ← 붕소 농도 편미분 계수
```

**MASTER 내부 보간 수식** (테일러 1차 전개):

\[
\sigma_a^{Xe}(B_u, T_f, T_m, \rho_m) \approx \sigma_{a,\text{ref}}^{Xe}(B_u) + \frac{\partial\sigma_a^{Xe}}{\partial\sqrt{T_f}}\Bigl(\sqrt{T_f}-\sqrt{T_{f,0}}\Bigr) + \sum_{k}\frac{\partial\sigma_a^{Xe}}{\partial T_{m,k}}\Delta T_{m,k} + \frac{\partial\sigma_a^{Xe}}{\partial\rho_m}(\rho_m - \rho_{m,0}) \quad \text{[Eq. 6]}
\]

***

## 5. MAS_XSL 기반 Python 구현 단계별 절차

### 5.1 파일 파싱 및 계수 테이블 로드

```python
def parse_mas_xsl_xe135(filepath):
    """
    MAS_XSL 파일에서 Xe-135 관련 항목 파싱.
    반환값: 번업 그리드, 기준 단면적, 편미분 계수 딕셔너리, 기준 상태값
    """
    # 1. 파일 열기, XE135 핵종 블록 위치 찾기
    # 2. 번업 포인트 수 및 그리드 읽기
    # 3. 기준 상태값 (Tf0, Tm0, rho_m0) 읽기
    # 4. 각 번업 포인트에서 sigma_ref, 편미분 계수 배열로 저장
    return {
        'burnup_grid': burnup_grid,       # shape: (N_BU,)
        'sigma_ref':   sigma_ref_table,   # shape: (N_BU,) [cm²]
        'Tf0':         Tf0,               # 기준 연료 온도 [K]
        'Tm0':         Tm0,               # 기준 감속재 온도 [K]
        'rho_m0':      rho_m0,            # 기준 감속재 밀도 [g/cm³]
        'd_sqrt_tf':   d_sqrt_tf_table,   # shape: (N_BU,)
        'd_Tm':        d_Tm_table,        # shape: (N_BU, N_Tm_seg)
        'd_rho':       d_rho_table,       # shape: (N_BU,)
    }
```

### 5.2 런타임 계산 함수 (추론 루프에서 매 스텝 호출)

```python
def compute_sigma_a_xe(Bu, Tf, Tm, rho_m, xsl_data):
    """
    현재 노드 상태로부터 σ_a^Xe 계산 (Eq. 6).
    입력: Bu [MWd/kgU], Tf [K], Tm [K], rho_m [g/cm³]
    출력: σ_a^Xe [cm²]
    """
    bg   = xsl_data['burnup_grid']

    # Step 1: 번업 보간으로 계수 추출
    sigma_ref  = np.interp(Bu, bg, xsl_data['sigma_ref'])
    d_sqrt_tf  = np.interp(Bu, bg, xsl_data['d_sqrt_tf'])
    d_rho      = np.interp(Bu, bg, xsl_data['d_rho'])
    d_Tm       = interp_Tm_segment(Bu, Tm, xsl_data)  # 구간별 처리

    # Step 2: 기준값 로드 (파일에서 1회 읽은 상수)
    Tf0    = xsl_data['Tf0']
    Tm0    = xsl_data['Tm0']
    rho_m0 = xsl_data['rho_m0']

    # Step 3: 테일러 전개 대입 (Eq. 6)
    sigma = sigma_ref
    sigma += d_sqrt_tf * (np.sqrt(Tf)  - np.sqrt(Tf0))   # 연료온도 보정
    sigma += d_Tm      * (Tm   - Tm0)                     # 감속재 온도 보정
    sigma += d_rho     * (rho_m - rho_m0)                 # 감속재 밀도 보정

    return sigma  # [cm²]
```

### 5.3 검증: MAS_NXS와 오차 비교

```python
# MASTER MAS_NXS 출력값으로 검증
sigma_nxs = XSXE35_g / NDXE                  # Ground Truth (Eq. 5)
sigma_xsl = compute_sigma_a_xe(Bu, Tf, Tm, rho_m, xsl_data)

rel_err = abs(sigma_xsl - sigma_nxs) / sigma_nxs * 100
print(f"상대 오차: {rel_err:.4f} %")          # 목표: < 0.1%
```

***

## 6. 추론 파이프라인 전체 흐름

```
[t 스텝 입력 상태 s_t]
  N_Xe, N_I, phi_th, Sigma_f,
  T_fuel, T_mod, rho_mod, burnup, rod_map
          │
          ▼
  [ViT3D Spatial Encoder]
    z_t = ViT3D(s_t)  →  shape: (B, D_latent)
          │
          ▼
  [Mamba Temporal Model]
    h_t, ŷ_{t+1} = Mamba(z_t, h_{t-1})
    ŷ_{t+1} = (T_fuel, T_mod, rho_mod, phi_th, Sigma_f, N_Xe_raw, N_I_raw, ...)
          │
          ▼
  [σᵃ_Xe 외부 계산 모듈]   ← MAS_XSL 계수 테이블 (1회 로드)
    σ_a^Xe(t+1) = compute_sigma_a_xe(Bu, T_fuel, T_mod, rho_mod)
          │
          ▼
  [Bateman ODE 수치 적분]   ← 물리 일관성 강제
    dN_I/dt   = γ_I·Σ_f·φ - λ_I·N_I
    dN_Xe/dt  = γ_Xe·Σ_f·φ + λ_I·N_I - λ_Xe·N_Xe - σ_a^Xe·φ·N_Xe
    적분 방법: Matrix Exponential (Δt = 5 min)
          │
          ▼
  [보정된 N_Xe(t+1), N_I(t+1)]
    → 다음 스텝 s_{t+1} 구성에 사용
```

### 6.1 Bateman ODE 수치 적분 — Matrix Exponential 권장

\[
\mathbf{N}(t+\Delta t) = \exp(M\,\Delta t)\,\mathbf{N}(t) + \text{소스항} \quad \text{[Eq. 7]}
\]

\[
M = \begin{pmatrix} -\lambda_I & 0 \\ \lambda_I & -(\lambda_{Xe} + \sigma_a^{Xe}\phi) \end{pmatrix}
\]

Matrix Exponential 방법은 강성(stiff) ODE에 대한 해석해로, Xe-135의 시정수(~9.14 h)와 I-135의 시정수(~6.57 h)가 혼재하는 강성 문제를 안정적으로 처리한다.[^13]

```python
from scipy.linalg import expm
import numpy as np

LAMBDA_I  = 2.87e-5   # s⁻¹
LAMBDA_XE = 2.10e-5   # s⁻¹
GAMMA_I   = 0.0639
GAMMA_XE  = 0.0023
DT        = 300.0     # 5분 = 300초

def bateman_step(N_I, N_Xe, phi, Sigma_f, sigma_a_xe):
    source_I  = GAMMA_I  * Sigma_f * phi
    source_Xe = GAMMA_XE * Sigma_f * phi

    M = np.array([
        [-LAMBDA_I,                         0.0],
        [ LAMBDA_I, -(LAMBDA_XE + sigma_a_xe * phi)]
    ])

    eM  = expm(M * DT)
    Minv_src = np.linalg.solve(M, np.array([source_I, source_Xe]))
    N_next = eM @ np.array([N_I, N_Xe]) - Minv_src + eM @ Minv_src  # 완전 해
    return N_next, N_next[^1]   # N_I(t+Δt), N_Xe(t+Δt)
```

***

## 7. 연소도(Bu) 추적 방법

연소도는 모델이 직접 예측하기보다 **물리적 누적 계산**으로 추적한다.

\[
B_u(t+\Delta t) = B_u(t) + \frac{\bar{Q}_{fis}(t) \cdot \Delta t}{M_{fuel} \cdot 10^6} \quad \text{[MWd/kgU]} \quad \text{[Eq. 8]}
\]

- \(\bar{Q}_{fis}\): 시간 평균 노심 핵분열 출력 [MW] (모델 출력 `Qabs`에서 획득)
- \(M_{fuel}\): 노심 핵연료 질량 [tHM]
- 분모의 10⁶: MW → W 변환, 초 → 일 변환 포함

**실용적 처리**: 부하추종 운전 시뮬레이션은 수십 시간 단위이므로 연소도 변화가 매우 작다(하루 ~1 MWd/kgU 내외). 초기 MASTER 해석 결과의 번업 분포를 초기값으로 사용하고, Eq. 8로 매 스텝 누적 갱신하는 방식이 충분히 정확하다.

***

## 8. 계산 시 주의사항 및 흔한 오해

### 8.1 ⚠️ 델타(Δ)는 시간 차이가 아님

테일러 전개에서 \(\Delta T_m = T_m^{\text{현재}} - T_{m,0}\)의 \(T_{m,0}\)은 **MAS_XSL 파일에 저장된 격자 계산 기준 상태**이다. 이전 타임스텝 값과의 차이가 아니며, 파일에서 1회 읽어 고정된 상수로 사용한다.

```python
# 올바른 사용
delta_sqrt_Tf = np.sqrt(T_fuel_now) - np.sqrt(xsl_data['Tf0'])   # ✅

# 잘못된 사용 (시간 차이로 오해)
delta_sqrt_Tf = np.sqrt(T_fuel_now) - np.sqrt(T_fuel_prev)       # ❌
```

### 8.2 ⚠️ 현재 스텝 단일 값만으로 계산 완결

`σᵃ_Xe` 계산에는 **이전 스텝 정보가 전혀 불필요**하다. 현재 스텝의 `(Bu, T_fuel, T_mod, rho_mod)`와 MAS_XSL에서 1회 로드한 계수 테이블만으로 완전히 계산된다. 2-스텝 이력을 유지할 필요가 없다.

### 8.3 ⚠️ T_fuel 단독으로 σᵃ_Xe 계산 불충분

`T_fuel`(연료 온도)은 주로 U-238의 Doppler broadening에 영향을 준다. Xe-135 흡수 단면적에 직접적으로 큰 영향을 주는 것은 **감속재 온도 \(T_{mod}\)와 감속재 밀도 \(\rho_{mod}\)**이다. 따라서 모델의 출력 특성에 `T_mod`(Tcool)와 `rho_mod`(rhocool)를 반드시 포함해야 한다.[^4][^5][^10]

### 8.4 ⚠️ 2-군 단면적과 ENDF 혼용 금지

MASTER의 \(\sigma_a^{Xe}\)는 격자 수송 코드가 에너지 스펙트럼 가중 평균으로 생성한 2-군 유효값이다. ENDF/B의 연속 에너지 단면적을 직접 Eq. 2에 대입하면 물리적으로 일관성이 없는 결과가 나온다. 반드시 MAS_XSL 또는 MAS_NXS에서 파생된 값을 사용해야 한다.[^6]

### 8.5 ⚠️ 테일러 전개 외삽 범위

Eq. 6은 기준 상태 근방의 1차 근사이므로, 기준 조건에서 크게 벗어난 상황(예: 완전 냉각 시동, 매우 높은 출력 트랜지언트)에서는 오차가 커질 수 있다. 이 경우에는 MAS_NXS 출력값을 직접 Ground Truth로 사용하는 것이 더 안전하다.[^7]

***

## 9. 대리모델(Surrogate) 접근: 적용 가능성 검토

### 9.1 DNN 대리모델 구성

테일러 전개 대신 DNN으로 \(\sigma_a^{Xe}\)를 직접 학습하는 방법이다. INL의 Griffin 코드는 이 방식으로 279개 핵종 단면적을 오차 0.01% 이내로 재현했다.[^6]

| 항목 | 내용 |
|------|------|
| 입력 차원 | 4 — \((B_u,\; \sqrt{T_{fuel}},\; T_{mod},\; \rho_{mod})\) |
| 출력 차원 | 1 — \(\sigma_a^{Xe}\) [cm²] |
| 학습 데이터 | MAS_NXS 추출값 (Eq. 5) 또는 MAS_XSL 기반 Eq. 6 평가 집합 |
| 권장 구조 | 3-hidden-layer MLP (각 28~64 뉴런, GeLU 활성화)[^6] |
| 정확도 목표 | Mean |RE| < 0.01%, 온도 계수 오차 < 0.3 pcm/K |

### 9.2 화이트박스 vs. DNN 대리모델 비교

| 기준 | MAS_XSL 테일러 전개 | DNN 대리모델 |
|------|---------------------|--------------|
| 구현 복잡도 | 파일 파싱 + 수식 구현 (저) | 학습 데이터 생성·훈련 필요 (중) |
| 물리 해석성 | 완전 명시적 | 제한적 |
| 비선형 처리 | 6구간 piecewise 선형 | 연속 비선형 직접 학습 |
| 외삽 안전성 | 입력 범위 명확, 보수적 | 훈련 범위 외 불안정 가능 |
| 업데이트 | MAS_XSL 교체만으로 즉시 갱신 | 재학습 필요 |
| 추론 속도 | μs 수준 | μs~ms 수준 |
| **권장 상황** | **기본 구현, 논문 설명 단순화** | **고정밀 비선형 효과 포착 필요 시** |

***

## 10. Q&A — 대화 중 핵심 의문 정리

### Q1. "MAS_XSL 파일 하나만 있으면 되는가? 계수를 별도로 도출해야 하는가?"

**A**: 별도 도출 불필요. MAS_XSL 파일 내부에 번업 포인트별 기준 단면적(\(\sigma_\text{ref}\))과 모든 편미분 계수(\(\partial\sigma/\partial\sqrt{T_f}\) 등)가 이미 저장되어 있다. 파일을 파싱하여 계수 테이블을 메모리에 올린 뒤, 현재 상태를 Eq. 6에 대입하면 즉시 계산 가능하다.

### Q2. "Δ 값 계산에 2스텝 정보가 필요한가?"

**A**: 불필요. 테일러 전개의 \(\Delta T_m = T_m^{\text{현재}} - T_{m,0}\)에서 \(T_{m,0}\)은 MAS_XSL 파일에 고정 저장된 격자 계산 기준값이다. 현재 스텝의 모델 예측값 하나만으로 계산이 완결된다.

### Q3. "T_fuel, T_mod, rho_mod는 어디서 가져오는가?"

**A**: 세 값 모두 **Mamba 모델의 출력값**에서 획득한다 (각각 `Tfuel`, `Tcool`, `rhocool` 출력 채널). 모델이 \(t\) 스텝 상태를 입력으로 받아 \(t+1\) 스텝의 열수력 상태를 예측하면, 그 예측값을 즉시 \(\sigma_a^{Xe}\) 계산 모듈에 전달한다.

### Q4. "연소도(Bu)를 모델이 예측해야 하는가?"

**A**: 불필요. 연소도는 Eq. 8에 따라 `Qabs` 모델 출력값으로 매 스텝 물리적으로 누적 계산한다. 부하추종 운전 시뮬레이션 시간 범위(수십 시간)에서 연소도 변화는 매우 작아, MASTER 초기값에서의 누적 추적으로 충분하다.

### Q5. "MASTER에서 어떤 파일을 언제 사용해야 하는가?"

**A**: 두 파일의 역할이 다르다.

| 파일 | 사용 시점 | 용도 |
|------|-----------|------|
| **MAS_NXS** | 학습 데이터 생성 시 | 매 시간 스텝·노드별 `XSXE35`/`NDXE` → Eq. 5로 Ground Truth 확보 |
| **MAS_XSL** | 추론 파이프라인 구현 시 | 테일러 전개 계수 1회 파싱 → 매 스텝 Eq. 6 계산 |

***

## 11. 구현 로드맵 및 우선순위

| 단계 | 작업 | 우선순위 |
|------|------|----------|
| **Step 1** | MASTER `%EDT_OPT inxs=1` 설정 → MAS_NXS 생성, `XSXE35`/`NDXE` 추출 → Eq. 5로 \(\sigma_a^{Xe}\) Ground Truth 데이터셋 구축 | 🔴 Critical |
| **Step 2** | MAS_XSL 파서 작성 → Xe-135 블록에서 번업 그리드, 기준 상태값, 편미분 계수 추출 | 🔴 Critical |
| **Step 3** | `compute_sigma_a_xe()` 함수 구현 (Eq. 6) → MAS_NXS 추출값과 상대 오차 < 0.1% 검증 | 🔴 Critical |
| **Step 4** | Bateman Matrix Exponential ODE 모듈 구현 → 추론 루프에 통합 | 🟡 High |
| **Step 5** | Mamba 추론 파이프라인에 `σᵃ_Xe 모듈` + `Bateman 모듈` 연결 | 🟡 High |
| **Step 6** | (선택) \((B_u, T_f, T_m, \rho_m) \to \sigma_a^{Xe}\) DNN 대리모델 학습 — INL Griffin 방식 참고[^6] | 🟢 Medium |

***

## 12. 논문 작성 시 핵심 강조 포인트

1. **완전한 마르코프 상태 정의의 중요성**: \(N_I\)와 \(\rho_{mod}\)를 상태 벡터에 포함하는 이유를 Eq. 1~2로 수식적으로 정당화.

2. **Physics-Informed 하이브리드 구조**: 순수 데이터 기반이 아닌, MASTER의 2-군 유효 단면적 체계를 외부 결정론적 모듈로 통합 → 물리 일관성 보장.

3. **σᵃ_Xe 처리 방법론**: 원시 핵자료(ENDF) 직접 사용을 회피하고 MASTER의 MAS_XSL 보간 수식을 화이트박스 서브모듈로 재현 → 격자 수송 코드 없이도 정확한 2-군 유효 단면적 사용 가능.

4. **Bateman ODE 후처리 보정기**: AI 모델이 직접 예측한 \(N_{Xe}\)와, 물리 ODE를 수치 적분한 \(N_{Xe}\)를 비교하는 **Physics Residual** 손실 함수 설계 가능성 — 모델 물리 일관성 강화.

---

## References

1. [[PDF] a,X ) a,X - canteach](https://canteach.candu.org/Content%20Library/20050613.pdf) - Xenon-135 has a microscopic absorption cross section of 3.5 x 10 6 barns and a total fission product...

2. [Theory and calculation of 135 Xe concentration time evolution for the ...](https://www.sciencedirect.com/science/article/abs/pii/S0306454911003744)

3. [Xenon-135 Reactor Poisoning - Stanford University](http://large.stanford.edu/courses/2014/ph241/alnoaimi2/) - The destruction of Xe-135 occurs mainly via neutron absorption. Most of it is burned off due to neut...

4. [A 2-D Physics Study for Understanding Moderator Temperature ...](https://www.tandfonline.com/doi/full/10.1080/00295639.2025.2455900?scroll=top&needAccess=true) - ... 135Xe and 10B, and their absorption XSs are also dependent on the spectrum change because of the...

5. [[PDF] Module 10 - Power Reactor Feedback Effects Rev 01.](https://www.nrc.gov/docs/ml1214/ml12142a130.pdf) - Xe135: peaks ~11.6 hr, then decays. • Xe135 capture competes with fission for neutrons. • Sm149 maxi...

6. [Deployment of neural-network-based neutron microscopic cross sections in the Griffin reactor physics application](https://www.osti.gov/servlets/purl/2570236)

7. [Improvement of cross section generation methodology for MASTER](https://www.osti.gov/etdeweb/servlets/purl/20163075)

8. [Neutron Cross Section of Xenon-135 as a Function of Energy](https://link.aps.org/doi/10.1103/PhysRev.102.823) - The neutron cross section of X e 1 3 5 as a function of energy was measured, using as velocity selec...

9. [Neutron Cross Section of Xenon-135 as a Function of Energy - ADS](https://ui.adsabs.harvard.edu/abs/1956PhRv..102..823B/abstract) - The neutron cross section of Xe 135 as a function of energy was measured, using as velocity selector...

10. [[PDF] Preliminary Solution of BEAVRS Hot Full Power at BOC by Monte ...](https://www.kns.org/files/pre_paper/36/16A-424%EC%9D%B4%ED%98%84%EC%84%9D.pdf) - Therefore, the microscopic absorption cross section of Xe-135 is smaller because of the spectrum har...

11. [Module 10 - Power Reactor Feedback Effects Rev 01. - NRC](https://www.nrc.gov/docs/ML1214/ML12142A130.pdf)

12. [On the Impact of Cross Section and Fission Product Yield Data on](https://www.djs.si/nene2024proceedings/pdf/NENE2024_105.pdf)

13. [Solving Bateman Equation for Xenon Transient Analysis Using Numerical Methods – DOAJ](https://doaj.org/article/6da5845eec6d4a66ba1668a1988b67a2) - After a nuclear reactor is shutdown, xenon-135, an isotope with a very high thermal neutron absorpti...

