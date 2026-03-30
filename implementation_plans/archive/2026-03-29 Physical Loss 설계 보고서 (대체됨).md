> [대체됨 2026-03-30] → `2026-03-30 Physical Loss 통합 레퍼런스.md`. Z=22→Z=20 오류(L11), Bateman Euler/해석해 둘 다 허용.

# Physical Loss 설계 보고서: V-SMR 부하추종 Core Surrogate Model

## 1. 개요

본 보고서는 SMART 원자로 부하추종 예측을 위한 Mamba 기반 물리 정보 내재 핵심 대리모델(Physics-Informed Core Surrogate Model)에 물리 기반 손실항(Physical Loss)을 어떻게 구성할지를 기술한다. 모델 아키텍처는 **Spatial CNN Encoder → Mamba Temporal SSM → Spatial Decoder** 파이프라인으로 구성되며, MASTER 코드의 출력 데이터를 기반으로 Xe-135/I-135 Bateman ODE, 제논 미시 흡수 단면적 1차 Taylor 전개, 중성자 확산방정식 잔차, K-eff Rayleigh 몫이라는 네 가지 물리 제약을 손실 함수에 통합한다.[^1][^2]

***

## 2. 가용 데이터

MASTER 코드 구동으로 생성되는 HDF5 형식의 시계열 데이터는 5분 간격, 1/4 대칭 노심(5×5×22 노드) 구조로, 노드별 다음의 물리량을 포함한다.[^3][^4]

### 2.1 가용 물리량 목록

| # | 기호 | 물리량 | 단위 | 비고 |
|---|------|--------|------|------|
| 1 | \(N_{Xe}(t)\) | Xe-135 수밀도 | #/barn-cm | MASTER 직접 출력 |
| 2 | \(N_I(t)\) | I-135 수밀도 | #/barn-cm | MASTER 직접 출력 |
| 3 | \(\phi_{g1}(t)\) | 1군 중성자속 | n/cm²·s | 에너지 고속군 |
| 4 | \(\phi_{g2}(t)\) | 2군 중성자속 | n/cm²·s | 에너지 열군 |
| 5 | \(\Sigma_{f,g1}\) | 1군 거시 핵분열 단면적 | cm⁻¹ | 격자 라이브러리 기반 Taylor 전개 갱신 |
| 6 | \(\Sigma_{f,g2}\) | 2군 거시 핵분열 단면적 | cm⁻¹ | 동상 |
| 7 | \(\sigma_{a}^{Xe}\) | Xe-135 미시 흡수 단면적 | barn | = XSXE35 / NDXE |
| 8 | \(T_f(t)\) | 연료 온도(Fuel Temperature) | K | Doppler 보정에 사용 |
| 9 | \(\rho_m(t)\) | 냉각재 밀도(Moderator Density) | g/cm³ | 밀도 보정에 사용 |
| 10 | \(C_B(t)\) | 붕소 농도 | ppm | |
| 11 | Rod(t) | 제어봉 위치 | step | |
| 12 | BU(t) | 연소도 | MWd/MTU | 거시단면적 보정 기준 |

> **참고**: `σ_a^Xe` 는 MASTER 내부에서 `XSXE35`(Xe-135 흡수 미시단면적, barn·단위)를 `NDXE`(Xe-135 수밀도)로 나누어 얻으며, 직접 출력되지 않을 경우 두 값으로부터 재계산한다.[^3]

***

## 3. Physical Loss 유형별 설계

본 모델에 통합되는 물리 손실항은 ODE 잔차 기반 두 항과 PDE 잔차 기반 두 항으로 구성된다.

### 3.1 Loss 1: Bateman ODE 잔차 손실 (\(\mathcal{L}_{Bateman}\))

**근거**: Xe-135/I-135의 시간 진화는 1차 연립 ODE(Markov 구조)로 정확히 기술된다. 모델이 예측한 \(\hat{N}_{Xe}(t+1)\), \(\hat{N}_I(t+1)\)이 물리 방정식을 만족하도록 강제한다.[^5]

**I-135 Bateman 방정식**:

\[\frac{dN_I}{dt} = \gamma_I \cdot \left(\Sigma_{f,g1}\phi_{g1} + \Sigma_{f,g2}\phi_{g2}\right) \times 10^{-24} - \lambda_I \cdot N_I \quad [{\rm \#/barn\text{-}cm/s}]\]

**Xe-135 Bateman 방정식**:

\[\frac{dN_{Xe}}{dt} = \gamma_{Xe} \cdot \Sigma_f\phi \times 10^{-24} + \lambda_I \cdot N_I - \left(\lambda_{Xe} + \sigma_a^{Xe} \cdot \phi\right) \cdot N_{Xe}\]

**수치 적분 방법**: 해석해(Analytical Solution) 방식을 권장한다. \(\Delta t\) 내 \(\phi\)를 상수로 고정하면 상수 계수 선형 ODE가 되어 행렬 지수함수 형태의 닫힌 해가 존재한다.[^6]

\[\mathbf{N}(t+\Delta t) = e^{\mathbf{A}\Delta t}\mathbf{N}(t) + \mathbf{A}^{-1}(e^{\mathbf{A}\Delta t} - \mathbf{I})\mathbf{S}\]

전진 오일러도 허용(\(\Delta t\) = 300s에서 국소 절단 오차 \(O(\Delta t^2) \approx 0.09\%\))되나, 해석해가 physical loss 타겟 자체에 수치 오차 없이 더 엄밀한 제약을 부여한다.[^6]

**손실 정의**:

\[\mathcal{L}_{Bateman} = \frac{1}{B \cdot V} \sum_{b,v} \left[ \left(\hat{N}_{Xe,b,v}(t+1) - N_{Xe,b,v}^{phys}\right)^2 + \left(\hat{N}_{I,b,v}(t+1) - N_{I,b,v}^{phys}\right)^2 \right]\]

여기서 \(N^{phys}\)는 현재 스텝 GT 값으로 계산한 ODE 적분 결과이고, \(\hat{N}\)은 모델 예측값이다.[^3]

**필요 입력 데이터**:

| 물리량 | 출처 | 용도 |
|--------|------|------|
| \(N_{Xe}(t)\), \(N_I(t)\) | MASTER GT | ODE 초기조건 |
| \(\phi_{g1}(t)\), \(\phi_{g2}(t)\) | MASTER GT | 생산항·소멸항 계수 |
| \(\Sigma_{f,g1}\), \(\Sigma_{f,g2}\) | MASTER GT | 핵분열 생산항 |
| \(\sigma_a^{Xe}(t)\) | MASTER 계산 | Xe 소멸항 |
| \(\gamma_I\), \(\gamma_{Xe}\), \(\lambda_I\), \(\lambda_{Xe}\) | 물리 상수 | 고정 계수 |

***

### 3.2 Loss 2: Taylor 전개 기반 \(\sigma_a^{Xe}\) 보정 손실 (\(\mathcal{L}_{Taylor}\))

**근거**: 모델이 예측한 \(\hat{T}_f(t+1)\), \(\hat{\rho}_m(t+1)\)로부터 계산한 Xe 미시 흡수 단면적이 실제 물리값과 일치해야 한다. 이는 단면적 재구성 공식의 정확성을 보장한다.[^4]

**Taylor 전개 공식** (1차 근사):

\[\sigma_{Taylor}(g) = \sigma_0(g) + \frac{\partial\sigma}{\partial\sqrt{T_f}}(g) \cdot \left(\sqrt{\hat{T}_f + 273.15} - \sqrt{T_{f,ref} + 273.15}\right) + \mathrm{interp}\!\left(\frac{\partial\sigma}{\partial M}(g,:),\, \hat{\rho}_m\right) \cdot \left(\hat{\rho}_m - \rho_{m,ref}\right)\]

**손실 정의**:

\[\mathcal{L}_{Taylor} = \frac{1}{B \cdot V \cdot G}\sum_{b,v,g} \left(\sigma_{Taylor,b,v}^{(g)} - \sigma_{a,b,v}^{Xe,GT,(g)}\right)^2\]

**필요 입력 데이터**:

| 물리량 | 출처 | 용도 |
|--------|------|------|
| \(\hat{T}_f(t+1)\) | 모델 예측값 | Taylor 입력 |
| \(\hat{\rho}_m(t+1)\) | 모델 예측값 | Taylor 입력 |
| \(\sigma_0(g)\), \(\partial\sigma/\partial\sqrt{T_f}\), \(\partial\sigma/\partial M\) | 격자 라이브러리 (고정) | Taylor 계수 |
| \(\sigma_a^{Xe,GT}\) | MASTER 출력 | 보정 타겟 |

> **주의**: 기준 조건(\(T_{f,ref}\), \(\rho_{m,ref}\))에서 크게 벗어나면 1차 근사 오차가 증가하므로, 격자 라이브러리를 충분히 촘촘하게 구성하는 것이 중요하다.[^4]

***

### 3.3 Loss 3: 중성자 확산방정식 잔차 손실 (\(\mathcal{L}_{diffusion}\))

**근거**: 모델이 예측한 중성자속 분포 \(\hat{\phi}\)가 2군 중성자 확산방정식을 만족하도록 PDE 잔차를 손실에 통합한다. 이는 순수 data-driven 방식 대비 물리적 일관성을 크게 향상시킨다.[^2][^7]

**2군 확산방정식 잔차**:

\[R_g(\hat{\phi}) = -\nabla \cdot D_g \nabla\hat{\phi}_g + \Sigma_{r,g}\hat{\phi}_g - \frac{\chi_g}{k_{eff}}\sum_{g'}\nu\Sigma_{f,g'}\hat{\phi}_{g'} - \sum_{g'\neq g}\Sigma_{s,g'\to g}\hat{\phi}_{g'}\]

\[\mathcal{L}_{diffusion} = \frac{1}{B \cdot V} \sum_{b,v,g} R_g(\hat{\phi})^2\]

**필요 입력 데이터**:

| 물리량 | 출처 | 용도 |
|--------|------|------|
| \(\hat{\phi}_{g1}\), \(\hat{\phi}_{g2}\) | 모델 예측값 | 잔차 계산 |
| \(D_{g1}\), \(D_{g2}\) | MASTER GT / 라이브러리 | 확산 계수 |
| \(\Sigma_{a,g}\), \(\Sigma_{r,g}\) | MASTER GT | 흡수·제거 단면적 |
| \(\nu\Sigma_{f,g}\) | MASTER GT | 핵분열 생산 |
| \(k_{eff}\) | Rayleigh 몫 계산 또는 GT | 임계도 |

> **구현 주의**: 공간 미분(\(\nabla^2\))은 유한 차분(FDM) 또는 노달법의 격자 라플라시안으로 처리하고, TF2 GradientTape 자동 미분이 아닌 명시적 차분 행렬 곱으로 구현하는 것이 VRAM 효율상 유리하다.[^1]

***

### 3.4 Loss 4: K-eff Rayleigh 몫 손실 (\(\mathcal{L}_{keff}\))

**근거**: 유효증배계수(K-eff)는 노심 임계도의 핵심 지표로, 예측 플럭스에서 Rayleigh 몫으로 직접 계산하여 GT K-eff와 비교한다.[^2]

\[k_{pred} = \frac{\langle\hat{\phi},\; \nu\Sigma_f \hat{\phi}\rangle}{\langle\hat{\phi},\; M\hat{\phi}\rangle} = \frac{\sum_{v,g}\nu\Sigma_{f,g,v}\hat{\phi}_{g,v}^2}{\sum_{v,g}(D_g|\nabla\hat{\phi}_{g,v}|^2 + \Sigma_{a,g,v}\hat{\phi}_{g,v}^2)}\]

\[\mathcal{L}_{keff} = \left(k_{pred} - k_{GT}\right)^2\]

**필요 입력 데이터**:

| 물리량 | 출처 | 용도 |
|--------|------|------|
| \(\hat{\phi}_{g}\) | 모델 예측값 | 분자·분모 계산 |
| \(\nu\Sigma_{f,g}\) | MASTER GT | 분자 |
| \(D_g\), \(\Sigma_{a,g}\) | MASTER GT | 분모 |
| \(k_{GT}\) | MASTER 출력 | 타겟값 |

***

## 4. 전체 손실 함수 구성

\[\mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{Bateman}\mathcal{L}_{Bateman} + \lambda_{Taylor}\mathcal{L}_{Taylor} + \lambda_{diffusion}\mathcal{L}_{diffusion} + \lambda_{keff}\mathcal{L}_{keff}\]

### 4.1 손실항 특성 비교

| Loss 항 | 제약 방식 | 타겟 | 필요 추가 입력 | 연산 비용 | 우선순위 |
|---------|-----------|------|--------------|-----------|----------|
| \(\mathcal{L}_{rec}\) | 데이터 재구성 MSE | GT 전체 필드 | — | 낮음 | ① 필수 |
| \(\mathcal{L}_{Bateman}\) | ODE 잔차 (전진 적분) | GT 기반 ODE 해 | \(\gamma, \lambda\) 상수 | 낮음 | ② 핵심 |
| \(\mathcal{L}_{Taylor}\) | 단면적 보정 일치 | GT \(\sigma_a^{Xe}\) | 격자 라이브러리 편미분 계수 | 낮음 | ③ 핵심 |
| \(\mathcal{L}_{diffusion}\) | PDE 잔차 | 0 (잔차 → 0) | 확산·단면적 계수 | 중간 | ④ 선택 |
| \(\mathcal{L}_{keff}\) | 임계도 일치 | GT K-eff | — | 낮음 | ⑤ 선택 |

### 4.2 권장 가중치 초기값 및 Warm-up 전략

물리 손실항은 스케일 불일치(예: \(N_{Xe} \sim 10^{-7}\) barn·cm 단위 vs. \(\phi \sim 10^{13}\) n/cm²·s)가 크므로, 각 물리량을 노드별 평균으로 정규화한 뒤 손실을 계산해야 한다.[^1]

- **Warm-up 단계** (첫 \(N_{warmup}\) 에폭): \(\lambda_{Bateman} = \lambda_{Taylor} = 0.01\), \(\lambda_{diffusion} = \lambda_{keff} = 0\)
- **Ramp-up 단계**: 에폭마다 \(\lambda_{phys}\)를 선형 증가시켜 목표 가중치까지 도달
- **Full training 단계**: 권장 초기 가중치 예시 \(\lambda_{rec}=1.0\), \(\lambda_{Bateman}=0.5\), \(\lambda_{Taylor}=0.3\), \(\lambda_{diffusion}=0.1\), \(\lambda_{keff}=0.1\)

> Warm-up 없이 초기부터 높은 \(\lambda_{phys}\)를 적용하면 gradient conflict로 학습이 불안정해진다.[^8][^9]

***

## 5. TF2 구현 패턴

```python
import tensorflow as tf

# 물리 상수 (전역)
GAMMA_I  = 0.0639   # I-135 핵분열 수율
GAMMA_XE = 0.00228  # Xe-135 직접 수율
LAMBDA_I  = 2.878e-5  # s^-1
LAMBDA_XE = 2.092e-5  # s^-1
DT = 300.0            # 초 (5분)

def bateman_analytical(N_xe, N_I, phi, sigma_a_xe, Sigma_f_total):
    """해석해 기반 Bateman ODE 적분기 (Δt 내 phi 고정 가정)"""
    S_I   = GAMMA_I  * Sigma_f_total * phi * 1e-24
    S_Xe  = GAMMA_XE * Sigma_f_total * phi * 1e-24

    lambda_eff = LAMBDA_XE + sigma_a_xe * phi  # Xe 유효 소멸 상수

    # I-135 해석해
    exp_I   = tf.exp(-LAMBDA_I * DT)
    N_I_next = N_I * exp_I + (S_I / LAMBDA_I) * (1.0 - exp_I)

    # Xe-135 해석해 (I-135 붕괴항 포함)
    exp_eff = tf.exp(-lambda_eff * DT)
    denom   = lambda_eff - LAMBDA_I + 1e-30  # 분모 0 방지
    N_xe_next = (
        N_xe * exp_eff
        + (S_Xe + LAMBDA_I * N_I - S_I * LAMBDA_I / LAMBDA_I) / lambda_eff * (1.0 - exp_eff)
        + LAMBDA_I * (N_I - S_I / LAMBDA_I) / denom * (exp_I - exp_eff)
    )
    return N_xe_next, N_I_next

def taylor_sigma_xe(T_f_pred, rho_m_pred, sigma0, dsigma_dTf, dsigma_dM,
                    T_f_ref, rho_m_ref):
    """1차 Taylor 전개 기반 sigma_a^Xe 재계산"""
    delta_sqTf = tf.sqrt(T_f_pred + 273.15) - tf.sqrt(T_f_ref + 273.15)
    delta_rho  = rho_m_pred - rho_m_ref
    return sigma0 + dsigma_dTf * delta_sqTf + dsigma_dM * delta_rho

@tf.function
def compute_physical_loss(model_output, gt_data, lambda_weights, lib_params):
    N_xe_pred = model_output['N_xe']
    N_I_pred  = model_output['N_I']
    phi_g1    = gt_data['phi_g1']
    phi_g2    = gt_data['phi_g2']
    Sigma_f   = gt_data['Sigma_f_g1'] * phi_g1 + gt_data['Sigma_f_g2'] * phi_g2

    # Bateman Loss
    N_xe_phys, N_I_phys = bateman_analytical(
        gt_data['N_xe'], gt_data['N_I'],
        phi_g1 + phi_g2, gt_data['sigma_a_xe'], Sigma_f
    )
    L_bateman = tf.reduce_mean(
        (N_xe_pred - N_xe_phys)**2 + (N_I_pred - N_I_phys)**2
    )

    # Taylor Loss
    sigma_pred = taylor_sigma_xe(
        model_output['T_f'], model_output['rho_m'],
        lib_params['sigma0'], lib_params['dsigma_dTf'], lib_params['dsigma_dM'],
        lib_params['T_f_ref'], lib_params['rho_m_ref']
    )
    L_taylor = tf.reduce_mean((sigma_pred - gt_data['sigma_a_xe'])**2)

    return (lambda_weights['bateman'] * L_bateman +
            lambda_weights['taylor']  * L_taylor)
```

***

## 6. 데이터-물리량 매핑 전체 요약

| Physical Loss 항 | 필요 물리량 | MASTER 가용 여부 | 추가 계산 필요 |
|-----------------|------------|:---------------:|:--------------:|
| \(\mathcal{L}_{Bateman}\) | \(N_{Xe}, N_I, \phi_{g1}, \phi_{g2}, \Sigma_{f,g1}, \Sigma_{f,g2}, \sigma_a^{Xe}\) | ✅ 전부 가용 | 없음 (해석해 직접 적용) |
| \(\mathcal{L}_{Taylor}\) | \(T_f, \rho_m, \sigma_a^{Xe}\) (GT) + 라이브러리 편미분 계수 | ✅ 가용 | 격자 라이브러리에서 편미분 계수 추출 필요 |
| \(\mathcal{L}_{diffusion}\) | \(\hat{\phi}_g, D_g, \Sigma_{a,g}, \nu\Sigma_{f,g}, k_{GT}\) | ✅ 부분 가용 | \(\nabla^2\phi\) 유한 차분 행렬 구성 필요 |
| \(\mathcal{L}_{keff}\) | \(\hat{\phi}_g, \nu\Sigma_{f,g}, D_g, \Sigma_{a,g}, k_{GT}\) | ✅ 가용 | Rayleigh 몫 연산기 구현 필요 |

***

## 7. 구현 시 주의사항 및 권고

1. **수치 안정성**: Bateman 해석해에서 \(\lambda_{eff} \approx \lambda_I\)인 플럭스 조건(즉, \(\sigma_a^{Xe} \cdot \phi \approx 0\))에서 분모 \((\lambda_{eff} - \lambda_I)\)가 0에 접근하므로 `tf.where` 또는 small-ε 클리핑 처리가 필수다.[^6]

2. **단위 정규화**: \(N_{Xe} \sim 10^{-7}\) barn⁻¹·cm⁻¹ 규모와 \(\phi \sim 10^{13}\) n/cm²·s 규모가 7~8 자릿수 차이나므로, 각 손실항 내부에서 물리량을 노드별 평균으로 나눠 정규화한다.[^1]

3. **Mamba와의 연동**: Mamba SSM 자체가 선택적 상태공간 모델(Selective SSM)로 Markov 구조의 ODE를 자연스럽게 표현하므로, Bateman ODE residual을 별도 penalty가 아닌 SSM 전이 행렬 초기화에 반영하는 방식도 고려할 수 있다.[^10][^11]

4. **PINNMamba 참고**: PINNMamba는 Mamba(SSM)가 PDE의 시간 의존성을 연속 이산 불일치 없이 모델링할 수 있음을 보이며, 확산방정식 잔차 손실 적용 시 유용한 구조적 참고가 된다.[^11]

5. **L2 vs L∞ loss**: ODE 잔차의 경우 일반적 L2 손실이 적절하나, 국소적 플럭스 피킹(local power peaking) 구간에서 잔차가 급증할 경우 gradient clipping 또는 L∞ 기반 변형을 고려할 수 있다.[^9]

---

## References

1. [지금 아래 맥락에 대하여 구현하려고 하고있어. 공간 모델로 4차원(3차원 텐서 + N개의 물리량채널) 데이터를 latent vector로 압축하고, 이를 시계열 모델인 Mamba가 받아서 처리하도록 하고싶은데, 이 과정에서 Bateman Equation을 이용한 physical loss를 적용하고 싶어. 이 때 시뮬레이션의 시간간격(즉, 데이터를 얻은 포인트의 delta_t)는 5min으로 일정해. physical loss 등을 적용하고 싶은데...

...
## 요약


| 항목 | 판단 |
|------|:----:|
| $t \to t+1$ 매핑으로 학습 | ✅ 유효 (마르코프 ODE) |
| $\sigma_a^{Xe}(t)$를 입력 feature로 사용 | ✅ 유효 (소멸항 결정 인자) |
| Physical Loss (Bateman ODE 잔차) | ✅ 가능 (현재 스텝 GT 사용) |
| $\sigma_a^{Xe}(t+1)$ 예측 | 선택사항 — 테일러 전개 또는 신경망 출력 |
```](https://www.perplexity.ai/search/c95f7845-5bf5-497c-8e71-e65bafe3cb1d) - 설계 전략에 대한 포괄적인 리포트를 준비했습니다. 핵심 개념을 아래에 정리합니다.



세 단계의 파이프라인으로 구성됩니다:

1. Spatial Encoder — 각 타임스텝 \...

2. [아 근데 BatemanODE 및 Taylor 손실항까지는 내가 한건데 PINN은 그냥 시도해본거야 - 물리 법칙 내재화 기법 적용: Xe-135/I-135 Bateman ODE 및 제논 미시단면적 Taylor 전개 기반 물리기반 손실항(Physical Loss) 적용을 위한 노드별 12종 물리량 HDF5 형식 전처리 완료. 추가로 물리기반신경망(PINN) 구현 및 확인: 중성자 확산방정식 잔차를 손실함수에 통합, Rayleigh 몫 연산자 기반 유효증배계수(K-eff)·PPF·AO End-to-End 산출 체계 구현,](https://www.perplexity.ai/search/80626798-d219-49dd-94cc-f8d0fb2b6067) - 두 항목을 분리하면 아래와 같습니다.


물리 법칙 내재화 기법 적용: Xe-135/I-135 Bateman ODE 및 제논 미시단면적 Taylor 전개 기반 물리기반 손실항(Ph...

3. [근데 MASTER 코드 구동으로 얻을 수 있는건 특정 시점의 I-135, Xe-135, σ_a^Xe = XSXE35 / NDXE, flux 등인데 베이트만 방정식 등의 공식도 그냥 특정 시점(t)의 값으로 다음 시점(t+1)을 예측하는게 맞니?](https://www.perplexity.ai/search/0aa2d9c8-9b01-472e-8276-733c75bad1d0) - 네, 맞습니다. 그게 ODE의 본질입니다. 단, 중요한 뉘앙스가 하나 있습니다.



베이트만 방정식의 구조 자체가 바로 그 개념입니다.

\[
\frac{dN_{Xe}}{dt} ...

4. [노달법 기반 코드(MASTER, ASTRA 및 다양한 노심 해석 코드)가 거시핵반응단면적을 계산하기 위해 Σ_node(t)=Σ_fuel+ΔΣ_Burnup+ΔΣ_Tf+ΔΣ_Tm+ΔΣ_Boron+ΔΣ_Xe+Σ_Rod 등을 이용하는게 맞는지 철저한 검증을 부탁해. 제논의 경우 아까 말한대로 계산한게 맞는지와, 제논 계산처럼 Tf 및 Tm은 , Boron의 거시단면적은 어떻게 계산하는지 상세하게 알려주고, 근거 문서를 제공해줘](https://www.perplexity.ai/search/0f03383b-9178-41f1-901d-32cde1a61d4b) - 제시하신 수식의 각 항에 대해 MASTER 매뉴얼, PARCS Theory Manual, NESTLE Manual 등 7개의 1차 기술문서를 교차 검증한 결과를 보고서로 정리했습니...

5. [Differential Equations - Xenon 135 - Nuclear Power for Everybody](https://www.nuclear-power.com/nuclear-power/reactor-physics/reactor-operation/xenon-135/xenon-135-differential-equations/) - Note the coupling between the two equations. ... The burnup rate of xenon 135 depends on the neutron...

6. [MASTER의 경우 제논 베이트만 ODE를 해결하기위해 어떤 수치적 해법을 썼니? 해석해를 바로 쓸순 없을것같은데](https://www.perplexity.ai/search/7a8f3371-c914-45e0-ab4d-55d96dc43fe7) - 좋은 질문입니다. 이 부분이 핵심입니다. MASTER의 Bateman ODE 풀이 방식을 정확히 설명드리겠습니다.



MASTER를 포함한 대부분의 노심 해석 코드에서 Xe/I ...

7. [[PDF] A data-enabled physics-informed neural network with ... - arXiv](https://arxiv.org/pdf/2208.13483.pdf) - This work confirms the possibility of DEPINN for practical engineer- ing applications in nuclear rea...

8. [Solve ODE Using Physics-Informed Neural Network - MathWorks](https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html) - This example shows how to train a physics-informed neural network (PINN) to predict the solutions of...

9. [Is $L^2$ Physics Informed Loss Always Suitable for Training Physics ...](https://proceedings.neurips.cc/paper_files/paper/2022/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html) - In this paper, we challenge this common practice by investigating the relationship between the loss ...

10. [Mamba Integrated with Physics Principles Masters Long-term ... - arXiv](https://arxiv.org/html/2505.23863v2) - Here, we propose PhyxMamba, a framework that integrates a Mamba-based state-space model with physics...

11. [Sub-Sequential Physics-Informed Learning with State ...](https://arxiv.org/html/2502.00318v1)

