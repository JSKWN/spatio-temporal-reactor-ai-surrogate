# NVIDIA Modulus — 물리 인공지능 프레임워크

> **출처**: NVIDIA (2021-2025). *Modulus: An Open-Source Framework for Building, Training, and Fine-Tuning Physics-ML Models.* [GitHub](https://github.com/NVIDIA/modulus)
> **공개 수준**: 완전공개 (오픈소스)
> **우리 아키텍처 맥락**: PINN/FNO/DeepONet/GNN 등 물리 제약 학습 기법의 통합 프레임워크. Physical Loss 설계 및 학습 전략 참고

---

## 1. 시스템 개요

### 직관

Modulus는 "물리법칙을 신경망 학습에 통합하는" 도구 모음이다. PDE 잔차를 loss에 포함하는 PINN, 연산자 학습의 FNO/DeepONet, 메시 기반의 GNN 등 다양한 접근법을 제공한다.

### 지원 기법

| 기법 | 원리 | 공간 처리 | 시간 처리 |
|------|------|----------|----------|
| **PINN** | PDE 잔차를 loss에 추가 | 연속 좌표 입력 | 시간도 좌표로 입력 |
| **FNO** | Fourier 공간에서 연산자 학습 | 전역 스펙트럼 혼합 | 시간 축도 Fourier 또는 자기회귀 |
| **DeepONet** | Branch(입력 함수) + Trunk(출력 좌표) | Trunk이 공간 처리 | Branch가 시간 입력 처리 |
| **GNN** | 메시 위 message-passing | 그래프 구조 | 자기회귀 스텝 |
| **AFNO** | 적응형 Fourier 연산자 | O(N log N) 전역 혼합 | 자기회귀 |

---

## 2. 우리 문제와의 관련성

### 2.1 PINN과 우리 Physical Loss의 관계

PINN은 PDE 잔차를 **자동미분(autograd)**으로 계산하여 loss에 추가한다:
```
L_PINN = ||∂u/∂t + N[u]||²     N[u]는 PDE의 비선형 연산자
```

우리 Physical Loss는 PINN과 **유사하지만 다른** 접근:
| | PINN | 우리 Physical Loss |
|---|---|---|
| 미분 계산 | autograd (∂u/∂x 등) | **유한차분 근사** (CMFD 스텐실) |
| PDE 형태 | 임의 PDE | Bateman ODE + 확산 방정식 |
| 적용 위치 | 연속 좌표점 | **이산 격자 (720 cell)** |
| 학습 데이터 | 경계조건만 (data-free) | **MASTER GT 존재 (data-rich)** |

**핵심 차이**: PINN은 데이터 없이 PDE만으로 해를 학습하지만, 우리는 **GT 데이터 + 물리 제약**을 동시에 사용한다. 이것이 우리 접근의 강점이다 — 데이터 Loss가 전역 수렴을, Physical Loss가 물리 일관성을 각각 담당.

### 2.2 NeurIPS 2025 경고: 정밀도 문제

Modulus 관련 연구에서 보고된 경고:
- PINN이 수렴하려면 **FP64가 필요**할 수 있음 (FP32에서 gradient가 소실되는 사례)
- 특히 stiff PDE (시간 스케일이 크게 다른 시스템)에서 문제 발생

**우리 문제 관련**: L_diff_rel과 L_Bateman의 gradient가 FP32에서 소실되지 않는지 모니터링 필요. 현재 TF32(GPU 내부 최적화)를 사용 중이며, 필요 시 FP64 전환 검토.

### 2.3 Sobolev 학습 (Modulus의 확장)

Modulus는 PDE 잔차뿐 아니라 **도함수의 도함수** (고차 미분)도 loss에 포함하는 Sobolev 학습을 지원:
```
L_Sobolev = L_data + λ₁·||∂u/∂x - ∂u_GT/∂x||² + λ₂·||∂²u/∂x² - ...||²
```

**우리 문제 대응**: L_diff_rel이 이미 Sobolev 정규화 효과를 가진다 (확산 방정식이 2차 미분을 포함하므로, 이 잔차를 최소화하면 공간 매끄러움이 강제됨).

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| PINN PDE 잔차 loss | O (이미 유사) | L_diff_rel, L_Bateman이 PINN PDE 잔차와 동일 원리 |
| FP64 정밀도 모니터링 | O | L_diff_rel gradient 소실 시 FP64 전환 검토 |
| Sobolev 학습 | O (이미 내포) | L_diff_rel의 확산 방정식 잔차가 2차 미분 정규화 효과 |
| FNO/AFNO | △ | 우리 인코더/디코더의 Attention이 유사 역할. 직접 적용 이점 제한 |
| DeepONet | X | Branch/Trunk 분리가 우리 격자 구조와 맞지 않음 |

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| Modulus | NVIDIA. [GitHub](https://github.com/NVIDIA/modulus) |
| PINN | Raissi et al. JCP 2019. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |
