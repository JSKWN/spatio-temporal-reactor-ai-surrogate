# L_diffusion × Cross-Attention 이론 해석

> **작성일**: 2026-04-07
> **목적**: L_diffusion을 "잔차 최소화"가 아니라 **"attention map이 학습된 Green's function이 되도록 강제하는 신호"** 로 재해석. PINTO (Computer Physics Communications 2025) 자문 결과 정리.
> **도입 여부**: **미정**. 본 문서는 이론 가이드이며, 실제 cross-attention 기반 L_diffusion 변형 도입은 SD-Phase에서 결정.
> **범위**: 인코더 범위 밖. 디코더 / loss 가중치 설계 / attention map 사후 검증의 이론적 근거.
> **관련 문서**: `2026-03-30 Physical Loss 통합 레퍼런스.md` §3 (forward 잔차 정의), `archive/참고문헌/2026-04-07 PDE Transformer 자문 참고문헌.md` §2.7 (PINTO)

---

## 0. 한 줄 요약 (먼저 결론)

> 확산방정식 $L[\varphi] = S$ 의 풀이는 $L$ 의 **역연산자(Green's function)** 를 구하는 것과 같고, 이는 $\int G(x,x')\,S(x')\,dx'$ 적분으로 표현됨. Cross-attention의 $\sum_{x'} (Q\cdot K)\cdot V$ 구조가 정확히 이 적분과 동형이어서 **attention map = 학습된 Green's function** 으로 해석 가능. 7점 스텐실 L_diffusion은 forward 이산화이고 attention은 inverse 학습이므로, 둘은 같은 물리의 양면.

---

## 1. 비유로 시작하기 — "난로 100개의 방"

수학 식을 잔뜩 보기 전에 직관부터 잡고 가는 게 좋다.

### 1.1 그린 함수란? — "열이 퍼지는 규칙"

아주 크고 차가운 방이 있다고 상상하자. 방 한가운데(위치 $x'$)에 작은 난로 **딱 1개**를 켰다.

- 난로 근처는 아주 따뜻하다
- 멀어질수록 점점 덜 따뜻해진다

이때 **"$x'$ 위치에 켠 난로 하나 때문에 $x$ 위치의 온도가 얼마나 올라가는가?"** 를 계산해 놓은 **거리별 온도 변화 규칙**이 바로 **그린 함수 $G(x, x')$** 다.

> **그린 함수 = 어떤 한 점에서 발생한 영향력이 공간 전체에 어떻게 퍼져나가는지 보여주는 지도**

핵공학 식으로 옮기면:
- 난로 → 중성자 점 소스
- 온도 → flux
- 거리별 규칙 → 확산 길이 $L_d$ 에 따른 감쇠

### 1.2 적분은 왜 하는가? — "모든 난로의 열기 합치기"

이제 방 안에 난로가 1개가 아니라 **여기저기 100개**가 켜져 있다 (실제 원자로 = 노심 곳곳에서 핵분열 발생).

특정 위치 $x$ 의 최종 온도는 어떻게 구하나?
**100개 난로가 $x$ 에 보내는 열기를 모두 더하면 된다.**

```
최종 온도(x) = (1번 난로 세기) × (1번 난로 → x 거리별 규칙)
            + (2번 난로 세기) × (2번 난로 → x 거리별 규칙)
            + ...
            + (100번 난로 세기) × (100번 난로 → x 거리별 규칙)
```

수학에서 **"공간 전체에 있는 것들을 싹 다 더하는 작업"** 을 적분 $\int$ 이라 부른다. 그래서 그린 함수 표현에 항상 적분 기호와 곱하기(난로 세기 × 거리 규칙)가 등장한다.

### 1.3 6면 스텐실과 그린 함수의 관계 — 두 가지 방법

핵심 정정 사항: **그린 함수 ≠ 6면 면적분**. 둘은 방의 온도를 구하는 **서로 다른 두 가지 방법**이다.

| 방법 A: 6면 스텐실 | 방법 B: 그린 함수 |
|---|---|
| 바둑판 타일을 가정 | 방 전체를 한 번에 봄 |
| 내 타일 온도 변화는 앞/뒤/좌/우/위/아래 6개 이웃 타일에서만 옴 | 방 끝의 난로가 내 타일에 보내는 영향을 직접 계산 |
| **"바로 옆 동네"** 만 봄 | **"전국구 지도"** 를 봄 |
| 우리 L_diffusion 7점 스텐실 잔차 = 이 방식 | 학습된 attention 가중치 = 이 방식 |

→ 두 방법은 **같은 물리(확산방정식)의 두 측면**. 한쪽이 옳고 다른 쪽이 틀린 것이 아니라 **forward(국소 미분 ↔ 이웃) ↔ inverse(비국소 적분 ↔ 전체)** 의 관계.

---

## 2. 수학적 유도 — "왜 ∫ G·S dx' 형태인가"

비유는 직관을 잡는 데까지만 유효하다. 왜 정확히 적분 형태가 나오는지를 단계별로 본다.

### 2.1 확산방정식을 연산자 형태로

정상상태 1군 확산방정식 (단순화):

$$
-\nabla\cdot(D\nabla\varphi) + \Sigma_a \varphi = S
$$

좌변을 하나의 **연산자** $L$ 로 묶는다:

$$
L[\varphi] = S, \qquad L = -\nabla\cdot(D\nabla) + \Sigma_a
$$

$L$ 은 "$\varphi$ 를 받아서 새 함수를 내놓는 기계" — 미분 연산자(differential operator).

### 2.2 L의 선형성 — 핵심 성질

$L$ 은 **선형(linear)** 이다:

$$
L[a\varphi_1 + b\varphi_2] = a\,L[\varphi_1] + b\,L[\varphi_2]
$$

선형 방정식 $Ax = b$ 를 풀 때 $x = A^{-1}b$ 라고 쓰듯, **미분 연산자에도 같은 발상을 적용**:

$$
L[\varphi] = S \;\Longrightarrow\; \varphi = L^{-1}[S]
$$

문제는 $L^{-1}$ 이 행렬이 아니라 함수 ↔ 함수의 변환이라는 것. 하지만 **선형 변환**이므로 적분 형태로 표현 가능:

$$
\varphi(x) = \int G(x, x')\cdot S(x')\,dx'
$$

이 $G(x, x')$ 가 **그린 함수**이고, **$L$ 의 역연산자를 적분 커널로 표현한 것**이다.

### 2.3 점 소스 응답으로 G 정의

소스가 위치 $x'$ **한 점**에만 있다고 하자 (디랙 델타):

$$
S_{\text{point}}(\xi) = \delta(\xi - x')
$$

이때 확산방정식의 해를 $G(x, x')$ 라고 정의:

$$
L_x[G(x, x')] = \delta(x - x')
$$

> **물리 해석**: $G(x, x')$ = "위치 $x'$ 에 단위 강도의 점 소스를 놓았을 때 위치 $x$ 에서 측정되는 flux"

핵공학 직관:
> **어셈블리 $(z'=10, qy'=2, qx'=2)$ 에 중성자 1개를 던졌을 때, 어셈블리 $(z=15, qy=3, qx=2)$ 에서 보이는 flux 기여량**

### 2.4 중첩 원리 (Superposition)

진짜 소스 $S$ 는 한 점이 아니라 공간 전체에 분포한다. 그런데 $S$ 를 무한히 많은 점 소스의 합으로 분해할 수 있다 (델타 함수의 핵심 성질):

$$
S(x') = \int S(x')\cdot\delta(\xi - x')\,dx'
$$

각 점 소스는 $G(x, x')$ 만큼의 flux를 만들고, **$L$ 이 선형이므로 모두 더하면 끝**:

$$
\boxed{\,\varphi(x) = \int \underbrace{G(x, x')}_{\text{x'점 소스 응답}}\cdot \underbrace{S(x')}_{\text{x'에서의 소스 강도}}\,dx'\,}
$$

### 2.5 왜 곱셈인가?

$S(x')$ 는 "$x'$ 위치의 소스 강도"이고, 한 점 소스의 응답이 $G(x, x')$ 이니, **강도가 두 배면 응답도 두 배** (선형성). 그래서 곱셈.

> 만약 $S(x')$ 가 음수면 $G\cdot S$ 도 음수 → "이 위치에서 중성자가 빠져나간다" 의미. 곱셈이 부호까지 자연스럽게 처리.

### 2.6 왜 적분인가?

모든 소스 위치 $x'$ 에 걸쳐 응답을 다 더해야 하니까. **연속 공간에서의 합 = 적분**.

이산 격자(우리 모델)에서는 그냥 시그마 합:

$$
\varphi_i = \sum_j G_{ij}\cdot S_j
$$

여기서 $i, j$ 는 어셈블리 인덱스. 행렬-벡터 곱과 형태가 같다.

### 2.7 도함수는 어디 갔나? — G의 형태에 숨어 있음

$L$ 에 미분이 있었는데 적분식에는 안 보인다. **그 도함수는 G 함수의 모양 안에 숨어 있다.**

무한 균질 매질 1군 확산방정식 Green 함수의 닫힌 형태:

$$
G(x, x') = \frac{e^{-|x - x'|/L_d}}{4\pi D\,|x - x'|}, \qquad L_d = \sqrt{D/\Sigma_a}
$$

여기서:
- $L_d$ = **확산 길이(diffusion length)**
- **$L_d$ 는 D, $\Sigma_a$ 라는 미분 연산자의 계수가 결정**
- 즉 **G의 모양** (거리에 따라 어떻게 감쇠하는지) **자체가 확산방정식의 정보를 담고 있음**

> **"미분방정식을 풀었다"의 의미** = "L의 작용을 G라는 적분 커널로 한 번에 계산해놓은 것"

도함수는 G를 만드는 과정에 들어갔고, 일단 G가 만들어지면 적분만 하면 된다.

---

## 3. 7점 스텐실 vs Green 함수 — 두 시각의 정정

| 개념 | 본질 | 위치성 | 우리 모델에서 |
|---|---|---|---|
| **7점 스텐실** (6면 + 중심) | $L$ 의 forward **이산화** | **국소** — 인접 노드만 | 현재 L_diffusion 잔차 |
| **Green's function** | $L$ 의 **역(inverse)** | **비국소** — 모든 점 쌍 | (학습된 attention map의 해석) |

### 3.1 7점 스텐실 = L의 이산화

3D 라플라시안을 유한차분으로 근사하면 (Z 방향만 예시):

$$
(\nabla^2\varphi)_z \approx \frac{\varphi_{z+1} - 2\varphi_z + \varphi_{z-1}}{(\Delta z)^2}
$$

3D 확장 → 6면 이웃 + 중심 = **7점**. 우리 L_diffusion 잔차 계산:

$$
R_g = \underbrace{\sum_{\text{6면}} J_g \cdot A}_{\nabla\cdot(D\nabla\varphi)} + \underbrace{\Sigma_{r,g}\,\varphi_g\,V}_{\text{소멸}} - \underbrace{S_g\,V}_{\text{소스}}
$$

이건 **forward 연산** — "주어진 $\varphi$ 에서 잔차를 계산". $L$ 의 작용을 직접 흉내내는 것.

### 3.2 Green 함수 = L의 역

$G(x, x')$ 는 **모든 $x'$ 에서 $x$ 로의 영향을 한 번에 담은 비국소 함수**.
무한 매질에서 거리 $|x - x'|$ 에 따라 $\exp(-|x-x'|/L_d)$ 로 부드럽게 감쇠한다.

> 노드 $(z=10, qy=2, qx=2)$ 에 점 소스를 놓으면, 그 영향이 인접 6면에만 가는 게 아니라 **20×5×5 격자 전체에 부드럽게 퍼진다**. 거리가 멀어지면 지수적으로 약해질 뿐, 0이 아니다.

### 3.3 두 시각의 관계

- **7점 스텐실**: "이 노드의 다음 시간 변화는 6면 이웃에만 의존" (국소 forward)
- **Green 함수**: "이 노드의 정상상태 flux는 노심 전체 소스 분포에 의존" (비국소 적분)

이 둘은 **모순이 아니라 동일한 물리의 두 측면**. forward 연산을 무한히 적용한 결과가 inverse(Green)이고, inverse를 미분 연산자에 통과시킨 결과가 forward.

```
L · G = I  (Green 정의)
L⁻¹ = ∫ G(·,·) (·) dx'  (역연산자 적분 표현)
```

→ "L_diffusion forward 잔차를 줄이는 것" ⊂ "inverse 커널을 학습하는 것"

---

## 4. Cross-Attention = 학습된 Green's function

### 4.1 Cross-attention 식 복습

$$
u(x) = \sum_{x'} \text{softmax}\bigl(Q(x)\cdot K(x')\bigr) \cdot V(x')
$$

### 4.2 Green 적분 식

$$
\varphi(x) = \int G(x, x')\cdot S(x')\,dx'
$$

### 4.3 구조적 동형성

| Cross-attention | Green 적분 | 의미 |
|---|---|---|
| $u(x)$ | $\varphi(x)$ | 출력 (얻고 싶은 값) |
| $V(x')$ | $S(x')$ | 입력 (각 위치의 신호) |
| $\text{softmax}\bigl(Q(x)\cdot K(x')\bigr)$ | $G(x, x')$ | "$x'$ 의 신호가 $x$ 에 얼마나 기여" 가중치 |
| $\sum_{x'}$ | $\int dx'$ | 전체 위치 합산 |

→ **Cross-attention = 학습된 적분 커널**

### 4.4 핵심 결론

> **attention map = 학습된 Green's function**
>
> attention 가중치 행렬 $W_{ij} = \text{softmax}(Q_i\cdot K_j)$ 는 데이터로부터 학습된 "노드 $j$ 의 정보가 노드 $i$ 에 얼마나 결합되는가"의 표.
>
> 만약 학습이 잘 되면, $W_{ij}$ 는 **물리적 그린 함수에 가까운 패턴**을 보일 것 — 거리에 따른 감쇠, 확산 길이 비례, 제어봉 방향 비대칭 등.

### 4.5 7점 스텐실의 한계와 attention의 강점

| | 7점 스텐실 | Cross-attention |
|---|---|---|
| 커널 형태 | 미리 정해짐 (6면 이웃) | 데이터로 학습됨 |
| 위치성 | 국소 (인접만) | 비국소 (전체 가능) |
| 비균일 매질 적응 | 어려움 (계수만 변함) | 자연스러움 (커널 자체가 적응) |
| **간접 비국소 효과** (k_eff 결합, 열수력 피드백) | 표현 불가 | 표현 가능 |

> **주의**: PWR 어셈블리에서 확산 길이 $L_d$ ≈ 3-10 cm < 어셈블리 피치(21.6 cm)이므로, **직접 transport 결합은 사실상 국소적**. 따라서 cross-attention의 진짜 가치는 "원거리 직접 transport"가 아니라 **k_eff power iteration의 전역 재분배**, **열수력 피드백 chain**, **MASTER NEM 보정의 implicit 학습** — 이들은 7점 스텐실로 표현 불가하지만 attention이 학습할 수 있는 영역. 자세한 한계 분석은 §5.5 참조.

---

## 5. 우리 모델에서의 함의 — 좌표 없이도 적용 가능

### 5.1 연속 좌표 입력 부재 — 대용품 정리

PINTO 원본은 도메인 좌표 $(x, y, z)$ 를 명시적으로 토큰화한다. 우리는 그게 없다.

| 우리가 가진 것 | 값 | 활용 |
|---|---|---|
| 정수 인덱스 | $(z, qy, qx)$, $z \in [0,19]$, $qy, qx \in [0,4]$ | 토큰 ID |
| 어셈블리 피치 | 21.608 cm (XY 상수) | 인덱스 → 거리 변환 |
| Z 메시 | 10 cm (현재 균일, 추후 반사체 포함 시 30+10·n+30 비균일) | Z 거리 변환 |
| 상대 거리 | 인덱스 차 × 피치 | 위치 인코딩 입력 |
| 확산 길이 | $L_d = \sqrt{D/\Sigma_a}$ from xs_fuel | 노드별 결합 범위 추정 |

→ **STRING / RPE 같은 상대 위치 인코딩이 거리 정보를 attention에 주입**하면, attention map이 자연스럽게 "거리 의존 결합"을 학습. 이게 학습된 Green 함수 역할을 한다.

좌표 없음 ≠ 위치 정보 없음. **상대 거리만 있으면 Green 함수의 핵심 정보(거리 의존 감쇠)는 표현 가능**.

### 5.2 L_diffusion의 의미 재해석

| 시각 | 해석 |
|---|---|
| **기존 시각** | "디코더 출력 $\varphi$ 가 7점 스텐실 PDE를 만족하게 강제" |
| **PINTO 시각** | "디코더 attention 가중치가 진짜 Green 함수가 되도록 강제" |

이 둘이 **다른 학습 신호가 아니다**. forward 잔차를 줄이는 것이 inverse 커널을 학습하는 것의 구체화 — 같은 그래디언트로 같은 방향으로 끌고 간다. 다만 해석이 풍부해지고, 학습 후 진단/검증 도구가 달라진다.

### 5.3 검증 기준 — 학습 후 attention map 진단

cross-attention 기반 L_diffusion 변형을 도입하지 않더라도, **기존 모델의 attention map 자체가 학습된 Green 함수로 해석 가능**하므로 다음 진단이 가능:

| 진단 항목 | 기대 패턴 | 의미 |
|---|---|---|
| 거리별 attention 강도 | $\exp(-r/L_d)$ 비례 감쇠 | 학습된 커널이 물리적 그린 함수 모양 |
| 노드별 결합 범위 | $L_d$ 큰 노드(thermal)는 더 멀리 | xs_fuel과 일관성 |
| 제어봉 인접 노드 | 비대칭 결합 (제어봉 방향 약함) | 흡수체 효과 학습 여부 |
| z=0, z=19 경계 | 반사체 albedo 방향 결합 패턴 | BC 학습 여부 |

이 검증은 인코더/디코더 attention 모두에 적용 가능. **L_diffusion이 attention을 물리 방향으로 끌어당겼는지 사후 확인**.

### 5.4 차용 우선순위 (참고문헌 §2.7과 일관)

| PINTO 요소 | 채택 가능성 | 적용 위치 |
|---|---|---|
| Cross-attention = Green 함수 해석 | **개념적 가이드** | L_diffusion 가중치 설계, attention map 검증 기준 |
| 좌표 쿼리 cross-attention | **검토 가치** | 디코더 face flux 보간 (조화평균 대체 후보) |
| 반복 적용 (Picard) | **IIET와 통합 가능** | SD-Phase 반복 디코더 + iteration별 L_diffusion 부과 |
| Data-free 학습 | **❌ 비채택** | 우리는 MASTER GT 데이터 학습이 메인 |
| 백본 전체 | **❌ 비채택** | Mamba와 호환 안 됨, 우리 구조 부적합 |

### 5.5 한계 및 우려사항 — 본질적 정보 손실 ★

> 사용자 우려 (2026-04-07): "어셈블리 평균 flux만 사용하고, MASTER NEM의 4차 곡률 정보가 데이터에 없으며, MFP < 어셈블리 피치라 직접 transport 결합도 짧다. Cross-attention/Green 함수가 정말 효과가 있을까?"

이 우려는 **이론적으로 정확**하다. 정직한 한계 분석:

#### 5.5.1 거리 척도 비교 — "직접 결합은 사실상 국소적"

| 거리 | 값 | 의미 |
|---|---:|---|
| Mean free path (MFP) | fast ~3-5 cm, thermal ~0.5-2 cm | 1회 산란/흡수 전 평균 거리 |
| 확산 길이 $L_d$ | fast ~10 cm, thermal ~3-5 cm | 흡수까지 RMS 직선거리 |
| Migration length $M$ | ~10-12 cm | fast→thermal→흡수 전체 |
| **어셈블리 피치** | **21.608 cm** | 우리 격자 단위 |
| **$L_d$ / 피치** | **~0.5** | **$L_d$ < 피치** |

→ **진짜 Green 함수는 사실상 1-2 어셈블리 이내에서 끝남**. 7점 스텐실로 표현 가능한 영역이 거의 전부.

#### 5.5.2 그래도 비국소가 발생하는 진짜 메커니즘 3가지

| 메커니즘 | 본질 | 거리 | 7점으로 표현? | Attention으로 표현? |
|---|---|---|:---:|:---:|
| **(a) Multiple scattering chain** | 여러 어셈블리 거쳐 전파 | ~1-2 어셈블리 | ✅ | ✅ |
| **(b) k_eff 고유치 결합** | Power iteration 노심 전체 재분배 | **전역** | ❌ | **✅** |
| **(c) 열수력 피드백** | 한 노드 power → T_f, ρ_m → XS → 전 노드 | **전역** (시간 지연) | ❌ | **✅** |

**핵심**: 직접 transport는 단거리지만, **k_eff 정규화 + 열수력 피드백이 격자 전체를 묶음**. 이 두 효과가 cross-attention의 진짜 활용 영역.

#### 5.5.3 NEM 4차 곡률 — 정보 이론적 한계

MASTER NEM은 어셈블리 내부 flux 분포가 4차 다항식임을 사용:

$$
\varphi(\xi) = \sum_{n=0}^{4} a_n P_n(\xi), \quad \xi \in [-h/2, h/2]
$$

CMFD는 어셈블리 평균만 보고 1차 차분하지만, NEM은 4차 곡률 → 정확한 면 flux → $\hat{D}$ correction → CMFD 결합 강도 보정.

**우리 데이터의 한계**: `MAS_OUT` flux_3d는 **어셈블리 체적평균 flux만** 보유. 4차 다항식 계수 $a_n$ 은 HDF5에 **없음**.

| Cross-attention이 학습 가능 | 학습 불가능 |
|---|---|
| 어셈블리 평균만으로 effective $\hat{D}_{ij}$ 추정 (implicit NEM) | 어셈블리 내부 4차 곡률 자체의 복원 |
| 데이터에서 보존된 정보의 한계까지 | MASTER NEM의 진짜 $\hat{D}$ 와 정확히 일치 |
| Albedo 경계 학습 | 정보 자체가 input에 없는 것 |

→ **정보 이론적 floor 존재**. 어떤 모델도 input에 없는 정보는 만들 수 없음.

#### 5.5.4 잔차 감소 정량 추정

| 잔차 원인 | 기여도 (추정) | Cross-attention 보정 가능? |
|---|---|---|
| CMFD 1차 근사 (NEM 부재) | ~3-5% | **부분 가능** — implicit $\hat{D}$ 학습 |
| Albedo 경계 처리 근사 | ~1-2% | **가능** (현재 학습 가능 albedo로 처리 중) |
| 데이터 보간/패딩 | <1% | 불가 |
| **NEM 4차 정보 부재** | ~2-3% | **불가** — 본질적 floor |

| 항목 | 현재 | Cross-attention 도입 후 (이론 추정) |
|---|---:|---:|
| g1 median | 2.24% | ~1.5-2% |
| g2 median | 6.44% | ~3-4% |
| 본질적 하한선 | — | ~2-3% (NEM 정보 부재) |

→ 6% → 3-4% 수준 개선 기대. 하지만 0%는 불가능.

#### 5.5.5 그래서 가치가 있는가?

**조건부 가치 있음**:

| 시각 | 평가 |
|---|---|
| **잔차 감소 자체를 목표** | 6% → 3-4% 개선이 power 예측 정확도에 의미 있는지는 별도 실험 필요. 비용 대비 이득 불확실 |
| **Attention map = 학습 진단 도구** | 잔차가 아닌 **"학습된 effective $\hat{D}$"** 자체에 의미 부여. 학습 후 attention map 시각화로 모델이 진짜 물리(거리 감쇠, $L_d$ 비례, 제어봉 비대칭)를 학습했는지 진단. **구조 변경 없이도 가능** |
| **간접 결합 표현 (k_eff, 열수력)** | 7점 스텐실이 표현 못 하는 진짜 비국소 효과. **여기가 진짜 가치 영역** |

#### 5.5.6 정직한 결론

> **사용자 우려는 정확하다.** $L_d$ < 어셈블리 피치라 직접 transport는 거의 국소적이고, NEM 4차 곡률 정보는 우리 데이터에 없다. 따라서 cross-attention/Green 함수 시각으로도 잔차 0%는 불가능하고, **본질적 floor (~2-3%)** 가 존재한다.
>
> 그러나 잔차 감소가 목적이 아니라 **(1) k_eff 전역 결합과 (2) 열수력 피드백 chain**이라는 진짜 비국소 효과를 학습 가능하다는 것이 핵심 가치다. 이건 7점 스텐실로는 표현 불가하다.
>
> NEM 4차 곡률 자체를 복원하려면 **데이터에 없는 정보를 외부에서 주입**해야 한다. AI 기법 검토는 §9 참조.

#### 5.5.7 단계적 권장 (현실 노선)

| 단계 | 액션 | 비용 | 의사결정 시점 |
|---|---|---|---|
| 1단계 (현재) | 기존 7점 스텐실 L_diffusion + Albedo BC 학습. 잔차 6% 수용 | 0 | 즉시 |
| 2단계 | 학습된 모델의 self-attention map 시각화 → 거리 감쇠/제어봉 비대칭 패턴 진단. **구조 변경 없음** | 분석 시간만 | SE/T-Phase 학습 후 |
| 3단계 (조건부) | 1단계 모델이 power 예측에 ~1% 이상 오차 + 패턴이 NEM correction 부재로 해석 가능하면, 디코더 cross-attention 추가 검토 | 구현 비용 | SD-Phase |
| 4단계 (장기) | IIET 반복 디코더 + iteration별 L_diffusion으로 power iteration 모방 | SD-Phase 별도 결정 | 장기 |

---

## 6. 추가 고려사항 — 에지 케이스

### 6.1 k_eff 고유치 문제

핵분열 소스가 $\nu\Sigma_f \varphi/k_{\text{eff}}$ 라서 사실은:

$$
L[\varphi] = \frac{1}{k_{\text{eff}}}\,F[\varphi]
$$

이건 **고유치 방정식**이고, Green 함수 형식이 더 복잡 (반복법 = Picard iteration 필요):

```
φ⁰ → 1차 fission source 추측
φ¹ = L⁻¹[F[φ⁰]/k⁰]
k¹ = ⟨F[φ¹]⟩ / ⟨F[φ⁰]⟩
... 수렴까지 반복
```

이게 정확히 **MASTER의 power iteration**이고, **IIET의 반복 디코더 구조와 직접 대응**한다. 우리가 IIET를 도입하면 디코더 자체가 power iteration을 학습하는 셈.

### 6.2 시간 의존

- 우리 모델: Mamba가 시간을 처리 → Green 함수는 **각 시점 정상상태 풀이**로 해석
- 시간 의존 Green 함수 (heat kernel) 는 별도 영역, 본 문서 범위 밖
- 시간 진화는 Mamba hidden state, 공간 결합은 Green/attention — **역할 분리**

### 6.3 2군 결합

우리 확산방정식은 g1(fast)와 g2(thermal)가 산란 항으로 결합:
- g1 → g2: $\Sigma_{s,12}\varphi_1$
- g2 → g1: 0 (역산란 무시)

→ Green 함수가 **2×2 행렬 커널**:

$$
\begin{pmatrix}\varphi_1\\\varphi_2\end{pmatrix}(x) = \int \begin{pmatrix}G_{11} & G_{12}\\G_{21} & G_{22}\end{pmatrix}(x,x')\begin{pmatrix}S_1\\S_2\end{pmatrix}(x')\,dx'
$$

multi-head attention에서 head별로 다른 g↔g' 결합을 학습하는 것이 자연스러운 대응. **head 분배 시 (g1→g1, g1→g2, g2→g2) 3개를 분리하는 설계 가능**.

---

## 7. 도입/비도입 정리

### 7.1 본 문서의 위치
- **인코더 범위 밖**. 본 문서는 SD-Phase 디코더 설계 / L_diffusion 가중치 설계 / attention map 사후 검증의 **이론적 가이드**
- 강요 사항이 아닌 이론 풀(pool)
- 모든 차용은 SD-Phase에서 별도 결정

### 7.2 단기 (SE-Phase, T-Phase)
- 기존 7점 스텐실 L_diffusion 그대로 유지
- 학습 후 attention map을 §5.3 기준으로 사후 검증
- **이론 해석만 도입, 구조 변경 없음**

### 7.3 중기 (SD-Phase 결정 사항)
- Cross-attention 기반 face flux 쿼리 (조화평균 대체) 검토
- IIET 반복 디코더와 power iteration 대응 검토
- iteration별 L_diffusion 부과 (반복마다 잔차 추가) 검토

### 7.4 비채택 (현 시점)
- PINTO 본체 (data-free physics-only 학습)
- PINTO 백본 전체 (Mamba 미호환)
- 좌표 입력을 위해 격자를 연속 좌표계로 재해석하는 것 (불필요한 복잡성)

---

## 8. 한 줄 결론 (재차)

> **확산방정식 $L[\varphi]=S$ 풀이 = $L$ 의 역연산자(=Green 함수)를 구하는 것 = $\int G\cdot S\,dx'$ 적분 표현. Cross-attention의 $(Q\cdot K)\cdot V$ 구조가 정확히 이 적분과 동형이므로 attention map = 학습된 Green 함수로 해석 가능. 7점 스텐실 L_diffusion은 forward 이산화이고 attention은 inverse 학습 — 둘은 같은 물리의 양면. 좌표 없이도 상대 거리(피치, Z 메시) + 확산 길이 $L_d$ 만으로 충분. 단, NEM 4차 곡률은 정보 이론적 한계로 복원 불가 (§5.5).**

---

## 9. NEM 효과를 반영하기 위한 AI 기법 검토 ★

> §5.5에서 확인한 본질적 정보 손실(어셈블리 내부 4차 곡률 부재)을 어떻게 우회하거나 보완할 수 있는지에 대한 AI 기법 매핑.

### 9.1 정보 이론적 정리 — "외부 정보 없이는 불가능"

NEM 4차 다항식 계수 $a_n$ ($n=0..4$)는 우리 입력에 없다. 이 사실은 변하지 않는다. 따라서 가능한 길은 **3가지**:

| 길 | 설명 | 비용 |
|---|---|---|
| **(A) 외부 정보 추가** | MASTER에서 4차 정보를 추출하거나, 더 fine한 격자로 데이터 재생성 | 데이터 재생성 비용 |
| **(B) 물리 지식 주입** | 4차 다항식 form을 모델 구조에 hardcode (analytical NEM) | 구현 복잡도 |
| **(C) Implicit 학습** | 어셈블리 평균만으로 effective parameter를 학습 | 데이터 풍부, 정보 floor 존재 |

이 3가지를 각각 어떤 AI 기법이 구현할 수 있는지 정리.

---

### 9.2 (A) 외부 정보 추가 — Multi-Fidelity / Super-Resolution 계열

#### 9.2.1 Multi-Fidelity Neural Networks (MFNN)

- **출처**: Meng & Karniadakis (2020), Penwarden et al. (2023)
- **핵심**: 풍부한 저해상도 데이터(어셈블리 평균) + 소량의 고해상도 데이터(노드 단위 또는 NEM 결과)로 학습
- **구조**:
  ```
  Low-fidelity NN(어셈블리 평균) → φ_LF
  High-fidelity NN(LF 출력 + 좌표) → φ_HF (잔차 학습)
  ```
- **우리 적용 가능성**:
  - **조건**: MASTER에서 일부 LP/스텝에 대해 노드 단위 또는 핀 단위 flux를 추출 가능해야 함
  - MAS_OUT의 `pin_power` 필드 또는 `node_flux` 옵션이 있는지 확인 필요
  - 가능하면 10% LP만 high-fidelity 추출 → 학습 후 90% LP에 일반화
- **한계**: 데이터 재생성 비용 발생, MASTER 옵션 설정 필요

#### 9.2.2 Super-Resolution (SR) 계열 — SRCNN, SwinIR, Diffusion SR

- **핵심**: 저해상도 → 고해상도 매핑 학습 (이미지 처리 표준 기법)
- **PDE 적용 사례**: SR-PINN, FNO Super-Resolution
- **우리 적용 가능성**:
  - 어셈블리 평균 (20,5,5) → 노드 (60,15,15) 또는 (200,50,50) 매핑
  - **단, GT 고해상도 데이터가 있어야 학습 가능** → MFNN과 동일 조건
- **한계**: GT 없이는 unsupervised SR이 되어 hallucination 위험

#### 9.2.3 Implicit Neural Representation (INR) — SIREN, NeRF

- **핵심**: 좌표 (x,y,z) → 물리량 매핑을 MLP로 표현. 연속 함수
- **PDE 적용**: PINN의 일종, NTFields, NIF (Neural Implicit Flow)
- **우리 적용 가능성**:
  - 어셈블리 평균 + 어셈블리 내부 좌표 $\xi$ → 내부 flux 분포
  - 학습은 NEM 4차 형태를 prior로 부여하면 됨
- **한계**: GT 내부 분포 없이는 학습 불가. (B)와 결합 필요.

---

### 9.3 (B) 물리 지식 주입 — Analytical NEM Layer

#### 9.3.1 Differentiable NEM Layer (가장 직접적)

- **핵심**: NEM 4차 다항식과 $\hat{D}$ correction을 **미분 가능한 layer로 구현**해서 모델에 삽입
- **구조**:
  ```
  인코더 → 어셈블리 평균 + edge moments 예측
       → DifferentiableNEM(평균, moments, XS) → face flux
       → L_diffusion (face flux 기반)
  ```
- **참고 논문**:
  - Differentiable Physics 분야: Holl et al. (NeurIPS 2020) "Learning to Control PDEs"
  - Hybrid neural-numerical solvers: Um et al. (2020) "Solver-in-the-loop"
  - Reactor physics 특화: 별로 없음 — 우리가 첫 적용 가능
- **우리 적용 가능성**:
  - NEM 알고리즘이 명시되어 있고 (방법론 §2.1.4) 미분 가능 (다항식 + 행렬 연산)
  - **모델이 어셈블리 평균만으로 NEM moments를 추정**하도록 학습
  - 추정된 moments → analytical NEM → face flux → L_diffusion
- **한계**:
  - moments는 GT가 없으므로 implicit 학습 → §5.5의 floor 동일
  - 단, 4차 form이 hardcode되어 있어 학습 효율↑
- **장점**: 가장 물리적으로 일관됨. 구현 가능성 높음

#### 9.3.2 Galerkin Neural Network

- **출처**: Ainsworth & Dong (2021)
- **핵심**: 솔루션을 basis function의 선형 결합으로 표현, 계수만 NN으로 학습
- **우리 적용**: NEM의 Legendre 기저를 그대로 사용 가능
  ```
  φ(ξ) = Σ_n c_n(어셈블리 평균) · P_n(ξ)
       = NN(features) · Legendre_basis(ξ)
  ```
- **한계**: NEM Layer (9.3.1)의 일반화 버전. 본질 동일

#### 9.3.3 Physics-Informed Neural Networks (PINN) with NEM constraint

- **핵심**: Loss에 NEM moment 균형식을 추가
- **우리 적용**: L_diffusion에 추가 항으로 "moment balance residual" 부과
- **한계**: GT moment 없이 self-consistency만 강제 → floor 비슷

---

### 9.4 (C) Implicit 학습 — Cross-Attention 본 문서의 메인 시각

이미 §4-5에서 다룬 내용. 요약:
- Cross-attention이 어셈블리 평균만으로 effective $\hat{D}_{ij}$ 를 학습
- 4차 곡률을 직접 복원하지는 못하지만, 그것의 "효과"인 face flux 결합 강도를 학습
- §5.5의 floor (~2-3%) 가 한계

#### 9.4.1 추가 Implicit 기법

| 기법 | 핵심 | 우리 적용 |
|---|---|---|
| **Closure modeling (LES 차용)** | Subgrid-scale 효과를 NN closure로 학습 | NEM correction을 closure로 해석 |
| **Reynolds-averaged + closure** | 평균 + 변동 모델링 | 어셈블리 평균 + NEM closure NN |
| **Multi-pole methods** | 원거리 결합을 저차 표현으로 압축 | k_eff 전역 결합 효율화 |
| **Graph Neural Operator (GNO)** | 메시 무관 연산자 학습 | 어셈블리 그래프 + edge feature |
| **MGNO (Multipole GNO)** | 다중 스케일 결합 | 단거리(7점) + 중거리(attention) + 전역(multipole) 분리 |

---

### 9.5 가장 유망한 조합 (순위)

| 순위 | 조합 | 기대 효과 | 비용 | 위험 |
|---|---|---|---|---|
| **1** | **현재 7점 + Cross-attention (§4-5)** | k_eff/열수력 결합 학습. 잔차 6%→3-4% | 낮음 | 낮음 |
| **2** | **Differentiable NEM Layer (9.3.1)** | 분석적 NEM 형태를 hardcode → 정합성↑, 잔차 추가 감소 | 중간 (구현) | 낮음 |
| **3** | **MFNN with MASTER node-level data (9.2.1)** | 정보 이론적 floor 자체를 낮춤 | 높음 (데이터 재생성) | 중간 |
| 4 | Galerkin NN (9.3.2) | NEM Layer와 동등 | 중간 | 낮음 |
| 5 | PINN with NEM constraint (9.3.3) | 추가 loss 항만 | 낮음 | 낮음 |
| 6 | Super-Resolution (9.2.2) | 데이터 의존 | 높음 | 높음 (hallucination) |

### 9.6 권장 — 단계별 진입

#### 즉시 (SE/T-Phase)
- **순위 1만 추진**: 기존 7점 L_diffusion 유지 + 학습된 attention map 사후 진단
- 추가 구현 0

#### 1차 모델 평가 후 (SD-Phase 결정)
- 1차 모델의 power 예측 잔차가 ~1% 이상이고, **실패 패턴이 NEM correction 부재로 해석되면**:
  - **순위 2 (Differentiable NEM Layer) 도입 검토**
  - 우리 데이터 (어셈블리 평균) 그대로 사용 가능
  - 분석적 NEM 4차 form을 hardcode → 잔차의 본질적 floor 자체를 낮춤
  - 구현 비용 중간, 정보 이론적 한계 일부 회피

#### 장기 (필요 시)
- 1차 모델이 충분치 않고 추가 데이터 추출 가능하면:
  - **순위 3 (MFNN + node-level data) 검토**
  - MASTER 옵션으로 일부 LP에 노드/핀 단위 flux 추출
  - 비용 크지만 floor 자체를 낮출 유일한 방법
- MASTER에 그런 옵션이 있는지 확인 필요 (`MAS_OUT pin_power`, `inppr` 카드 등)

### 9.7 우리에게 적합하지 않은 기법

| 기법 | 부적합 이유 |
|---|---|
| Pure Super-Resolution | GT 고해상도 없음 → hallucination |
| Pure INR/SIREN | 좌표→물리량 매핑이라 어셈블리 평균 입력과 매칭 안 됨 |
| Diffusion SR | 데이터 부족 + diffusion 학습 절차 별도 |
| Equivariant NN (E(3)) | 우리 격자는 고정, 3D 회전 불변성 불필요 |
| Mesh-Free Neural Operators | 우리는 격자 고정 |

### 9.8 핵심 결론

> **NEM 효과의 완전 복원은 불가능 (정보 이론적 한계).** 그러나 3가지 우회 경로가 있음:
> 1. **Cross-attention (Implicit)** — 즉시 가능, 본 문서 §4-5 내용
> 2. **Differentiable NEM Layer (Hardcode)** — SD-Phase 도입 검토 가치, NEM 4차 form을 모델 구조에 직접 삽입
> 3. **Multi-Fidelity (External info)** — 장기, MASTER에서 노드/핀 단위 데이터 추가 추출 가능 시
>
> **현실 노선**: 1단계 → 2단계 → (필요 시) 3단계. 우선은 §5.5.7의 단계적 권장 그대로 진행.

---

## 부록 A. 사용자 직관용 비유 (보존)

> 본 절은 사용자가 처음 직관을 잡을 때 받은 비유 설명을 보존한 것. 본문 §1과 중복되나, 향후 다른 사람에게 설명할 때 그대로 인용 가능.

### 그린 함수란? — "열이 퍼지는 규칙"
크고 차가운 방의 한가운데 난로 1개를 켜면, 난로 근처는 따뜻하고 멀어질수록 식는다. **"A 위치에 켠 난로 때문에 B 위치 온도가 얼마나 올라가는가"** 의 거리별 규칙이 그린 함수다.

### 적분이란? — "모든 난로의 열기 합치기"
방에 난로가 100개 있으면 (실제 원자로 = 곳곳에서 중성자 발생), 특정 위치의 최종 온도 = 100개 난로의 영향 합. 이 합산 작업이 적분.

### 6면 vs 그린 함수
- **6면 스텐실** = "바로 옆 동네만 보기" (인접 타일 6개로부터의 열 이동만 계산)
- **그린 함수** = "전국구 지도" (방 끝의 난로가 내 타일에 미치는 영향을 옆 타일 무시하고 직접 계산)

### Cross-Attention과의 연결
AI 어텐션 공식이 그린 함수 적분 공식과 형태가 동일. **AI가 데이터를 보고 "중성자가 퍼져나가는 규칙"(=그린 함수)을 스스로 학습**하는 것.

---

## 부록 B. 참고 문서

- `2026-03-30 Physical Loss 통합 레퍼런스.md` §3 — L_diffusion 7점 스텐실/CMFD 잔차 정의 (forward 이산화)
- `2026-04-01 L_diffusion 노드 vs 집합체 CMFD 비교 계획.md`
- `archive/참고문헌/2026-04-07 PDE Transformer 자문 참고문헌.md` §2.7 — PINTO 자문 정리
- `piecewise-test/2026-04-01_L_diffusion_endtoend_결과.md` — 검증 결과 (g1 median 2.24%, g2 median 6.44%)
- 외부 논문: PINTO (Computer Physics Communications, 2025), Picard iteration 표준 reactor physics 교재
