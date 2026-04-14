# 05a. Halo Cell에 의한 대칭 조건 구별 가능성 — 수학적 증명

> **작성일**: 2026-04-14
> **목적**: Quarter-core halo cell이 mirror 대칭과 rotational (90°) 대칭을 구별할 수 있는지에 대한 엄밀한 수학적 증명. 논문 서술 및 외부 검증 목적으로 최대한 정밀하게 기술
> **관련 문서**: `05_symmetry_mode.md` (halo cell 설계 결정), `data_transform.py` (quarter_to_fullcore 구현)

---

## 1. 배경 및 문제 정의

### 1.1. 배경

Quarter-core 원자로 대리모델에서 공간 인코더는 quarter (5×5×20) 데이터에 halo cell 1층을 추가한 (6×6×20) 텐서를 입력으로 받는다. Halo cell은 대칭 축 너머의 인접 cell 값을 제공하여 다음 두 가지 역할을 수행한다:

1. **경계 조건 정보 제공**: boundary inner cell의 attention 입력에 이웃 값을 포함
2. **대칭 조건 식별**: 모델이 halo cell의 값 패턴으로부터 대칭 유형 (mirror 또는 rotational) 을 암묵적으로 학습

본 문서는 역할 2에 대해 수학적으로 분석한다.

### 1.2. 문제

> **문제**: 비대칭 (non-octant-symmetric) quarter-core 데이터에 대해, mirror 방식으로 생성한 halo cell과 rotational (90°) 방식으로 생성한 halo cell이 항상 다른가? 동일한 반례가 존재하는가? 존재한다면 그 확률은?

---

## 2. 좌표계 정의

세 가지 좌표계를 사용한다. 모두 반경 방향 (XY 평면) 에 대해서만 정의하며, 축방향 (Z) 은 모든 매핑에서 보존되므로 이하 생략한다.

### 2.1. Full-core 좌표계 (F)

v-SMR 연료 영역은 9×9 격자이다. 0-인덱싱으로 row ∈ {0, ..., 8}, col ∈ {0, ..., 8}. 중심은 (r_c, c_c) = (4, 4).

```
         col →
         0  1  2  3  4  5  6  7  8
row  0 [ .  .  .  .  .  .  .  .  . ]
 ↓   1 [ .  .  .  .  .  .  .  .  . ]
     2 [ .  .  .  .  .  .  .  .  . ]
     3 [ .  .  .  .  .  .  .  .  . ]    Q2 (row<4, col≥4)
     4 [ .  .  .  . (C)  .  .  .  . ] ← center row
     5 [ .  .  .  .  .  .  .  .  . ]
     6 [ .  .  .  .  .  .  .  .  . ]    Q4 (row≥4, col≥4)
     7 [ .  .  .  .  .  .  .  .  . ]
     8 [ .  .  .  .  .  .  .  .  . ]
                    ↑ center col
```

사분면 정의 (대칭 축 위의 셀은 Q4에 포함):
- **Q1** (상우): row < 4, col ≥ 4
- **Q2** (상좌): row < 4, col < 4
- **Q3** (하좌): row ≥ 4, col < 4
- **Q4** (하우): row ≥ 4, col ≥ 4 — **저장 대상**

### 2.2. Quarter 좌표계 (Q)

Q4의 내부 인덱스. 원점은 full-core 중심에 대응한다.
$$qy = \text{row} - 4, \quad qx = \text{col} - 4, \quad (qy, qx) \in \{0, 1, 2, 3, 4\}^2$$

역변환: row = qy + 4, col = qx + 4.

### 2.3. Halo 좌표계 (H)

6×6 격자. Quarter 주위에 1층의 halo cell을 추가한 것.
$$h = qy + 1, \quad w = qx + 1, \quad (h, w) \in \{0, 1, 2, 3, 4, 5\}^2$$

- **Inner cell**: h ∈ {1, ..., 5}, w ∈ {1, ..., 5} — 저장된 Q4 데이터
- **Halo row**: h = 0, w ∈ {1, ..., 5} — Q1 하단 행에 대응
- **Halo col**: h ∈ {1, ..., 5}, w = 0 — Q3 우단 열에 대응
- **Halo corner**: (h, w) = (0, 0) — Q2 우하단 모서리에 대응

좌표 변환 요약 (H → F):
$$\text{row} = h + 3, \quad \text{col} = w + 3$$

---

## 3. 대칭 매핑 정의

### 3.1. Mirror 대칭

Full-core가 수평 축 (row = 4) 과 수직 축 (col = 4) 에 대해 반사 대칭이라 가정한다:

$$V_{\text{mirror}}(\text{row}, \text{col}) = V_{\text{mirror}}(8 - \text{row}, \text{col}) = V_{\text{mirror}}(\text{row}, 8 - \text{col})$$

이로부터 Q4 데이터만으로 전체 full-core를 복원할 수 있다:
- Q1 (상우): V(row, col) = V(8-row, col) — Q4의 상하 반전
- Q3 (하좌): V(row, col) = V(row, 8-col) — Q4의 좌우 반전
- Q2 (상좌): V(row, col) = V(8-row, 8-col) — Q4의 상하좌우 반전

### 3.2. Rotational (90°) 대칭

Full-core가 중심 (4, 4) 에 대해 4회 회전 대칭 (90° 단위) 이라 가정한다. 90° 시계 방향 (CW) 회전:

$$(\text{row}, \text{col}) \xrightarrow{90° \text{CW}} (4 + (\text{col} - 4),\; 4 - (\text{row} - 4)) = (\text{col},\; 8 - \text{row})$$

4회 적용하면 원래 위치로 복원된다. 대칭 조건:

$$V_{\text{rot}}(\text{row}, \text{col}) = V_{\text{rot}}(\text{col}, 8 - \text{row})$$

반복 적용으로 4가지 등가 위치가 존재한다:
$$V(r, c) = V(c, 8-r) = V(8-r, 8-c) = V(8-c, r) \tag{4-fold}$$

이로부터 Q4 데이터만으로 전체 full-core를 복원할 수 있다:
- Q3 (하좌): 90° CW 회전 — V(row, col) = V_Q4(col, 8-row)
- Q2 (상좌): 180° 회전 — V(row, col) = V_Q4(8-row, 8-col)
- Q1 (상우): 90° CCW 회전 — V(row, col) = V_Q4(8-col, row)

---

## 4. Halo Cell 매핑 유도

### 4.1. Halo Row (h = 0)

Halo row의 cell (h=0, w) 는 full-core 위치 (row=3, col=w+3) 에 대응한다 (§2.3).

이 위치는 Q1 영역 (row < 4, col ≥ 4) 에 속한다.

**Mirror 매핑**:

Q1에서의 mirror 복원 규칙 V(row, col) = V(8-row, col) 을 적용:

$$V_{\text{mirror}}(3, w+3) = V(8-3, w+3) = V(5, w+3)$$

Full-core (5, w+3) 은 Q4 내부이다: qy = 5-4 = 1, qx = w+3-4 = w-1. Halo 좌표로 h = 2, w = w.

$$\boxed{\text{halo}_{\text{mirror}}(0, w) = \text{inner}(2, w) \quad \text{for } w = 1, \ldots, 5} \tag{M-row}$$

**Rotational 매핑**:

Q1에서의 rotational 복원 규칙 V(row, col) = V(8-col, row) 을 적용:

$$V_{\text{rot}}(3, w+3) = V(8-(w+3), 3) = V(5-w, 3)$$

(5-w, 3) 은 Q3 영역 (col < 4) 이므로 Q4 내부가 아니다. 4-fold 성질 (Eq. 4-fold) 을 재적용한다:

$$V(5-w, 3) \overset{\text{4-fold}}{=} V(8-3, 5-w) = V(5, 5-w)$$

이것도 Q4 내부가 아닐 수 있으므로 (5-w < 4 일 때), 한 번 더 적용:

$$V(5, 5-w) \overset{\text{4-fold}}{=} V(8-(5-w), 5) = V(3+w, 5)$$

Full-core (3+w, 5) 는 Q4 내부이다: qy = 3+w-4 = w-1, qx = 5-4 = 1. Halo 좌표로 h = w, w = 2.

$$\boxed{\text{halo}_{\text{rot}}(0, w) = \text{inner}(w, 2) \quad \text{for } w = 1, \ldots, 5} \tag{R-row}$$

> **검증 (직접 경로)**: 90° CCW 회전은 Q4 → Q1을 매핑하므로, Q1 cell = rot90_CCW(Q4 cell) 이다. Q1 하단 행 (row=3, col=4+j) 에 대한 Q4 원본을 구하면: rot90_CW를 역으로 적용하여 (row=3, col=4+j) ← rot90_CCW(4+j, 5) → Q4에서 (qy=j, qx=1) = inner(j+1, 2). w = j+1 이므로 inner(w, 2). ✓

### 4.2. Halo Col (w = 0)

Halo col의 cell (h, w=0) 는 full-core 위치 (row=h+3, col=3) 에 대응한다.

이 위치는 Q3 영역 (row ≥ 4, col < 4) 에 속한다 (h ≥ 1 일 때 row ≥ 4).

**Mirror 매핑**:

Q3에서의 mirror 복원 규칙 V(row, col) = V(row, 8-col) 을 적용:

$$V_{\text{mirror}}(h+3, 3) = V(h+3, 8-3) = V(h+3, 5)$$

Full-core (h+3, 5) 는 Q4 내부이다: qy = h+3-4 = h-1, qx = 5-4 = 1. Halo 좌표로 h = h, w = 2.

$$\boxed{\text{halo}_{\text{mirror}}(h, 0) = \text{inner}(h, 2) \quad \text{for } h = 1, \ldots, 5} \tag{M-col}$$

**Rotational 매핑**:

Q3에서의 rotational 복원은 90° CW: Q4 → Q3. 따라서 Q3 cell = rot90_CW(Q4 cell).

Q3 cell (h+3, 3) 에 대응하는 Q4 원본을 구한다. 4-fold 성질 Eq. (4-fold) 적용:

$$V(h+3, 3) \overset{\text{4-fold}}{=} V(8-3, h+3) = V(5, h+3)$$

Full-core (5, h+3) 는 Q4 내부이다: qy = 5-4 = 1, qx = h+3-4 = h-1. Halo 좌표로 h = 2, w = h.

$$\boxed{\text{halo}_{\text{rot}}(h, 0) = \text{inner}(2, h) \quad \text{for } h = 1, \ldots, 5} \tag{R-col}$$

### 4.3. Halo Corner (h = 0, w = 0)

Full-core 위치 (3, 3) 은 Q2 영역이다.

**Mirror**: V(3, 3) = V(5, 5) → inner(2, 2).
**Rotational**: V(3, 3) = V(5, 5) (4-fold: V(3,3) = V(3,5) = V(5,5) = V(5,3), 그 중 (5,5)가 Q4) → inner(2, 2).

$$\boxed{\text{halo}_{\text{mirror}}(0, 0) = \text{halo}_{\text{rot}}(0, 0) = \text{inner}(2, 2)} \tag{Corner}$$

Corner는 **항상 동일**하다.

### 4.4. 매핑 요약

| Cell | Mirror | Rotational (90°) |
|---|---|---|
| halo(0, w), w=1..5 | inner(**2**, w) | inner(**w**, 2) |
| halo(h, 0), h=1..5 | inner(h, **2**) | inner(**2**, h) |
| halo(0, 0) | inner(2, 2) | inner(2, 2) |

> **핵심 관찰**: Mirror는 inner의 **행 h=2** (= qy=1 행) 와 **열 w=2** (= qx=1 열) 를 직접 복사한다. Rotational은 이 행과 열을 **교환** (transpose) 한다.

---

## 5. 정리 (Theorem) 및 증명

### 5.1. 정리

**Theorem 1 (Halo 구별 가능 조건)**:

quarter-core inner cell 데이터를 $A \in \mathbb{R}^{5 \times 5}$ 라 하자 (인덱스 $A_{h,w}$ for $h, w \in \{1, \ldots, 5\}$).

Mirror halo와 rotational halo가 동일한 것은 다음 조건과 필요충분이다:

$$\text{halo}_{\text{mirror}} = \text{halo}_{\text{rot}} \iff A_{2, w} = A_{w, 2} \quad \forall\, w \in \{1, \ldots, 5\}$$

즉, inner 행렬의 **2행과 2열이 전치 관계**일 때에만 (= qy=1 행과 qx=1 열의 원소가 대응 위치에서 동일할 때에만) 두 halo가 일치한다.

### 5.2. 증명

**(⟹ 방향: halo 동일 → 2행 = 2열)**

Halo가 동일하면, 특히 halo row가 동일해야 한다:
$$\text{halo}_{\text{mirror}}(0, w) = \text{halo}_{\text{rot}}(0, w) \quad \forall\, w \in \{1, \ldots, 5\}$$

Eq. (M-row) 과 Eq. (R-row) 으로부터:
$$A_{2, w} = A_{w, 2} \quad \forall\, w \in \{1, \ldots, 5\} \qquad \blacksquare$$

**(⟸ 방향: 2행 = 2열 → halo 동일)**

$A_{2, w} = A_{w, 2}$ for all $w$ 를 가정한다.

- **Halo row**: Eq. (M-row) 에서 $\text{halo}_{\text{mirror}}(0, w) = A_{2, w}$. Eq. (R-row) 에서 $\text{halo}_{\text{rot}}(0, w) = A_{w, 2} = A_{2, w}$. 따라서 동일. ✓
- **Halo col**: Eq. (M-col) 에서 $\text{halo}_{\text{mirror}}(h, 0) = A_{h, 2}$. Eq. (R-col) 에서 $\text{halo}_{\text{rot}}(h, 0) = A_{2, h}$. 가정에 의해 $A_{h, 2} = A_{2, h}$. 따라서 동일. ✓
- **Corner**: Eq. (Corner) 에 의해 항상 동일. ✓

모든 halo cell이 동일하므로 halo가 동일하다. $\qquad \blacksquare$

---

## 6. Octant 대칭과의 관계

### 6.1. 정의

**Octant 대칭 (대각선 대칭)**: $A_{i,j} = A_{j,i}$ for ALL $i, j \in \{1, \ldots, 5\}$. 즉 inner 행렬이 대칭 행렬 ($A = A^T$).

### 6.2. 관계

Octant 대칭은 Theorem 1의 조건보다 **강한 조건**이다:

$$A = A^T \implies A_{2,w} = A_{w,2}\; \forall w \quad \text{(역은 성립하지 않음)}$$

역이 성립하지 않는 이유: $A_{2,w} = A_{w,2}$ 조건은 행렬의 **2행과 2열만** 제약한다. 나머지 원소 (예: $A_{1,3}$ 과 $A_{3,1}$) 는 자유롭다.

따라서 다음 포함 관계가 성립한다:

$$\{\text{Octant symmetric } A\} \subsetneq \{A_{2,w} = A_{w,2}\; \forall w\} \subsetneq \{\text{모든 } 5 \times 5 \text{ 행렬}\}$$

---

## 7. 반례 (Counterexample) 구성

### 7.1. 구성

$A_{2,w} = A_{w,2}$ for all $w$ 이지만 $A \neq A^T$ 인 행렬:

$$A = \begin{pmatrix} 1 & 3 & 5 & 7 & 9 \\ 3 & 4 & 6 & 8 & 2 \\ 5 & 6 & 0 & 1 & - \\ \mathbf{99} & 8 & 1 & - & - \\ 9 & 2 & - & - & - \end{pmatrix}$$

(빈 칸(-)은 반사체/void 영역이며 fuel mask로 처리.)

**검증**:
- 2행: $(3, 4, 6, 8, 2)$
- 2열: $(3, 4, 6, 8, 2)$ — 동일 ✓
- $A_{4,1} = 99 \neq A_{1,4} = 7$ — 대각선 비대칭 ✓

이 행렬에 대해 mirror halo = rotational halo 이지만, octant 대칭은 아니다.

### 7.2. 이 반례가 실제 원자로 데이터에서 발생하는가?

**발생 확률: 측도론적으로 0 (measure zero)**

**논증**: 비-octant-대칭 quarter-core 데이터에서 물리량 (flux, temperature, xenon density 등) 은 연속 확률 변수이다. 각 cell의 값은 운전 조건 (rod 위치, 열수력 피드백, 연소도 등) 에 의해 결정되는 연속 함수이다.

Theorem 1의 동일 조건 $A_{2,w} = A_{w,2}$ 는 **4개의 등식** (w=1,3,4,5; w=2는 자명) 을 요구한다:

$$A_{2,1} = A_{1,2}, \quad A_{2,3} = A_{3,2}, \quad A_{2,4} = A_{4,2}, \quad A_{2,5} = A_{5,2}$$

비-octant-대칭 데이터에서 $A_{2,w}$와 $A_{w,2}$는 서로 다른 물리적 위치의 독립적인 연속 변수이다. 두 연속 확률 변수가 정확히 같을 확률은 연속 측도에서 0이다:

$$P(X = Y) = 0 \quad \text{for continuous } X, Y \text{ with } X \neq Y \text{ a.s.}$$

4개의 등식이 동시에 성립할 확률은:

$$P\bigl(A_{2,1} = A_{1,2} \;\land\; A_{2,3} = A_{3,2} \;\land\; A_{2,4} = A_{4,2} \;\land\; A_{2,5} = A_{5,2}\bigr) = 0$$

따라서 비-octant-대칭 연속 데이터에서 mirror halo = rotational halo 가 발생할 확률은 **정확히 0**이다.

### 7.3. 수치 정밀도 고려

실제 데이터는 float32 (유효숫자 ~7.2자리) 로 저장된다. 이론적 확률 0과 달리, float32 양자화로 인해 두 값이 "우연히" 같은 비트 패턴을 가질 확률이 미미하게 존재한다. 그러나:

- v-SMR 물리량의 값 범위가 넓고 (flux: ~10¹² ~ 10¹⁴, temperature: 300~1500 K) float32 표현 공간에서 두 독립 값이 같은 비트 패턴을 가질 확률은 ~ $2^{-23}$ ≈ $10^{-7}$ per cell pair
- 4개 등식이 동시에 성립할 확률: ~ $10^{-28}$
- 575 timestep × 100 LP × 20 scenario ≈ $10^6$ 샘플에서 1번이라도 발생할 확률: ~ $10^{-22}$

**사실상 발생하지 않는다.**

### 7.4. Halo가 제약하지 못하는 영역 (한계)

Theorem 1에 의해 halo의 구별 가능 여부는 **inner 2행과 2열** (= qy=1 행, qx=1 열) 의 관계에만 의존한다. 이는 halo cell이 inner 행렬의 **1층 이웃만** 노출하기 때문이다. inner 행렬에서 2행/2열에 포함되지 않는 cell 쌍의 대칭 관계는 halo로 알 수 없다.

**제약/미제약 쌍의 시각적 분류**:

inner 행렬 $A$ 에서 대칭 쌍 $(A_{i,j}, A_{j,i})$ 를 분류한다 (연료 cell만 표시, --는 반사체/void):

```
     w=1  w=2  w=3  w=4  w=5
h=1:  -    ●    ○    ○    ○       ● = halo가 이 쌍의 대칭 관계를 노출
h=2:  ●    -    ●    ●    ●       ○ = halo가 노출하지 못함
h=3:  ○    ●    -    ○    --      - = 대각 원소 (자명하게 대칭)
h=4:  ○    ●    ○    --   --     -- = 반사체/void
h=5:  ○    ●    --   --   --
```

| 분류 | 제약 조건 | 쌍 목록 | 개수 |
|---|---|---|:---:|
| **Halo 제약** (●) | i=2 또는 j=2 | (1,2)↔(2,1), (2,3)↔(3,2), (2,4)↔(4,2), (2,5)↔(5,2) | 4 |
| **Halo 미제약** (○, 연료만) | i≠2 AND j≠2, 연료 cell | (1,3)↔(3,1), (1,4)↔(4,1), (1,5)↔(5,1), (3,4)↔(4,3) | **4** |
| **Halo 미제약** (○, 반사체 포함) | i≠2 AND j≠2, 반사체 | (3,5)↔(5,3), (4,5)↔(5,4) | 2 |
| **자명** (-) | i=j (대각 원소) | (1,1), (2,2), (3,3), (4,4), (5,5) | 5 |

**미제약 연료 cell 쌍 4개의 물리적 위치** (quarter 좌표):

```
(qy=0, qx=2) ↔ (qy=2, qx=0):  중심 행/열에서 2칸 떨어진 대각 쌍
(qy=0, qx=3) ↔ (qy=3, qx=0):  중심에서 3칸 떨어진 대각 쌍
(qy=0, qx=4) ↔ (qy=4, qx=0):  Q4 모서리의 대각 쌍
(qy=2, qx=3) ↔ (qy=3, qx=2):  내부 대각 쌍
```

이 4쌍은 모두 **qy=1/qx=1 축에서 떨어진 영역**에 위치하며, halo 1층으로는 접근 불가능하다.

**한계의 의미**: 반례 (§7.1) 가 확률 0으로 발생하지 않더라도, halo가 "확인하는" 대칭 정보의 범위가 inner 행렬 전체가 아닌 **2행/2열에 한정**된다는 구조적 한계가 존재한다. 이는 halo cell이 1층만 추가되는 것에서 비롯되는 본질적 제약이다.

**완화 방안**: §8.5에서 채널 추가 없이 대칭 유형을 명시적으로 전달하는 방안을 분석한다.

---

## 8. 실용적 함의

### 8.1. 비-octant-대칭 데이터에서의 구별 가능성

**Theorem 1** 과 **§7** 의 결합으로부터:

> 비-octant-대칭 연속 데이터에서 mirror halo ≠ rotational halo 가 **확률 1로 (almost surely)** 성립한다.

따라서 신경망 모델은 halo cell의 값 패턴으로부터 대칭 유형을 **암묵적으로 학습** 할 수 있다.

구체적으로:
- Mirror: halo_row = inner 2행, halo_col = inner 2열
- Rotational: halo_row = inner 2열 (전치), halo_col = inner 2행 (전치)
- 두 패턴은 행과 열이 **교환 (transpose)** 되는 관계
- 비대칭 데이터에서 이 교환은 관측 가능한 차이를 만듦

### 8.2. Octant 대칭 데이터에서의 구별 불가능성

Octant 대칭 데이터 ($A = A^T$) 에서는 mirror halo = rotational halo 이다. 그러나:

- Octant 대칭이면 **물리적으로도** mirror와 rotational이 동등한 full-core를 생성한다
- 따라서 모델이 구별할 **필요가 없다** — 어느 쪽을 가정해도 동일한 예측이 올바르다

### 8.3. v-SMR 데이터에서의 적용

| 데이터 유형 | Octant 대칭? | Mirror = Rotational? | 구별 필요? |
|---|:---:|:---:|:---:|
| **xs_fuel** (BOC, 연료 물성) | ✓ (LP가 90° 회전 대칭) | ✓ | ✗ |
| **CRS state** (대칭 rod 위치) | ✓ (대칭 운전 조건) | ✓ | ✗ |
| **Branch state** (비대칭 rod offset) | **△** (bank별 비대칭 시) | **△** | **△** |
| **Transient state** (Xe 진동 등) | **✗** (비대칭 가능) | **✗** | **✓** |

현재 v-SMR 데이터셋 (100 LP, 대칭 rod 운전) 에서는 대부분 octant 대칭이므로 실용적 차이가 미미하다. 비대칭 운전 조건 (예: bank별 독립 rod 이동, asymmetric Xe oscillation) 을 포함하는 확장 데이터에서 halo cell의 대칭 구별 능력이 본격적으로 활용된다.

### 8.3a. Fullcore 대칭 유형 정보 보존 체계 (Critical)

> **심각도: 높음** — 대칭 유형 정보가 데이터 파이프라인 전 단계에서 정확히 보존되지 않으면, halo_expand, L_diff_rel, Conditional LAPE 모두 오작동할 수 있다.

#### 문제: `lp_geometry` ≠ `symmetry_type`

현재 HDF5 metadata에 `quarter_crop_mode: 'mirror'` 가 기록되어 있으나 (MANUAL.md §5.3), 이는 **"어떤 방식으로 crop했는가"** 를 의미할 뿐이다. Quarter crop 자체는 단순 슬라이싱 (`fullcore[center:, center:]`) 으로 대칭 유형과 무관하게 동일한 결과를 생성한다.

모델이 실제로 필요로 하는 정보는 **"원래 fullcore가 어떤 대칭이었는가"** 이다. 이 정보가 없으면:
- `halo_expand()` 가 잘못된 모드 (mirror vs rotational) 로 호출될 수 있음
- `Conditional LAPE` 가 잘못된 테이블을 선택할 수 있음
- `L_diff_rel` 의 stencil ghost neighbor가 비물리적 값을 사용할 수 있음

두 정보를 명확히 분리한다:

| 필드 | 의미 | 가능한 값 | 결정 시점 |
|---|---|---|---|
| `lp_geometry` | quarter core 추출 방식 | `'quarter'`, `'full'`, `'none'` | 전처리 시 |
| `symmetry_type` | LP 연료 배치의 대칭 유형 | `'quarter_rotational'`, `'quarter_mirror'`, `'octant'`, `'none'` | **LP 생성 시** |

#### 현재 100LP 데이터셋의 정확한 대칭 유형

> **MANUAL.md line 416 의 문구는 부정확하다**: "LP 생성 시점의 `_compute_orbit()`은 rotational을 사용하지만..." 은 **현재 코드 상태를 기술**한 것이며, **실제 100LP 생산 시점의 상태가 아니다**.

타임라인 (git 커밋 검증):
```
2026-03-26  LP 데이터 생산 (workspace_lf_20260326_40LP + 60LP)
            → _compute_orbit()이 mirror 로직 사용
            → symmetry_type = 'quarter_mirror'

2026-03-30  commit 14a07f9: _compute_orbit() 를 rotational 90° 로 변경
            → 이후 생산되는 LP는 symmetry_type = 'quarter_rotational'

2026-04-02  전처리 실행 → lf_dataset_100lp_mirrored_2026-04-02.h5
            → metadata에 crop_mode='mirror' 기록 (의도: mirror 대칭 LP임을 표시)
```

**결론**: 현재 100LP 데이터셋의 `symmetry_type = 'quarter_mirror'` (rotational이 아님).

#### 현재 파이프라인의 정보 흐름 — 문제점

```
[LP 생성] config.yaml: symmetry='quarter', _compute_orbit()=mirror (당시)
    │
    │ ★ symmetry_type이 LP 산출물에 기록되지 않음 ★
    │
    ▼
[MASTER 시뮬레이션] fullcore 직접 시뮬레이션
    │
    ▼
[전처리] config_preproc.yaml: crop_mode='mirror'
    │
    ▼
[HDF5] metadata: quarter_crop_mode='mirror' ← crop 방식과 대칭 유형이 혼재!
    │
    ▼
[모델] ★ symmetry_type을 명시적으로 읽을 수 없음 ★
```

#### 수정된 정보 보존 체계

```
[Stage 1: LP 생성]
  config.yaml:
    symmetry: 'quarter'              # orbit 함수가 사용하는 대칭 모드
  
  → 수정: 생산 config의 symmetry 값 + orbit 함수 유형을
    dataset_builder가 HDF5 생성 시 metadata에 기록

[Stage 2: 전처리]
  config_preproc.yaml (수정):
    lp_geometry: 'quarter'                        # 기존 crop_mode rename (crop 방식만)
    symmetry_type: 'quarter_mirror'    # 현재 100LP 데이터 기준
    # (향후 rotational 생산 LP 전처리 시 'quarter_rotational'로 변경)

  dataset_builder.py _write_metadata() (수정):
    meta.attrs['lp_geometry'] = 'quarter'
    meta.attrs['symmetry_type'] = config['symmetry_type']

[Stage 3: 모델 학습]
  dataloader:
    sym_type = h5f['metadata'].attrs['symmetry_type']
    → halo_expand(quarter, sym=sym_type)
    → Conditional LAPE(x, sym_type)

  configs/model.yaml:
    data.symmetry: 자동 (HDF5 metadata에서 읽음)
    # 수동 설정 시 HDF5와 불일치하면 에러

[Stage 4: 추론]
  동일 sym_type 사용 보장 (HDF5 metadata → 모델)
```

#### 현재 100LP 데이터셋 즉시 조치

```python
import h5py
with h5py.File('lf_dataset_100lp_mirrored_2026-04-02.h5', 'a') as f:
    f['metadata'].attrs['symmetry_type'] = 'quarter_mirror'  # ★ mirror!
    f['metadata'].attrs['lp_geometry'] = 'quarter'
```

#### MANUAL.md 정정 필요 사항

MANUAL.md line 416 의 현재 문구:
> "LP 생성 시점의 `_compute_orbit()`은 **rotational**을 사용하지만..."

정정:
> "**현재 코드**의 `_compute_orbit()`은 rotational을 사용하지만, **100LP 생산 시점 (2026-03-26) 에는 mirror 로직이었다**. 따라서 현재 100LP 데이터셋의 `symmetry_type = 'quarter_mirror'`."

#### 구현 대상 코드 파일 (별도 구현 단계에서 수행)

| 파일 | 변경 내용 | 저장소 |
|---|---|---|
| `config_preproc.yaml` | `crop_mode` → `lp_geometry` rename + `symmetry_type` 추가 | v-smr_load_following |
| `dataset_builder.py` `_write_metadata()` | `symmetry_type` attrs 추가 | v-smr_load_following |
| `MANUAL.md` line 416 | "rotational" → "mirror (100LP 기준)" 정정 | v-smr_load_following |
| `halo_expand.py` (신규) | HDF5 metadata에서 sym_type 읽어 mode 결정 | spatio-temporal-reactor-ai-surrogate |
| `Conditional LAPE` (신규) | HDF5 metadata에서 sym_type 읽어 테이블 선택 | spatio-temporal-reactor-ai-surrogate |
| `configs/model.yaml` (신규) | HDF5 metadata와 일치 검증 | spatio-temporal-reactor-ai-surrogate |

### 8.4. 모델 설계 시사점

1. **halo_expand() 함수는 올바른 `sym` 옵션으로 호출되어야 한다**: 잘못된 대칭 유형으로 halo를 생성하면, 모델이 부정확한 boundary 정보를 받는다. HDF5 metadata에 symmetry_type을 기록하고 dataloader에서 검증해야 한다.

2. **모델은 대칭 유형을 명시적 입력으로 받지 않는다**: 대칭 정보는 halo cell의 값 패턴에 **암묵적으로 인코딩**된다. 이는 encoder의 attention 레이어가 halo row ↔ halo col 간의 상관 패턴을 학습함으로써 달성된다.

3. **L_diff_rel에서의 영향**: L_diff_rel의 CMFD stencil은 halo cell을 ghost neighbor로 사용한다. 잘못된 대칭으로 halo를 생성하면 stencil neighbor가 부정확해지나, **양변에 동일한 halo를 사용하므로** systematic bias가 ε_total(t) 에 흡수되어 cancel된다 (`Physical Loss 통합 레퍼런스 §3.6` 참조).

### 8.5. 대칭 유형 명시 전달 방안

§7.4에서 식별한 한계 (halo가 2행/2열만 제약, 미제약 연료 cell 쌍 4개 존재) 를 완화하기 위해, 대칭 유형을 모델에 명시적으로 전달하는 5가지 방안을 분석한다.

> **본 프로젝트 아키텍처 제약 조건** (평가 기준):
> - 인코더: **Pre-LN + LayerNorm** 사용. BatchNorm은 명시적으로 거부됨 (`04_normalization_omitted_options.md` §1.2)
> - 인코더: **p_load 미주입**. 조건 변조는 디코더 AdaLN-Zero에서만 (`04_normalization_omitted_options.md` §1.4)
> - 인코더: **FiLM 미사용**. rod_map은 21ch 입력에 채널 concat (`02_cell_embedder.md`)
> - CellEmbedder: Conv3D(1,1,1), 21→128, spatial mixing 없음
> - Mamba SSM: cell-wise 공유 가중치, S6 selective scan (B, C, Δ 입력 의존)
> - 디코더: AdaLN-Zero (p_load 스칼라 조건 변조), 상세 미설계

#### 방안 1: sym_indicator 입력 채널 추가 (Baseline)

입력 21ch에 상수 채널 1개 (mirror=0, rotational=1) 추가하여 22ch으로 확장.

```
기존 입력: (B, 20, 6, 6, 21)     state(10) + xs_fuel(10) + rod_map(1)
변경 후:   (B, 20, 6, 6, 22)     + sym_indicator(1)
```

**한계점**: PDE 대리 모델 문헌에서 파라미터 (레이놀즈 수, 확산계수 등) 를 추가 입력 채널로 주입하는 방식은 표준적이나 (FNO, Li et al. 2021), **공간적으로 균일한 상수 채널은 Conv+Attention 이후 단계에서 효과가 희석**될 수 있다. 특히:
- CellEmbedder Conv3D(1,1,1) 은 채널 간 선형 결합만 수행 → 상수 채널은 global bias로 작용
- Attention 이후 여러 layer를 거치며 상수 신호가 residual stream에서 점차 약화
- 2-class 이진 정보에 128차원의 채널 1개를 할당하는 것은 정보 효율이 낮음

> **참고**: Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. arXiv:2010.08895.

**판정**: 가장 단순하나 효과 희석 우려. 다른 방안과 비교 시 차선.

#### 방안 A/B 통합: 조건부 Affine 변조 (Conditional LN / FiLM)

> **FiLM과 AdaLN-Zero의 관계**: FiLM (Perez et al. 2018) 의 핵심 연산은 `output = γ(z)·input + β(z)` 이며, AdaLN-Zero (Peebles & Xie 2023) 는 이를 LayerNorm 출력에 적용한 것이다: `output = γ(z)·LN(x) + β(z)`. 즉 **AdaLN = LayerNorm + FiLM**. 두 방안은 본질적으로 동일한 원리 (조건부 affine 변조) 이며, 적용 위치만 다르다.

본 프로젝트는 BatchNorm을 사용하지 않으므로, Conditional LayerNorm (= AdaLN-Zero의 2-class 특수 사례) 형태로 적용한다. `sym_type`은 2-class discrete 변수로 DiT의 class label conditioning과 구조적으로 동일.

```python
class SymmetryCondLN(tf.keras.layers.Layer):
    """Pre-LN의 LayerNorm을 sym_type 조건부로 교체. AdaLN = LN + FiLM."""
    def __init__(self, d_model):
        super().__init__()
        self.gamma_table = tf.keras.layers.Embedding(2, d_model)  # (2, D)
        self.beta_table  = tf.keras.layers.Embedding(2, d_model)  # (2, D)

    def call(self, x, sym_type):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + 1e-5)
        gamma = tf.gather(self.gamma_table.embeddings, sym_type)[:, None, :]
        beta = tf.gather(self.beta_table.embeddings, sym_type)[:, None, :]
        return gamma * x_norm + beta
```

**적용 위치 옵션**:

| 위치 | 방식 | 특징 |
|---|---|---|
| **인코더 각 Pre-LN (6곳)** | LN을 SymmetryCondLN으로 교체 | 매 layer에서 재적용, 효과 강함 |
| **디코더 AdaLN-Zero 입력에 합산** | `p_load_emb + sym_emb → MLP → γ, β` | 기존 AdaLN-Zero 경로 재활용, 인코더 변경 없음 |
| **인코더 FiLM (신설)** | SSM Phase 1 Branch rod conditioning 경로에 sym_type 합산 | Branch FiLM 도입 시 함께 적용 |

**적합성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 입력 채널 보존 (21ch) | ✓ | 입력 텐서 구조 변경 없음 |
| Pre-LN 구조 호환 | **△** | 인코더 LN 교체 시 **"조건 변조 없음" 설계 원칙에 위배**. 디코더 경로 재활용 시 인코더 무변경 |
| 파라미터 추가 | 인코더 LN 교체: 2×D×2×6 ≈ **3K** (0.5%). 디코더 합산: MLP 입력 +4 dim ≈ **무시** | 미미 |
| 물리적 해석 | 정규화 통계의 조건부 모드 전환 | |

> **참고**:
> - Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers.* ICCV 2023. arXiv:2212.09748.
> - Hang, T., et al. (2026). *DP-aware AdaLN-Zero: Taming Conditioning-Induced Heavy-Tailed Noise in Diffusion Transformers.* arXiv:2602.22610.
> - Perez, E., et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. arXiv:1709.07871.

**판정**: 인코더에 적용 시 설계 원칙 위반 여부 의사 결정 필요. 디코더 AdaLN-Zero 합산 경로는 비용 0으로 적용 가능.

#### 방안 C: Conditional LAPE — 대칭 유형별 별도 위치 임베딩

LAPE (Learned Absolute Position Encoding) 가 이미 각 위치의 고유 정체성을 학습한다. sym_type에 따라 **다른 LAPE 테이블** 을 선택하면, 위치 임베딩 자체에 대칭 정보가 내재된다. 이는 BERT의 segment type embedding (문장 A/B 구분) 이 위치 표현을 조건부로 분기하는 것과 구조적으로 유사.

```python
class ConditionalLAPE3D(tf.keras.layers.Layer):
    def __init__(self, z_dim, qh_dim, qw_dim, d_model):
        super().__init__()
        self.lape_mirror = self.add_weight(
            shape=(z_dim, qh_dim, qw_dim, d_model),
            initializer='random_normal', name='lape_mirror')
        self.lape_rot = self.add_weight(
            shape=(z_dim, qh_dim, qw_dim, d_model),
            initializer='random_normal', name='lape_rot')

    def call(self, x, sym_type):
        lape = tf.where(
            sym_type[:, None, None, None, None] == 0,
            self.lape_mirror[None, ...],
            self.lape_rot[None, ...]
        )
        return x + lape
```

**적합성 평가**:

| 기준 | 평가 | 비고 |
|---|:---:|---|
| 입력 채널 보존 (21ch) | ✓ | 입력 텐서 구조 변경 없음 |
| Pre-LN 구조 호환 | **✓** | LAPE는 Pre-LN 이전, LayerNorm 자체를 건드리지 않음 |
| 인코더 설계 원칙 호환 | **✓** | LAPE 확장이지 조건 변조가 아님 |
| 파라미터 추가 | LAPE 테이블 1세트 추가 = **92,160개** (LAPE 현재의 2배) | |
| 물리적 해석 | **가장 자연스러움** — "mirror일 때의 위치 정체성"과 "rotational일 때의 위치 정체성"이 다름 |  |

**학계 선례**:

대칭 유형에 따라 위치 인코딩 (또는 잠재 표현) 이 달라져야 한다는 관점은 다수의 선행 연구에서 지지된다:

- **Symmetric Embedding Network (SEN)** (Park et al., ICML 2022): 대칭 변환군 (rotation, reflection 등) 에 따라 다른 임베딩 공간을 학습. 같은 입력이라도 적용되는 대칭군에 따라 다른 표현을 갖는다는 개념이 Conditional LAPE와 동일한 철학을 공유.
- **Group Equivariant ViT (GE-ViT)** (Xu et al., 2023): ViT의 위치 인코딩이 데이터 내재 등변성 (equivariance) 학습을 방해한다는 문제를 지적하고, 군 등변 위치 인코딩으로 대체. 대칭군 선택에 따라 위치 인코딩이 달라져야 한다는 근거를 제공.
- **Platonic Transformer** (Niazoys et al., 2025): RoPE를 복수의 reference frame (이산 대칭군 $\mathcal{G} \subset O(n)$의 원소) 에 대해 병렬 적용하여 Euclidean 등변성 달성. 대칭 유형에 따라 위치 인코딩이 다른 reference frame을 선택한다는 점에서 Conditional LAPE와 구조적으로 유사.

> **참고**: 
> - Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019. arXiv:1810.04805.
> - Park, J. Y., et al. (2022). *Learning Symmetric Embeddings for Equivariant World Models.* ICML 2022. arXiv:2204.11371.
> - Xu, R., et al. (2023). *Group Equivariant Vision Transformer.* arXiv:2306.06722.
> - Niazoys, M., et al. (2025). *Platonic Transformers: A Solid Choice For Equivariance.* arXiv:2510.03511.

**판정**: 본 프로젝트 아키텍처에 가장 적합. 설계 원칙 무위반 + 물리적 해석 자연스러움 + 학계 선례 풍부.

#### 방안 비교 요약

| | 방안 1 (입력 채널) | 방안 A/B (조건부 Affine) | **방안 C (Cond. LAPE)** |
|---|---|---|---|
| **메커니즘** | 상수 채널 concat | γ(sym)·LN(x) + β(sym) | LAPE 테이블 2개 중 선택 |
| **입력 채널 보존** | ✗ (21→22) | ✓ | ✓ |
| **인코더 설계 원칙** | ✓ | △ (Pre-LN 조건화 시) | **✓** (LAPE 확장) |
| **효과 희석** | △ (상수 채널, layer 거치며 약화) | ✓ (매 LN 위치에서 재적용) | ✓ (residual stream 전달) |
| **파라미터** | 128 (0.02%) | 3K (0.5%) | 92K (15%) |
| **물리적 해석** | bias-like | 정규화 모드 전환 (AdaLN = LN+FiLM) | **위치 정체성의 대칭 의존** |
| **학계 선례** | FNO (Li 2021) | DiT (Peebles 2023), FiLM (Perez 2018) | **SEN, GE-ViT, Platonic** |
| **권장도** | ★★ | ★★★ | **★★★★** |

> **Note**: 방안 A/B 통합 근거 — FiLM (Perez 2018) 과 AdaLN-Zero (Peebles 2023) 는 본질적으로 동일한 원리 (조건부 affine 변조: γ·x + β) 이며, AdaLN = LayerNorm + FiLM 이다. 두 방안의 차이는 메커니즘이 아닌 적용 위치 (LN 내부 vs 임의 feature map) 뿐이므로 통합하여 비교한다.

---

## 9. 결론

1. **Theorem 1**: Mirror halo = Rotational halo ⟺ inner 2행 = inner 2열 (전치)

2. **Octant 대칭 → 조건 성립** (구별 불가하나, 구별 불필요)

3. **비-octant-대칭에서 조건 성립 확률 = 0** (연속 측도): 반례가 수학적으로 존재하나, 연속 실수 데이터에서 발생하지 않음

4. **따라서**: halo cell은 비대칭 데이터에서 **확률 1로** mirror와 rotational을 구별할 수 있으며, 대칭 데이터에서는 구별이 불필요하다. Halo cell의 **대칭 조건 식별** 기능은 수학적으로 보장된다.

---

## 10. 참고

### 관련 문서
- `05_symmetry_mode.md` — halo cell 설계 결정 (§1 파이프라인, §3 학습 신호 흐름, §6 symmetry 정보 흐름)
- `data_transform.py:134-200` — `quarter_to_fullcore(quarter, mode='mirror'|'rotational')` 구현
- `2026-04-05_quarter_symmetry_analysis.md` — v-SMR LP의 대칭 유형 분석 + round-trip 검증
- `piecewise-test/2026-04-01_L_diffusion_endtoend_test.py:187-195` — mirror 분기 inline lookup 코드

### 기호 정리
| 기호 | 의미 |
|---|---|
| $V(r, c)$ | Full-core 위치 $(r, c)$의 물리량 값 (스칼라 또는 벡터) |
| $A_{h,w}$ | Inner cell (h, w) 의 값 = $V(h+3, w+3)$ |
| $\text{halo}(h, w)$ | Halo 좌표 $(h, w)$에 배치된 값 |
| Q4 | Quarter core 저장 영역 (row ≥ 4, col ≥ 4) |
| qy, qx | Quarter 좌표 (0-indexed) |
| h, w | 6×6 halo 격자 좌표 (h=qy+1, w=qx+1) |
