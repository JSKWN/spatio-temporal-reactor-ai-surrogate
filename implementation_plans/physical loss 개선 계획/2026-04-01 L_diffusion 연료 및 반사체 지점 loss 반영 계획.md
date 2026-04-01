# L_diffusion 연료 및 반사체 인접면 loss 반영 계획

**작성일**: 2026-04-01
**참조**: Physical Loss 통합 레퍼런스 §3, Albedo 캘리브레이션 종합 보고서 (확정)

---

## 1. 개요

L_diffusion은 모델 예측 flux가 확산방정식의 **공간 밸런스**를 만족하는지 검사:
```
R_g = [6면 누설] + [제거 × V] - [생성 × V] = 0 (이상적)
```

각 연료 노드의 6면은 이웃에 따라 **누설 계산 방식이 다름**:
- 이웃이 연료 → CMFD (유한차분)
- 이웃이 반사체 → **Albedo BC** (Marshak α 또는 행렬 C)
- 이웃이 대칭면 → Mirror CMFD (REFLECT: ghost=mirror neighbor)

---

## 2. 면 유형 분류 (boundary_mask)

### 면 유형 정의

각 연료 노드 (z, y, x)의 6면(N, S, E, W, T, B)을 다음 유형으로 분류:

| 유형 | 코드 | 이웃 | 누설 계산 |
|------|:---:|------|---------|
| **내부** | 0 | 연료 | CMFD: D̃×(φ̂_nb - φ̂_center)/h × A |
| **ortho 반사체** | 1 | R1, R2 | Marshak: α_ortho×D/(αh/2+D)×φ̂ × A |
| **diag 반사체** | 2 | R3~R6 | Marshak: α_diag×D/(αh/2+D)×φ̂ × A |
| **bottom 반사체** | 3 | K=1 | 행렬 C: C_bottom × [φ̂_g1, φ̂_g2] × A/V |
| **top 반사체** | 4 | K=22 | 행렬 C: C_top × [φ̂_g1, φ̂_g2] × A/V |
| **대칭면** | 5 | quarter 경계 | Mirror CMFD (REFLECT): ghost=mirror neighbor |
| **빈공간** | 6 | o (존재하지 않음) | 해당 면 L_diffusion 미적용 |

### Quarter-core (20, 5, 5)에서의 경계면 위치 (R4 검증 완료)

**XY 맵 (Z 중앙 평면 기준)**:
```
         x=0   x=1   x=2   x=3   x=4
y=0  [   ■     ■     ■     ■    E(o) ]   A2  A3  A2  A2  B3
y=1  [   ■     ■     ■     ■   sE(od)]   A3  A2  A3  A2  A3
y=2  [   ■     ■     ■    se(dd) ·   ]   A2  A3  B5  A3  R3
y=3  [   ■     ■    se(dd)  ·     ·   ]   A2  A2  A3  R5  R6
y=4  [  S(o)  Se(od)  ·     ·     ·   ]   B3  A3  R3  R6  o

■ = 내부 (6면 모두 연료 또는 대칭)
알파벳 = 반사체 인접면 (대문자=ortho, 소문자=diag)
· = 반사체/빈공간 (L_diffusion 미적용)
```

**경계면 상세** (R4 boundary verification 결과):
| 위치 (y,x) | 어셈블리 | 반사체면 | 유형 |
|:---:|:---:|------|:---:|
| (0,4) | B3 | E면 → R1 | ortho |
| (1,4) | A3 | S면 → R3, E면 → R2 | diag+ortho |
| (2,3) | A3 | S면 → R5, E면 → R3 | diag+diag |
| (3,2) | A3 | S면 → R3, E면 → R5 | diag+diag |
| (4,0) | B3 | S면 → R1 | ortho |
| (4,1) | A3 | S면 → R2, E면 → R3 | ortho+diag |

**Z축 경계면**:
- z=0 (K=2): B면 = bottom 반사체 (모든 연료 XY 위치)
- z=19 (K=21): T면 = top 반사체 (모든 연료 XY 위치)
- z=1~18: B면, T면 모두 내부 (연료-연료)

**대칭면** (quarter-core):
- y=0인 모든 노드의 N면 = 대칭 (Mirror CMFD: ghost=qy=1)
- x=0인 모든 노드의 W면 = 대칭 (Mirror CMFD: ghost=qx=1)

---

## 3. Trainable 파라미터 (12개)

### Radial — 스칼라 α (4개)

```python
# R5 캘리브레이션 확정값 (40LP × CRS 10스텝 = 400 시나리오)
alpha_ortho_g1 = tf.Variable(0.108, trainable=True)   # β=0.805, R²=0.998
alpha_ortho_g2 = tf.Variable(0.453, trainable=True)   # β=0.377, R²=0.998
alpha_diag_g1  = tf.Variable(0.082, trainable=True)   # β=0.849, R²=0.988
alpha_diag_g2  = tf.Variable(0.513, trainable=True)   # β=0.322, R²=0.997
```

Marshak FD 공식: `J_g = α_g × D_g / (α_g × h/2 + D_g) × φ̂_g`
- D_g: 해당 연료 노드의 확산계수 = 1/(3Σ_tr) (xs_fuel, 노드마다 다름)
- h: 어셈블리 피치 21.608 cm
- φ̂_g: 모델 예측 flux

### Axial — 행렬 C (8개)

```python
# R5 캘리브레이션 확정값 (40LP)
C_bottom = tf.Variable([[+0.155, -0.135],    # R²=(0.999, 0.992)
                         [-0.025, +0.078]], trainable=True)
C_top    = tf.Variable([[+0.174, -0.097],    # R²=(0.993, 0.925)
                         [-0.036, +0.080]], trainable=True)
```

행렬 BC: `[J_g1, J_g2] = C × [φ̂_g1, φ̂_g2]`
- C₂₁ ≠ 0: 고속→열 감속 반환 효과 (경수 반사체 특성)
- TF2: `leak = tf.matmul(C, phi_2group)`
- 단순화 옵션: C₂₁=0 → 스칼라 α로 환원 가능 (정확도↓)

### 초기값 주의

- **R5 (40LP × CRS 10스텝 = 400 시나리오)** 기반 확정값
- R4(2LP) 대비: g1 안정 (<1%), g2 6~9% 감소, top C₁₂ 부호 반전 (-0.097)
- 추후 60LP 추가 생산 완료 시 재확인 가능 (현재 40LP로 충분히 안정)
- trainable이므로 학습 중 자동 보정됨

---

## 4. Loss 계산 흐름

### 의사코드

```python
def compute_L_diffusion(phi_pred, xs_fuel, keff, boundary_mask,
                        alpha_ortho, alpha_diag, C_bottom, C_top):
    """
    phi_pred: (B, 2, 20, 5, 5) — 모델 예측 flux [g1, g2]
    xs_fuel:  (20, 5, 5, 10) — 거시단면적 (고정)
    keff:     (B,) — GT keff
    boundary_mask: (20, 5, 5, 6) — 면 유형 코드 (사전 생성, 고정)
    """
    D_g1 = 1 / (3 * xs_fuel[..., 3])   # 확산계수 g1
    D_g2 = 1 / (3 * xs_fuel[..., 8])   # 확산계수 g2
    h = 21.608  # 어셈블리 피치 [cm]

    total_loss = 0
    for 각 연료 노드 (z, y, x):
        leak_g1, leak_g2 = 0, 0

        for 각 면 f in [N, S, E, W, T, B]:
            face_type = boundary_mask[z, y, x, f]

            if face_type == 0:  # 내부면
                nb = get_neighbor(z, y, x, f)
                D_harm = 2*D[z,y,x]*D[nb] / (D[z,y,x]+D[nb])
                leak_g1 += D_harm*(phi_nb_g1-phi_g1)/h * A
                leak_g2 += D_harm*(phi_nb_g2-phi_g2)/h * A

            elif face_type in [1, 2]:  # radial 반사체
                alpha = alpha_ortho if face_type==1 else alpha_diag
                leak_g1 += alpha[g1]*D_g1/(alpha[g1]*h/2+D_g1)*phi_g1 * A
                leak_g2 += alpha[g2]*D_g2/(alpha[g2]*h/2+D_g2)*phi_g2 * A

            elif face_type == 3:  # bottom 반사체
                J = C_bottom @ [phi_g1, phi_g2]
                leak_g1 += J[0] * A
                leak_g2 += J[1] * A

            elif face_type == 4:  # top 반사체
                J = C_top @ [phi_g1, phi_g2]
                leak_g1 += J[0] * A
                leak_g2 += J[1] * A

            elif face_type == 5:  # 대칭면 → CMFD with mirror neighbor
                nb = get_mirror_neighbor(z, y, x, f)
                # mirror symmetry: REFLECT (qy=-1→qy=1, qx=-1→qx=1)
                # rotational symmetry: transpose 매핑 (추후 변경)
                D_harm = 2*D[z,y,x]*D[nb] / (D[z,y,x]+D[nb])
                leak_g1 += D_harm*(phi_nb_g1-phi_g1)/h * A
                leak_g2 += D_harm*(phi_nb_g2-phi_g2)/h * A

        # 밸런스 잔차
        R_g1 = leak_g1 + Sigma_r1*phi_g1*V - (1/keff)*fission_src*V
        R_g2 = leak_g2 + Sigma_a2*phi_g2*V - Sigma_s12*phi_g1*V

        total_loss += R_g1**2 + R_g2**2

    return total_loss / (B * N_fuel)
```

### TF2 벡터화 구현 참고

실제 TF2에서는 loop 대신 **텐서 마스크 연산**으로 일괄 계산:
```python
# 면 유형별 마스크 생성
ortho_mask = (boundary_mask == 1)  # (20,5,5,6) bool
diag_mask  = (boundary_mask == 2)

# 각 유형별 누설 텐서 계산 후 마스크 적용하여 합산
leak_cmfd = ...      # 내부면 CMFD 누설
leak_ortho = ...     # ortho Marshak 누설
leak_diag = ...      # diag Marshak 누설
leak_bottom = ...    # bottom 행렬 C 누설
leak_top = ...       # top 행렬 C 누설

total_leak = leak_cmfd + leak_ortho + leak_diag + leak_bottom + leak_top
```

---

## 5. 검증 계획

1. **boundary_mask 검증**: R4 boundary verification 맵과 일치 확인
2. **GT flux 잔차**: GT flux로 L_diffusion 계산 시 기존 결과와 비교
   - 내부 노드만 (기존): g1 2.3%, g2 7.5%
   - 전체 연료 (Albedo BC 포함): 유사하거나 개선 기대
3. **trainable 수렴**: 학습 후 α/C가 캘리브레이션 초기값 근처 유지 확인
4. **gradient 전파**: 경계 연료 노드의 flux에 실제 gradient 제공 확인

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `Physical Loss 통합 레퍼런스 §3.5` | Albedo BC 공식 + 확정값 |
| `Albedo 캘리브레이션 종합 보고서 (확정).md` | R4 확정값 + 물리적 해석 |
| `JNET0 N-S 매핑 오류 발견 및 해결.md` | 좌표 매핑 (J↑=남쪽) |
| `R4 boundary verification` | 경계면 위치 + 맵 검증 |
