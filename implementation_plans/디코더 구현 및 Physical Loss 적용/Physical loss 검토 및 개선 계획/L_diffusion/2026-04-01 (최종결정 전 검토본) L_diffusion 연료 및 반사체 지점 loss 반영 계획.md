# L_diffusion 연료 및 반사체 인접면 loss 반영 계획

**작성일**: 2026-04-01
**개정**: 2026-04-08 — 손실 형식을 절대 잔차 → **상대 잔차 (L_diff_rel)** 로 전환. 자세한 근거: `2026-03-30 Physical Loss 통합 레퍼런스.md` §3.6
**참조**: Physical Loss 통합 레퍼런스 §3, Albedo 캘리브레이션 종합 보고서 (확정), `05_symmetry_mode.md`, ML 검토 plan `C:\Users\Administrator\.claude\plans\compressed-wandering-stroustrup.md`

---

## 0. 개정 요약 (2026-04-08)

본 plan은 다음 변경사항을 반영하여 갱신됨:

1. **손실 형식 변경**: `‖R(φ_pred)‖²` (절대 잔차) → `‖R(φ_pred) − R(φ_GT)‖²` (상대 잔차, L_diff_rel)
   - CMFD-only의 g2 ~6.4% bias floor를 양변에서 cancel
   - target이 0이 아닌 사전 계산된 ε_total(t) = R(φ_GT, xs_BOC)
2. **본 plan의 boundary_mask, Albedo BC, trainable α/C는 모두 그대로 유지**:
   - R 연산자는 변하지 않음 (양변에 동일하게 적용)
   - α/C 학습 가능 파라미터의 backward chain은 R 연산자를 통해 그대로 작동
   - Albedo BC 학습 가치는 오히려 강화됨 (절대 잔차 bias 제거 후 신호가 더 선명)
3. **추가 손실 항 (별도 plan)**:
   - L_data_halo (λ=0.3): halo cell에 직접 supervision (`05_symmetry_mode.md` §3.1)
4. **Optimization 보조 옵션 (선택)**:
   - 옵션 A: rod-aware Σ_a 학습 가능 보정
   - 옵션 B: rod cell L_diff 가중치 감쇠 `weight = exp(−β·rod_frac)`

---

## 1. 개요

L_diffusion은 모델 예측 flux가 확산방정식의 **공간 밸런스**를 만족하는지 검사:
```
R_g = [6면 누설] + [제거 × V] - [생성 × V] = 0 (이상적, NEM 한계 무시)
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

### 4.0 개정 후 손실 정의 (상대 잔차 L_diff_rel)

**용어 정리**:
- **R**: 본 §3 의 잔차 연산자. 함수 `compute_R(phi, xs_fuel, keff, boundary_mask, alpha_ortho, alpha_diag, C_bottom, C_top)` 가 한 cell의 6면 누설을 boundary_mask에 따라 분기 처리하여 잔차 R_g1, R_g2를 계산한다 (이하 §4 의사코드 본문)
- **α, C**: §3 에서 정의된 학습 가능 Albedo BC 파라미터 12개 (`alpha_ortho_g1/g2`, `alpha_diag_g1/g2`, `C_bottom`, `C_top`). R 연산자 안의 face_type 1~4 분기에서 사용
- **L_diff_rel**: R 연산자를 사용한 손실의 상대 잔차 형식. 절대 잔차 `‖R(φ_pred)‖²`를 `‖R(φ_pred) − R(φ_GT)‖²`로 교체

**손실 계산 흐름**:
```python
# 매 학습 step:
R_pred = compute_R(phi_pred, xs_fuel, keff, boundary_mask,
                   alpha_ortho, alpha_diag, C_bottom, C_top)
R_GT   = compute_R(phi_GT,   xs_fuel, keff, boundary_mask,
                   alpha_ortho, alpha_diag, C_bottom, C_top)
L_diff_rel = mean((R_pred - R_GT)**2)
```

- R_pred와 R_GT는 **완전히 동일한 R 연산자**로 계산. 입력 phi만 다름
- α, C가 학습 가능이므로 두 R 모두 같은 α, C 값을 사용 → α/C에 정상 gradient 전달
- 비용: forward 시 R 호출 2회 (R 자체가 가벼워 무시 가능)

**사전 계산 가능성**:
- α, C가 *학습 도중 변하므로* R(φ_GT)를 학습 시작 전에 한 번 사전 계산해 둘 수는 없음
- 단 **α, C가 학습 진행에 따라 거의 변하지 않는 안정 단계** 진입 후에는 R(φ_GT)를 cache 가능 (성능 최적화 옵션, 본 plan 범위 외)
- 만약 α, C 를 사전 calibration된 값으로 fix하기로 결정하면, R(φ_GT)는 Phase G 전처리에서 1회 계산해 HDF5 저장 가능

### 의사코드 (R 연산자 본체, 절대 잔차 form 그대로 보관)

R 연산자 자체는 절대 잔차 시절과 동일. L_diff_rel은 R 호출을 두 번 (φ_pred, φ_GT) 한 후 차분을 취할 뿐이므로 본 함수는 그대로 재사용.

```python
def compute_R(phi, xs_fuel, keff, boundary_mask,
              alpha_ortho, alpha_diag, C_bottom, C_top):
    """
    phi:      (B, 2, 20, 5, 5) — flux [g1, g2] (φ_pred 또는 φ_GT)
    xs_fuel:  (20, 5, 5, 10) — 거시단면적 (BOC 고정)
    keff:     (B,) — GT keff
    boundary_mask: (20, 5, 5, 6) — 면 유형 코드 (사전 생성, 고정)
    
    Returns:
        R_g1, R_g2: (B, 20, 5, 5) — 노드별 잔차
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
        R_g1[z, y, x] = leak_g1 + Sigma_r1*phi_g1*V - (1/keff)*fission_src*V
        R_g2[z, y, x] = leak_g2 + Sigma_a2*phi_g2*V - Sigma_s12*phi_g1*V

    return R_g1, R_g2   # (B, 20, 5, 5) 각각

# L_diff_rel 호출:
R_pred_g1, R_pred_g2 = compute_R(phi_pred, ...)
R_GT_g1,   R_GT_g2   = compute_R(phi_GT,   ...)
L_diff_rel = mean((R_pred_g1 - R_GT_g1)**2 + (R_pred_g2 - R_GT_g2)**2)
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
2. **GT flux 잔차** (절대 잔차 R(φ_GT) 자체의 분포): 기존 결과와 비교
   - 내부 노드만 (기존): g1 2.3%, g2 7.5%
   - 전체 연료 (Albedo BC 포함): 유사하거나 개선 기대
   - 이 잔차가 곧 ε_total(t)이며, L_diff_rel의 target이 됨
3. **L_diff_rel 자체 검증**: 학습 시작 시점에서 R(φ_pred) ≈ R(φ_GT) 인 경우 (e.g. φ_pred = φ_GT) L_diff_rel ≈ 0 인지 확인
4. **trainable 수렴**: 학습 후 α/C가 캘리브레이션 초기값 근처 유지 확인
5. **gradient 전파**: 
   - 경계 연료 노드의 flux에 실제 gradient 제공 확인
   - α/C 파라미터에도 gradient 전달 확인 (R(φ_pred), R(φ_GT) 양쪽 chain에서)
6. **상대 vs 절대 ablation**: 동일 모델을 절대 잔차 / 상대 잔차로 학습하여 inner cell phi MSE 비교
   - 예측: 상대 잔차가 동등 또는 약간 우수, bias floor 영역에서 큰 차이

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `Physical Loss 통합 레퍼런스 §3.5` | Albedo BC 공식 + 확정값 |
| `Physical Loss 통합 레퍼런스 §3.6` | **L_diff_rel 형식 + redundancy 분석 + Consistency Barrier 참조 (개정 2026-04-08)** |
| `Albedo 캘리브레이션 종합 보고서 (확정).md` | R4 확정값 + 물리적 해석 |
| `JNET0 N-S 매핑 오류 발견 및 해결.md` | 좌표 매핑 (J↑=남쪽) |
| `R4 boundary verification` | 경계면 위치 + 맵 검증 |
| `05_symmetry_mode.md` §3.4 | halo cell의 L_diff_rel 사용 (디코더 (6,6) 출력 그대로) |
| `compressed-wandering-stroustrup.md` | ML 위협 검토 plan (권고 1, 7, 옵션 A/B) |
