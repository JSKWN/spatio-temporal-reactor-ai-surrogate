# CMFD 방향 규약 정합성 검증 보고서

> ⚠️ **본 파일 은 (검토 시안)** — 결론 미확정 상태 의 초기 검증 보고서.
> 최종 확정 결과 는 다음 자료 참조:
> - `piecewise-test/JNET0(MAS_NXS)의 공식 정합성 검증(오차0%확인)/2026-03-31_step0_abs_jnet0_result.txt` (A1)
> - `piecewise-test/2026-04-01_L_diffusion_코드검증_및_잔차원인분석.md` (A6)
> - 분류 기록: `project_fnc/v-smr_load_following/data_preprocess/archive/작성 내용 및 계획/2026-04-23 CMFD 모듈 검증 기록.md` §3

## 검증 일자: 2026-04-02

---

## 1. 검증 목적

CMFD(Coarse Mesh Finite Difference) 유한차분 계산의 방향 규약이 JNET0 및 물리 방정식과 일관되는지 확인:
- CMFD 면 전류 공식의 방향 정의 (J = D̃ × (φ_center - φ_neighbor) / h)
- 조화평균 확산계수 적용 확인
- 6면 일관된 외향(outward) 규약 적용
- Mirror CMFD 대칭면 처리 확인
- CMFD 계산 결과가 JNET0 ×2 밸런스와 동일한 방향 규약 사용 확인

---

## 2. 확정 결과 (Executive Summary)

### 2.1 CMFD 면 전류 공식 (검증 필요)

```
┌──────────────────────────────────────────────────────────┐
│ CMFD 면 전류 (outward from center):                      │
│                                                            │
│ J_face = D̃ × (φ_neighbor - φ_center) / h                │
│                                                            │
│ - 양수(+): 중심→이웃 방향 누설 (outward)                 │
│ - 음수(-): 이웃→중심 방향 유입 (inward)                  │
│ - D̃: 조화평균 확산계수 = 2×D_a×D_b/(D_a+D_b)           │
│ - h: 노드 간 거리 [cm]                                    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 6면 방향 규약 (검증 필요)

**XY 평면 단면도** (K=k 고정):

```
              North (J-1)
              φ[k,j-1,i]
                  ↑
                  │
    West ─────────●─────────→ East
  φ[k,j,i-1]   (k,j,i)    φ[k,j,i+1]
                  │
                  ↓
              South (J+1)
              φ[k,j+1,i]

주의: J 증가 = 남쪽(South) 방향
```

**Z축 단면도** (I=i 고정):

```
              Top (K+1)
              φ[k+1,j,i]
                  ↑
                  │
    North ────────●─────────→ South
  φ[k,j-1,i]   (k,j,i)    φ[k,j+1,i]
                  │
                  ↓
              Bottom (K-1)
              φ[k-1,j,i]
```

**6면 이웃 노드 요약**:

| 면 | 인덱스 | 방향 | 설명 |
|---|--------|------|------|
| East | φ[k, j, i+1] | I 증가 | 오른쪽 |
| West | φ[k, j, i-1] | I 감소 | 왼쪽 |
| North | φ[k, j-1, i] | J 감소 | 위쪽 (J 증가=남쪽) |
| South | φ[k, j+1, i] | J 증가 | 아래쪽 |
| Top | φ[k+1, j, i] | K 증가 | 상부 |
| Bottom | φ[k-1, j, i] | K 감소 | 하부 |

**CMFD 면 전류 공식** (검증 필요):

```python
# East 면 (I+1 방향, 오른쪽)
J_E = D̃_E × (φ[k,j,i+1] - φ[k,j,i]) / dx
# φ_east > φ_center → J_E > 0 (중심→동쪽 누설, outward)

# West 면 (I-1 방향, 왼쪽)
J_W = D̃_W × (φ[k,j,i-1] - φ[k,j,i]) / dx
# φ_west > φ_center → J_W > 0 (중심→서쪽 누설, outward)

# North 면 (J-1 방향, 위쪽, J 증가=남쪽)
J_N = D̃_N × (φ[k,j-1,i] - φ[k,j,i]) / dy
# φ_north > φ_center → J_N > 0 (중심→북쪽 누설, outward)

# South 면 (J+1 방향, 아래쪽)
J_S = D̃_S × (φ[k,j+1,i] - φ[k,j,i]) / dy
# φ_south > φ_center → J_S > 0 (중심→남쪽 누설, outward)

# Top 면 (K+1 방향, 상부)
J_T = D̃_T × (φ[k+1,j,i] - φ[k,j,i]) / h_z
# φ_top > φ_center → J_T > 0 (중심→상부 누설, outward)

# Bottom 면 (K-1 방향, 하부)
J_B = D̃_B × (φ[k-1,j,i] - φ[k,j,i]) / h_z
# φ_bottom > φ_center → J_B > 0 (중심→하부 누설, outward)
```

**조화평균 확산계수**:
```python
D̃ = 2 × D_center × D_neighbor / (D_center + D_neighbor)
```

**체적 적분 밸런스** (검증 필요):
```
-Σ_faces(J_face × A_face) + Σ_r × φ̄ × V = Source × V
```

**주의**: 누설항에 **음수 부호(-)** 가 붙는 이유는 J > 0 (outward)일 때 노드에서 중성자가 빠져나가므로 제거항처럼 작용하기 때문.

**면적 계산**:
```
A_NS = dx × dz  (North/South 면)
A_WE = dy × dz  (West/East 면)
A_BT = dx × dy  (Bottom/Top 면)
```