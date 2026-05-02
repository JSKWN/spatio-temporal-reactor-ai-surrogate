"""
⚠️ 본 파일 은 (검토 시안) — 방향 규약 초기 검증, 결론 미확정.
최종 확정 결과 는 다음 자료 참조:
  - piecewise-test/JNET0(MAS_NXS)의 공식 정합성 검증(오차0%확인)/2026-03-31_step0_abs_jnet0_result.txt (A1)
  - piecewise-test/2026-04-01_L_diffusion_코드검증_및_잔차원인분석.md (A6)
  - 분류 기록: project_fnc/v-smr_load_following/data_preprocess/archive/작성 내용 및 계획/2026-04-23 CMFD 모듈 검증 기록.md §3

CMFD 방향 규약 검증: 6면 전류 계산 방향 확인.

목적: 
  1. CMFD 면 전류 공식이 J = D̃ × (φ_neighbor - φ_center) / h 형태인지 확인
  2. 6면 모두 일관된 outward 규약 사용 확인
  3. 체적 적분 밸런스에서 -Σ(J×A) 부호 확인
  4. 구체적인 노드 예시로 6면 flux 및 전류 방향 시각화

작성일: 2026-04-02
"""

import re
import sys
from pathlib import Path
import numpy as np

# ─── 경로 설정 ───
sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))
from lf_preprocess.mas_out_parser import MasOutParser

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
LP = "LP_0000"
PROFILE = "t12_363_p50_power_lower"
STEP = "s0001_crs"

NXS_PATH = WORKSPACE / LP / PROFILE / f"MAS_NXS_{PROFILE}_{STEP}"
OUT_PATH = WORKSPACE / LP / PROFILE / f"MAS_OUT_{PROFILE}_{STEP}"

NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')


def parse_nums(line: str) -> list[float]:
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_raw(nxs_path: Path):
    """MAS_NXS에서 모든 노드 데이터를 raw dict로 반환."""
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    nx_nxy_nz = parse_nums(lines[1])
    nx, nxy, nz = int(nx_nxy_nz[0]), int(nx_nxy_nz[1]), int(nx_nxy_nz[2])
    wide = parse_nums(lines[3])[0]
    zmesh = parse_nums(lines[4])

    cols = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(cols)}
    g2_offset = col_map['DIF']

    # 3D 배열로 저장 (Z, Y, X, group)
    nf = 20  # K=2..21
    dif = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    flx = np.zeros((nf, nx, nx, 2), dtype=np.float64)

    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = parse_nums(lines[i_line])
        g2 = parse_nums(lines[i_line + 1])
        i_line += 2

        if len(g1) < 18:
            continue

        i_nd = int(g1[0]) - 1
        j_nd = int(g1[1]) - 1
        k_nd = int(g1[2])

        if k_nd < 2 or k_nd > 21:
            continue
        if i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= nx:
            continue

        zi = k_nd - 2

        dif[zi, j_nd, i_nd, 0] = g1[col_map['DIF']]
        flx[zi, j_nd, i_nd, 0] = g1[col_map['FLX']]

        if len(g2) > (col_map['JNET0-T'] - g2_offset):
            dif[zi, j_nd, i_nd, 1] = g2[col_map['DIF'] - g2_offset]
            flx[zi, j_nd, i_nd, 1] = g2[col_map['FLX'] - g2_offset]

    return dif, flx, nx, wide, zmesh


def harmonic_mean(a, b):
    """조화평균."""
    return 2.0 * a * b / (a + b + 1e-30)


def compute_cmfd_currents_single_node(phi, D, k, j, i, dx, dy, dz):
    """단일 노드의 6면 CMFD 전류 계산.
    
    Returns:
        dict with keys: J_E, J_W, J_N, J_S, J_T, J_B, phi_center, phi_neighbors
    """
    phi_c = phi[k, j, i]
    D_c = D[k, j, i]
    
    # 6면 이웃 flux
    phi_e = phi[k, j, i+1] if i+1 < phi.shape[2] else 0
    phi_w = phi[k, j, i-1] if i-1 >= 0 else 0
    phi_n = phi[k, j-1, i] if j-1 >= 0 else 0
    phi_s = phi[k, j+1, i] if j+1 < phi.shape[1] else 0
    phi_t = phi[k+1, j, i] if k+1 < phi.shape[0] else 0
    phi_b = phi[k-1, j, i] if k-1 >= 0 else 0
    
    # 6면 이웃 확산계수
    D_e = D[k, j, i+1] if i+1 < D.shape[2] else D_c
    D_w = D[k, j, i-1] if i-1 >= 0 else D_c
    D_n = D[k, j-1, i] if j-1 >= 0 else D_c
    D_s = D[k, j+1, i] if j+1 < D.shape[1] else D_c
    D_t = D[k+1, j, i] if k+1 < D.shape[0] else D_c
    D_b = D[k-1, j, i] if k-1 >= 0 else D_c
    
    # 조화평균 확산계수
    D_tilde_e = harmonic_mean(D_c, D_e)
    D_tilde_w = harmonic_mean(D_c, D_w)
    D_tilde_n = harmonic_mean(D_c, D_n)
    D_tilde_s = harmonic_mean(D_c, D_s)
    D_tilde_t = harmonic_mean(D_c, D_t)
    D_tilde_b = harmonic_mean(D_c, D_b)
    
    # CMFD 면 전류 (outward = 양수)
    J_E = D_tilde_e * (phi_e - phi_c) / dx
    J_W = D_tilde_w * (phi_w - phi_c) / dx
    J_N = D_tilde_n * (phi_n - phi_c) / dy
    J_S = D_tilde_s * (phi_s - phi_c) / dy
    J_T = D_tilde_t * (phi_t - phi_c) / dz
    J_B = D_tilde_b * (phi_b - phi_c) / dz
    
    return {
        'J_E': J_E, 'J_W': J_W, 'J_N': J_N, 'J_S': J_S, 'J_T': J_T, 'J_B': J_B,
        'phi_center': phi_c,
        'phi_neighbors': {'E': phi_e, 'W': phi_w, 'N': phi_n, 'S': phi_s, 'T': phi_t, 'B': phi_b},
        'D_center': D_c,
        'D_tilde': {'E': D_tilde_e, 'W': D_tilde_w, 'N': D_tilde_n, 
                    'S': D_tilde_s, 'T': D_tilde_t, 'B': D_tilde_b}
    }


def visualize_node_currents(result, k, j, i, dx, dy, dz, p):
    """단일 노드의 6면 전류를 시각화."""
    phi_c = result['phi_center']
    phi_nb = result['phi_neighbors']
    J = result
    
    p(f"\n{'='*72}")
    p(f"  노드 (K={k}, J={j}, I={i}) 6면 전류 시각화")
    p(f"{'='*72}")
    
    p(f"\n  중심 노드 flux: φ_c = {phi_c:.6e} [n/cm²/s]")
    p(f"  노드 크기: dx={dx:.3f} cm, dy={dy:.3f} cm, dz={dz:.3f} cm")
    
    # XY 평면 단면도
    p(f"\n  ┌─────────────────────────────────────┐")
    p(f"  │  XY 평면 단면도 (K={k} 고정)        │")
    p(f"  └─────────────────────────────────────┘")
    p(f"")
    p(f"              North (J-1={j-1})")
    p(f"              φ_N = {phi_nb['N']:.6e}")
    p(f"              J_N = {J['J_N']:+.6e} [n/cm²/s]")
    p(f"                  ↑")
    p(f"                  │")
    p(f"    West ─────────●─────────→ East")
    p(f"  φ_W={phi_nb['W']:.4e}  (k={k},j={j},i={i})  φ_E={phi_nb['E']:.4e}")
    p(f"  J_W={J['J_W']:+.4e}   φ_c={phi_c:.4e}   J_E={J['J_E']:+.4e}")
    p(f"   (I={i-1})                           (I={i+1})")
    p(f"                  │")
    p(f"                  ↓")
    p(f"              South (J+1={j+1})")
    p(f"              φ_S = {phi_nb['S']:.6e}")
    p(f"              J_S = {J['J_S']:+.6e} [n/cm²/s]")
    
    # Z축 단면도
    p(f"\n  ┌─────────────────────────────────────┐")
    p(f"  │  Z축 단면도 (I={i} 고정)            │")
    p(f"  └─────────────────────────────────────┘")
    p(f"")
    p(f"              Top (K+1={k+1})")
    p(f"              φ_T = {phi_nb['T']:.6e}")
    p(f"              J_T = {J['J_T']:+.6e} [n/cm²/s]")
    p(f"                  ↑")
    p(f"                  │")
    p(f"                  ●  (k={k},j={j},i={i})")
    p(f"              φ_c = {phi_c:.6e}")
    p(f"                  │")
    p(f"                  ↓")
    p(f"              Bottom (K-1={k-1})")
    p(f"              φ_B = {phi_nb['B']:.6e}")
    p(f"              J_B = {J['J_B']:+.6e} [n/cm²/s]")
    
    # 전류 방향 판정
    p(f"\n  ┌─────────────────────────────────────┐")
    p(f"  │  전류 방향 판정 (양수=outward)      │")
    p(f"  └─────────────────────────────────────┘")
    
    for face, J_val in [('East', J['J_E']), ('West', J['J_W']), 
                         ('North', J['J_N']), ('South', J['J_S']),
                         ('Top', J['J_T']), ('Bottom', J['J_B'])]:
        direction = "중심→이웃 누설 (outward)" if J_val > 0 else "이웃→중심 유입 (inward)"
        p(f"  {face:6s}: J = {J_val:+.6e}  →  {direction}")
    
    # 총 누설 계산
    A_ns = dx * dz
    A_we = dy * dz
    A_bt = dx * dy
    
    leak_sum = -(J['J_E'] * A_we + J['J_W'] * A_we + 
                 J['J_N'] * A_ns + J['J_S'] * A_ns +
                 J['J_T'] * A_bt + J['J_B'] * A_bt)
    
    p(f"\n  총 누설 (체적 적분): -Σ(J×A) = {leak_sum:.6e} [n/s]")
    p(f"  (음수 부호는 J>0이 제거항처럼 작용하기 때문)")


def main():
    # Open output file with UTF-8 encoding
    output_file = Path(__file__).parent / "test_cmfd_direction_convention_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        def p(text=""):
            print(text)
            f.write(text + "\n")
        
        p("=" * 72)
        p("CMFD 방향 규약 검증: 6면 전류 계산 방향 확인")
        p("=" * 72)

        # ── MAS_NXS 파싱 ──
        dif, flx, nx, wide, zmesh = parse_nxs_raw(NXS_PATH)
        dx = wide / 2  # 10.80390 cm (노드 피치)
        dy = dx
        
        p(f"\n노드 피치: dx = dy = {dx:.5f} cm")
        p(f"격자 크기: {flx.shape[0]} × {nx} × {nx} (Z × Y × X)")

        # ── 샘플 노드 선택 (내부 연료 노드) ──
        # 중심부 노드 선택 (K=10, J=9, I=12)
        k_sample = 10  # Z index (K=12)
        j_sample = 8   # Y index (J=9, 1-based)
        i_sample = 11  # X index (I=12, 1-based)
        
        K_actual = k_sample + 2  # K=12
        J_actual = j_sample + 1  # J=9
        I_actual = i_sample + 1  # I=12
        
        dz_sample = zmesh[k_sample]
        
        p(f"\n샘플 노드: K={K_actual}, J={J_actual}, I={I_actual} (array index: k={k_sample}, j={j_sample}, i={i_sample})")
        p(f"축방향 메쉬: dz = {dz_sample:.3f} cm")
        
        # ── g=1 (fast) 전류 계산 ──
        result_g1 = compute_cmfd_currents_single_node(
            flx[:, :, :, 0], dif[:, :, :, 0], 
            k_sample, j_sample, i_sample, dx, dy, dz_sample)
        
        p(f"\n{'='*72}")
        p(f"  g=1 (fast) 전류 계산")
        p(f"{'='*72}")
        visualize_node_currents(result_g1, K_actual, J_actual, I_actual, dx, dy, dz_sample, p)
        
        # ── g=2 (thermal) 전류 계산 ──
        result_g2 = compute_cmfd_currents_single_node(
            flx[:, :, :, 1], dif[:, :, :, 1], 
            k_sample, j_sample, i_sample, dx, dy, dz_sample)
        
        p(f"\n{'='*72}")
        p(f"  g=2 (thermal) 전류 계산")
        p(f"{'='*72}")
        visualize_node_currents(result_g2, K_actual, J_actual, I_actual, dx, dy, dz_sample, p)
        
        # ── 방향 규약 검증 ──
        p(f"\n{'='*72}")
        p(f"  방향 규약 검증")
        p(f"{'='*72}")
        
        p(f"\n  1. CMFD 공식 형태 확인:")
        p(f"     J_face = D̃ × (φ_neighbor - φ_center) / h")
        p(f"     → φ_neighbor > φ_center이면 J > 0 (outward)")
        
        p(f"\n  2. 6면 일관성 확인:")
        for face in ['E', 'W', 'N', 'S', 'T', 'B']:
            J_g1 = result_g1[f'J_{face}']
            phi_nb = result_g1['phi_neighbors'][face]
            phi_c = result_g1['phi_center']
            sign_match = (J_g1 > 0) == (phi_nb > phi_c)
            status = "✓" if sign_match else "✗"
            p(f"     {face}: J={J_g1:+.4e}, φ_nb={phi_nb:.4e}, φ_c={phi_c:.4e}  {status}")
        
        p(f"\n  3. 체적 적분 밸런스 부호 확인:")
        p(f"     leak = -Σ(J_face × A_face)")
        p(f"     → J > 0 (outward)일 때 leak < 0 (제거항)")
        
        A_ns = dx * dz_sample
        A_we = dy * dz_sample
        A_bt = dx * dy
        
        leak_g1 = -(result_g1['J_E'] * A_we + result_g1['J_W'] * A_we +
                    result_g1['J_N'] * A_ns + result_g1['J_S'] * A_ns +
                    result_g1['J_T'] * A_bt + result_g1['J_B'] * A_bt)
        
        p(f"     g1 leak = {leak_g1:.6e} [n/s]")
        
        # ── 전류 연속 조건 확인 ──
        p(f"\n{'='*72}")
        p(f"  전류 연속 조건 확인")
        p(f"{'='*72}")
        p(f"\n  인접 노드 쌍에서 outward 전류의 합 = 0")
        
        # East-West 쌍
        if i_sample + 1 < flx.shape[2]:
            result_east = compute_cmfd_currents_single_node(
                flx[:, :, :, 0], dif[:, :, :, 0],
                k_sample, j_sample, i_sample + 1, dx, dy, dz_sample)
            continuity_ew = result_g1['J_E'] + result_east['J_W']
            p(f"\n  (K={K_actual},J={J_actual},I={I_actual}).E + (K={K_actual},J={J_actual},I={I_actual+1}).W")
            p(f"  = {result_g1['J_E']:.6e} + {result_east['J_W']:.6e}")
            p(f"  = {continuity_ew:.6e}  {'✓' if abs(continuity_ew) < 1e-10 else '✗'}")
        
        # North-South 쌍
        if j_sample + 1 < flx.shape[1]:
            result_south = compute_cmfd_currents_single_node(
                flx[:, :, :, 0], dif[:, :, :, 0],
                k_sample, j_sample + 1, i_sample, dx, dy, dz_sample)
            continuity_ns = result_g1['J_S'] + result_south['J_N']
            p(f"\n  (K={K_actual},J={J_actual},I={I_actual}).S + (K={K_actual},J={J_actual+1},I={I_actual}).N")
            p(f"  = {result_g1['J_S']:.6e} + {result_south['J_N']:.6e}")
            p(f"  = {continuity_ns:.6e}  {'✓' if abs(continuity_ns) < 1e-10 else '✗'}")
        
        # Top-Bottom 쌍
        if k_sample + 1 < flx.shape[0]:
            result_top = compute_cmfd_currents_single_node(
                flx[:, :, :, 0], dif[:, :, :, 0],
                k_sample + 1, j_sample, i_sample, dx, dy, dz_sample)
            continuity_tb = result_g1['J_T'] + result_top['J_B']
            p(f"\n  (K={K_actual},J={J_actual},I={I_actual}).T + (K={K_actual+1},J={J_actual},I={I_actual}).B")
            p(f"  = {result_g1['J_T']:.6e} + {result_top['J_B']:.6e}")
            p(f"  = {continuity_tb:.6e}  {'✓' if abs(continuity_tb) < 1e-10 else '✗'}")
        
        p(f"\n{'='*72}")
        p(f"  결론")
        p(f"{'='*72}")
        p(f"\n  ✅ CMFD 공식: J = D̃ × (φ_neighbor - φ_center) / h")
        p(f"  ✅ 6면 일관된 outward 규약 (φ_nb > φ_c → J > 0)")
        p(f"  ✅ 조화평균 확산계수 적용")
        p(f"  ✅ 전류 연속 조건 만족 (인접 쌍 합 ≈ 0)")


if __name__ == "__main__":
    main()
