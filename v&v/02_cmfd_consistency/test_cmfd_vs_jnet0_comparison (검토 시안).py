"""
CMFD vs JNET0 비교 검증: 방향 규약 동치성 확인.

목적:
  1. CMFD 계산: J_CMFD = D̃ × (φ_neighbor - φ_center) / h
  2. JNET0 × 2: J_JNET0 = 2 × JNET0_face (MAS_NXS)
  3. 두 값이 일치하는지 확인 → 방향 규약 동치성 검증
  4. 공식 방향 확인: (φ_nb - φ_c) vs (φ_c - φ_nb)

작성일: 2026-04-02
"""

import re
import sys
from pathlib import Path
import numpy as np

# ─── 경로 설정 ───
sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
LP = "LP_0000"
PROFILE = "t12_363_p50_power_lower"
STEP = "s0001_crs"

NXS_PATH = WORKSPACE / LP / PROFILE / f"MAS_NXS_{PROFILE}_{STEP}"

NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')


def parse_nums(line: str) -> list[float]:
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_full(nxs_path: Path):
    """MAS_NXS에서 DIF, FLX, JNET0 모두 추출."""
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    nx_nxy_nz = parse_nums(lines[1])
    nx, nxy, nz = int(nx_nxy_nz[0]), int(nx_nxy_nz[1]), int(nx_nxy_nz[2])
    wide = parse_nums(lines[3])[0]
    zmesh = parse_nums(lines[4])

    cols = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(cols)}
    g2_offset = col_map['DIF']

    # 3D 배열 (Z, Y, X, group)
    nf = 20  # K=2..21
    dif = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    flx = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_n = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_s = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_w = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_e = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_b = np.zeros((nf, nx, nx, 2), dtype=np.float64)
    jnet_t = np.zeros((nf, nx, nx, 2), dtype=np.float64)

    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = parse_nums(lines[i_line])
        g2 = parse_nums(lines[i_line + 1])
        i_line += 2

        if len(g1) < col_map['JNET0-T'] + 1:
            continue

        i_nd = int(g1[0]) - 1
        j_nd = int(g1[1]) - 1
        k_nd = int(g1[2])

        if k_nd < 2 or k_nd > 21:
            continue
        if i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= nx:
            continue

        zi = k_nd - 2

        # g=1
        dif[zi, j_nd, i_nd, 0] = g1[col_map['DIF']]
        flx[zi, j_nd, i_nd, 0] = g1[col_map['FLX']]
        jnet_n[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-N']]
        jnet_s[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-S']]
        jnet_w[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-W']]
        jnet_e[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-E']]
        jnet_b[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-B']]
        jnet_t[zi, j_nd, i_nd, 0] = g1[col_map['JNET0-T']]

        # g=2
        if len(g2) > (col_map['JNET0-T'] - g2_offset):
            dif[zi, j_nd, i_nd, 1] = g2[col_map['DIF'] - g2_offset]
            flx[zi, j_nd, i_nd, 1] = g2[col_map['FLX'] - g2_offset]
            jnet_n[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-N'] - g2_offset]
            jnet_s[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-S'] - g2_offset]
            jnet_w[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-W'] - g2_offset]
            jnet_e[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-E'] - g2_offset]
            jnet_b[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-B'] - g2_offset]
            jnet_t[zi, j_nd, i_nd, 1] = g2[col_map['JNET0-T'] - g2_offset]

    return {
        'dif': dif, 'flx': flx,
        'jnet_n': jnet_n, 'jnet_s': jnet_s, 'jnet_w': jnet_w,
        'jnet_e': jnet_e, 'jnet_b': jnet_b, 'jnet_t': jnet_t,
        'nx': nx, 'wide': wide, 'zmesh': zmesh
    }


def harmonic_mean(a, b):
    """조화평균."""
    return 2.0 * a * b / (a + b + 1e-30)


def compare_cmfd_jnet0_single_node(data, k, j, i, g, dx, dy, dz, p):
    """단일 노드에서 CMFD vs JNET0 × 2 비교.
    
    Args:
        k, j, i: array index (0-based)
        g: group index (0 or 1)
    """
    phi = data['flx'][:, :, :, g]
    D = data['dif'][:, :, :, g]
    
    phi_c = phi[k, j, i]
    D_c = D[k, j, i]
    
    # 6면 이웃 flux 및 D
    phi_e = phi[k, j, i+1] if i+1 < phi.shape[2] else phi_c
    phi_w = phi[k, j, i-1] if i-1 >= 0 else phi_c
    phi_n = phi[k, j-1, i] if j-1 >= 0 else phi_c
    phi_s = phi[k, j+1, i] if j+1 < phi.shape[1] else phi_c
    phi_t = phi[k+1, j, i] if k+1 < phi.shape[0] else phi_c
    phi_b = phi[k-1, j, i] if k-1 >= 0 else phi_c
    
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
    
    # CMFD 면 전류 계산 (두 가지 공식 시도)
    # 가설 A: J = D̃ × (φ_neighbor - φ_center) / h
    J_CMFD_A = {
        'E': D_tilde_e * (phi_e - phi_c) / dx,
        'W': D_tilde_w * (phi_w - phi_c) / dx,
        'N': D_tilde_n * (phi_n - phi_c) / dy,
        'S': D_tilde_s * (phi_s - phi_c) / dy,
        'T': D_tilde_t * (phi_t - phi_c) / dz,
        'B': D_tilde_b * (phi_b - phi_c) / dz,
    }
    
    # 가설 B: J = D̃ × (φ_center - φ_neighbor) / h
    J_CMFD_B = {
        'E': D_tilde_e * (phi_c - phi_e) / dx,
        'W': D_tilde_w * (phi_c - phi_w) / dx,
        'N': D_tilde_n * (phi_c - phi_n) / dy,
        'S': D_tilde_s * (phi_c - phi_s) / dy,
        'T': D_tilde_t * (phi_c - phi_t) / dz,
        'B': D_tilde_b * (phi_c - phi_b) / dz,
    }
    
    # JNET0 × 2 (MAS_NXS)
    J_JNET0 = {
        'E': 2.0 * data['jnet_e'][k, j, i, g],
        'W': 2.0 * data['jnet_w'][k, j, i, g],
        'N': 2.0 * data['jnet_n'][k, j, i, g],
        'S': 2.0 * data['jnet_s'][k, j, i, g],
        'T': 2.0 * data['jnet_t'][k, j, i, g],
        'B': 2.0 * data['jnet_b'][k, j, i, g],
    }
    
    # 비교
    K_actual = k + 2
    J_actual = j + 1
    I_actual = i + 1
    
    p(f"\n{'='*72}")
    p(f"  노드 (K={K_actual}, J={J_actual}, I={I_actual}), g={g+1}")
    p(f"{'='*72}")
    
    p(f"\n  중심 flux: φ_c = {phi_c:.6e} [n/cm²/s]")
    p(f"  중심 확산계수: D_c = {D_c:.6f} cm")
    
    p(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    p(f"  │  6면 비교: CMFD vs JNET0 × 2                                │")
    p(f"  └─────────────────────────────────────────────────────────────┘")
    p(f"")
    p(f"  {'면':<6} {'φ_nb':>12} {'D̃':>8} {'J_CMFD_A':>14} {'J_CMFD_B':>14} {'JNET0×2':>14} {'오차A':>10} {'오차B':>10}")
    p(f"  {'-'*72}")
    
    phi_nbs = {'E': phi_e, 'W': phi_w, 'N': phi_n, 'S': phi_s, 'T': phi_t, 'B': phi_b}
    D_tildes = {'E': D_tilde_e, 'W': D_tilde_w, 'N': D_tilde_n, 
                'S': D_tilde_s, 'T': D_tilde_t, 'B': D_tilde_b}
    
    errors_A = []
    errors_B = []
    
    for face in ['E', 'W', 'N', 'S', 'T', 'B']:
        phi_nb = phi_nbs[face]
        D_tilde = D_tildes[face]
        J_A = J_CMFD_A[face]
        J_B = J_CMFD_B[face]
        J_ref = J_JNET0[face]
        
        # 상대 오차 계산
        if abs(J_ref) > 1e-10:
            err_A = abs(J_A - J_ref) / abs(J_ref) * 100
            err_B = abs(J_B - J_ref) / abs(J_ref) * 100
        else:
            err_A = 0.0 if abs(J_A) < 1e-10 else 999.0
            err_B = 0.0 if abs(J_B) < 1e-10 else 999.0
        
        errors_A.append(err_A)
        errors_B.append(err_B)
        
        p(f"  {face:<6} {phi_nb:>12.6e} {D_tilde:>8.4f} {J_A:>14.6e} {J_B:>14.6e} {J_ref:>14.6e} {err_A:>9.4f}% {err_B:>9.4f}%")
    
    # 판정
    p(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    p(f"  │  판정                                                        │")
    p(f"  └─────────────────────────────────────────────────────────────┘")
    
    mean_err_A = np.mean(errors_A)
    mean_err_B = np.mean(errors_B)
    
    p(f"\n  가설 A: J = D̃ × (φ_neighbor - φ_center) / h")
    p(f"    평균 오차: {mean_err_A:.6f}%")
    
    p(f"\n  가설 B: J = D̃ × (φ_center - φ_neighbor) / h")
    p(f"    평균 오차: {mean_err_B:.6f}%")
    
    if mean_err_A < 0.01:
        p(f"\n  ✅ 가설 A 확정: J = D̃ × (φ_neighbor - φ_center) / h")
        p(f"     CMFD와 JNET0 × 2가 동치 (오차 < 0.01%)")
        return 'A'
    elif mean_err_B < 0.01:
        p(f"\n  ✅ 가설 B 확정: J = D̃ × (φ_center - φ_neighbor) / h")
        p(f"     CMFD와 JNET0 × 2가 동치 (오차 < 0.01%)")
        return 'B'
    else:
        p(f"\n  ⚠️  두 가설 모두 오차 > 0.01% → 추가 조사 필요")
        return None


def main():
    # Open output file with UTF-8 encoding
    output_file = Path(__file__).parent / "test_cmfd_vs_jnet0_comparison_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        def p(text=""):
            print(text)
            f.write(text + "\n")
        
        p("=" * 72)
        p("CMFD vs JNET0 비교 검증: 방향 규약 동치성 확인")
        p("=" * 72)
        p("")
        p("목적:")
        p("  1. CMFD 공식 방향 확인: (φ_nb - φ_c) vs (φ_c - φ_nb)")
        p("  2. CMFD 계산 결과와 JNET0 × 2 비교")
        p("  3. 방향 규약 동치성 검증")
        p("")
        p("=" * 72)

        # ── MAS_NXS 파싱 ──
        data = parse_nxs_full(NXS_PATH)
        dx = data['wide'] / 2  # 10.80390 cm
        dy = dx
        
        p(f"\n노드 피치: dx = dy = {dx:.5f} cm")
        p(f"격자 크기: {data['flx'].shape[0]} × {data['nx']} × {data['nx']} (Z × Y × X)")

        # ── 샘플 노드 3개 선택 ──
        samples = [
            (10, 8, 11),   # K=12, J=9, I=12 (중심부)
            (5, 10, 15),   # K=7, J=11, I=16 (다른 위치)
            (15, 6, 8),    # K=17, J=7, I=9 (상부)
        ]
        
        results = []
        for k, j, i in samples:
            dz = data['zmesh'][k]
            
            # g=1
            result_g1 = compare_cmfd_jnet0_single_node(data, k, j, i, 0, dx, dy, dz, p)
            results.append(result_g1)
            
            # g=2
            result_g2 = compare_cmfd_jnet0_single_node(data, k, j, i, 1, dx, dy, dz, p)
            results.append(result_g2)
        
        # ── 종합 판정 ──
        p(f"\n{'='*72}")
        p(f"  종합 판정")
        p(f"{'='*72}")
        
        if all(r == 'A' for r in results if r is not None):
            p(f"\n  ✅ CMFD 공식 확정:")
            p(f"     J_face = D̃ × (φ_neighbor - φ_center) / h")
            p(f"")
            p(f"  ✅ JNET0 × 2와 동치 확인:")
            p(f"     J_CMFD ≈ JNET0 × 2 (오차 < 0.01%)")
            p(f"")
            p(f"  ✅ 방향 규약 일관성:")
            p(f"     - φ_neighbor > φ_center → J > 0 (outward)")
            p(f"     - CMFD와 JNET0 모두 동일한 outward 규약 사용")
        elif all(r == 'B' for r in results if r is not None):
            p(f"\n  ✅ CMFD 공식 확정:")
            p(f"     J_face = D̃ × (φ_center - φ_neighbor) / h")
            p(f"")
            p(f"  ⚠️  주의: 이 경우 φ_center > φ_neighbor → J > 0 (outward)")
            p(f"     JNET0와 부호 규약이 반대일 수 있음")
        else:
            p(f"\n  ⚠️  일관성 없음 → 추가 조사 필요")


if __name__ == "__main__":
    main()
