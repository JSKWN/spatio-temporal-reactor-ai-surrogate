"""
L_diffusion Outlier 진단 분석.

목적: end-to-end 테스트(10LP)의 outlier 원인 추적.
  - LP별 잔차 비교, XY 위치 시각화, 축방향 flux 프로파일
  - 면별 누설 분해, XS 비교, LP간 동일 위치 비교, CRS 스텝 안정성

작성일: 2026-04-01
"""

import sys
import os
from pathlib import Path
import numpy as np

VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))

from lf_preprocess.mas_out_parser import MasOutParser
from lf_preprocess.core_geometry import CoreGeometry

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780
SAVE_DIR = Path(__file__).parent

ALPHA_ORTHO = {'g1': 0.108, 'g2': 0.453}
ALPHA_DIAG = {'g1': 0.082, 'g2': 0.513}
C_BOTTOM = np.array([[+0.155, -0.135], [-0.025, +0.078]])
C_TOP = np.array([[+0.174, -0.097], [-0.036, +0.080]])

TEST_LPS = [
    ("LP_0000", "t12_363_p50_power_lower"),
    ("LP_0001", "t12_363_p50_power_upper"),
    ("LP_0002", "t12_363_p50_ramp_down"),
    ("LP_0003", "t12_363_p50_ramp_up"),
    ("LP_0004", "t12_8_p50_power_lower"),
    ("LP_0005", "t12_8_p50_power_upper"),
    ("LP_0006", "t12_8_p50_ramp_down"),
    ("LP_0007", "t12_8_p50_ramp_up"),
    ("LP_0008", "t14_262_p50_power_lower"),
    ("LP_0009", "t14_262_p50_power_upper"),
]

N_LP, N_STEP, N_Z, N_QY, N_QX = 10, 10, 20, 5, 5

# 11x11 full-core assembly map
ASM_11 = np.array([r.split() for r in """o o o R4 R2 R1 R2 R4 o o o
o o R6 R3 A3 B3 A3 R3 R6 o o
o R6 R5 A3 A2 A2 A2 A3 R5 R6 o
R4 R3 A3 B5 A3 A2 A3 B5 A3 R3 R4
R2 A3 A2 A3 A2 A3 A2 A3 A2 A3 R2
R1 B3 A2 A2 A3 A2 A3 A2 A2 B3 R1
R2 A3 A2 A3 A2 A3 A2 A3 A2 A3 R2
R4 R3 A3 B5 A3 A2 A3 B5 A3 R3 R4
o R6 R5 A3 A2 A2 A2 A3 R5 R6 o
o o R6 R3 A3 B3 A3 R3 R6 o o
o o o R4 R2 R1 R2 R4 o o o""".strip().split('\n')])

IS_FUEL_11 = np.array([[not c.startswith('R') and c != 'o' for c in r] for r in ASM_11])
IS_ORTHO_11 = np.array([[c in ('R1', 'R2') for c in r] for r in ASM_11])
IS_DIAG_11 = np.array([[c in ('R3', 'R4', 'R5', 'R6') for c in r] for r in ASM_11])

FACE_NAMES = ['N', 'S', 'E', 'W', 'B', 'T']


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def load_xs_fuel(geom, data_dir):
    from lf_preprocess.xs_voxel_builder import build_xs_voxel
    fuel_mask_quarter = np.ones((5, 5), dtype=bool)
    xsl_path = Path(data_dir) / "MAS_XSL"
    return build_xs_voxel(geom, xsl_path, fuel_mask_quarter)


def get_face_type(qy, qx, face):
    fj = qy + 5
    fi = qx + 5
    dj = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
    di = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
    nj, ni = fj + dj, fi + di
    if nj < 0 or nj >= 11 or ni < 0 or ni >= 11:
        return 'void'
    if IS_FUEL_11[nj, ni]:
        return 'fuel'
    if IS_ORTHO_11[nj, ni]:
        return 'ortho'
    if IS_DIAG_11[nj, ni]:
        return 'diag'
    return 'void'


def get_face_type_label(qy, qx, face):
    """6면 유형 문자열 반환 (대칭면 구분 포함)."""
    ft = get_face_type(qy, qx, face)
    if ft == 'fuel':
        dqy = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
        dqx = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
        nqy, nqx = qy + dqy, qx + dqx
        if not (0 <= nqy < 5 and 0 <= nqx < 5):
            return 'sym'
        return 'fuel'
    return ft


def collect_all_data():
    """10LP x 10step 전체 데이터 수집."""
    shape = (N_LP, N_STEP, N_Z, N_QY, N_QX)

    phi_g1 = np.zeros(shape)
    phi_g2 = np.zeros(shape)
    rel_g1 = np.full(shape, np.nan)
    rel_g2 = np.full(shape, np.nan)
    keff_arr = np.zeros((N_LP, N_STEP))
    fl_g1 = np.zeros((*shape, 6))  # face leakage g1 [N,S,E,W,B,T]
    fl_g2 = np.zeros((*shape, 6))
    rem_g1 = np.zeros(shape)
    rem_g2 = np.zeros(shape)
    src_g1 = np.zeros(shape)
    src_g2 = np.zeros(shape)
    is_bnd = np.zeros(shape, dtype=bool)
    xs_all = np.zeros((N_LP, N_Z, N_QY, N_QX, 10))

    dx = dy = WIDE

    for lp_idx, (lp_id, profile) in enumerate(TEST_LPS):
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            print(f"  SKIP {lp_id} (not found)")
            continue

        first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
        with open(first_out, "r", encoding="utf-8", errors="replace") as f:
            geom_lines = f.readlines()
        geom = CoreGeometry.from_lines(geom_lines)
        xs_fuel = load_xs_fuel(geom, data_dir).astype(np.float64)
        xs_all[lp_idx] = xs_fuel

        D_g1 = 1.0 / (3.0 * xs_fuel[..., 3] + 1e-30)
        D_g2 = 1.0 / (3.0 * xs_fuel[..., 8] + 1e-30)
        Sr_g1 = (xs_fuel[..., 2] + xs_fuel[..., 1]) + xs_fuel[..., 4]
        Sa_g2 = xs_fuel[..., 7] + xs_fuel[..., 6]
        nuSf_g1 = xs_fuel[..., 0]
        nuSf_g2 = xs_fuel[..., 5]
        Ss12 = xs_fuel[..., 4]

        dz_arr = np.array([geom.zmesh[k] for k in range(1, 21)], dtype=np.float64)

        for s_idx in range(N_STEP):
            s = s_idx + 1
            out_path = data_dir / f"MAS_OUT_{profile}_s{s:04d}_crs"
            if not out_path.exists():
                continue

            out_data = MasOutParser.parse(out_path)
            keff = out_data.keff
            keff_arr[lp_idx, s_idx] = keff
            phi_full = out_data.flux_3d
            pq_g1 = phi_full[:, 4:, 4:, 0].astype(np.float64)
            pq_g2 = phi_full[:, 4:, 4:, 1].astype(np.float64)

            for z in range(N_Z):
                dz = dz_arr[z]
                V = dx * dy * dz
                Az = dx * dy

                for qy in range(N_QY):
                    for qx in range(N_QX):
                        p1 = pq_g1[z, qy, qx]
                        p2 = pq_g2[z, qy, qx]
                        phi_g1[lp_idx, s_idx, z, qy, qx] = p1
                        phi_g2[lp_idx, s_idx, z, qy, qx] = p2
                        if p1 < 1e-10:
                            continue

                        leak1 = np.zeros(6)
                        leak2 = np.zeros(6)
                        inner = True

                        # 4 radial faces
                        for fi, face in enumerate(['N', 'S', 'E', 'W']):
                            Af = dy * dz if face in ('E', 'W') else dx * dz
                            ft = get_face_type(qy, qx, face)
                            dqy = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
                            dqx = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
                            nqy, nqx = qy + dqy, qx + dqx

                            if ft == 'fuel' and 0 <= nqy < 5 and 0 <= nqx < 5:
                                Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[z, nqy, nqx])
                                Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[z, nqy, nqx])
                                leak1[fi] = Dh1 * (p1 - pq_g1[z, nqy, nqx]) / dx * Af
                                leak2[fi] = Dh2 * (p2 - pq_g2[z, nqy, nqx]) / dx * Af
                            elif ft == 'fuel':
                                inner = False
                                # symmetry: mirror CMFD (REFLECT)
                            elif ft in ('ortho', 'diag'):
                                inner = False
                                alpha = ALPHA_ORTHO if ft == 'ortho' else ALPHA_DIAG
                                d1 = D_g1[z, qy, qx]
                                d2 = D_g2[z, qy, qx]
                                a1 = alpha['g1']
                                a2 = alpha['g2']
                                leak1[fi] = a1 * d1 / (a1 * dx / 2 + d1) * p1 * Af
                                leak2[fi] = a2 * d2 / (a2 * dx / 2 + d2) * p2 * Af
                            else:
                                inner = False

                        # 2 axial faces
                        for ai, (face, dz_idx) in enumerate([('B', z - 1), ('T', z + 1)]):
                            fi = 4 + ai  # B=4, T=5
                            if 0 <= dz_idx < 20:
                                hz = 0.5 * (dz_arr[z] + dz_arr[dz_idx])
                                Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[dz_idx, qy, qx])
                                Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[dz_idx, qy, qx])
                                leak1[fi] = Dh1 * (p1 - pq_g1[dz_idx, qy, qx]) / hz * Az
                                leak2[fi] = Dh2 * (p2 - pq_g2[dz_idx, qy, qx]) / hz * Az
                            else:
                                inner = False
                                C = C_BOTTOM if face == 'B' else C_TOP
                                phi_vec = np.array([p1, p2])
                                J_vec = C @ phi_vec
                                leak1[fi] = J_vec[0] * Az
                                leak2[fi] = J_vec[1] * Az

                        fl_g1[lp_idx, s_idx, z, qy, qx] = leak1
                        fl_g2[lp_idx, s_idx, z, qy, qx] = leak2

                        fission = nuSf_g1[z, qy, qx] * p1 + nuSf_g2[z, qy, qx] * p2
                        rv1 = Sr_g1[z, qy, qx] * p1 * V
                        rv2 = Sa_g2[z, qy, qx] * p2 * V
                        sv1 = (1.0 / keff) * fission * V
                        sv2 = Ss12[z, qy, qx] * p1 * V

                        rem_g1[lp_idx, s_idx, z, qy, qx] = rv1
                        rem_g2[lp_idx, s_idx, z, qy, qx] = rv2
                        src_g1[lp_idx, s_idx, z, qy, qx] = sv1
                        src_g2[lp_idx, s_idx, z, qy, qx] = sv2
                        is_bnd[lp_idx, s_idx, z, qy, qx] = not inner

                        R1 = leak1.sum() + rv1 - sv1
                        R2 = leak2.sum() + rv2 - sv2

                        r1 = abs(R1) / abs(rv1) * 100 if abs(rv1) > 1e-10 else np.nan
                        r2 = abs(R2) / abs(rv2) * 100 if abs(rv2) > 1e-10 else np.nan
                        rel_g1[lp_idx, s_idx, z, qy, qx] = r1
                        rel_g2[lp_idx, s_idx, z, qy, qx] = r2

        print(f"  {lp_id} ({profile}) done")

    return {
        'phi_g1': phi_g1, 'phi_g2': phi_g2,
        'rel_g1': rel_g1, 'rel_g2': rel_g2,
        'keff': keff_arr,
        'fl_g1': fl_g1, 'fl_g2': fl_g2,
        'rem_g1': rem_g1, 'rem_g2': rem_g2,
        'src_g1': src_g1, 'src_g2': src_g2,
        'is_bnd': is_bnd,
        'xs_all': xs_all,
    }


def section1_lp_comparison(d):
    """Section 1: LP별 잔차 비교."""
    print(f"\n{'='*78}")
    print("Section 1: LP별 잔차 비교")
    print('='*78)

    hdr = f"  {'LP':>8s}  {'profile':<30s}  {'keff_min':>8s}  {'keff_max':>8s}  {'g1_med':>7s}  {'g1_max':>7s}  {'g2_med':>7s}  {'g2_max':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    g1_maxs = []
    g2_maxs = []
    for li in range(N_LP):
        lp_id, profile = TEST_LPS[li]
        kmin = d['keff'][li].min()
        kmax = d['keff'][li].max()
        g1_vals = d['rel_g1'][li]
        g2_vals = d['rel_g2'][li]
        valid_g1 = g1_vals[np.isfinite(g1_vals)]
        valid_g2 = g2_vals[np.isfinite(g2_vals)]
        g1m = np.nanmedian(valid_g1)
        g1x = np.nanmax(valid_g1)
        g2m = np.nanmedian(valid_g2)
        g2x = np.nanmax(valid_g2)
        g1_maxs.append(g1x)
        g2_maxs.append(g2x)

        mark = ""
        if g1x == max(g1_maxs):
            mark = " <<< g1"
        if g2x >= max(g2_maxs):
            mark = " <<< g2" if not mark else " <<< g1+g2"

        print(f"  {lp_id:>8s}  {profile:<30s}  {kmin:8.5f}  {kmax:8.5f}  {g1m:6.2f}%  {g1x:6.2f}%  {g2m:6.2f}%  {g2x:6.2f}%{mark}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(N_LP)
    labels = [TEST_LPS[i][0] for i in range(N_LP)]

    axes[0].bar(x, g1_maxs, color=['#e74c3c' if i == 7 else '#3498db' for i in range(N_LP)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('g1 max residual (%)')
    axes[0].set_title('LP별 g1 max 잔차')
    axes[0].axhline(y=np.median(g1_maxs), color='gray', ls='--', alpha=0.5)

    axes[1].bar(x, g2_maxs, color=['#e74c3c' if i == 9 else '#2ecc71' for i in range(N_LP)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('g2 max residual (%)')
    axes[1].set_title('LP별 g2 max 잔차')
    axes[1].axhline(y=np.median(g2_maxs), color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    path = SAVE_DIR / "outlier_lp_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> {path}")


def section2_xy_position_map(d):
    """Section 2: Outlier 위치 시각화."""
    print(f"\n{'='*78}")
    print("Section 2: Outlier 위치 시각화 (Quarter-core XY 맵)")
    print('='*78)

    # ASCII map
    print("\n  Quarter-core 5x5 어셈블리 맵 + 경계면 유형:")
    print(f"  {'':>6s}", end="")
    for qx in range(5):
        print(f"  x={qx:<8d}", end="")
    print()

    for qy in range(5):
        asm_row = ""
        face_row = ""
        for qx in range(5):
            asm_name = ASM_11[qy + 5, qx + 5]
            if not IS_FUEL_11[qy + 5, qx + 5]:
                asm_row += f"  {'  ·  ':<10s}"
                face_row += f"  {'     ':<10s}"
            else:
                mark = ""
                if (qy, qx) == (2, 0):
                    mark = "*g1"
                elif (qy, qx) == (0, 2):
                    mark = "*g2"
                asm_row += f"  [{asm_name:>2s}]{mark:<5s}"

                faces_str = ""
                for face in ['N', 'S', 'E', 'W']:
                    fl = get_face_type_label(qy, qx, face)
                    if fl == 'sym':
                        faces_str += face.lower() + "s"
                    elif fl in ('ortho', 'diag'):
                        faces_str += face.upper() + fl[0]
                    elif fl == 'void':
                        faces_str += face + "v"
                    else:
                        faces_str += ".."
                face_row += f"  {faces_str:<10s}"

        print(f"  y={qy:<3d} {asm_row}")
        print(f"  {'':>6s}{face_row}")

    print("\n  범례: ns/ws = symmetry(Mirror CMFD), No/So/Eo = ortho reflector")
    print("        nd/sd/ed = diag reflector, .. = fuel(CMFD)")
    print("        *g1 = g1 outlier (LP_0007, z=4)")
    print("        *g2 = g2 outlier (LP_0009, z=17)")

    # Outlier 상세
    for label, qy, qx, desc in [
        ("g1 outlier", 2, 0, "LP_0007, z=4"),
        ("g2 outlier", 0, 2, "LP_0009, z=17"),
    ]:
        fj, fi = qy + 5, qx + 5
        asm = ASM_11[fj, fi]
        print(f"\n  [{label}] (qy={qy}, qx={qx}) -> full(j={fj}, i={fi}) = {asm}")
        print(f"    {desc}")
        for face in ['N', 'S', 'E', 'W']:
            fl = get_face_type_label(qy, qx, face)
            dqy = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
            dqx = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
            nj, ni = fj + dqy, fi + dqx
            nb_name = ASM_11[nj, ni] if 0 <= nj < 11 and 0 <= ni < 11 else "outside"
            print(f"    {face}면: {fl:<8s} -> ({nj},{ni}) = {nb_name}")
        print(f"    B면: {'bottom' if True else ''} (z=0 only)")
        print(f"    T면: {'top' if True else ''} (z=19 only)")

    # matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    asm_colors = {'A2': '#4ecdc4', 'A3': '#45b7d1', 'B3': '#f9ca24', 'B5': '#f0932b'}

    for qy in range(5):
        for qx in range(5):
            fj, fi = qy + 5, qx + 5
            if not IS_FUEL_11[fj, fi]:
                ax.add_patch(plt.Rectangle((qx - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                           facecolor='#ddd', edgecolor='#999', lw=0.5))
                ax.text(qx, 4 - qy, ASM_11[fj, fi], ha='center', va='center',
                        fontsize=8, color='#999')
                continue

            asm = ASM_11[fj, fi]
            color = asm_colors.get(asm, '#ccc')
            ax.add_patch(plt.Rectangle((qx - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                       facecolor=color, edgecolor='black', lw=1.5))
            ax.text(qx, 4 - qy, asm, ha='center', va='center', fontsize=11, fontweight='bold')

            # Mark outlier positions
            if (qy, qx) == (2, 0):
                ax.annotate('g1 outlier\n(W=sym)', xy=(qx - 0.45, 4 - qy),
                            xytext=(qx - 1.5, 4 - qy + 0.8),
                            fontsize=9, color='red', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))
            if (qy, qx) == (0, 2):
                ax.annotate('g2 outlier\n(N=sym)', xy=(qx, 4 - qy + 0.45),
                            xytext=(qx + 0.8, 4 - qy + 1.5),
                            fontsize=9, color='blue', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

            # Draw boundary face indicators
            for face in ['N', 'S', 'E', 'W']:
                fl = get_face_type_label(qy, qx, face)
                if fl == 'sym':
                    if face == 'N':
                        ax.plot([qx - 0.45, qx + 0.45], [4 - qy + 0.45, 4 - qy + 0.45],
                                'r-', lw=3)
                    elif face == 'W':
                        ax.plot([qx - 0.45, qx - 0.45], [4 - qy - 0.45, 4 - qy + 0.45],
                                'r-', lw=3)
                elif fl in ('ortho', 'diag'):
                    c = 'orange' if fl == 'ortho' else 'purple'
                    if face == 'S':
                        ax.plot([qx - 0.45, qx + 0.45], [4 - qy - 0.45, 4 - qy - 0.45],
                                color=c, lw=3)
                    elif face == 'E':
                        ax.plot([qx + 0.45, qx + 0.45], [4 - qy - 0.45, 4 - qy + 0.45],
                                color=c, lw=3)

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 6)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'qx={i}' for i in range(5)])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'qy={4-i}' for i in range(5)])
    ax.set_aspect('equal')
    ax.set_title('Quarter-core XY Map + Outlier Positions')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#4ecdc4', label='A2'), Patch(facecolor='#45b7d1', label='A3'),
        Patch(facecolor='#f9ca24', label='B3'), Patch(facecolor='#f0932b', label='B5'),
        Line2D([0], [0], color='red', lw=3, label='Symmetry (Mirror CMFD)'),
        Line2D([0], [0], color='orange', lw=3, label='Ortho reflector'),
        Line2D([0], [0], color='purple', lw=3, label='Diag reflector'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    path = SAVE_DIR / "outlier_xy_map.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> {path}")


def section3_axial_profile(d):
    """Section 3: 축방향 flux 프로파일."""
    print(f"\n{'='*78}")
    print("Section 3: 축방향 flux 프로파일")
    print('='*78)

    # g1 outlier: LP_0007 (idx=7), step=1 (idx=0), (qy=2, qx=0)
    # g2 outlier: LP_0009 (idx=9), step=1 (idx=0), (qy=0, qx=2)
    # ref: inner node, e.g. (qy=1, qx=1) = A2 inner

    cases = [
        ("g1 outlier: LP_0007, (qy=2,qx=0)", 7, 0, 2, 0, 4),
        ("g2 outlier: LP_0009, (qy=0,qx=2)", 9, 0, 0, 2, 17),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ci, (title, li, si, oqy, oqx, z_peak) in enumerate(cases):
        print(f"\n  [{title}]")
        ax = axes[ci]

        phi1_out = d['phi_g1'][li, si, :, oqy, oqx]
        phi2_out = d['phi_g2'][li, si, :, oqy, oqx]
        r1_out = d['rel_g1'][li, si, :, oqy, oqx]
        r2_out = d['rel_g2'][li, si, :, oqy, oqx]

        # reference: interior node (qy=1, qx=1)
        phi1_ref = d['phi_g1'][li, si, :, 1, 1]
        phi2_ref = d['phi_g2'][li, si, :, 1, 1]
        r1_ref = d['rel_g1'][li, si, :, 1, 1]
        r2_ref = d['rel_g2'][li, si, :, 1, 1]

        zz = np.arange(N_Z)

        print(f"  {'z':>3s}  {'phi_g1_out':>12s}  {'phi_g1_ref':>12s}  {'r_g1_out':>9s}  {'r_g1_ref':>9s}  {'phi_g2_out':>12s}  {'phi_g2_ref':>12s}  {'r_g2_out':>9s}  {'r_g2_ref':>9s}")
        for z in range(N_Z):
            p1o = phi1_out[z]
            p1r = phi1_ref[z]
            r1o = r1_out[z] if np.isfinite(r1_out[z]) else 0
            r1r = r1_ref[z] if np.isfinite(r1_ref[z]) else 0
            p2o = phi2_out[z]
            p2r = phi2_ref[z]
            r2o = r2_out[z] if np.isfinite(r2_out[z]) else 0
            r2r = r2_ref[z] if np.isfinite(r2_ref[z]) else 0
            mark = " <<<" if z == z_peak else ""
            print(f"  {z:3d}  {p1o:12.5e}  {p1r:12.5e}  {r1o:8.2f}%  {r1r:8.2f}%  {p2o:12.5e}  {p2r:12.5e}  {r2o:8.2f}%  {r2r:8.2f}%{mark}")

        # Gradient
        print(f"\n  축방향 flux gradient |dphi/dz| (10cm 간격):")
        for z in range(1, N_Z):
            grad1 = abs(phi1_out[z] - phi1_out[z - 1])
            grad2 = abs(phi2_out[z] - phi2_out[z - 1])
            if z == z_peak or z == z_peak + 1:
                print(f"    z={z-1}->{z}: g1={grad1:.4e}, g2={grad2:.4e}  <<<")

        # Plot
        if ci == 0:
            ax.plot(zz, phi1_out, 'ro-', ms=4, label=f'g1 outlier ({oqy},{oqx})')
            ax.plot(zz, phi1_ref, 'b^-', ms=4, label='g1 ref (1,1)')
            ax.set_ylabel('phi_g1')
        else:
            ax.plot(zz, phi2_out, 'ro-', ms=4, label=f'g2 outlier ({oqy},{oqx})')
            ax.plot(zz, phi2_ref, 'b^-', ms=4, label='g2 ref (1,1)')
            ax.set_ylabel('phi_g2')

        ax.axvline(x=z_peak, color='gray', ls='--', alpha=0.5, label=f'z={z_peak}')
        ax.set_xlabel('z layer')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / "axial_flux_profile.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> {path}")


def section4_face_leakage(d):
    """Section 4: 면별 누설 분해."""
    print(f"\n{'='*78}")
    print("Section 4: 면별 누설 분해")
    print('='*78)

    cases = [
        ("g1 worst: LP_0007, z=4, (2,0)", 7, 0, 4, 2, 0, 'g1'),
        ("g2 worst: LP_0009, z=17, (0,2)", 9, 0, 17, 0, 2, 'g2'),
    ]

    for title, li, si, z, qy, qx, group in cases:
        print(f"\n  [{title}]")

        fl1 = d['fl_g1'][li, si, z, qy, qx]  # (6,)
        fl2 = d['fl_g2'][li, si, z, qy, qx]
        rv1 = d['rem_g1'][li, si, z, qy, qx]
        rv2 = d['rem_g2'][li, si, z, qy, qx]
        sv1 = d['src_g1'][li, si, z, qy, qx]
        sv2 = d['src_g2'][li, si, z, qy, qx]

        R1 = fl1.sum() + rv1 - sv1
        R2 = fl2.sum() + rv2 - sv2

        if group == 'g1':
            fl, rv, sv, R = fl1, rv1, sv1, R1
        else:
            fl, rv, sv, R = fl2, rv2, sv2, R2

        print(f"  {'':>12s}  {'g1':>14s}  {'g2':>14s}  {'face_type':>10s}")
        print(f"  {'':>12s}  {'='*14}  {'='*14}  {'='*10}")
        for fi, fn in enumerate(FACE_NAMES):
            ft = get_face_type_label(qy, qx, fn) if fi < 4 else ('bottom/top' if fi == 4 else 'bottom/top')
            if fi == 4:
                ft = 'bottom' if z == 0 else 'fuel'
            elif fi == 5:
                ft = 'top' if z == 19 else 'fuel'
            else:
                ft = get_face_type_label(qy, qx, fn)
            print(f"  leak_{fn:>5s}  {fl1[fi]:+14.4e}  {fl2[fi]:+14.4e}  {ft:>10s}")

        print(f"  {'leak_total':>12s}  {fl1.sum():+14.4e}  {fl2.sum():+14.4e}")
        print(f"  {'removal':>12s}  {rv1:+14.4e}  {rv2:+14.4e}")
        print(f"  {'source':>12s}  {-sv1:+14.4e}  {-sv2:+14.4e}")
        print(f"  {'='*54}")
        print(f"  {'RESIDUAL':>12s}  {R1:+14.4e}  {R2:+14.4e}")
        print(f"  {'rel%':>12s}  {abs(R1)/abs(rv1)*100:13.2f}%  {abs(R2)/abs(rv2)*100:13.2f}%")

        # Fraction analysis
        print(f"\n  Removal 대비 각 면 누설 비율 ({group}):")
        ref = abs(rv1) if group == 'g1' else abs(rv2)
        for fi, fn in enumerate(FACE_NAMES):
            frac = fl[fi] / ref * 100 if ref > 0 else 0
            bar = '#' * int(abs(frac) * 2)
            sign = '+' if fl[fi] >= 0 else '-'
            print(f"    {fn:>5s}: {sign}{abs(frac):6.2f}%  {bar}")
        print(f"    total: {fl.sum()/ref*100:+7.2f}%")


def section5_xs_comparison(d):
    """Section 5: XS 비교."""
    print(f"\n{'='*78}")
    print("Section 5: XS 비교 (outlier vs 연료 평균)")
    print('='*78)

    xs_names = ['nuSf1', 'Sf1', 'Sc1', 'Str1', 'Ss12', 'nuSf2', 'Sf2', 'Sc2', 'Str2', 'pad']
    derived = [
        ('D_g1', lambda xs: 1.0 / (3.0 * xs[..., 3] + 1e-30)),
        ('D_g2', lambda xs: 1.0 / (3.0 * xs[..., 8] + 1e-30)),
        ('Sr_g1', lambda xs: xs[..., 2] + xs[..., 1] + xs[..., 4]),
        ('Sa_g2', lambda xs: xs[..., 7] + xs[..., 6]),
    ]

    cases = [
        ("LP_0007 (g1 outlier)", 7, 4, 2, 0),
        ("LP_0009 (g2 outlier)", 9, 17, 0, 2),
    ]

    # 연료 위치 마스크 (quarter-core 5x5 중 실제 연료)
    fuel_mask_q = np.array([[IS_FUEL_11[qy + 5, qx + 5] for qx in range(5)] for qy in range(5)])

    for title, li, z, qy, qx in cases:
        print(f"\n  [{title}] z={z}, (qy={qy},qx={qx}) = {ASM_11[qy+5, qx+5]}")
        xs_node = d['xs_all'][li, z, qy, qx]  # (10,)
        # 연료 위치만 평균 (비연료 XS=0 제외)
        xs_lp = d['xs_all'][li]  # (20, 5, 5, 10)
        xs_fuel_only = xs_lp[:, fuel_mask_q, :]  # (20, N_fuel, 10)
        xs_mean = np.mean(xs_fuel_only, axis=(0, 1))  # (10,)

        print(f"  {'channel':>10s}  {'outlier':>12s}  {'fuel_avg':>12s}  {'ratio':>8s}")
        print(f"  {'='*46}")
        for ci, cn in enumerate(xs_names[:9]):
            ratio = xs_node[ci] / xs_mean[ci] if xs_mean[ci] != 0 else float('inf')
            print(f"  {cn:>10s}  {xs_node[ci]:12.5e}  {xs_mean[ci]:12.5e}  {ratio:8.4f}")

        print(f"\n  Derived:")
        for name, func in derived:
            val_node = func(d['xs_all'][li, z:z+1, qy:qy+1, qx:qx+1]).item()
            vals_fuel = func(xs_fuel_only)  # (20, N_fuel)
            val_mean = np.mean(vals_fuel)
            ratio = val_node / val_mean if val_mean != 0 else float('inf')
            print(f"  {name:>10s}  {val_node:12.5e}  {val_mean:12.5e}  {ratio:8.4f}")


def section6_cross_lp(d):
    """Section 6: 동일 위치 LP간 비교."""
    print(f"\n{'='*78}")
    print("Section 6: 동일 위치 LP간 비교 (position-specific vs LP-specific)")
    print('='*78)

    positions = [
        ("(qy=2,qx=0), z=4  [g1 outlier position]", 4, 2, 0),
        ("(qy=0,qx=2), z=17 [g2 outlier position]", 17, 0, 2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for pi, (pos_label, z, qy, qx) in enumerate(positions):
        print(f"\n  [{pos_label}]")
        print(f"  {'LP':>8s}  {'profile':<30s}  {'g1_med':>7s}  {'g1_max':>7s}  {'g2_med':>7s}  {'g2_max':>7s}")

        g1_maxs = []
        g2_maxs = []
        for li in range(N_LP):
            lp_id, profile = TEST_LPS[li]
            g1v = d['rel_g1'][li, :, z, qy, qx]
            g2v = d['rel_g2'][li, :, z, qy, qx]
            valid1 = g1v[np.isfinite(g1v)]
            valid2 = g2v[np.isfinite(g2v)]
            if len(valid1) == 0:
                continue
            g1m = np.median(valid1)
            g1x = np.max(valid1)
            g2m = np.median(valid2)
            g2x = np.max(valid2)
            g1_maxs.append(g1x)
            g2_maxs.append(g2x)
            print(f"  {lp_id:>8s}  {profile:<30s}  {g1m:6.2f}%  {g1x:6.2f}%  {g2m:6.2f}%  {g2x:6.2f}%")

        ax = axes[pi]
        x = np.arange(len(g1_maxs))
        w = 0.35
        ax.bar(x - w/2, g1_maxs, w, label='g1 max', color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, g2_maxs, w, label='g2 max', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([TEST_LPS[i][0] for i in range(len(g1_maxs))], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('max residual (%)')
        ax.set_title(pos_label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / "cross_lp_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> {path}")


def section7_crs_stability(d):
    """Section 7: CRS 스텝간 유사성 원인."""
    print(f"\n{'='*78}")
    print("Section 7: CRS 스텝간 유사성 원인")
    print('='*78)

    cases = [
        ("g1 outlier: LP_0007, z=4, (2,0)", 7, 4, 2, 0),
        ("g2 outlier: LP_0009, z=17, (0,2)", 9, 17, 0, 2),
    ]

    for title, li, z, qy, qx in cases:
        print(f"\n  [{title}]")
        print(f"  {'step':>5s}  {'keff':>8s}  {'phi_g1':>12s}  {'phi_g2':>12s}  {'rel_g1':>8s}  {'rel_g2':>8s}")

        keffs = d['keff'][li]
        p1s = d['phi_g1'][li, :, z, qy, qx]
        p2s = d['phi_g2'][li, :, z, qy, qx]
        r1s = d['rel_g1'][li, :, z, qy, qx]
        r2s = d['rel_g2'][li, :, z, qy, qx]

        for si in range(N_STEP):
            print(f"  {si+1:5d}  {keffs[si]:8.5f}  {p1s[si]:12.5e}  {p2s[si]:12.5e}  {r1s[si]:7.2f}%  {r2s[si]:7.2f}%")

        # CV
        cv_p1 = np.std(p1s) / np.mean(p1s) * 100 if np.mean(p1s) > 0 else 0
        cv_p2 = np.std(p2s) / np.mean(p2s) * 100 if np.mean(p2s) > 0 else 0
        valid_r1 = r1s[np.isfinite(r1s)]
        valid_r2 = r2s[np.isfinite(r2s)]
        cv_r1 = np.std(valid_r1) / np.mean(valid_r1) * 100 if len(valid_r1) > 0 and np.mean(valid_r1) > 0 else 0
        cv_r2 = np.std(valid_r2) / np.mean(valid_r2) * 100 if len(valid_r2) > 0 and np.mean(valid_r2) > 0 else 0

        print(f"\n  CV (std/mean):")
        print(f"    phi_g1: {cv_p1:.3f}%  phi_g2: {cv_p2:.3f}%")
        print(f"    rel_g1: {cv_r1:.3f}%  rel_g2: {cv_r2:.3f}%")

        if cv_r1 < 1 and cv_r2 < 1:
            print(f"  => 잔차 CV < 1%: flux 크기는 변하지만 잔차 비율은 고정")
            print(f"     -> 잔차는 geometry+BC의 구조적 오차 (스텝/flux 크기에 무관)")
        else:
            print(f"  => 잔차 CV > 1%: 스텝에 따라 잔차 비율도 변동")


def main():
    print("=" * 78)
    print("L_diffusion Outlier 진단 분석")
    print("데이터: 10 LP x 10 CRS steps = 100 시나리오")
    print("=" * 78)

    print("\n[데이터 수집 중...]")
    d = collect_all_data()

    section1_lp_comparison(d)
    section2_xy_position_map(d)
    section3_axial_profile(d)
    section4_face_leakage(d)
    section5_xs_comparison(d)
    section6_cross_lp(d)
    section7_crs_stability(d)

    print(f"\n{'='*78}")
    print("분석 완료")
    print("=" * 78)


if __name__ == "__main__":
    main()
