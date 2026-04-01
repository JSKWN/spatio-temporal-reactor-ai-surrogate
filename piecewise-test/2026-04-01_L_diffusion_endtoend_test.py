"""
L_diffusion end-to-end 테스트: 집합체 단위, R5 albedo, GT flux.

목적: 캘리브레이션된 albedo로 L_diffusion 전체(내부+경계)를 계산하여
      경계 노드의 잔차 수준 확인.

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

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780

# R5 확정 albedo (40LP)
ALPHA_ORTHO = {'g1': 0.108, 'g2': 0.453}
ALPHA_DIAG = {'g1': 0.082, 'g2': 0.513}
C_BOTTOM = np.array([[+0.155, -0.135], [-0.025, +0.078]])
C_TOP = np.array([[+0.174, -0.097], [-0.036, +0.080]])


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def load_xs_fuel(geom, data_dir):
    from lf_preprocess.xs_voxel_builder import build_xs_voxel
    fuel_mask_quarter = np.ones((5, 5), dtype=bool)
    xsl_path = Path(data_dir) / "MAS_XSL"
    return build_xs_voxel(geom, xsl_path, fuel_mask_quarter)


def expand_quarter_to_full(xs_q):
    Z, _, _, C = xs_q.shape
    full = np.zeros((Z, 9, 9, C), dtype=xs_q.dtype)
    full[:, 4:, 4:, :] = xs_q
    full[:, 4:, :5, :] = xs_q[:, :, ::-1, :]
    full[:, :5, 4:, :] = xs_q[:, ::-1, :, :]
    full[:, :5, :5, :] = xs_q[:, ::-1, ::-1, :]
    return full


def main():
    print("=" * 70)
    print("L_diffusion end-to-end 테스트 (집합체 단위, R5 albedo)")
    print("=" * 70)

    dx = dy = WIDE
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

    # Quarter-core 어셈블리 맵 (5×5) — 반사체 인접 판별용
    # 11×11 fullcore에서 우하단 quarter: j_full=5~9, i_full=5~9
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

    # Quarter (5×5) 추출: q_y=0~4 → full_j=5~9, q_x=0~4 → full_i=5~9
    # 면 유형 판별 함수
    def get_face_type(qy, qx, face):
        """Quarter-core (y,x)의 face 유형 반환.
        face: 'N'(-y), 'S'(+y), 'E'(+x), 'W'(-x)
        """
        fj = qy + 5  # full j
        fi = qx + 5  # full i
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

    grand_inner_g1, grand_inner_g2 = [], []
    grand_boundary_g1, grand_boundary_g2 = [], []
    face_type_counts = {}
    # outlier 추적용
    inner_detail = []   # (rel1, rel2, lp_id, s, z, qy, qx)
    boundary_detail = []

    for lp_id, profile in TEST_LPS:
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            continue

        first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
        with open(first_out, "r", encoding="utf-8", errors="replace") as f:
            geom_lines = f.readlines()
        geom = CoreGeometry.from_lines(geom_lines)
        xs_fuel = load_xs_fuel(geom, data_dir).astype(np.float64)  # (20,5,5,10)

        # Quarter-core XS
        D_g1 = 1.0 / (3.0 * xs_fuel[..., 3] + 1e-30)
        D_g2 = 1.0 / (3.0 * xs_fuel[..., 8] + 1e-30)
        Sr_g1 = (xs_fuel[..., 2] + xs_fuel[..., 1]) + xs_fuel[..., 4]
        Sa_g2 = xs_fuel[..., 7] + xs_fuel[..., 6]
        nuSf_g1 = xs_fuel[..., 0]
        nuSf_g2 = xs_fuel[..., 5]
        Ss12 = xs_fuel[..., 4]

        # 연료 K=2~21 → zmesh index 1~20 (zmesh[0]=K1=bottom reflector, zmesh[21]=K22=top reflector)
        dz_arr = np.array([geom.zmesh[k] for k in range(1, 21)], dtype=np.float64)

        # Flux: quarter (20, 5, 5)
        for s in range(1, 11):
            out_path = data_dir / f"MAS_OUT_{profile}_s{s:04d}_crs"
            if not out_path.exists():
                continue

            out_data = MasOutParser.parse(out_path)
            keff = out_data.keff
            phi_full = out_data.flux_3d  # (20, 9, 9, 2)
            phi_q_g1 = phi_full[:, 4:, 4:, 0].astype(np.float64)  # quarter
            phi_q_g2 = phi_full[:, 4:, 4:, 1].astype(np.float64)

            for z in range(20):
                dz = dz_arr[z]
                V = dx * dy * dz
                Az = dx * dy

                for qy in range(5):
                    for qx in range(5):
                        phi1 = phi_q_g1[z, qy, qx]
                        phi2 = phi_q_g2[z, qy, qx]
                        if phi1 < 1e-10:
                            continue

                        leak_g1, leak_g2 = 0.0, 0.0
                        is_inner = True
                        node_faces = []

                        # 4 radial faces
                        for face in ['N', 'S', 'E', 'W']:
                            Af = dy * dz if face in ('E', 'W') else dx * dz
                            ft = get_face_type(qy, qx, face)
                            node_faces.append(ft)

                            # 이웃 quarter 좌표
                            dqy = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
                            dqx = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
                            nqy, nqx = qy + dqy, qx + dqx

                            if ft == 'fuel' and 0 <= nqy < 5 and 0 <= nqx < 5:
                                # 내부면: CMFD (outward convention)
                                Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[z, nqy, nqx])
                                Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[z, nqy, nqx])
                                leak_g1 += Dh1 * (phi1 - phi_q_g1[z, nqy, nqx]) / dx * Af
                                leak_g2 += Dh2 * (phi2 - phi_q_g2[z, nqy, nqx]) / dx * Af
                            elif ft == 'fuel':
                                # 대칭면: mirror 이웃의 flux로 CMFD 계산
                                # mirror: nqy=-1 → qy=1, nqx=-1 → qx=1
                                is_inner = False
                                mqy = -nqy if nqy < 0 else nqy
                                mqx = -nqx if nqx < 0 else nqx
                                Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[z, mqy, mqx])
                                Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[z, mqy, mqx])
                                leak_g1 += Dh1 * (phi1 - phi_q_g1[z, mqy, mqx]) / dx * Af
                                leak_g2 += Dh2 * (phi2 - phi_q_g2[z, mqy, mqx]) / dx * Af
                            elif ft in ('ortho', 'diag'):
                                # 반사체면: Marshak BC
                                is_inner = False
                                alpha = ALPHA_ORTHO if ft == 'ortho' else ALPHA_DIAG
                                d1 = D_g1[z, qy, qx]
                                d2 = D_g2[z, qy, qx]
                                a1 = alpha['g1']
                                a2 = alpha['g2']
                                leak_g1 += a1 * d1 / (a1 * dx / 2 + d1) * phi1 * Af
                                leak_g2 += a2 * d2 / (a2 * dx / 2 + d2) * phi2 * Af
                            else:
                                # void: L_diffusion 미적용? 일단 J=0
                                is_inner = False
                                pass

                        # 2 axial faces
                        for face, dz_idx in [('B', z - 1), ('T', z + 1)]:
                            if 0 <= dz_idx < 20:
                                hz = 0.5 * (dz_arr[z] + dz_arr[dz_idx])
                                Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[dz_idx, qy, qx])
                                Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[dz_idx, qy, qx])
                                leak_g1 += Dh1 * (phi1 - phi_q_g1[dz_idx, qy, qx]) / hz * Az
                                leak_g2 += Dh2 * (phi2 - phi_q_g2[dz_idx, qy, qx]) / hz * Az
                            else:
                                is_inner = False
                                C = C_BOTTOM if face == 'B' else C_TOP
                                phi_vec = np.array([phi1, phi2])
                                J_vec = C @ phi_vec
                                leak_g1 += J_vec[0] * Az
                                leak_g2 += J_vec[1] * Az
                                node_faces.append('bottom' if face == 'B' else 'top')

                        # 밸런스 잔차
                        fission = nuSf_g1[z, qy, qx] * phi1 + nuSf_g2[z, qy, qx] * phi2
                        rem_g1 = Sr_g1[z, qy, qx] * phi1 * V
                        rem_g2 = Sa_g2[z, qy, qx] * phi2 * V
                        R_g1 = leak_g1 + rem_g1 - (1.0 / keff) * fission * V
                        R_g2 = leak_g2 + rem_g2 - Ss12[z, qy, qx] * phi1 * V

                        rel1 = abs(R_g1) / abs(rem_g1) * 100 if abs(rem_g1) > 1e-10 else np.nan
                        rel2 = abs(R_g2) / abs(rem_g2) * 100 if abs(rem_g2) > 1e-10 else np.nan

                        if np.isfinite(rel1):
                            if is_inner:
                                grand_inner_g1.append(rel1)
                                grand_inner_g2.append(rel2)
                                inner_detail.append((rel1, rel2, lp_id, s, z, qy, qx))
                            else:
                                grand_boundary_g1.append(rel1)
                                grand_boundary_g2.append(rel2)
                                boundary_detail.append((rel1, rel2, lp_id, s, z, qy, qx))

    print(f"\n  내부 노드 (6면 모두 연료):")
    if grand_inner_g1:
        g1 = np.array(grand_inner_g1)
        g2 = np.array(grand_inner_g2)
        print(f"    N={len(g1)}")
        print(f"    g1: median={np.median(g1):.3f}%, mean={g1.mean():.3f}%, p95={np.percentile(g1,95):.3f}%, max={g1.max():.3f}%")
        print(f"    g2: median={np.median(g2):.3f}%, mean={g2.mean():.3f}%, p95={np.percentile(g2,95):.3f}%, max={g2.max():.3f}%")

    print(f"\n  경계 노드 (Albedo BC 적용):")
    if grand_boundary_g1:
        g1 = np.array(grand_boundary_g1)
        g2 = np.array(grand_boundary_g2)
        print(f"    N={len(g1)}")
        print(f"    g1: median={np.median(g1):.3f}%, mean={g1.mean():.3f}%, p95={np.percentile(g1,95):.3f}%, max={g1.max():.3f}%")
        print(f"    g2: median={np.median(g2):.3f}%, mean={g2.mean():.3f}%, p95={np.percentile(g2,95):.3f}%, max={g2.max():.3f}%")

    print(f"\n  전체 (내부 + 경계):")
    all_g1 = np.array(grand_inner_g1 + grand_boundary_g1)
    all_g2 = np.array(grand_inner_g2 + grand_boundary_g2)
    print(f"    N={len(all_g1)}")
    print(f"    g1: median={np.median(all_g1):.3f}%, mean={all_g1.mean():.3f}%, max={all_g1.max():.3f}%")
    print(f"    g2: median={np.median(all_g2):.3f}%, mean={all_g2.mean():.3f}%, max={all_g2.max():.3f}%")

    # ── Outlier 분석 ──
    print(f"\n{'='*70}")
    print("Outlier 분석: g1 잔차 상위 10개")
    print("="*70)

    all_detail = inner_detail + boundary_detail
    all_detail_sorted = sorted(all_detail, key=lambda x: x[0], reverse=True)
    print(f"  {'rank':>4s}  {'g1%':>7s}  {'g2%':>7s}  {'LP':>8s}  {'step':>4s}  {'z':>2s}  {'qy':>2s}  {'qx':>2s}  {'type':>8s}")
    for i, (r1, r2, lp, s, z, qy, qx) in enumerate(all_detail_sorted[:10]):
        tp = "inner" if (r1, r2, lp, s, z, qy, qx) in inner_detail else "boundary"
        print(f"  {i+1:4d}  {r1:7.3f}  {r2:7.3f}  {lp:>8s}  {s:4d}  {z:2d}  {qy:2d}  {qx:2d}  {tp:>8s}")

    print(f"\n  g2 잔차 상위 10개:")
    all_detail_g2 = sorted(all_detail, key=lambda x: x[1], reverse=True)
    print(f"  {'rank':>4s}  {'g1%':>7s}  {'g2%':>7s}  {'LP':>8s}  {'step':>4s}  {'z':>2s}  {'qy':>2s}  {'qx':>2s}  {'type':>8s}")
    for i, (r1, r2, lp, s, z, qy, qx) in enumerate(all_detail_g2[:10]):
        tp = "inner" if (r1, r2, lp, s, z, qy, qx) in inner_detail else "boundary"
        print(f"  {i+1:4d}  {r1:7.3f}  {r2:7.3f}  {lp:>8s}  {s:4d}  {z:2d}  {qy:2d}  {qx:2d}  {tp:>8s}")

    # ── Z층별 분포 ──
    print(f"\n{'='*70}")
    print("Z층별 잔차 분포 (전체 노드)")
    print("="*70)
    print(f"  {'z':>2s}  {'N':>5s}  {'g1_med':>7s}  {'g1_mean':>8s}  {'g1_max':>7s}  {'g2_med':>7s}  {'g2_mean':>8s}  {'g2_max':>7s}")
    for zz in range(20):
        z_nodes = [(r1, r2) for r1, r2, lp, s, z, qy, qx in all_detail if z == zz]
        if not z_nodes:
            continue
        zg1 = np.array([x[0] for x in z_nodes])
        zg2 = np.array([x[1] for x in z_nodes])
        print(f"  {zz:2d}  {len(zg1):5d}  {np.median(zg1):7.3f}  {zg1.mean():8.3f}  {zg1.max():7.3f}  {np.median(zg2):7.3f}  {zg2.mean():8.3f}  {zg2.max():7.3f}")


if __name__ == "__main__":
    main()
