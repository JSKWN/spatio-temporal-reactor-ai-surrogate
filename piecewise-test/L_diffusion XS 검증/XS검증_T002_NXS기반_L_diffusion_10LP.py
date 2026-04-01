"""
XS검증_T002: MAS_NXS XS 기반 L_diffusion CMFD 잔차 10LP 재계산.

목적: endtoend_test.py와 동일한 CMFD 잔차 계산을 수행하되,
      XS를 MAS_XSL(BOC 고정) 대신 MAS_NXS(스텝별 feedback)로 교체.
      두 결과를 나란히 비교하여 XS 불일치가 잔차에 미치는 영향 정량화.

비교:
  [A] MAS_XSL XS 기반 CMFD 잔차 (기존 endtoend_test 결과)
  [B] MAS_NXS XS 기반 CMFD 잔차 (스텝별 feedback XS)
  → 차이 = XS 불일치 기여분, [B]의 잔차 = 순수 CMFD(D-hat) 한계

대상: 10 LP × 10 CRS steps = 100 시나리오

작성일: 2026-04-01
"""

import sys
import os
import re
import gc
from pathlib import Path
import numpy as np

VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))

from lf_preprocess.mas_out_parser import MasOutParser
from lf_preprocess.core_geometry import CoreGeometry
from lf_preprocess.xs_voxel_builder import build_xs_voxel

sys.path.insert(0, str(Path(__file__).parent))
from XS검증_T001_XSL_vs_NXS_채널비교 import parse_nxs_diffusion_xs, IS_FUEL_9x9

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780

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


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def get_face_type(qy, qx, face):
    fj, fi = qy + 5, qx + 5
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


def compute_residuals(phi_q_g1, phi_q_g2, D_g1, D_g2, Sr_g1, Sa_g2,
                      nuSf_g1, nuSf_g2, Ss12, dz_arr, keff, dx):
    """endtoend_test.py와 동일한 CMFD 잔차 계산 (Mirror CMFD 포함)."""
    dy = dx
    inner_g1, inner_g2 = [], []
    boundary_g1, boundary_g2 = [], []

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

                for face in ['N', 'S', 'E', 'W']:
                    Af = dy * dz if face in ('E', 'W') else dx * dz
                    ft = get_face_type(qy, qx, face)
                    dqy = {'N': -1, 'S': +1, 'E': 0, 'W': 0}[face]
                    dqx = {'N': 0, 'S': 0, 'E': +1, 'W': -1}[face]
                    nqy, nqx = qy + dqy, qx + dqx

                    if ft == 'fuel' and 0 <= nqy < 5 and 0 <= nqx < 5:
                        Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[z, nqy, nqx])
                        Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[z, nqy, nqx])
                        leak_g1 += Dh1 * (phi1 - phi_q_g1[z, nqy, nqx]) / dx * Af
                        leak_g2 += Dh2 * (phi2 - phi_q_g2[z, nqy, nqx]) / dx * Af
                    elif ft == 'fuel':
                        is_inner = False
                        mqy = -nqy if nqy < 0 else nqy
                        mqx = -nqx if nqx < 0 else nqx
                        Dh1 = harmonic_mean(D_g1[z, qy, qx], D_g1[z, mqy, mqx])
                        Dh2 = harmonic_mean(D_g2[z, qy, qx], D_g2[z, mqy, mqx])
                        leak_g1 += Dh1 * (phi1 - phi_q_g1[z, mqy, mqx]) / dx * Af
                        leak_g2 += Dh2 * (phi2 - phi_q_g2[z, mqy, mqx]) / dx * Af
                    elif ft in ('ortho', 'diag'):
                        is_inner = False
                        alpha = ALPHA_ORTHO if ft == 'ortho' else ALPHA_DIAG
                        d1, d2 = D_g1[z, qy, qx], D_g2[z, qy, qx]
                        leak_g1 += alpha['g1'] * d1 / (alpha['g1'] * dx / 2 + d1) * phi1 * Af
                        leak_g2 += alpha['g2'] * d2 / (alpha['g2'] * dx / 2 + d2) * phi2 * Af
                    else:
                        is_inner = False

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
                        J_vec = C @ np.array([phi1, phi2])
                        leak_g1 += J_vec[0] * Az
                        leak_g2 += J_vec[1] * Az

                fission = nuSf_g1[z, qy, qx] * phi1 + nuSf_g2[z, qy, qx] * phi2
                rem_g1 = Sr_g1[z, qy, qx] * phi1 * V
                rem_g2 = Sa_g2[z, qy, qx] * phi2 * V
                R_g1 = leak_g1 + rem_g1 - (1.0 / keff) * fission * V
                R_g2 = leak_g2 + rem_g2 - Ss12[z, qy, qx] * phi1 * V

                rel1 = abs(R_g1) / abs(rem_g1) * 100 if abs(rem_g1) > 1e-10 else np.nan
                rel2 = abs(R_g2) / abs(rem_g2) * 100 if abs(rem_g2) > 1e-10 else np.nan

                if np.isfinite(rel1):
                    if is_inner:
                        inner_g1.append(rel1)
                        inner_g2.append(rel2)
                    else:
                        boundary_g1.append(rel1)
                        boundary_g2.append(rel2)

    return inner_g1, inner_g2, boundary_g1, boundary_g2


def print_stats(label, ig1, ig2, bg1, bg2):
    g1i = np.array(ig1) if ig1 else np.array([0])
    g2i = np.array(ig2) if ig2 else np.array([0])
    g1b = np.array(bg1) if bg1 else np.array([0])
    g2b = np.array(bg2) if bg2 else np.array([0])
    g1a = np.concatenate([g1i, g1b])
    g2a = np.concatenate([g2i, g2b])

    print(f"\n  [{label}]")
    print(f"  {'구분':>8s}  {'N':>6s}  {'g1 med':>7s}  {'g1 mean':>8s}  {'g1 p95':>7s}  {'g1 max':>7s}  {'g2 med':>7s}  {'g2 mean':>8s}  {'g2 p95':>7s}  {'g2 max':>7s}")
    print(f"  {'내부':>8s}  {len(g1i):6d}  {np.median(g1i):6.2f}%  {g1i.mean():7.2f}%  {np.percentile(g1i,95):6.2f}%  {g1i.max():6.2f}%  {np.median(g2i):6.2f}%  {g2i.mean():7.2f}%  {np.percentile(g2i,95):6.2f}%  {g2i.max():6.2f}%")
    print(f"  {'경계':>8s}  {len(g1b):6d}  {np.median(g1b):6.2f}%  {g1b.mean():7.2f}%  {np.percentile(g1b,95):6.2f}%  {g1b.max():6.2f}%  {np.median(g2b):6.2f}%  {g2b.mean():7.2f}%  {np.percentile(g2b,95):6.2f}%  {g2b.max():6.2f}%")
    print(f"  {'전체':>8s}  {len(g1a):6d}  {np.median(g1a):6.2f}%  {g1a.mean():7.2f}%  {np.percentile(g1a,95):6.2f}%  {g1a.max():6.2f}%  {np.median(g2a):6.2f}%  {g2a.mean():7.2f}%  {np.percentile(g2a,95):6.2f}%  {g2a.max():6.2f}%")

    return {
        'inner_g1_med': np.median(g1i), 'inner_g2_med': np.median(g2i),
        'bnd_g1_med': np.median(g1b), 'bnd_g2_med': np.median(g2b),
        'all_g1_med': np.median(g1a), 'all_g2_med': np.median(g2a),
        'all_g1_max': g1a.max(), 'all_g2_max': g2a.max(),
    }


def main():
    print("=" * 78)
    print("XS검증_T002: MAS_NXS XS 기반 L_diffusion CMFD 잔차 (10LP)")
    print("비교: [A] MAS_XSL(BOC) vs [B] MAS_NXS(스텝별 feedback)")
    print("=" * 78)

    dx = WIDE

    # 두 방식의 결과를 각각 수집
    xsl_inner_g1, xsl_inner_g2 = [], []
    xsl_bnd_g1, xsl_bnd_g2 = [], []
    nxs_inner_g1, nxs_inner_g2 = [], []
    nxs_bnd_g1, nxs_bnd_g2 = [], []

    for lp_id, profile in TEST_LPS:
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            print(f"  SKIP {lp_id}")
            continue

        first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
        with open(first_out, "r", encoding="utf-8", errors="replace") as f:
            geom = CoreGeometry.from_lines(f.readlines())
        dz_arr = np.array([geom.zmesh[k] for k in range(1, 21)], dtype=np.float64)

        # XSL XS (LP 고정)
        xs_xsl = build_xs_voxel(geom, data_dir / "MAS_XSL",
                                np.ones((5, 5), dtype=bool)).astype(np.float64)
        D_xsl_g1 = 1.0 / (3.0 * xs_xsl[..., 3] + 1e-30)
        D_xsl_g2 = 1.0 / (3.0 * xs_xsl[..., 8] + 1e-30)
        Sr_xsl_g1 = xs_xsl[..., 2] + xs_xsl[..., 1] + xs_xsl[..., 4]
        Sa_xsl_g2 = xs_xsl[..., 7] + xs_xsl[..., 6]

        for s in range(1, 11):
            out_path = data_dir / f"MAS_OUT_{profile}_s{s:04d}_crs"
            nxs_path = data_dir / f"MAS_NXS_{profile}_s{s:04d}_crs"
            if not out_path.exists() or not nxs_path.exists():
                continue

            out_data = MasOutParser.parse(out_path)
            keff = out_data.keff
            phi_q_g1 = out_data.flux_3d[:, 4:, 4:, 0].astype(np.float64)
            phi_q_g2 = out_data.flux_3d[:, 4:, 4:, 1].astype(np.float64)

            # [A] XSL 기반 잔차
            ig1, ig2, bg1, bg2 = compute_residuals(
                phi_q_g1, phi_q_g2, D_xsl_g1, D_xsl_g2,
                Sr_xsl_g1, Sa_xsl_g2,
                xs_xsl[..., 0], xs_xsl[..., 5], xs_xsl[..., 4],
                dz_arr, keff, dx)
            xsl_inner_g1.extend(ig1); xsl_inner_g2.extend(ig2)
            xsl_bnd_g1.extend(bg1); xsl_bnd_g2.extend(bg2)

            # [B] NXS 기반 잔차
            nxs = parse_nxs_diffusion_xs(nxs_path)
            nxs_q = {t: nxs[t][:, 4:, 4:, :] for t in nxs}

            D_nxs_g1 = nxs_q['DIF'][:, :, :, 0]
            D_nxs_g2 = nxs_q['DIF'][:, :, :, 1]
            Sr_nxs_g1 = nxs_q['ABS'][:, :, :, 0] + nxs_q['SCA'][:, :, :, 0]
            Sa_nxs_g2 = nxs_q['ABS'][:, :, :, 1]
            nuSf_nxs_g1 = nxs_q['NFS'][:, :, :, 0]
            nuSf_nxs_g2 = nxs_q['NFS'][:, :, :, 1]
            Ss12_nxs = nxs_q['SCA'][:, :, :, 0]

            ig1, ig2, bg1, bg2 = compute_residuals(
                phi_q_g1, phi_q_g2, D_nxs_g1, D_nxs_g2,
                Sr_nxs_g1, Sa_nxs_g2,
                nuSf_nxs_g1, nuSf_nxs_g2, Ss12_nxs,
                dz_arr, keff, dx)
            nxs_inner_g1.extend(ig1); nxs_inner_g2.extend(ig2)
            nxs_bnd_g1.extend(bg1); nxs_bnd_g2.extend(bg2)

            del out_data, nxs, nxs_q
            gc.collect()

        gc.collect()
        print(f"  {lp_id} done")

    # 결과 출력
    print(f"\n{'='*78}")
    print("결과 비교")
    print("=" * 78)

    s_xsl = print_stats("A: MAS_XSL (BOC 고정)", xsl_inner_g1, xsl_inner_g2, xsl_bnd_g1, xsl_bnd_g2)
    s_nxs = print_stats("B: MAS_NXS (스텝별 feedback)", nxs_inner_g1, nxs_inner_g2, nxs_bnd_g1, nxs_bnd_g2)

    print(f"\n{'='*78}")
    print("개선율 (XSL → NXS)")
    print("=" * 78)
    for key, label in [
        ('all_g1_med', '전체 g1 median'),
        ('all_g2_med', '전체 g2 median'),
        ('all_g1_max', '전체 g1 max'),
        ('all_g2_max', '전체 g2 max'),
        ('inner_g1_med', '내부 g1 median'),
        ('inner_g2_med', '내부 g2 median'),
        ('bnd_g1_med', '경계 g1 median'),
        ('bnd_g2_med', '경계 g2 median'),
    ]:
        xv = s_xsl[key]
        nv = s_nxs[key]
        imp = (xv - nv) / xv * 100 if xv != 0 else 0
        print(f"  {label:>20s}: {xv:7.2f}% → {nv:7.2f}%  ({imp:+.1f}%)")

    print(f"\n  [B]의 잔차 = 순수 CMFD(D-hat) 한계 (XS 불일치 제거 후)")


if __name__ == "__main__":
    main()
