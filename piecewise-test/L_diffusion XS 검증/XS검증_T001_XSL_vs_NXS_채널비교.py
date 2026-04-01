"""
XS검증_T001: MAS_XSL vs MAS_NXS 거시단면적 값 직접 비교.

목적: CMFD 계산 없이, 두 소스의 거시단면적(macroscopic XS) 값 자체를 비교.
      - MAS_XSL: BOC(초기) 고정 단면적 라이브러리
      - MAS_NXS: 각 CRS 스텝에서 MASTER가 실제 사용한 단면적 (Xe/Sm/온도 feedback 반영)

비교 대상 (CMFD에 사용되는 핵심 XS):
  DIF  = 확산계수 D [cm]
  ABS  = 흡수 거시단면적 Σ_a [/cm] = Σ_c + Σ_f
  SCA  = 산란 거시단면적 Σ_s12 [/cm] (g1→g2 downscatter)
  NFS  = nu-fission Σ νΣ_f [/cm]
  FIS  = 핵분열 거시단면적 Σ_f [/cm]

비교 방법: |XSL - NXS| / |XSL| × 100% (상대 차이율)

대상: LP_0000 step 1 (1개 시나리오, quarter-core 5×5)
      → 어셈블리 레벨 (ndivxy=2 서브노드 평균)

작성일: 2026-04-01
"""

import sys
import os
import re
from pathlib import Path
import numpy as np

VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))

from lf_preprocess.core_geometry import CoreGeometry
from lf_preprocess.xs_voxel_builder import build_xs_voxel

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
_NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')

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
IS_FUEL_9x9 = np.array([[not ASM_11[j+1][i+1].startswith('R') and ASM_11[j+1][i+1] != 'o'
                          for i in range(9)] for j in range(9)])


def parse_nxs_diffusion_xs(nxs_path):
    """MAS_NXS에서 DIF, ABS, SCA, NFS, FIS 추출 → assembly 평균 (20, 9, 9, 2).

    mas_nxs_parser.py의 파싱 패턴 재사용 (동적 헤더, node→assembly 집계).
    """
    with open(nxs_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    nums = [float(x) for x in _NUM_RE.findall(lines[1])]
    nx = int(nums[0])
    nxy = int(nums[1])
    nz = int(nums[2])
    ny = nx

    header = lines[7].split()
    col = {name: idx for idx, name in enumerate(header)}
    targets = ['DIF', 'ABS', 'SCA', 'FIS', 'NFS']
    g1_idx = {t: col[t] for t in targets}
    g2_off = col['DIF']
    g2_idx = {t: g1_idx[t] - g2_off for t in targets}
    ci, cj, ck = col['I'], col['J'], col['K']

    ndivxy = 2
    n_asm = nx // ndivxy  # 11
    nr = nc = 9
    nf = 20

    asm_sum = {t: np.zeros((nf, nr, nc, 2), dtype=np.float64) for t in targets}
    asm_cnt = np.zeros((nf, nr, nc), dtype=np.float64)

    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = [float(x) for x in _NUM_RE.findall(lines[i_line])]
        g2 = [float(x) for x in _NUM_RE.findall(lines[i_line + 1])]
        i_line += 2

        if len(g1) < g1_idx['NFS'] + 1:
            continue
        ni = int(g1[ci]) - 1
        nj = int(g1[cj]) - 1
        nk = int(g1[ck])
        if nk < 2 or nk > 21:
            continue
        pi = nk - 2
        ai = ni // ndivxy - 1
        aj = nj // ndivxy - 1
        if ai < 0 or ai >= nc or aj < 0 or aj >= nr:
            continue
        if not IS_FUEL_9x9[aj, ai]:
            continue

        for t in targets:
            asm_sum[t][pi, aj, ai, 0] += g1[g1_idx[t]]
            if g2_idx[t] < len(g2):
                asm_sum[t][pi, aj, ai, 1] += g2[g2_idx[t]]
        asm_cnt[pi, aj, ai] += 1

    asm_cnt = np.maximum(asm_cnt, 1)
    result = {}
    for t in targets:
        result[t] = asm_sum[t].copy()
        result[t][..., 0] /= asm_cnt
        result[t][..., 1] /= asm_cnt
    return result


def main():
    print("=" * 78)
    print("XS검증_T001: MAS_XSL vs MAS_NXS 채널별 비교")
    print("=" * 78)

    lp_id, profile = "LP_0000", "t12_363_p50_power_lower"
    data_dir = WORKSPACE / lp_id / profile

    # MAS_XSL 로드
    first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
    with open(first_out, "r", encoding="utf-8", errors="replace") as f:
        geom = CoreGeometry.from_lines(f.readlines())
    xs_xsl = build_xs_voxel(geom, data_dir / "MAS_XSL",
                            np.ones((5, 5), dtype=bool)).astype(np.float64)

    # MAS_NXS 로드
    nxs_path = data_dir / f"MAS_NXS_{profile}_s0001_crs"
    nxs = parse_nxs_diffusion_xs(nxs_path)

    # Quarter crop: 9x9 → [4:,4:]
    nxs_q = {t: nxs[t][:, 4:, 4:, :] for t in nxs}

    # XSL → NXS 매핑
    pairs = [
        ("DIF g1",   1.0 / (3.0 * xs_xsl[..., 3] + 1e-30), nxs_q['DIF'][..., 0]),
        ("DIF g2",   1.0 / (3.0 * xs_xsl[..., 8] + 1e-30), nxs_q['DIF'][..., 1]),
        ("ABS g1",   xs_xsl[..., 2] + xs_xsl[..., 1],       nxs_q['ABS'][..., 0]),
        ("ABS g2",   xs_xsl[..., 7] + xs_xsl[..., 6],       nxs_q['ABS'][..., 1]),
        ("NFS g1",   xs_xsl[..., 0],                         nxs_q['NFS'][..., 0]),
        ("NFS g2",   xs_xsl[..., 5],                         nxs_q['NFS'][..., 1]),
        ("SCA g1",   xs_xsl[..., 4],                         nxs_q['SCA'][..., 0]),
        ("FIS g1",   xs_xsl[..., 1],                         nxs_q['FIS'][..., 0]),
        ("FIS g2",   xs_xsl[..., 6],                         nxs_q['FIS'][..., 1]),
    ]

    # Fuel mask
    fuel_q = np.array([[IS_FUEL_9x9[4 + qy, 4 + qx] for qx in range(5)] for qy in range(5)])

    print(f"\n  대상: {lp_id}/{profile}, step 1")
    print(f"  XSL: {data_dir / 'MAS_XSL'}")
    print(f"  NXS: {nxs_path}")

    print(f"\n  {'channel':>10s}  {'XSL mean':>12s}  {'NXS mean':>12s}  {'mean_diff%':>10s}  {'max_diff%':>10s}  {'p95_diff%':>10s}")
    print(f"  {'='*68}")

    for name, xsl, nxs_v in pairs:
        mask = fuel_q[np.newaxis, :, :] & (np.abs(xsl) > 1e-30) & (np.abs(nxs_v) > 1e-30)
        mask_3d = np.broadcast_to(mask, xsl.shape) if mask.ndim < xsl.ndim else mask
        if mask_3d.sum() == 0:
            continue
        diff = np.abs(xsl[mask_3d] - nxs_v[mask_3d]) / np.abs(xsl[mask_3d]) * 100
        print(f"  {name:>10s}  {xsl[mask_3d].mean():12.5e}  {nxs_v[mask_3d].mean():12.5e}  "
              f"{diff.mean():9.3f}%  {diff.max():9.3f}%  {np.percentile(diff, 95):9.3f}%")

    # XY 위치별 NFS g2 차이 (가장 큰 차이를 보였던 채널)
    print(f"\n{'='*78}")
    print("NFS g2 (nuSigma_f thermal) — XY 위치별 차이 (z 평균)")
    print("=" * 78)

    nfs_xsl = xs_xsl[..., 5]  # (20, 5, 5)
    nfs_nxs = nxs_q['NFS'][:, :, :, 1]

    print(f"\n  {'':>6s}", end="")
    for qx in range(5):
        print(f"  qx={qx:d}    ", end="")
    print()

    for qy in range(5):
        row = f"  qy={qy}  "
        for qx in range(5):
            if not fuel_q[qy, qx]:
                row += f"{'---':>8s}  "
                continue
            xsl_v = nfs_xsl[:, qy, qx].mean()
            nxs_v = nfs_nxs[:, qy, qx].mean()
            diff = (nxs_v - xsl_v) / xsl_v * 100
            row += f"{diff:+7.2f}%  "
        asm_row = " ".join(ASM_11[5 + qy, 5 + qx] for qx in range(5))
        print(row + f"  ({asm_row})")

    print(f"\n  => 음수 = NXS < XSL (feedback으로 fission 감소)")
    print(f"  => Xe-135 흡수가 열중성자 핵분열을 억제하는 효과")


if __name__ == "__main__":
    main()
