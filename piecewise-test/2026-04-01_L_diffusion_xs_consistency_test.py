"""
L_diffusion XS 일관성 검증: MAS_XSL(BOC 고정) vs MAS_NXS(스텝별).

목적: 리뷰어 지적 — "Source-Removal 불균형이 크다면 XS 불일치 의심"
      MAS_NXS의 스텝별 XS와 MAS_XSL의 BOC XS를 비교하여
      잔차의 원인이 XS 불일치인지, CMFD 한계인지 규명.

MAS_NXS 필드:
  g1: DIF ABS SCA FIS NFS KFS ABS-B10 ABS-XEN ABS-SAM ADF FLX JNET0-*
  g2: DIF ABS SCA FIS NFS KFS ABS-B10 ABS-XEN ABS-SAM ADF FLX JNET0-*

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
WIDE = 21.60780

_NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')


def parse_nxs_xs(nxs_path, n_fuel_rows=9, n_fuel_cols=9):
    """MAS_NXS에서 DIF, ABS, SCA, NFS, FLX 추출 (노드→어셈블리 평균).

    Returns:
        dict with keys 'DIF', 'ABS', 'SCA', 'NFS', 'FLX'
        각 shape: (20, 9, 9, 2) — [z, j_asm, i_asm, group]
    """
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    nx_nxy_nz = [float(x) for x in _NUM_RE.findall(lines[1])]
    nx = int(nx_nxy_nz[0])
    nxy = int(nx_nxy_nz[1])
    nz = int(nx_nxy_nz[2])
    ny = nx

    # Header parsing
    header_tokens = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(header_tokens)}

    # Target columns
    targets = ['DIF', 'ABS', 'SCA', 'NFS', 'FLX']
    g1_idx = {t: col_map[t] for t in targets}
    g2_offset = col_map['DIF']
    g2_idx = {t: g1_idx[t] - g2_offset for t in targets}

    g1_i = col_map['I']
    g1_j = col_map['J']
    g1_k = col_map['K']

    # Assembly-level arrays (20, 9, 9) — accumulate then average
    node_data = {t: np.zeros((20, 9, 9, 2), dtype=np.float64) for t in targets}
    count = np.zeros((20, 9, 9), dtype=np.float64)

    i_line = 8
    n_nodes = nxy * nz

    for _ in range(n_nodes):
        if i_line + 1 >= len(lines):
            break
        g1_vals = [float(x) for x in _NUM_RE.findall(lines[i_line])]
        g2_vals = [float(x) for x in _NUM_RE.findall(lines[i_line + 1])]
        i_line += 2

        if len(g1_vals) < g1_idx['FLX'] + 1:
            continue

        ni = int(g1_vals[g1_i]) - 1
        nj = int(g1_vals[g1_j]) - 1
        nk = int(g1_vals[g1_k])

        if nk < 2 or nk > 21:
            continue
        z_idx = nk - 2

        # Fuel region only (skip reflector nodes)
        n_asm = nx // 2  # 22//2 = 11
        ndivxy = 2
        ai = ni // ndivxy
        aj = nj // ndivxy
        if ai >= 11 or aj >= 11:
            continue

        # Map to 9x9 fuel grid (skip reflector assemblies)
        # In 11x11, fuel rows/cols are 1~9 (0-indexed)
        fi = ai - 1
        fj = aj - 1
        if fi < 0 or fi >= 9 or fj < 0 or fj >= 9:
            continue

        for t in targets:
            g1v = g1_vals[g1_idx[t]] if g1_idx[t] < len(g1_vals) else 0.0
            g2v = g2_vals[g2_idx[t]] if g2_idx[t] < len(g2_vals) else 0.0
            node_data[t][z_idx, fj, fi, 0] += g1v
            node_data[t][z_idx, fj, fi, 1] += g2v
        count[z_idx, fj, fi] += 1

    # Divide by count
    count = np.maximum(count, 1)
    for t in targets:
        node_data[t][..., 0] /= count
        node_data[t][..., 1] /= count

    return node_data


def main():
    print("=" * 78)
    print("L_diffusion XS 일관성 검증: MAS_XSL vs MAS_NXS")
    print("=" * 78)

    lp_id, profile = "LP_0000", "t12_363_p50_power_lower"
    data_dir = WORKSPACE / lp_id / profile

    # 1. MAS_XSL XS 로드
    first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
    with open(first_out, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    geom = CoreGeometry.from_lines(lines)
    xs_xsl = build_xs_voxel(geom, data_dir / "MAS_XSL",
                            np.ones((5, 5), dtype=bool)).astype(np.float64)
    # xs_xsl: (20, 5, 5, 10) — quarter
    # Channels: [nuSf1, Sf1, Sc1, Str1, Ss12, nuSf2, Sf2, Sc2, Str2, pad]

    # 2. MAS_NXS XS 로드 (step 1)
    nxs_path = data_dir / f"MAS_NXS_{profile}_s0001_crs"
    print(f"\n  MAS_NXS: {nxs_path}")
    print(f"  MAS_XSL: {data_dir / 'MAS_XSL'}")

    nxs_data = parse_nxs_xs(nxs_path)
    # nxs_data: dict with shape (20, 9, 9, 2) each

    # Quarter crop (j=4:, i=4:)
    nxs_q = {t: nxs_data[t][:, 4:, 4:, :] for t in nxs_data}
    # shape: (20, 5, 5, 2)

    # 3. 비교: XSL vs NXS
    print(f"\n{'='*78}")
    print("1. XS 채널별 비교 (quarter 전체 평균)")
    print("=" * 78)

    # XSL → NXS 매핑
    # XSL[3] = Str1, NXS DIF g1 = 1/(3*Str1) → Str1 = 1/(3*DIF)
    # XSL[2] = Sc1, XSL[1] = Sf1 → ABS = Sc1 + Sf1
    # XSL[4] = Ss12 → SCA ≈ Ss12 (in 2-group, SCA = downscatter)
    # XSL[0] = nuSf1 → NFS

    # Derived from XSL
    D_xsl_g1 = 1.0 / (3.0 * xs_xsl[..., 3] + 1e-30)  # (20,5,5)
    D_xsl_g2 = 1.0 / (3.0 * xs_xsl[..., 8] + 1e-30)
    ABS_xsl_g1 = xs_xsl[..., 2] + xs_xsl[..., 1]  # Sc1 + Sf1 = Sigma_a1
    ABS_xsl_g2 = xs_xsl[..., 7] + xs_xsl[..., 6]  # Sc2 + Sf2 = Sigma_a2
    NFS_xsl_g1 = xs_xsl[..., 0]  # nuSf1
    NFS_xsl_g2 = xs_xsl[..., 5]  # nuSf2
    SCA_xsl_g1 = xs_xsl[..., 4]  # Ss12

    # From NXS (already assembly-averaged)
    D_nxs_g1 = nxs_q['DIF'][:, :, :, 0]
    D_nxs_g2 = nxs_q['DIF'][:, :, :, 1]
    ABS_nxs_g1 = nxs_q['ABS'][:, :, :, 0]
    ABS_nxs_g2 = nxs_q['ABS'][:, :, :, 1]
    NFS_nxs_g1 = nxs_q['NFS'][:, :, :, 0]
    NFS_nxs_g2 = nxs_q['NFS'][:, :, :, 1]
    SCA_nxs_g1 = nxs_q['SCA'][:, :, :, 0]

    # 연료 마스크 (non-zero flux)
    fuel_mask = D_nxs_g1 > 0

    comparisons = [
        ("DIF g1", D_xsl_g1, D_nxs_g1),
        ("DIF g2", D_xsl_g2, D_nxs_g2),
        ("ABS g1", ABS_xsl_g1, ABS_nxs_g1),
        ("ABS g2", ABS_xsl_g2, ABS_nxs_g2),
        ("NFS g1", NFS_xsl_g1, NFS_nxs_g1),
        ("NFS g2", NFS_xsl_g2, NFS_nxs_g2),
        ("SCA g1", SCA_xsl_g1, SCA_nxs_g1),
    ]

    print(f"\n  {'channel':>10s}  {'XSL mean':>12s}  {'NXS mean':>12s}  {'diff%':>8s}  {'max_diff%':>10s}")
    print(f"  {'='*56}")
    for name, xsl, nxs in comparisons:
        mask = fuel_mask & (np.abs(xsl) > 1e-30)
        if mask.sum() == 0:
            continue
        diff = np.abs(xsl[mask] - nxs[mask]) / np.abs(xsl[mask]) * 100
        print(f"  {name:>10s}  {xsl[mask].mean():12.5e}  {nxs[mask].mean():12.5e}  "
              f"{diff.mean():7.3f}%  {diff.max():9.3f}%")

    # 4. 생성-소멸 밸런스 비교
    print(f"\n{'='*78}")
    print("2. 생성-소멸 밸런스: XSL vs NXS (누설 제외)")
    print("=" * 78)

    from lf_preprocess.mas_out_parser import MasOutParser
    out = MasOutParser.parse(data_dir / f"MAS_OUT_{profile}_s0001_crs")
    phi_q_g1 = out.flux_3d[:, 4:, 4:, 0].astype(np.float64)
    phi_q_g2 = out.flux_3d[:, 4:, 4:, 1].astype(np.float64)
    keff = out.keff

    dx = dy = WIDE
    dz_arr = np.array([geom.zmesh[k] for k in range(1, 21)], dtype=np.float64)

    # Sr_g1 = ABS_g1 + SCA_g1 (removal = absorption + downscatter)
    Sr_xsl_g1 = ABS_xsl_g1 + SCA_xsl_g1
    Sr_nxs_g1 = ABS_nxs_g1 + SCA_nxs_g1

    # For each node: Source and Removal
    net_xsl = np.full((20, 5, 5), np.nan)
    net_nxs = np.full((20, 5, 5), np.nan)

    for z in range(20):
        dz = dz_arr[z]
        V = dx * dy * dz
        for qy in range(5):
            for qx in range(5):
                p1 = phi_q_g1[z, qy, qx]
                p2 = phi_q_g2[z, qy, qx]
                if p1 < 1e-10:
                    continue

                # XSL
                src_xsl = (1.0 / keff) * (NFS_xsl_g1[z, qy, qx] * p1 +
                                            NFS_xsl_g2[z, qy, qx] * p2) * V
                rem_xsl = Sr_xsl_g1[z, qy, qx] * p1 * V
                net_xsl[z, qy, qx] = (src_xsl - rem_xsl) / rem_xsl * 100

                # NXS
                src_nxs = (1.0 / keff) * (NFS_nxs_g1[z, qy, qx] * p1 +
                                            NFS_nxs_g2[z, qy, qx] * p2) * V
                rem_nxs = Sr_nxs_g1[z, qy, qx] * p1 * V
                net_nxs[z, qy, qx] = (src_nxs - rem_nxs) / rem_nxs * 100

    valid = np.isfinite(net_xsl) & np.isfinite(net_nxs)
    diff_net = np.abs(net_xsl[valid] - net_nxs[valid])

    print(f"\n  (Source - Removal) / Removal × 100% 통계:")
    print(f"  {'':>12s}  {'median':>8s}  {'mean':>8s}  {'max':>8s}")
    print(f"  {'XSL (BOC)':>12s}  {np.nanmedian(net_xsl[valid]):7.2f}%  {np.nanmean(net_xsl[valid]):7.2f}%  {np.nanmax(np.abs(net_xsl[valid])):7.2f}%")
    print(f"  {'NXS (step)':>12s}  {np.nanmedian(net_nxs[valid]):7.2f}%  {np.nanmean(net_nxs[valid]):7.2f}%  {np.nanmax(np.abs(net_nxs[valid])):7.2f}%")
    print(f"  {'|XSL-NXS|':>12s}  {np.median(diff_net):7.2f}%  {np.mean(diff_net):7.2f}%  {np.max(diff_net):7.2f}%")

    # 5. Outlier 노드 상세
    print(f"\n{'='*78}")
    print("3. Outlier 노드 상세: (z=4, qy=2, qx=0)")
    print("=" * 78)

    z, qy, qx = 4, 2, 0
    p1 = phi_q_g1[z, qy, qx]
    p2 = phi_q_g2[z, qy, qx]
    dz = dz_arr[z]
    V = dx * dy * dz

    print(f"\n  phi_g1 = {p1:.4e}, phi_g2 = {p2:.4e}, keff = {keff:.6f}")
    print(f"  V = {V:.1f} cm3")

    for label, D1, ABS1, SCA1, NFS1, NFS2 in [
        ("XSL", D_xsl_g1[z,qy,qx], ABS_xsl_g1[z,qy,qx], SCA_xsl_g1[z,qy,qx], NFS_xsl_g1[z,qy,qx], NFS_xsl_g2[z,qy,qx]),
        ("NXS", D_nxs_g1[z,qy,qx], ABS_nxs_g1[z,qy,qx], SCA_nxs_g1[z,qy,qx], NFS_nxs_g1[z,qy,qx], NFS_nxs_g2[z,qy,qx]),
    ]:
        Sr = ABS1 + SCA1
        src = (1.0 / keff) * (NFS1 * p1 + NFS2 * p2) * V
        rem = Sr * p1 * V
        net = src - rem
        print(f"\n  [{label}]")
        print(f"    D_g1     = {D1:.5e}")
        print(f"    ABS_g1   = {ABS1:.5e}")
        print(f"    SCA_g1   = {SCA1:.5e}")
        print(f"    Sr_g1    = {Sr:.5e}")
        print(f"    NFS_g1   = {NFS1:.5e}")
        print(f"    NFS_g2   = {NFS2:.5e}")
        print(f"    Source   = {src:.4e}")
        print(f"    Removal  = {rem:.4e}")
        print(f"    Net(S-R) = {net:+.4e}  ({net/rem*100:+.2f}% of Removal)")

    print(f"\n{'='*78}")
    print("결론")
    print("=" * 78)
    if np.mean(diff_net) > 1.0:
        print("  XSL과 NXS의 생성-소멸 밸런스가 유의미하게 다름 (>1%)")
        print("  -> XS 불일치가 잔차의 주요 원인일 가능성 높음")
        print("  -> MAS_NXS의 스텝별 XS 사용 시 잔차 개선 기대")
    else:
        print("  XSL과 NXS의 생성-소멸 밸런스 차이 미미 (<1%)")
        print("  -> XS 불일치는 잔차의 주 원인 아님")
        print("  -> CMFD(D-hat 부재)가 본질적 한계")


if __name__ == "__main__":
    main()
