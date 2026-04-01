"""
L_diffusion 잔차 비교: 노드 단위(MAS_NXS) vs 집합체 단위(xs_fuel)

목적:
  격자 세분화(10.8cm vs 21.6cm)가 CMFD 잔차를 얼마나 줄이는지 정량 비교.
  → 집합체 단위 L_diffusion physical loss 전략 결정 근거.

A. 노드 단위 CMFD (18×18, MAS_NXS 기반):
  - DIF, ABS, SCA, NFS, FLX 직접 사용
  - dx = dy = WIDE/2 = 10.804 cm
  - Σ_r1 = ABS + SCA, Σ_a2 = ABS (Step 0 확정)

B. 집합체 단위 CMFD (9×9, xs_fuel 기반):
  - xs_fuel 10채널: [νΣf₁, Σf₁, Σc₁, Σtr₁, Σs₁₂, νΣf₂, Σf₂, Σc₂, Σtr₂, 0]
  - D_g = 1/(3×Σ_tr)
  - MAS_OUT flux_3d
  - dx = dy = 21.60780 cm (피치 수정)

작성일: 2026-03-31
"""

import re
import sys
import os
from pathlib import Path

import numpy as np

# ─── 경로 설정 ───
VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))

from lf_preprocess.mas_out_parser import MasOutParser
from lf_preprocess.core_geometry import CoreGeometry

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780  # 어셈블리 피치 [cm] (Step 0 확정)

TEST_LPS = [
    ("LP_0000", "t12_363_p50_power_lower"),
    ("LP_0001", "t12_363_p50_power_upper"),
]
CRS_STEPS = list(range(1, 11))

NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════
# MAS_NXS 파싱 (노드 단위 DIF, ABS, SCA, NFS, FLX)
# ═══════════════════════════════════════════════════════════

def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_diffusion(nxs_path):
    """MAS_NXS에서 노드 단위 확산 물리량 추출.

    Returns:
        dict with keys: dif, abs_xs, sca, nfs, flx (각 ndarray),
        wide, zmesh, ndivxy, dx
    """
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    nx = int(parse_nums(lines[1])[0])
    nxy = int(parse_nums(lines[1])[1])
    nz = int(parse_nums(lines[1])[2])
    ny = nx
    wide = parse_nums(lines[3])[0]
    zmesh = np.array(parse_nums(lines[4]), dtype=np.float64)

    cols = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(cols)}
    g2_offset = col_map['DIF']

    # g1 인덱스
    ci = col_map
    g1_dif, g1_abs, g1_sca = ci['DIF'], ci['ABS'], ci['SCA']
    g1_nfs, g1_flx = ci['NFS'], ci['FLX']

    nf = 20  # K=2..21
    z = np.float64
    dif = np.zeros((nf, ny, nx, 2), dtype=z)
    abs_xs = np.zeros((nf, ny, nx, 2), dtype=z)
    sca = np.zeros((nf, ny, nx), dtype=z)
    nfs = np.zeros((nf, ny, nx, 2), dtype=z)
    flx = np.zeros((nf, ny, nx, 2), dtype=z)

    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = parse_nums(lines[i_line])
        g2 = parse_nums(lines[i_line + 1])
        i_line += 2

        if len(g1) < ci['JNET0-T'] + 1:
            continue

        i_nd = int(g1[0]) - 1
        j_nd = int(g1[1]) - 1
        k_nd = int(g1[2])

        if k_nd < 2 or k_nd > 21:
            continue
        if i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= ny:
            continue

        pi = k_nd - 2

        dif[pi, j_nd, i_nd, 0] = g1[g1_dif]
        abs_xs[pi, j_nd, i_nd, 0] = g1[g1_abs]
        sca[pi, j_nd, i_nd] = g1[g1_sca]
        nfs[pi, j_nd, i_nd, 0] = g1[g1_nfs]
        flx[pi, j_nd, i_nd, 0] = g1[g1_flx]

        g2_jt = ci['JNET0-T'] - g2_offset
        if len(g2) > g2_jt:
            dif[pi, j_nd, i_nd, 1] = g2[ci['DIF'] - g2_offset]
            abs_xs[pi, j_nd, i_nd, 1] = g2[ci['ABS'] - g2_offset]
            nfs[pi, j_nd, i_nd, 1] = g2[ci['NFS'] - g2_offset]
            flx[pi, j_nd, i_nd, 1] = g2[ci['FLX'] - g2_offset]

    # 연료 영역 슬라이싱
    dif_sum = dif[:, :, :, 0].sum(axis=0)
    nz_j = np.where(dif_sum.sum(axis=1) > 0)[0]
    nz_i = np.where(dif_sum.sum(axis=0) > 0)[0]

    if len(nz_j) == 0:
        return None

    j0, i0 = nz_j[0], nz_i[0]
    fuel_ny = nz_j[-1] - j0 + 1
    fuel_nx = nz_i[-1] - i0 + 1
    ndivxy = fuel_ny // 9  # 18 / 9 = 2
    dx_node = wide / ndivxy

    return {
        'dif': dif[:, j0:j0+fuel_ny, i0:i0+fuel_nx, :],
        'abs_xs': abs_xs[:, j0:j0+fuel_ny, i0:i0+fuel_nx, :],
        'sca': sca[:, j0:j0+fuel_ny, i0:i0+fuel_nx],
        'nfs': nfs[:, j0:j0+fuel_ny, i0:i0+fuel_nx, :],
        'flx': flx[:, j0:j0+fuel_ny, i0:i0+fuel_nx, :],
        'zmesh': zmesh, 'ndivxy': ndivxy, 'dx': dx_node,
        'fuel_ny': fuel_ny, 'fuel_nx': fuel_nx,
    }


# ═══════════════════════════════════════════════════════════
# CMFD 잔차 계산 (공통 로직)
# ═══════════════════════════════════════════════════════════

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def compute_cmfd_residual(phi_g1, phi_g2, D_g1, D_g2, Sr_g1, Sa_g2,
                          nuSf_g1, nuSf_g2, Ss_12, keff, dz_arr, dx, dy):
    """CMFD 체적 적분 잔차 (inner-only). 범용: 노드/집합체 공용."""
    Az = dx * dy

    inner = (slice(1, -1), slice(1, -1), slice(1, -1))

    def _leak(phi, D):
        Zi = phi.shape[0] - 2
        Hi = phi.shape[1] - 2
        Wi = phi.shape[2] - 2
        leak = np.zeros((Zi, Hi, Wi), dtype=np.float64)

        for ki in range(Zi):
            k = ki + 1
            hz_p = 0.5 * (dz_arr[k] + dz_arr[k + 1])
            hz_m = 0.5 * (dz_arr[k - 1] + dz_arr[k])
            D_top = harmonic_mean(D[k, 1:-1, 1:-1], D[k+1, 1:-1, 1:-1])
            D_bot = harmonic_mean(D[k, 1:-1, 1:-1], D[k-1, 1:-1, 1:-1])
            J_top = D_top * (phi[k+1, 1:-1, 1:-1] - phi[k, 1:-1, 1:-1]) / hz_p
            J_bot = D_bot * (phi[k, 1:-1, 1:-1] - phi[k-1, 1:-1, 1:-1]) / hz_m
            leak[ki] += -(J_top - J_bot) * Az

        D_n = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 2:, 1:-1])
        D_s = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, :-2, 1:-1])
        J_n = D_n * (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
        J_s = D_s * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, :-2, 1:-1]) / dy
        for ki in range(Zi):
            Ay_k = dx * dz_arr[ki + 1]
            leak[ki] += -(J_n[ki] - J_s[ki]) * Ay_k

        D_e = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 1:-1, 2:])
        D_w = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 1:-1, :-2])
        J_e = D_e * (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dx
        J_w = D_w * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, 1:-1, :-2]) / dx
        for ki in range(Zi):
            Ax_k = dy * dz_arr[ki + 1]
            leak[ki] += -(J_e[ki] - J_w[ki]) * Ax_k

        return leak

    leak_g1 = _leak(phi_g1, D_g1)
    leak_g2 = _leak(phi_g2, D_g2)

    phi1_in = phi_g1[inner]
    phi2_in = phi_g2[inner]
    Zi = phi_g1.shape[0] - 2
    V = np.zeros_like(phi1_in)
    for ki in range(Zi):
        V[ki] = dx * dy * dz_arr[ki + 1]

    fission_src = nuSf_g1[inner] * phi1_in + nuSf_g2[inner] * phi2_in
    removal_g1_V = Sr_g1[inner] * phi1_in * V
    removal_g2_V = Sa_g2[inner] * phi2_in * V

    R_g1 = leak_g1 + removal_g1_V - (1.0 / keff) * fission_src * V
    R_g2 = leak_g2 + removal_g2_V - Ss_12[inner] * phi1_in * V

    # fuel mask: 자기 + 6면 이웃이 모두 연료인 노드만 평가
    # 연료 판정: nuSf_g1 > 0 (핵분열 있는 노드)
    is_fuel = nuSf_g1 > 1e-10  # (Z, Y, X)
    fuel_mask = (
        is_fuel[1:-1, 1:-1, 1:-1]       # 자기 자신
        & is_fuel[2:,   1:-1, 1:-1]      # top (+Z)
        & is_fuel[:-2,  1:-1, 1:-1]      # bottom (-Z)
        & is_fuel[1:-1, 2:,   1:-1]      # north (+Y)
        & is_fuel[1:-1, :-2,  1:-1]      # south (-Y)
        & is_fuel[1:-1, 1:-1, 2:]        # east (+X)
        & is_fuel[1:-1, 1:-1, :-2]       # west (-X)
    )

    return R_g1, R_g2, removal_g1_V, removal_g2_V, fuel_mask


def relative_residual(R_g, removal_V, fuel_mask=None):
    """상대잔차 계산. fuel_mask로 비연료 노드 제외."""
    scale = np.abs(removal_V)
    valid = scale > 1e-10
    if fuel_mask is not None:
        valid = valid & fuel_mask
    rel = np.full_like(R_g, np.nan)
    rel[valid] = np.abs(R_g[valid]) / scale[valid] * 100
    return rel


def print_fuel_map(nuSf_g1, label, dz_arr=None):
    """연료 영역 XY맵을 텍스트로 시각화 (중간 Z 평면 기준).

    ■ = 6면 모두 연료 (평가 대상)
    □ = 연료이나 이웃에 비연료 있음 (평가 제외)
    · = 비연료
    """
    Z, H, W = nuSf_g1.shape
    mid_z = Z // 2
    is_fuel = nuSf_g1 > 1e-10

    # 6면 이웃 모두 연료인 마스크 (inner)
    full_fuel = (
        is_fuel[1:-1, 1:-1, 1:-1]
        & is_fuel[2:,   1:-1, 1:-1]
        & is_fuel[:-2,  1:-1, 1:-1]
        & is_fuel[1:-1, 2:,   1:-1]
        & is_fuel[1:-1, :-2,  1:-1]
        & is_fuel[1:-1, 1:-1, 2:]
        & is_fuel[1:-1, 1:-1, :-2]
    )

    print(f"\n  [{label}] 연료 맵 (Z={mid_z}, {H}×{W})")
    print(f"  ■ = 6면 모두 연료(평가대상), □ = 연료(경계), · = 비연료")

    # XY 맵 출력 (mid_z에 해당하는 inner index = mid_z - 1)
    inner_z = mid_z - 1 if mid_z > 0 else 0
    for j in range(H):
        row = "    "
        for i in range(W):
            is_f = is_fuel[mid_z, j, i]
            is_inner = (1 <= j < H - 1) and (1 <= i < W - 1)
            is_full = False
            if is_inner and 0 <= inner_z < full_fuel.shape[0]:
                is_full = full_fuel[inner_z, j - 1, i - 1]
            if is_full:
                row += "■ "
            elif is_f:
                row += "□ "
            else:
                row += "· "
        print(row)

    # 통계
    n_fuel_xy = is_fuel[mid_z].sum()
    n_eval = full_fuel[inner_z].sum() if inner_z < full_fuel.shape[0] else 0
    n_eval_total = full_fuel.sum()
    print(f"  Z={mid_z} 연료: {n_fuel_xy}, 평가대상(■): {n_eval}")
    print(f"  전체 평가대상 노드: {n_eval_total} (Z×Y×X)")

    # Z축 평가 노드 분포
    z_counts = [full_fuel[z].sum() for z in range(full_fuel.shape[0])]
    print(f"  Z축 평가 노드 수: {z_counts[:3]}...{z_counts[-3:]}")


# ═══════════════════════════════════════════════════════════
# 집합체 단위: xs_fuel 로드
# ═══════════════════════════════════════════════════════════

def load_xs_fuel(geom, data_dir):
    from lf_preprocess.xs_voxel_builder import build_xs_voxel
    fuel_mask_quarter = np.ones((5, 5), dtype=bool)
    xsl_path = Path(data_dir) / "MAS_XSL"
    return build_xs_voxel(geom, xsl_path, fuel_mask_quarter)


def expand_quarter_to_full(xs_q):
    """quarter (20,5,5,C) → full (20,9,9,C) 대칭 복원."""
    Z, _, _, C = xs_q.shape
    full = np.zeros((Z, 9, 9, C), dtype=xs_q.dtype)
    full[:, 4:, 4:, :] = xs_q
    full[:, 4:, :5, :] = xs_q[:, :, ::-1, :]
    full[:, :5, 4:, :] = xs_q[:, ::-1, :, :]
    full[:, :5, :5, :] = xs_q[:, ::-1, ::-1, :]
    return full


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def print_stats(label, g1, g2):
    if len(g1) == 0:
        print(f"  [{label}] 데이터 없음")
        return
    print(f"  [{label}] 유효 노드: {len(g1)}")
    print(f"  {'':20} {'g=1':>10}  {'g=2':>10}")
    print(f"  {'mean':20} {g1.mean():>8.3f}%  {g2.mean():>8.3f}%")
    print(f"  {'median':20} {np.median(g1):>8.3f}%  {np.median(g2):>8.3f}%")
    print(f"  {'p95':20} {np.percentile(g1,95):>8.3f}%  {np.percentile(g2,95):>8.3f}%")
    print(f"  {'max':20} {g1.max():>8.3f}%  {g2.max():>8.3f}%")


def main():
    print("=" * 72)
    print("L_diffusion 잔차 비교: 노드(MAS_NXS) vs 집합체(xs_fuel)")
    print("=" * 72)

    grand_node_g1, grand_node_g2 = [], []
    grand_asm_g1, grand_asm_g2 = [], []

    for lp_id, profile in TEST_LPS:
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            print(f"\n[SKIP] {lp_id}/{profile}")
            continue

        print(f"\n{'='*72}")
        print(f"  {lp_id} / {profile}")
        print(f"{'='*72}")

        # Geometry + xs_fuel (집합체용)
        first_out = data_dir / f"MAS_OUT_{profile}_s0001_crs"
        with open(first_out, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        geom = CoreGeometry.from_lines(lines)
        xs_fuel = load_xs_fuel(geom, data_dir).astype(np.float64)

        # xs_fuel: quarter (20,5,5,10) → full (20,9,9,10)
        xs_full = expand_quarter_to_full(xs_fuel)

        # 집합체 XS 분해
        asm_D_g1 = 1.0 / (3.0 * xs_full[..., 3] + 1e-30)
        asm_D_g2 = 1.0 / (3.0 * xs_full[..., 8] + 1e-30)
        asm_Sr_g1 = (xs_full[..., 2] + xs_full[..., 1]) + xs_full[..., 4]
        asm_Sa_g2 = xs_full[..., 7] + xs_full[..., 6]
        asm_nuSf_g1 = xs_full[..., 0]
        asm_nuSf_g2 = xs_full[..., 5]
        asm_Ss12 = xs_full[..., 4]

        # dz
        fuel_k_min, fuel_k_max = 2, 21
        dz_arr = np.array([geom.zmesh[k] for k in range(fuel_k_min, fuel_k_max + 1)],
                          dtype=np.float64)

        for s in CRS_STEPS:
            suffix = f"s{s:04d}_crs"
            out_path = data_dir / f"MAS_OUT_{profile}_{suffix}"
            nxs_path = data_dir / f"MAS_NXS_{profile}_{suffix}"

            if not out_path.exists() or not nxs_path.exists():
                continue

            # keff
            out_data = MasOutParser.parse(out_path)
            keff = out_data.keff
            if keff <= 0:
                continue

            # ── A. 노드 단위 CMFD ──
            nxs = parse_nxs_diffusion(nxs_path)
            if nxs is None:
                continue

            dx_node = nxs['dx']
            node_D_g1 = nxs['dif'][:, :, :, 0]
            node_D_g2 = nxs['dif'][:, :, :, 1]
            node_Sr_g1 = nxs['abs_xs'][:, :, :, 0] + nxs['sca']
            node_Sa_g2 = nxs['abs_xs'][:, :, :, 1]
            node_nuSf_g1 = nxs['nfs'][:, :, :, 0]
            node_nuSf_g2 = nxs['nfs'][:, :, :, 1]
            node_Ss12 = nxs['sca']
            node_phi_g1 = nxs['flx'][:, :, :, 0]
            node_phi_g2 = nxs['flx'][:, :, :, 1]

            Rn1, Rn2, remn1, remn2, fuel_mask_node = compute_cmfd_residual(
                node_phi_g1, node_phi_g2, node_D_g1, node_D_g2,
                node_Sr_g1, node_Sa_g2, node_nuSf_g1, node_nuSf_g2,
                node_Ss12, keff, dz_arr, dx_node, dx_node)

            # 첫 스텝에서 fuel map 시각화
            if s == 1 and lp_id == "LP_0000":
                print_fuel_map(node_nuSf_g1, f"노드 {node_nuSf_g1.shape[1]}×{node_nuSf_g1.shape[2]}", dz_arr)

            rn1 = relative_residual(Rn1, remn1, fuel_mask_node)
            rn2 = relative_residual(Rn2, remn2, fuel_mask_node)
            vn1 = rn1[np.isfinite(rn1)]
            vn2 = rn2[np.isfinite(rn2)]

            # ── B. 집합체 단위 CMFD (fullcore 9×9) ──
            phi_full = out_data.flux_3d  # (20, 9, 9, 2)
            asm_phi_g1 = phi_full[:, :, :, 0].astype(np.float64)
            asm_phi_g2 = phi_full[:, :, :, 1].astype(np.float64)

            Ra1, Ra2, rema1, rema2, fuel_mask_asm = compute_cmfd_residual(
                asm_phi_g1, asm_phi_g2, asm_D_g1, asm_D_g2,
                asm_Sr_g1, asm_Sa_g2, asm_nuSf_g1, asm_nuSf_g2,
                asm_Ss12, keff, dz_arr, WIDE, WIDE)

            # 첫 스텝에서 어셈블리 fuel map 시각화
            if s == 1 and lp_id == "LP_0000":
                print_fuel_map(asm_nuSf_g1, "집합체 9×9", dz_arr)

            ra1 = relative_residual(Ra1, rema1, fuel_mask_asm)
            ra2 = relative_residual(Ra2, rema2, fuel_mask_asm)
            va1 = ra1[np.isfinite(ra1)]
            va2 = ra2[np.isfinite(ra2)]

            if len(vn1) > 0 and len(va1) > 0:
                print(f"  {suffix}  "
                      f"노드: g1={np.median(vn1):6.2f}% g2={np.median(vn2):6.2f}%  "
                      f"집합체: g1={np.median(va1):6.2f}% g2={np.median(va2):6.2f}%  "
                      f"keff={keff:.5f}")
                grand_node_g1.append(vn1)
                grand_node_g2.append(vn2)
                grand_asm_g1.append(va1)
                grand_asm_g2.append(va2)

    # ── 종합 비교 ──
    print("\n" + "=" * 72)
    print("종합 비교 결과")
    print("=" * 72)

    if not grand_node_g1:
        print("  데이터 없음")
        return

    all_node_g1 = np.concatenate(grand_node_g1)
    all_node_g2 = np.concatenate(grand_node_g2)
    all_asm_g1 = np.concatenate(grand_asm_g1)
    all_asm_g2 = np.concatenate(grand_asm_g2)

    print(f"\n  A. 노드 단위 CMFD (MAS_NXS, dx={WIDE/2:.3f} cm)")
    print_stats("노드 CMFD", all_node_g1, all_node_g2)

    print(f"\n  B. 집합체 단위 CMFD (xs_fuel, dx={WIDE:.3f} cm)")
    print_stats("집합체 CMFD", all_asm_g1, all_asm_g2)

    # 비교 표
    print(f"\n  {'─'*60}")
    print(f"  비교 요약:")
    print(f"  {'메트릭':<12} {'노드(10.8cm)':>14} {'집합체(21.6cm)':>14} {'비율':>8}")
    print(f"  {'─'*60}")
    med_n1, med_n2 = np.median(all_node_g1), np.median(all_node_g2)
    med_a1, med_a2 = np.median(all_asm_g1), np.median(all_asm_g2)
    print(f"  {'g1 median':<12} {med_n1:>12.3f}% {med_a1:>12.3f}% {med_n1/med_a1:>7.2f}x")
    print(f"  {'g2 median':<12} {med_n2:>12.3f}% {med_a2:>12.3f}% {med_n2/med_a2:>7.2f}x")
    mean_n1, mean_n2 = all_node_g1.mean(), all_node_g2.mean()
    mean_a1, mean_a2 = all_asm_g1.mean(), all_asm_g2.mean()
    print(f"  {'g1 mean':<12} {mean_n1:>12.3f}% {mean_a1:>12.3f}% {mean_n1/mean_a1:>7.2f}x")
    print(f"  {'g2 mean':<12} {mean_n2:>12.3f}% {mean_a2:>12.3f}% {mean_n2/mean_a2:>7.2f}x")

    # 판정
    print(f"\n  O(h²) 스케일링 예측: 격자 2배 → 잔차 1/4 (0.25x)")
    print(f"  실측: g1 {med_n1/med_a1:.2f}x, g2 {med_n2/med_a2:.2f}x")

    # 결과 저장
    result_path = os.path.join(OUT_DIR,
                               "2026-03-31_diffusion_residual_comparison_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("L_diffusion 잔차 비교: 노드 vs 집합체\n")
        f.write("=" * 60 + "\n")
        f.write(f"노드: dx=dy={WIDE/2:.3f} cm (MAS_NXS)\n")
        f.write(f"집합체: dx=dy={WIDE:.3f} cm (xs_fuel, 피치 수정)\n\n")
        f.write(f"{'메트릭':<12} {'노드':>12} {'집합체':>12} {'비율':>8}\n")
        f.write(f"{'g1 median':<12} {med_n1:>10.3f}% {med_a1:>10.3f}% {med_n1/med_a1:>7.2f}x\n")
        f.write(f"{'g2 median':<12} {med_n2:>10.3f}% {med_a2:>10.3f}% {med_n2/med_a2:>7.2f}x\n")
        f.write(f"{'g1 mean':<12} {mean_n1:>10.3f}% {mean_a1:>10.3f}% {mean_n1/mean_a1:>7.2f}x\n")
        f.write(f"{'g2 mean':<12} {mean_n2:>10.3f}% {mean_a2:>10.3f}% {mean_n2/mean_a2:>7.2f}x\n")
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    main()
