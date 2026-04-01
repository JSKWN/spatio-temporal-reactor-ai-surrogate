"""
L_diffusion 사전 검증: MASTER GT phi에 유한차분 확산잔차 계산

목적:
  MASTER GT의 중성자속 phi에 7점 유한차분 스텐실을 적용하여
  2군 확산방정식 잔차 크기를 확인한다.
  → L_diffusion 도입 가능성 판단 (잔차 < 1%: PASS, 1~5%: MARGINAL, > 5%: FAIL)

대상:
  D:\workspace_lf_20260326_40LP\LP_0000\t12_363_p50_power_lower
  - MAS_NXS: phi(g1,g2), Sigma_f(g1,g2) — mas_nxs_parser.py
  - MAS_XSL: xs_fuel 10ch (Sigma_tr -> D_g) — xs_voxel_builder.py
  - MAS_OUT: keff, ZMESH — mas_out_parser.py, core_geometry.py

작성일: 2026-03-30
"""

import sys
import os
from pathlib import Path
import numpy as np

# v-smr_load_following 모듈 import
VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))

from lf_preprocess.mas_nxs_parser import parse_nxs
from lf_preprocess.mas_out_parser import MasOutParser
from lf_preprocess.core_geometry import CoreGeometry

# ─── 설정 ───
WORKSPACE = r"D:\workspace_lf_20260326_40LP"
ASSEMBLY_PITCH_CM = 21.504  # SMART 어셈블리 피치 [cm]

TEST_LPS = [
    ("LP_0000", "t12_363_p50_power_lower"),
    ("LP_0001", "t12_363_p50_power_upper"),
]
CRS_STEPS = list(range(1, 11))  # s0001~s0010
BRANCH_STEP = 1  # s0001에서 branch 테스트

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sep(char="=", n=72):
    print(char * n)


def load_mas_out(data_dir, profile, suffix):
    """MAS_OUT 파싱 (phi, keff)"""
    out_path = Path(data_dir) / f"MAS_OUT_{profile}_{suffix}"
    if not out_path.exists():
        return None
    parser = MasOutParser()
    return parser.parse(out_path)


def load_xs_fuel(geom, data_dir):
    """MAS_XSL에서 xs_fuel 10채널 구성 (quarter crop)"""
    from lf_preprocess.xs_voxel_builder import build_xs_voxel
    fuel_mask_quarter = np.ones((5, 5), dtype=bool)
    xsl_path = Path(data_dir) / "MAS_XSL"
    return build_xs_voxel(geom, xsl_path, fuel_mask_quarter)  # (20, 5, 5, 10)


def harmonic_mean(a, b):
    """계면 조화평균: 인접 노드간 유효 확산계수 (직렬 통과 원리)"""
    return 2.0 * a * b / (a + b + 1e-30)


def compute_cmfd_residual(phi_g1, phi_g2, xs_fuel, keff, dz_arr, dx, dy):
    """
    CMFD 체적 적분 형태 확산잔차 (inner-only).

    방법론 근거:
      Eq. 2.1-2a: 노드 체적 적분 균형식
      Eq. 2.1-37: CMFD 노드간 커플링 (D_tilde, 계면 조화평균)

    MASTER의 NNEM = CMFD(1단계) + NEM 보정(2단계) 반복.
    이 코드는 1단계(CMFD)만 재현. NEM 보정(D_hat)은 재현 불가 → 잔차 5~20% 예상.

    체적 적분 잔차:
      R = 누설(6면 순 중성자류 합 × 면적) + 소멸(Σ_r × φ × V) - 소스(핵분열+산란 × V)
      R = 0 이면 확산방정식 만족.

    inner-only: [1:-1, 1:-1, 1:-1] 내부 노드만 평가.
                경계 노드 제외 → 경계 조건(albedo) 불필요.

    반환: R_g1, R_g2 — inner 노드 잔차 (Z-2, H-2, W-2)
    """
    # ── 단면적 추출 ──
    nuSf_g1 = xs_fuel[..., 0].astype(np.float64)
    Sf_g1   = xs_fuel[..., 1].astype(np.float64)
    Sc_g1   = xs_fuel[..., 2].astype(np.float64)
    Str_g1  = xs_fuel[..., 3].astype(np.float64)
    Ss_12   = xs_fuel[..., 4].astype(np.float64)
    nuSf_g2 = xs_fuel[..., 5].astype(np.float64)
    Sf_g2   = xs_fuel[..., 6].astype(np.float64)
    Sc_g2   = xs_fuel[..., 7].astype(np.float64)
    Str_g2  = xs_fuel[..., 8].astype(np.float64)

    D_g1 = 1.0 / (3.0 * Str_g1 + 1e-30)
    D_g2 = 1.0 / (3.0 * Str_g2 + 1e-30)
    Sr_g1 = (Sc_g1 + Sf_g1) + Ss_12   # 제거 = 흡수 + 산란유출
    Sa_g2 = Sc_g2 + Sf_g2              # g2 흡수

    # ── 체적/면적 ──
    # dz_arr: (Z,) 각 노드 높이. Z축 비균일 가능하나 현재 10cm 균일.
    Az = dx * dy          # XY면 면적 (Z방향 누설용)
    Ay = dx               # XZ면 (Y방향) — dz는 노드별
    Ax = dy               # YZ면 (X방향) — dz는 노드별

    inner = (slice(1, -1), slice(1, -1), slice(1, -1))

    def _compute_leakage(phi, D, dz_arr, dx, dy):
        """
        6면 순 누설 합 × 면적 (inner 노드, 체적 적분 형태).

        면 중성자류: J = D_harmonic × (φ_neighbor - φ_center) / h
        순 누설 = (J_right - J_left) × Area
        """
        Zi = phi.shape[0] - 2
        Hi = phi.shape[1] - 2
        Wi = phi.shape[2] - 2
        leak = np.zeros((Zi, Hi, Wi), dtype=np.float64)

        # Z축 누설 (비균일 메시)
        for ki in range(Zi):
            k = ki + 1
            hz_p = 0.5 * (dz_arr[k] + dz_arr[k + 1])  # center→top 거리
            hz_m = 0.5 * (dz_arr[k - 1] + dz_arr[k])    # bottom→center 거리

            D_top = harmonic_mean(D[k, 1:-1, 1:-1], D[k+1, 1:-1, 1:-1])
            D_bot = harmonic_mean(D[k, 1:-1, 1:-1], D[k-1, 1:-1, 1:-1])

            J_top = D_top * (phi[k+1, 1:-1, 1:-1] - phi[k, 1:-1, 1:-1]) / hz_p
            J_bot = D_bot * (phi[k, 1:-1, 1:-1] - phi[k-1, 1:-1, 1:-1]) / hz_m
            # 순 누설 = (J_out_top - J_in_bot) × Az  ← 체적 적분
            # J_top: center→top 방향 순 중성자류 (양이면 유출)
            # J_bot: bottom→center 방향 순 중성자류
            # net_z = -(J_top - J_bot) × Az  = outflow × Az
            leak[ki] += -(J_top - J_bot) * Az

        # Y축 누설 (균일 메시)
        D_north = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 2:, 1:-1])
        D_south = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, :-2, 1:-1])
        J_north = D_north * (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
        J_south = D_south * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, :-2, 1:-1]) / dy
        # 각 inner 노드의 dz (Z 방향 높이)
        for ki in range(Zi):
            k = ki + 1
            Ay_k = dx * dz_arr[k]  # XZ면 면적 (노드별 dz)
            leak[ki] += -(J_north[ki] - J_south[ki]) * Ay_k

        # X축 누설 (균일 메시)
        D_east = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 1:-1, 2:])
        D_west = harmonic_mean(D[1:-1, 1:-1, 1:-1], D[1:-1, 1:-1, :-2])
        J_east = D_east * (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dx
        J_west = D_west * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, 1:-1, :-2]) / dx
        for ki in range(Zi):
            k = ki + 1
            Ax_k = dy * dz_arr[k]  # YZ면 면적
            leak[ki] += -(J_east[ki] - J_west[ki]) * Ax_k

        return leak  # (Zi, Hi, Wi) 누설 [n/s] (체적 적분)

    # ── 누설 계산 ──
    leak_g1 = _compute_leakage(phi_g1, D_g1, dz_arr, dx, dy)
    leak_g2 = _compute_leakage(phi_g2, D_g2, dz_arr, dx, dy)

    # ── 체적 × 단면적 항 (inner) ──
    phi1_in = phi_g1[inner]
    phi2_in = phi_g2[inner]
    Sr1_in = Sr_g1[inner]
    Sa2_in = Sa_g2[inner]
    nuSf1_in = nuSf_g1[inner]
    nuSf2_in = nuSf_g2[inner]
    Ss12_in = Ss_12[inner]

    # 노드별 체적 (inner Z planes)
    Zi = phi_g1.shape[0] - 2
    V = np.zeros_like(phi1_in)
    for ki in range(Zi):
        k = ki + 1
        V[ki] = dx * dy * dz_arr[k]

    fission_source = nuSf1_in * phi1_in + nuSf2_in * phi2_in

    # ── 잔차: R = 누설 + 소멸×V - 소스×V ──
    # R=0이면 확산방정식 만족
    removal_g1_V = Sr1_in * phi1_in * V
    removal_g2_V = Sa2_in * phi2_in * V

    R_g1 = leak_g1 + removal_g1_V - (1.0 / keff) * fission_source * V
    R_g2 = leak_g2 + removal_g2_V - Ss12_in * phi1_in * V

    return R_g1, R_g2, removal_g1_V, removal_g2_V


def relative_residual(R_g, removal_times_V):
    """
    체적 적분 잔차를 소멸항(Σ_r × φ × V) 대비 상대값으로 변환.
    R_g와 removal_times_V 모두 체적 적분 스케일 [n/s].
    비연료 노드(소멸≈0) 제외.
    """
    scale = np.abs(removal_times_V)
    valid = scale > 1e-10
    rel = np.full_like(R_g, np.nan)
    rel[valid] = np.abs(R_g[valid]) / scale[valid] * 100  # %
    return rel


def run_residual_batch(label, data_dir, profile, xs_fuel, dz_arr, dx, dy, suffixes):
    """
    여러 suffix(CRS 또는 branch)에 대해 확산잔차를 계산하고 통계 반환.
    """
    all_g1, all_g2 = [], []

    for suffix in suffixes:
        out = load_mas_out(data_dir, profile, suffix)
        if out is None:
            continue
        phi_full = out.flux_3d
        if phi_full is None or phi_full.max() == 0:
            continue

        phi_g1 = phi_full[:, 4:, 4:, 0].astype(np.float64)
        phi_g2 = phi_full[:, 4:, 4:, 1].astype(np.float64)
        keff = out.keff

        R_g1, R_g2, rem_g1_V, rem_g2_V = compute_cmfd_residual(
            phi_g1, phi_g2, xs_fuel.astype(np.float64), keff, dz_arr, dx, dy
        )

        rel_g1 = relative_residual(R_g1, rem_g1_V)
        rel_g2 = relative_residual(R_g2, rem_g2_V)

        v_g1 = rel_g1[np.isfinite(rel_g1)]
        v_g2 = rel_g2[np.isfinite(rel_g2)]
        nan1 = np.isnan(rel_g1).sum()

        if len(v_g1) > 0:
            print(f"    {suffix:<25} g1 mean={v_g1.mean():7.3f}% med={np.median(v_g1):7.3f}%  "
                  f"g2 mean={v_g2.mean():7.3f}% med={np.median(v_g2):7.3f}%  "
                  f"NaN={nan1}/{rel_g1.size}  keff={keff:.5f}")
            all_g1.append(v_g1)
            all_g2.append(v_g2)

    return (np.concatenate(all_g1) if all_g1 else np.array([]),
            np.concatenate(all_g2) if all_g2 else np.array([]))


def print_stats(label, g1, g2):
    """통합 통계 출력"""
    if len(g1) == 0:
        print(f"  [{label}] 데이터 없음")
        return
    print(f"\n  [{label}] 유효 노드: {len(g1)}")
    print(f"  {'':25} {'g=1 (fast)':>12}  {'g=2 (therm)':>12}")
    print(f"  {'mean':25} {g1.mean():>10.3f}%  {g2.mean():>10.3f}%")
    print(f"  {'median':25} {np.median(g1):>10.3f}%  {np.median(g2):>10.3f}%")
    print(f"  {'p95':25} {np.percentile(g1,95):>10.3f}%  {np.percentile(g2,95):>10.3f}%")
    print(f"  {'max':25} {g1.max():>10.3f}%  {g2.max():>10.3f}%")
    print(f"  {'< 5%':25} {(g1<5).sum():>6}/{len(g1):<6}  {(g2<5).sum():>6}/{len(g2):<6}")
    print(f"  {'< 10%':25} {(g1<10).sum():>6}/{len(g1):<6}  {(g2<10).sum():>6}/{len(g2):<6}")


def main():
    sep()
    print("L_diffusion 사전 검증: MASTER GT 확산잔차 테스트")
    print("  CRS 10스텝 + Branch 제어봉 위치 + 다중 LP")
    sep()

    dx, dy = ASSEMBLY_PITCH_CM, ASSEMBLY_PITCH_CM
    grand_g1, grand_g2 = [], []
    results_lines = []

    for lp_id, profile in TEST_LPS:
        data_dir = os.path.join(WORKSPACE, lp_id, profile)
        if not os.path.isdir(data_dir):
            print(f"\n[SKIP] {lp_id}/{profile}: 디렉토리 없음")
            continue

        print(f"\n{'='*72}")
        print(f"  {lp_id} / {profile}")
        print(f"{'='*72}")

        # Geometry + xs_fuel
        first_out = Path(data_dir) / f"MAS_OUT_{profile}_s0001_crs"
        with open(first_out, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        geom = CoreGeometry.from_lines(lines)
        xs_fuel = load_xs_fuel(geom, data_dir)

        fuel_k_min, fuel_k_max = 2, 21
        dz_arr = np.array([geom.zmesh[k] for k in range(fuel_k_min, fuel_k_max + 1)], dtype=np.float64)
        print(f"  xs_fuel: {xs_fuel.shape}, dz: {dz_arr[:3]}..., dx=dy={dx}")

        # ── T1: CRS 10스텝 ──
        print(f"\n  [CRS 10스텝] s0001~s0010:")
        crs_suffixes = [f"s{s:04d}_crs" for s in CRS_STEPS]
        g1_crs, g2_crs = run_residual_batch("CRS", data_dir, profile, xs_fuel, dz_arr, dx, dy, crs_suffixes)
        print_stats(f"{lp_id} CRS", g1_crs, g2_crs)

        # ── T2: Branch (s0001, 제어봉 위치별) ──
        print(f"\n  [Branch s{BRANCH_STEP:04d}] 제어봉 위치별:")
        # 자동 탐색: s0001의 branch 파일
        branch_files = sorted(Path(data_dir).glob(f"MAS_OUT_{profile}_s{BRANCH_STEP:04d}_r*"))
        branch_suffixes = [f.name.replace(f"MAS_OUT_{profile}_", "") for f in branch_files]
        if not branch_suffixes:
            print("    branch 파일 없음")
            g1_br, g2_br = np.array([]), np.array([])
        else:
            print(f"    발견: {len(branch_suffixes)}개 branch")
            g1_br, g2_br = run_residual_batch("Branch", data_dir, profile, xs_fuel, dz_arr, dx, dy, branch_suffixes[:10])
            print_stats(f"{lp_id} Branch", g1_br, g2_br)

        # 합산
        for arr in [g1_crs, g1_br]:
            if len(arr) > 0:
                grand_g1.append(arr)
        for arr in [g2_crs, g2_br]:
            if len(arr) > 0:
                grand_g2.append(arr)

    # ── 전체 종합 ──
    sep()
    print("\n종합 판정 (전체 LP + CRS + Branch)")
    sep("-")

    if not grand_g1:
        print("  데이터 없음")
        return

    all_g1 = np.concatenate(grand_g1)
    all_g2 = np.concatenate(grand_g2)
    print_stats("TOTAL", all_g1, all_g2)

    med_max = max(np.median(all_g1), np.median(all_g2))
    if med_max < 1:
        verdict, detail = "PASS", "유한차분이 노달법을 잘 근사 -> L_diffusion 도입 가능"
    elif med_max < 5:
        verdict, detail = "MARGINAL", "경계 보정 필요하나 도입 가능성 있음"
    else:
        verdict, detail = "FAIL", "유한차분-노달법 불일치 심각 -> L_diffusion 그대로 도입 부적합"

    print(f"\n  median 기준: g1={np.median(all_g1):.3f}%, g2={np.median(all_g2):.3f}%")
    print(f"  판정: [{verdict}] — {detail}")
    sep()

    # 결과 저장
    result_path = os.path.join(OUT_DIR, "2026-03-30_diffusion_residual_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"L_diffusion 확산잔차 사전 검증 결과\n{'='*60}\n")
        f.write(f"스텐실: 7점 유한차분, dx=dy={dx} cm, dz=ZMESH\n")
        f.write(f"LP: {[lp for lp,_ in TEST_LPS]}\n\n")
        f.write(f"종합 (전체 LP + CRS + Branch):\n")
        f.write(f"  유효 노드: {len(all_g1)}\n")
        f.write(f"  g1: mean={all_g1.mean():.3f}% median={np.median(all_g1):.3f}% p95={np.percentile(all_g1,95):.3f}% max={all_g1.max():.3f}%\n")
        f.write(f"  g2: mean={all_g2.mean():.3f}% median={np.median(all_g2):.3f}% p95={np.percentile(all_g2,95):.3f}% max={all_g2.max():.3f}%\n")
        f.write(f"\n판정: [{verdict}]\n")
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    main()
