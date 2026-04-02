"""
JNET0 α=1 공간 분포 분석: XY 맵 및 Z축 프로파일.

목적: α=1 적용 시 공간 패턴(Z축 경계 급증, XY 체커보드)이 나타나고 ~3% 잔차 발생함을 확인.

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

    nodes = []
    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = parse_nums(lines[i_line])
        g2 = parse_nums(lines[i_line + 1])
        i_line += 2

        if len(g1) < 18:
            continue

        node = {
            'i': int(g1[0]), 'j': int(g1[1]), 'k': int(g1[2]),
            'abs_g1': g1[col_map['ABS']],
            'sca_g1': g1[col_map['SCA']],
            'nfs_g1': g1[col_map['NFS']],
            'flx_g1': g1[col_map['FLX']],
            'jn_g1': g1[col_map['JNET0-N']],
            'jw_g1': g1[col_map['JNET0-W']],
            'je_g1': g1[col_map['JNET0-E']],
            'js_g1': g1[col_map['JNET0-S']],
            'jb_g1': g1[col_map['JNET0-B']],
            'jt_g1': g1[col_map['JNET0-T']],
        }
        if len(g2) > (col_map['JNET0-T'] - g2_offset):
            node['abs_g2'] = g2[col_map['ABS'] - g2_offset]
            node['sca_g2'] = g2[col_map['SCA'] - g2_offset]
            node['nfs_g2'] = g2[col_map['NFS'] - g2_offset]
            node['flx_g2'] = g2[col_map['FLX'] - g2_offset]
            node['jn_g2'] = g2[col_map['JNET0-N'] - g2_offset]
            node['jw_g2'] = g2[col_map['JNET0-W'] - g2_offset]
            node['je_g2'] = g2[col_map['JNET0-E'] - g2_offset]
            node['js_g2'] = g2[col_map['JNET0-S'] - g2_offset]
            node['jb_g2'] = g2[col_map['JNET0-B'] - g2_offset]
            node['jt_g2'] = g2[col_map['JNET0-T'] - g2_offset]
        nodes.append(node)

    return nodes, nx, nxy, nz, wide, zmesh


def compute_balance_alpha1(node, keff, dz):
    """α=1 적용한 밸런스 잔차 계산 (가설: A+pos+A)."""
    phi1 = node['flx_g1']
    phi2 = node['flx_g2']
    dx = 21.60780 / 2
    dy = dx
    V = dx * dy * dz

    A_ns = dx * dz
    A_we = dy * dz
    A_bt = dx * dy
    
    alpha = 1.0  # α=1 기준 (스케일 보정 전)
    jnet_sum_g1 = alpha * (node['jn_g1'] * A_ns + node['js_g1'] * A_ns +
                           node['jw_g1'] * A_we + node['je_g1'] * A_we +
                           node['jb_g1'] * A_bt + node['jt_g1'] * A_bt)
    jnet_sum_g2 = alpha * (node['jn_g2'] * A_ns + node['js_g2'] * A_ns +
                           node['jw_g2'] * A_we + node['je_g2'] * A_we +
                           node['jb_g2'] * A_bt + node['jt_g2'] * A_bt)

    sigma_r1 = node['abs_g1'] + node['sca_g1']
    sigma_a2 = node['abs_g2']

    removal_g1 = sigma_r1 * phi1 * V
    removal_g2 = sigma_a2 * phi2 * V

    fission_src = (1.0 / keff) * (node['nfs_g1'] * phi1 + node['nfs_g2'] * phi2) * V
    scatter_src_g2 = node['sca_g1'] * phi1 * V

    R_g1 = jnet_sum_g1 + removal_g1 - fission_src
    R_g2 = jnet_sum_g2 + removal_g2 - scatter_src_g2

    rel_g1 = abs(R_g1) / abs(removal_g1) * 100 if abs(removal_g1) > 1e-20 else float('nan')
    rel_g2 = abs(R_g2) / abs(removal_g2) * 100 if abs(removal_g2) > 1e-20 else float('nan')

    return rel_g1, rel_g2


def print_xy_map(residuals, nx, ny, title, p):
    """XY 맵 출력 (범례 포함) - α=1 기준 범위."""
    p(f"\n  {title}")
    p(f"    범례: ·=비연료  0=<1%  1=1~2%  2=2~5%  3=5~10%  X=>10%")
    
    for j in range(1, ny + 1):
        line = "    "
        for i in range(1, nx + 1):
            if (i, j) not in residuals:
                line += "· "
            else:
                val = residuals[(i, j)]
                if val < 1.0:
                    line += "0 "
                elif val < 2.0:
                    line += "1 "
                elif val < 5.0:
                    line += "2 "
                elif val < 10.0:
                    line += "3 "
                else:
                    line += "X "
        p(line)
    
    vals = [v for v in residuals.values() if not np.isnan(v)]
    if vals:
        p(f"    통계: median={np.median(vals):.4f}% mean={np.mean(vals):.4f}% max={np.max(vals):.4f}% n={len(vals)}")


def main():
    # Open output file with UTF-8 encoding
    output_file = Path(__file__).parent / "test_jnet0_alpha1_current_continuity_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        def p(text=""):
            print(text)
            f.write(text + "\n")
        
        p("=" * 72)
        p("JNET0 밸런스 잔차 공간 분포 분석 — α=1 기준 (스케일 보정 전)")
        p("=" * 72)
        p("※ α=1 사용 시 ~3% 잔차 발생, 경계에서 10%+ 급증.")
        p("※ XY 체커보드 패턴 및 Z축 상하부 경계 급증 패턴 관찰.")
        p("=" * 72)

        # ── keff 추출 ──
        out_data = MasOutParser.parse(OUT_PATH)
        keff = out_data.keff
        p(f"keff = {keff:.6f}")

        # ── MAS_NXS 파싱 ──
        nodes, nx, nxy, nz, wide, zmesh = parse_nxs_raw(NXS_PATH)
        ny = nx

        # ── 연료 노드 선택 ──
        fuel_nodes = [n for n in nodes
                      if 2 <= n['k'] <= 21
                      and n.get('abs_g1', 0) > 0 and n['flx_g1'] > 0
                      and 'flx_g2' in n]

        # ── 공간 분포 계산 ──
        residuals_g1 = {}  # {(k, j, i): rel_g1}
        residuals_g2 = {}

        for nd in fuel_nodes:
            dz = zmesh[nd['k'] - 1]
            rel1, rel2 = compute_balance_alpha1(nd, keff, dz)
            if not np.isnan(rel1):
                residuals_g1[(nd['k'], nd['j'], nd['i'])] = rel1
                residuals_g2[(nd['k'], nd['j'], nd['i'])] = rel2

        # ── XY 맵 출력 (선택된 Z 평면) ──
        p("\n" + "=" * 72)
        p("  g=1 (fast) 잔차 XY 분포")
        p("=" * 72)

        for k_sel, z_idx in [(2, 0), (7, 5), (11, 9), (16, 14), (21, 19)]:
            xy_g1 = {(i, j): residuals_g1.get((k_sel, j, i), np.nan)
                     for j in range(1, ny + 1) for i in range(1, nx + 1)}
            dz = zmesh[k_sel - 1]
            print_xy_map(xy_g1, nx, ny, f"Z={z_idx} (K={k_sel}, dz={dz:.0f} cm):", p)

        p("\n" + "=" * 72)
        p("  g=2 (thermal) 잔차 XY 분포")
        p("=" * 72)

        for k_sel, z_idx in [(2, 0), (7, 5), (11, 9), (16, 14), (21, 19)]:
            xy_g2 = {(i, j): residuals_g2.get((k_sel, j, i), np.nan)
                     for j in range(1, ny + 1) for i in range(1, nx + 1)}
            dz = zmesh[k_sel - 1]
            print_xy_map(xy_g2, nx, ny, f"Z={z_idx} (K={k_sel}, dz={dz:.0f} cm):", p)

        # ── Z축 프로파일 ──
        p("\n" + "=" * 72)
        p("  Z축 잔차 프로파일 (평면별 median)")
        p("=" * 72)
        p(f"    {'Z':>2s}  {'K':>3s}   {'g1 med':>8s}  {'g1 mean':>8s}   {'g2 med':>8s}  {'g2 mean':>8s}")

        for k in range(2, 22):
            z_idx = k - 2
            z_g1 = [residuals_g1.get((k, j, i), np.nan) 
                    for j in range(1, ny + 1) for i in range(1, nx + 1)]
            z_g2 = [residuals_g2.get((k, j, i), np.nan) 
                    for j in range(1, ny + 1) for i in range(1, nx + 1)]
            
            z_g1 = [v for v in z_g1 if not np.isnan(v)]
            z_g2 = [v for v in z_g2 if not np.isnan(v)]
            
            if z_g1:
                med1 = np.median(z_g1)
                mean1 = np.mean(z_g1)
                med2 = np.median(z_g2)
                mean2 = np.mean(z_g2)
                p(f"   {z_idx:2d}  {k:3d}   {med1:8.4f}  {mean1:8.4f}   {med2:8.4f}  {mean2:8.4f}")


if __name__ == "__main__":
    main()
