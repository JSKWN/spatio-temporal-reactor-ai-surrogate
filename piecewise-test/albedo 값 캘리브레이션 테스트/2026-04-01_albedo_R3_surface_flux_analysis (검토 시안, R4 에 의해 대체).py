"""
R3 면 flux 역산: JNET0 역전 현상의 물리적 검증.

목적:
  JNET0 < 0 (유입)인 반사체 인접면에서, 면 flux(surface flux)를 역산하여
  φ_fuel_surface < φ_refl_surface 인지 확인.

방법:
  half-node Fick: φ_surface = φ̄ - J_net × h / (2D)
  J_net = 2 × JNET0 (외향 방향, ×2 보정)

작성일: 2026-04-01
"""

import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780
dx = dy = WIDE / 2
NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')
FACE_DIR = {'N': (1, 0), 'S': (-1, 0), 'E': (0, 1), 'W': (0, -1)}


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def main():
    nxs_path = WORKSPACE / "LP_0000/t12_363_p50_power_lower/MAS_NXS_t12_363_p50_power_lower_s0001_crs"
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    nx = int(parse_nums(lines[1])[0])
    nxy = int(parse_nums(lines[1])[1])
    nz = int(parse_nums(lines[1])[2])
    cols = lines[7].split()
    col_map = {n: i for i, n in enumerate(cols)}
    g2_off = col_map['DIF']
    zmesh = np.array(parse_nums(lines[4]))

    # K=11 (중앙 Z) 모든 노드 파싱
    nodes = {}
    i_line = 8
    for _ in range(nxy * nz):
        if i_line + 1 >= len(lines):
            break
        g1 = parse_nums(lines[i_line])
        g2 = parse_nums(lines[i_line + 1])
        i_line += 2
        if len(g1) < col_map['JNET0-T'] + 1:
            continue
        i_nd, j_nd, k_nd = int(g1[0]) - 1, int(g1[1]) - 1, int(g1[2])
        if k_nd != 11:
            continue
        g2_jt = col_map['JNET0-T'] - g2_off
        if len(g2) <= g2_jt:
            continue
        nodes[(j_nd, i_nd)] = {
            'fuel': g1[col_map['NFS']] > 1e-10,
            'flx1': g1[col_map['FLX']], 'flx2': g2[col_map['FLX'] - g2_off],
            'dif1': g1[col_map['DIF']], 'dif2': g2[col_map['DIF'] - g2_off],
            'jn1': g1[col_map['JNET0-N']], 'js1': g1[col_map['JNET0-S']],
            'je1': g1[col_map['JNET0-E']], 'jw1': g1[col_map['JNET0-W']],
            'jn2': g2[col_map['JNET0-N'] - g2_off], 'js2': g2[col_map['JNET0-S'] - g2_off],
            'je2': g2[col_map['JNET0-E'] - g2_off], 'jw2': g2[col_map['JNET0-W'] - g2_off],
        }

    h = dx  # 노드 피치

    print("=" * 72)
    print("R3 면 flux 역산: JNET0 역전 현상 물리적 검증")
    print("=" * 72)
    print(f"방법: half-node Fick → φ_surface = φ̄ - J_net × h/(2D)")
    print(f"      J_net = 2 × JNET0 (외향 방향)")
    print(f"      h = {h:.3f} cm (노드 피치)")

    print(f"\n{'='*72}")
    print("g=1 (고속군) — radial 경계면 전체")
    print(f"{'='*72}")

    results_pos = []  # JNET0 > 0 면
    results_neg = []  # JNET0 < 0 면

    for (j, i), nd in nodes.items():
        if not nd['fuel']:
            continue
        for face, (dj, di) in FACE_DIR.items():
            nj, ni = j + dj, i + di
            refl = nodes.get((nj, ni))
            if refl is None or refl['fuel']:
                continue

            for g in [1, 2]:
                phi_fuel = nd[f'flx{g}']
                phi_refl = refl[f'flx{g}']
                D_fuel = nd[f'dif{g}']
                D_refl = refl[f'dif{g}']
                jnet = 2.0 * nd[f'j{face.lower()}{g}']  # 외향 × 2

                # 면 flux 역산: φ_surface = φ̄ - J × h/(2D)
                # 연료측: J가 외향(양수)이면 면 flux < 노드 평균
                phi_fuel_surface = phi_fuel - jnet * h / (2 * D_fuel)

                # 반사체측: 반사체에서 보면 이 면은 "반대편" 면
                # 반사체의 해당 면 JNET0는 연료 방향 유입
                # 간단히: 반사체 면 flux ≈ φ̄_refl + |J| × h/(2D_refl)
                # (유입 방향이므로 면 flux > 노드 평균)
                phi_refl_surface = phi_refl + abs(jnet) * h / (2 * D_refl)

                row = (j, i, face, g, jnet, phi_fuel, phi_refl,
                       D_fuel, D_refl, phi_fuel_surface, phi_refl_surface)

                if g == 1:
                    if jnet > 0:
                        results_pos.append(row)
                    else:
                        results_neg.append(row)

    # 출력: JNET0 > 0 (누설면)
    print(f"\n  ── JNET0 > 0 (누설면, {len(results_pos)}면) ──")
    print(f"  {'(j,i)':<8} {'face':<5} {'J_net':>10} {'φ_fuel':>10} {'φ_refl':>10} "
          f"{'φ_f_surf':>10} {'φ_r_surf':>10} {'f_surf>r_surf':>13}")
    for row in results_pos[:10]:
        j, i, face, g, jnet, pf, pr, df, dr, pfs, prs = row
        print(f"  ({j:2d},{i:2d}) {face:<5} {jnet:+.3e} {pf:.3e} {pr:.3e} "
              f"{pfs:.3e} {prs:.3e} {'✓' if pfs > prs else '✗':>5}")

    # 통계
    if results_pos:
        arr = np.array([(r[9], r[10]) for r in results_pos])
        n_fick_ok = np.sum(arr[:, 0] > arr[:, 1])
        print(f"  φ_f_surface > φ_r_surface (Fick ✓): {n_fick_ok}/{len(results_pos)}")

    # 출력: JNET0 < 0 (역전면)
    print(f"\n  ── JNET0 < 0 (역전면, {len(results_neg)}면) ──")
    print(f"  {'(j,i)':<8} {'face':<5} {'J_net':>10} {'φ_fuel':>10} {'φ_refl':>10} "
          f"{'φ_f_surf':>10} {'φ_r_surf':>10} {'r_surf>f_surf':>13}")
    for row in results_neg[:10]:
        j, i, face, g, jnet, pf, pr, df, dr, pfs, prs = row
        print(f"  ({j:2d},{i:2d}) {face:<5} {jnet:+.3e} {pf:.3e} {pr:.3e} "
              f"{pfs:.3e} {prs:.3e} {'✓' if prs > pfs else '✗':>5}")

    if results_neg:
        arr = np.array([(r[9], r[10]) for r in results_neg])
        n_reversed = np.sum(arr[:, 1] > arr[:, 0])
        print(f"  φ_r_surface > φ_f_surface (역전 확인): {n_reversed}/{len(results_neg)}")

    # 비율 분석
    print(f"\n  ── 면 flux 하락률 분석 ──")
    if results_neg:
        fuel_drops = [(r[5] - r[9]) / r[5] * 100 for r in results_neg]  # (φ̄ - φ_surf)/φ̄
        refl_rises = [(r[10] - r[6]) / r[6] * 100 for r in results_neg]
        print(f"  역전면 연료 면 flux 하락: {np.median(fuel_drops):.1f}% (median)")
        print(f"  역전면 반사체 면 flux 상승: {np.median(refl_rises):.1f}% (median)")
        print(f"  (하락 = 노드 평균 대비 면에서 flux가 낮아짐)")
        print(f"  (상승 = 반사체에서 면 근처 flux가 노드 평균보다 높음 = '봉우리')")


if __name__ == "__main__":
    main()
