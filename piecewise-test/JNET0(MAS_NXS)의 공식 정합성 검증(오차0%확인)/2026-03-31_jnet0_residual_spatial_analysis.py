"""
JNET0 밸런스 잔차 공간 분포 분석.

목적: JNET0 밸런스 잔차가 0이 아닌 원인 파악
  - 무작위 분포 → 수렴 허용오차
  - 경계 근처 체계적 패턴 → 횡방향 누설(Transverse Leakage) 근사 오차
  - 특정 위치 집중 → XS 불일치 또는 연소도 보정 미반영

방법: A+pos+A 가설 (ABS=Σ_c+Σ_f, JNET+=outward, per-area) 사용
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))
from lf_preprocess.mas_out_parser import MasOutParser

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
NXS_PATH = WORKSPACE / "LP_0000/t12_363_p50_power_lower/MAS_NXS_t12_363_p50_power_lower_s0001_crs"
OUT_PATH = WORKSPACE / "LP_0000/t12_363_p50_power_lower/MAS_OUT_t12_363_p50_power_lower_s0001_crs"

NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')
WIDE = 21.60780


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def main():
    print("=" * 72)
    print("JNET0 밸런스 잔차 공간 분포 분석")
    print("=" * 72)

    # keff
    out_data = MasOutParser.parse(OUT_PATH)
    keff = out_data.keff
    print(f"keff = {keff:.6f}")

    # MAS_NXS 파싱
    lines = NXS_PATH.read_text(encoding='utf-8', errors='ignore').splitlines()
    nx = int(parse_nums(lines[1])[0])
    ny = nx
    nz = int(parse_nums(lines[1])[2])
    nxy = int(parse_nums(lines[1])[1])
    zmesh = np.array(parse_nums(lines[4]), dtype=np.float64)

    cols = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(cols)}
    g2_off = col_map['DIF']

    nf = 20  # K=2..21
    dx = WIDE / 2
    dy = dx

    # 노드 데이터 수집
    dif = np.zeros((nf, ny, nx, 2))
    abs_xs = np.zeros((nf, ny, nx, 2))
    sca = np.zeros((nf, ny, nx))
    nfs = np.zeros((nf, ny, nx, 2))
    flx = np.zeros((nf, ny, nx, 2))
    jnet = np.zeros((nf, ny, nx, 2, 6))  # (Z,Y,X,G,face) N,W,E,S,B,T

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
        if k_nd < 2 or k_nd > 21 or i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= ny:
            continue
        pi = k_nd - 2

        dif[pi, j_nd, i_nd, 0] = g1[col_map['DIF']]
        abs_xs[pi, j_nd, i_nd, 0] = g1[col_map['ABS']]
        sca[pi, j_nd, i_nd] = g1[col_map['SCA']]
        nfs[pi, j_nd, i_nd, 0] = g1[col_map['NFS']]
        flx[pi, j_nd, i_nd, 0] = g1[col_map['FLX']]
        for fi, face in enumerate(['JNET0-N', 'JNET0-W', 'JNET0-E', 'JNET0-S', 'JNET0-B', 'JNET0-T']):
            jnet[pi, j_nd, i_nd, 0, fi] = g1[col_map[face]]

        g2_jt = col_map['JNET0-T'] - g2_off
        if len(g2) > g2_jt:
            dif[pi, j_nd, i_nd, 1] = g2[col_map['DIF'] - g2_off]
            abs_xs[pi, j_nd, i_nd, 1] = g2[col_map['ABS'] - g2_off]
            nfs[pi, j_nd, i_nd, 1] = g2[col_map['NFS'] - g2_off]
            flx[pi, j_nd, i_nd, 1] = g2[col_map['FLX'] - g2_off]
            for fi, face in enumerate(['JNET0-N', 'JNET0-W', 'JNET0-E', 'JNET0-S', 'JNET0-B', 'JNET0-T']):
                jnet[pi, j_nd, i_nd, 1, fi] = g2[col_map[face] - g2_off]

    # 연료 마스크
    is_fuel = nfs[:, :, :, 0] > 1e-10

    # JNET0 밸런스 잔차 (모든 연료 노드)
    rel_g1 = np.full((nf, ny, nx), np.nan)
    rel_g2 = np.full((nf, ny, nx), np.nan)

    for pi in range(nf):
        dz = zmesh[pi + 1]  # K = pi+2, zmesh index = pi+1
        V = dx * dy * dz
        A_ns = dx * dz
        A_we = dy * dz
        A_bt = dx * dy

        for j in range(ny):
            for i in range(nx):
                if not is_fuel[pi, j, i]:
                    continue

                for g in range(2):
                    phi = flx[pi, j, i, g]
                    if phi < 1e-10:
                        continue

                    # JNET0 누설 (per-area × face area, positive=outward)
                    leak = (jnet[pi, j, i, g, 0] * A_ns +  # N
                            jnet[pi, j, i, g, 1] * A_we +  # W
                            jnet[pi, j, i, g, 2] * A_we +  # E
                            jnet[pi, j, i, g, 3] * A_ns +  # S
                            jnet[pi, j, i, g, 4] * A_bt +  # B
                            jnet[pi, j, i, g, 5] * A_bt)   # T

                    if g == 0:
                        removal = (abs_xs[pi, j, i, 0] + sca[pi, j, i]) * phi * V
                        source = (1.0 / keff) * (nfs[pi, j, i, 0] * flx[pi, j, i, 0] +
                                                  nfs[pi, j, i, 1] * flx[pi, j, i, 1]) * V
                    else:
                        removal = abs_xs[pi, j, i, 1] * phi * V
                        source = sca[pi, j, i] * flx[pi, j, i, 0] * V

                    R = leak + removal - source
                    scale = abs(removal)
                    if scale > 1e-10:
                        rel = abs(R) / scale * 100
                        if g == 0:
                            rel_g1[pi, j, i] = rel
                        else:
                            rel_g2[pi, j, i] = rel

    # ── XY 맵 출력 (중간 Z 평면) ──
    for g, (rel_map, label) in enumerate([(rel_g1, "g=1 (fast)"), (rel_g2, "g=2 (thermal)")]):
        print(f"\n{'='*72}")
        print(f"  {label} 잔차 XY 분포")
        print(f"{'='*72}")

        for z_idx in [0, 5, 9, 14, 19]:  # 하부, 중하부, 중앙, 중상부, 상부
            k_actual = z_idx + 2
            print(f"\n  Z={z_idx} (K={k_actual}, dz={zmesh[z_idx+1]:.0f} cm):")

            # 연료 영역 범위 확인
            valid = ~np.isnan(rel_map[z_idx])
            if not valid.any():
                print("    (연료 노드 없음)")
                continue

            # XY 맵 (잔차 크기를 문자로 표시)
            print("    범례: ·=비연료  0=<1%  1=1~2%  2=2~5%  3=5~10%  X=>10%")
            for j in range(ny):
                row = "    "
                for i_nd in range(nx):
                    v = rel_map[z_idx, j, i_nd]
                    if np.isnan(v):
                        row += "· "
                    elif v < 1:
                        row += "0 "
                    elif v < 2:
                        row += "1 "
                    elif v < 5:
                        row += "2 "
                    elif v < 10:
                        row += "3 "
                    else:
                        row += "X "
                print(row)

            vals = rel_map[z_idx][valid]
            print(f"    통계: median={np.median(vals):.2f}% mean={vals.mean():.2f}% "
                  f"max={vals.max():.2f}% n={len(vals)}")

    # ── Z축 프로파일 ──
    print(f"\n{'='*72}")
    print("  Z축 잔차 프로파일 (평면별 median)")
    print(f"{'='*72}")
    print(f"  {'Z':>3} {'K':>3}  {'g1 med':>8} {'g1 mean':>8} {'g2 med':>8} {'g2 mean':>8}")
    for pi in range(nf):
        v1 = rel_g1[pi][~np.isnan(rel_g1[pi])]
        v2 = rel_g2[pi][~np.isnan(rel_g2[pi])]
        if len(v1) > 0:
            print(f"  {pi:3d} {pi+2:3d}  {np.median(v1):8.2f} {v1.mean():8.2f} "
                  f"{np.median(v2):8.2f} {v2.mean():8.2f}")


if __name__ == "__main__":
    main()
