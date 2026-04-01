"""
Albedo R1: 2군 행렬 Albedo C 캘리브레이션 (least squares).

목적:
  반사체 경계면에서 J_net = C × φ 형태의 2×2 행렬 C를 역산.
  C₂₁(고속→열 cross-group) 크기로 행렬 BC 도입 필요성 판단.

수학:
  J⁻ = β × J⁺ (2×2 albedo matrix)
  J_net = C × φ, C = ½(I+β)⁻¹(I-β)

  역산: [J1(s)]   [C11 C12] [φ1(s)]
        [J2(s)] = [C21 C22] [φ2(s)]  → 행별 독립 least squares

작성일: 2026-04-01
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))
sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess\lf_preprocess")))

from lf_preprocess.mas_out_parser import MasOutParser

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780
NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')

TEST_LPS = [
    ("LP_0000", "t12_363_p50_power_lower"),
    ("LP_0001", "t12_363_p50_power_upper"),
]
CRS_STEPS = list(range(1, 11))


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_boundary_data(nxs_path):
    """MAS_NXS에서 경계면 (φ1, φ2, J1, J2) 샘플 수집."""
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    nx = int(parse_nums(lines[1])[0])
    ny = nx
    nxy = int(parse_nums(lines[1])[1])
    nz = int(parse_nums(lines[1])[2])
    zmesh = np.array(parse_nums(lines[4]))
    cols = lines[7].split()
    col_map = {n: i for i, n in enumerate(cols)}
    g2_off = col_map['DIF']
    dx = dy = WIDE / 2

    # 모든 노드 파싱
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
        if i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= ny:
            continue
        if k_nd < 1 or k_nd > 22:
            continue
        if k_nd < 2 or k_nd > 21:
            nodes[(k_nd - 2, j_nd, i_nd)] = {'fuel': False}
            continue
        pi = k_nd - 2
        g2_jt = col_map['JNET0-T'] - g2_off
        if len(g2) <= g2_jt:
            continue
        nodes[(pi, j_nd, i_nd)] = {
            'fuel': g1[col_map['NFS']] > 1e-10,
            'flx1': g1[col_map['FLX']], 'flx2': g2[col_map['FLX'] - g2_off],
            'jn1': g1[col_map['JNET0-N']], 'js1': g1[col_map['JNET0-S']],
            'je1': g1[col_map['JNET0-E']], 'jw1': g1[col_map['JNET0-W']],
            'jb1': g1[col_map['JNET0-B']], 'jt1': g1[col_map['JNET0-T']],
            'jn2': g2[col_map['JNET0-N'] - g2_off], 'js2': g2[col_map['JNET0-S'] - g2_off],
            'je2': g2[col_map['JNET0-E'] - g2_off], 'jw2': g2[col_map['JNET0-W'] - g2_off],
            'jb2': g2[col_map['JNET0-B'] - g2_off], 'jt2': g2[col_map['JNET0-T'] - g2_off],
        }

    return nodes


def main():
    print("=" * 72)
    print("Albedo R1: 2군 행렬 C 캘리브레이션 (least squares)")
    print("=" * 72)

    # 방향별 샘플 수집: (φ1, φ2, J_net1, J_net2)
    samples = {
        'radial': [], 'bottom': [], 'top': [],
    }

    for lp_id, profile in TEST_LPS:
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            continue
        print(f"\n  {lp_id} / {profile}")

        for s in CRS_STEPS:
            suffix = f"s{s:04d}_crs"
            nxs_path = data_dir / f"MAS_NXS_{profile}_{suffix}"
            if not nxs_path.exists():
                continue

            nodes = parse_nxs_boundary_data(nxs_path)

            for (pi, j, i), nd in nodes.items():
                if not nd.get('fuel', False):
                    continue

                neighbors = {
                    'N': (pi, j + 1, i), 'S': (pi, j - 1, i),
                    'E': (pi, j, i + 1), 'W': (pi, j, i - 1),
                    'T': (pi + 1, j, i), 'B': (pi - 1, j, i),
                }

                for face, nkey in neighbors.items():
                    neighbor = nodes.get(nkey)
                    is_reflector = (neighbor is None) or (not neighbor.get('fuel', False))
                    if not is_reflector:
                        continue

                    phi1 = nd['flx1']
                    phi2 = nd['flx2']
                    if phi1 < 1e-10 or phi2 < 1e-10:
                        continue

                    # JNET0 × 2 = GT net current (per-unit-area)
                    j1 = 2.0 * nd[f'j{face.lower()}1']
                    j2 = 2.0 * nd[f'j{face.lower()}2']

                    if face in ('N', 'S', 'E', 'W'):
                        direction = 'radial'
                    elif face == 'B':
                        direction = 'bottom'
                    else:
                        direction = 'top'

                    samples[direction].append((phi1, phi2, j1, j2))

    # ── 행렬 C 역산 (least squares) ──
    print("\n" + "=" * 72)
    print("행렬 C 캘리브레이션 결과: J_net = C × φ")
    print("=" * 72)

    for direction in ['radial', 'bottom', 'top']:
        data = np.array(samples[direction])
        if len(data) == 0:
            print(f"\n  [{direction}] 데이터 없음")
            continue

        Phi = data[:, :2]       # (N, 2): [φ1, φ2]
        J1 = data[:, 2]         # (N,): J_net_g1
        J2 = data[:, 3]         # (N,): J_net_g2

        # 행별 독립 least squares
        C_row1, res1, _, _ = np.linalg.lstsq(Phi, J1, rcond=None)  # [C11, C12]
        C_row2, res2, _, _ = np.linalg.lstsq(Phi, J2, rcond=None)  # [C21, C22]
        C = np.array([C_row1, C_row2])  # (2, 2)

        # β 역산: β = (I + 2C)⁻¹(I - 2C)
        I2 = np.eye(2)
        try:
            beta = np.linalg.solve(I2 + 2 * C, I2 - 2 * C)
        except np.linalg.LinAlgError:
            beta = np.full((2, 2), np.nan)

        # R² (결정계수)
        ss_res1 = np.sum((J1 - Phi @ C_row1) ** 2)
        ss_tot1 = np.sum((J1 - J1.mean()) ** 2)
        r2_g1 = 1 - ss_res1 / ss_tot1 if ss_tot1 > 0 else 0

        ss_res2 = np.sum((J2 - Phi @ C_row2) ** 2)
        ss_tot2 = np.sum((J2 - J2.mean()) ** 2)
        r2_g2 = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0

        print(f"\n  [{direction}] 샘플 수: {len(data)}")
        print(f"  C 행렬:")
        print(f"    [C11={C[0, 0]:+.6e}  C12={C[0, 1]:+.6e}]")
        print(f"    [C21={C[1, 0]:+.6e}  C22={C[1, 1]:+.6e}]")
        print(f"  R² (적합도): g1={r2_g1:.6f}, g2={r2_g2:.6f}")
        print(f"  β 행렬:")
        print(f"    [β11={beta[0, 0]:+.6f}  β12={beta[0, 1]:+.6f}]")
        print(f"    [β21={beta[1, 0]:+.6f}  β22={beta[1, 1]:+.6f}]")

        # C₂₁ 유의미성 판단
        c21_ratio = abs(C[1, 0]) / abs(C[1, 1]) if abs(C[1, 1]) > 1e-30 else float('inf')
        c12_ratio = abs(C[0, 1]) / abs(C[0, 0]) if abs(C[0, 0]) > 1e-30 else float('inf')
        print(f"\n  비대각 성분 유의미성:")
        print(f"    |C12|/|C11| = {c12_ratio:.4f}  (β12: 열→고속, ≈0 기대)")
        print(f"    |C21|/|C22| = {c21_ratio:.4f}  (β21: 고속→열, 핵심 판단)")

        if c21_ratio > 0.1:
            print(f"    → C21 유의미 ({c21_ratio:.1%} of C22) → 행렬 BC 권장")
        else:
            print(f"    → C21 무시 가능 ({c21_ratio:.1%} of C22) → 스칼라 유지 가능")


if __name__ == "__main__":
    main()
