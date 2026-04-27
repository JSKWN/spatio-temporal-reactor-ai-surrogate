"""
Albedo R2: radial 직각/대각 분리 + 행렬 C 재캘리브레이션.

radial을 MAS_XSL 반사체 배치에 따라 분리:
  - 직각(ortho): R1, R2 인접면 (높은 SS 비율)
  - 대각(diag):  R3~R6 인접면 (높은 물 비율)

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
TEST_LPS = [("LP_0000", "t12_363_p50_power_lower"), ("LP_0001", "t12_363_p50_power_upper")]
CRS_STEPS = list(range(1, 11))

# ── 어셈블리 맵 (11×11) ──
ASM_MAP = [
    "o   o   o   R4  R2  R1  R2  R4  o   o   o".split(),
    "o   o   R6  R3  A3  B3  A3  R3  R6  o   o".split(),
    "o   R6  R5  A3  A2  A2  A2  A3  R5  R6  o".split(),
    "R4  R3  A3  B5  A3  A2  A3  B5  A3  R3  R4".split(),
    "R2  A3  A2  A3  A2  A3  A2  A3  A2  A3  R2".split(),
    "R1  B3  A2  A2  A3  A2  A3  A2  A2  B3  R1".split(),
    "R2  A3  A2  A3  A2  A3  A2  A3  A2  A3  R2".split(),
    "R4  R3  A3  B5  A3  A2  A3  B5  A3  R3  R4".split(),
    "o   R6  R5  A3  A2  A2  A2  A3  R5  R6  o".split(),
    "o   o   R6  R3  A3  B3  A3  R3  R6  o   o".split(),
    "o   o   o   R4  R2  R1  R2  R4  o   o   o".split(),
]
ASM_MAP = np.array(ASM_MAP)
IS_FUEL_ASM = np.array([[not c.startswith('R') and c != 'o' for c in row] for row in ASM_MAP])
IS_ORTHO = np.array([[c in ('R1', 'R2') for c in row] for row in ASM_MAP])
IS_DIAG = np.array([[c in ('R3', 'R4', 'R5', 'R6') for c in row] for row in ASM_MAP])


def classify_radial_face(j_node, i_node, dj, di):
    """노드 좌표에서 이웃 반사체가 직각(ortho) vs 대각(diag) 판별."""
    nj, ni = j_node + dj, i_node + di
    if nj < 0 or nj >= 22 or ni < 0 or ni >= 22:
        return None
    naj, nai = nj // 2, ni // 2
    if naj >= 11 or nai >= 11:
        return None
    if IS_FUEL_ASM[naj, nai]:
        return None  # 이웃이 연료
    if IS_ORTHO[naj, nai]:
        return 'ortho'
    if IS_DIAG[naj, nai]:
        return 'diag'
    return None  # 'o' (빈 공간)


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def main():
    print("=" * 72)
    print("Albedo R2: radial 직각/대각 분리 + 행렬 C 캘리브레이션")
    print("=" * 72)

    samples = {'ortho': [], 'diag': [], 'bottom': [], 'top': []}

    for lp_id, profile in TEST_LPS:
        data_dir = WORKSPACE / lp_id / profile
        if not data_dir.is_dir():
            continue
        print(f"\n  {lp_id} / {profile}")

        for s in CRS_STEPS:
            nxs_path = data_dir / f"MAS_NXS_{profile}_s{s:04d}_crs"
            if not nxs_path.exists():
                continue

            lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()
            nx = int(parse_nums(lines[1])[0])
            cols = lines[7].split()
            col_map = {n: i for i, n in enumerate(cols)}
            g2_off = col_map['DIF']
            zmesh = np.array(parse_nums(lines[4]))

            nodes = {}
            i_line = 8
            nxy = int(parse_nums(lines[1])[1])
            nz = int(parse_nums(lines[1])[2])
            for _ in range(nxy * nz):
                if i_line + 1 >= len(lines):
                    break
                g1 = parse_nums(lines[i_line])
                g2 = parse_nums(lines[i_line + 1])
                i_line += 2
                if len(g1) < col_map['JNET0-T'] + 1:
                    continue
                i_nd, j_nd, k_nd = int(g1[0]) - 1, int(g1[1]) - 1, int(g1[2])
                if k_nd < 2 or k_nd > 21:
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

            for (pi, j, i), nd in nodes.items():
                if not nd.get('fuel', False):
                    continue
                phi1, phi2 = nd['flx1'], nd['flx2']
                if phi1 < 1e-10 or phi2 < 1e-10:
                    continue

                for face, (dj, di) in [('N', (1, 0)), ('S', (-1, 0)), ('E', (0, 1)), ('W', (0, -1))]:
                    cat = classify_radial_face(j, i, dj, di)
                    if cat is not None:
                        j1 = 2.0 * nd[f'j{face.lower()}1']
                        j2 = 2.0 * nd[f'j{face.lower()}2']
                        samples[cat].append((phi1, phi2, j1, j2))

                # 축방향
                for face, direction in [('B', 'bottom'), ('T', 'top')]:
                    dpi = -1 if face == 'B' else 1
                    neighbor = nodes.get((pi + dpi, j, i))
                    if neighbor is None or not neighbor.get('fuel', False):
                        j1 = 2.0 * nd[f'j{face.lower()}1']
                        j2 = 2.0 * nd[f'j{face.lower()}2']
                        samples[direction].append((phi1, phi2, j1, j2))

    # ── 결과 ──
    print("\n" + "=" * 72)
    print("행렬 C 캘리브레이션 결과 (R2: radial 직각/대각 분리)")
    print("=" * 72)

    for direction in ['ortho', 'diag', 'bottom', 'top']:
        data = np.array(samples[direction])
        if len(data) == 0:
            print(f"\n  [{direction}] 데이터 없음")
            continue

        Phi = data[:, :2]
        J1, J2 = data[:, 2], data[:, 3]

        C_row1, _, _, _ = np.linalg.lstsq(Phi, J1, rcond=None)
        C_row2, _, _, _ = np.linalg.lstsq(Phi, J2, rcond=None)
        C = np.array([C_row1, C_row2])

        I2 = np.eye(2)
        try:
            beta = np.linalg.solve(I2 + 2 * C, I2 - 2 * C)
        except np.linalg.LinAlgError:
            beta = np.full((2, 2), np.nan)

        ss_res1 = np.sum((J1 - Phi @ C_row1) ** 2)
        ss_tot1 = np.sum((J1 - J1.mean()) ** 2)
        r2_g1 = 1 - ss_res1 / ss_tot1 if ss_tot1 > 0 else 0

        ss_res2 = np.sum((J2 - Phi @ C_row2) ** 2)
        ss_tot2 = np.sum((J2 - J2.mean()) ** 2)
        r2_g2 = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0

        c21_ratio = abs(C[1, 0]) / abs(C[1, 1]) if abs(C[1, 1]) > 1e-30 else float('inf')

        print(f"\n  [{direction}] 샘플 수: {len(data)}")
        print(f"  C 행렬:")
        print(f"    [C11={C[0, 0]:+.6e}  C12={C[0, 1]:+.6e}]")
        print(f"    [C21={C[1, 0]:+.6e}  C22={C[1, 1]:+.6e}]")
        print(f"  R² (적합도): g1={r2_g1:.6f}, g2={r2_g2:.6f}")
        print(f"  β 행렬:")
        print(f"    [β11={beta[0, 0]:+.6f}  β12={beta[0, 1]:+.6f}]")
        print(f"    [β21={beta[1, 0]:+.6f}  β22={beta[1, 1]:+.6f}]")
        print(f"  |C₂₁|/|C₂₂| = {c21_ratio:.4f}")


if __name__ == "__main__":
    main()
