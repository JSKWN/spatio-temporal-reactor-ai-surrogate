"""
Albedo 통합 캘리브레이션: 방향별(radial_ortho/diag, bottom, top) C 행렬 도출.

모델 A: [φ₁, φ₂] (2변수) — 기본 선형
모델 C: [φ₁, φ₂, JNET_opp_g1, JNET_opp_g2] (4변수) — 반대면 전류 포함

각 방향에 대해 C 행렬, β 행렬, R², scalar α(Marshak 역산)를 모두 산출.

작성일: 2026-04-01
"""

import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))
sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess\lf_preprocess")))

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
WIDE = 21.60780
NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')
TEST_LPS = [("LP_0000", "t12_363_p50_power_lower"), ("LP_0001", "t12_363_p50_power_upper")]
CRS_STEPS = list(range(1, 11))

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
IS_FUEL_ASM = np.array([[not c.startswith('R') and c != 'o' for c in r] for r in ASM_MAP])
IS_ORTHO = np.array([[c in ('R1', 'R2') for c in r] for r in ASM_MAP])
IS_DIAG = np.array([[c in ('R3', 'R4', 'R5', 'R6') for c in r] for r in ASM_MAP])

OPP_FACE = {'N': 's', 'S': 'n', 'E': 'w', 'W': 'e', 'B': 't', 'T': 'b'}
FACE_DIR = {'N': (1, 0), 'S': (-1, 0), 'E': (0, 1), 'W': (0, -1)}


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def classify_radial(j_node, i_node, dj, di):
    nj, ni = j_node + dj, i_node + di
    if nj < 0 or nj >= 22 or ni < 0 or ni >= 22:
        return None
    naj, nai = nj // 2, ni // 2
    if naj >= 11 or nai >= 11:
        return None
    if IS_FUEL_ASM[naj, nai]:
        return None
    if IS_ORTHO[naj, nai]:
        return 'radial_ortho'
    if IS_DIAG[naj, nai]:
        return 'radial_diag'
    return None


def r2_score(Phi, J):
    C, _, _, _ = np.linalg.lstsq(Phi, J, rcond=None)
    ss_res = np.sum((J - Phi @ C) ** 2)
    ss_tot = np.sum((J - J.mean()) ** 2)
    return (1 - ss_res / ss_tot if ss_tot > 0 else 0), C


def main():
    print("=" * 72)
    print("Albedo 통합 캘리브레이션: 방향별 C 행렬 도출")
    print("모델 A: [φ₁, φ₂],  모델 C: [φ₁, φ₂, JNET_opp_g1, JNET_opp_g2]")
    print("=" * 72)

    # 방향별 샘플: {direction: [(φ1, φ2, j_opp_g1, j_opp_g2, j_refl_g1, j_refl_g2)]}
    samples = {d: [] for d in ['radial_ortho', 'radial_diag', 'bottom', 'top']}

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
            nxy = int(parse_nums(lines[1])[1])
            nz = int(parse_nums(lines[1])[2])
            cols = lines[7].split()
            col_map = {n: i for i, n in enumerate(cols)}
            g2_off = col_map['DIF']

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

                # Radial faces
                for face, (dj, di) in FACE_DIR.items():
                    cat = classify_radial(j, i, dj, di)
                    if cat is None:
                        continue
                    j_refl_g1 = 2.0 * nd[f'j{face.lower()}1']
                    j_refl_g2 = 2.0 * nd[f'j{face.lower()}2']
                    opp = OPP_FACE[face]
                    j_opp_g1 = 2.0 * nd[f'j{opp}1']
                    j_opp_g2 = 2.0 * nd[f'j{opp}2']
                    samples[cat].append((phi1, phi2, j_opp_g1, j_opp_g2, j_refl_g1, j_refl_g2))

                # Axial faces
                for face, direction in [('B', 'bottom'), ('T', 'top')]:
                    dpi = -1 if face == 'B' else 1
                    neighbor = nodes.get((pi + dpi, j, i))
                    if neighbor is not None and neighbor.get('fuel', False):
                        continue
                    j_refl_g1 = 2.0 * nd[f'j{face.lower()}1']
                    j_refl_g2 = 2.0 * nd[f'j{face.lower()}2']
                    opp = OPP_FACE[face]
                    j_opp_g1 = 2.0 * nd[f'j{opp}1']
                    j_opp_g2 = 2.0 * nd[f'j{opp}2']
                    samples[direction].append((phi1, phi2, j_opp_g1, j_opp_g2, j_refl_g1, j_refl_g2))

    # ── 결과 출력 ──
    print("\n" + "=" * 72)
    print("방향별 C 행렬 도출 결과")
    print("=" * 72)

    for direction in ['radial_ortho', 'radial_diag', 'bottom', 'top']:
        data = np.array(samples[direction])
        if len(data) == 0:
            print(f"\n  [{direction}] 데이터 없음")
            continue

        phi = data[:, :2]               # [φ₁, φ₂]
        phi_aug = data[:, :4]           # [φ₁, φ₂, JNET_opp_g1, JNET_opp_g2]
        J1 = data[:, 4]                 # J_net_reflector_g1
        J2 = data[:, 5]                 # J_net_reflector_g2

        # 모델 A: J = C_A × [φ₁, φ₂]
        r2_A1, CA1 = r2_score(phi, J1)
        r2_A2, CA2 = r2_score(phi, J2)
        CA = np.array([CA1, CA2])

        # 모델 C: J = C_C × [φ₁, φ₂, JNET_opp_g1, JNET_opp_g2]
        r2_C1, CC1 = r2_score(phi_aug, J1)
        r2_C2, CC2 = r2_score(phi_aug, J2)
        CC = np.array([CC1, CC2])

        # β 역산 (모델 A)
        I2 = np.eye(2)
        try:
            beta_A = np.linalg.solve(I2 + 2 * CA, I2 - 2 * CA)
        except np.linalg.LinAlgError:
            beta_A = np.full((2, 2), np.nan)

        print(f"\n  [{direction}] 샘플 수: {len(data)}")
        print(f"  ┌─ 모델 A: J = C × [φ₁, φ₂]")
        print(f"  │  C = [{CA[0,0]:+.4e}  {CA[0,1]:+.4e}]    R²: g1={r2_A1:.4f}  g2={r2_A2:.4f}")
        print(f"  │      [{CA[1,0]:+.4e}  {CA[1,1]:+.4e}]")
        print(f"  │  β = [{beta_A[0,0]:+.4f}  {beta_A[0,1]:+.4f}]")
        print(f"  │      [{beta_A[1,0]:+.4f}  {beta_A[1,1]:+.4f}]")
        print(f"  │  |C₂₁|/|C₂₂| = {abs(CA[1,0])/abs(CA[1,1]):.4f}" if abs(CA[1,1]) > 1e-30 else "")
        print(f"  └─")
        print(f"  ┌─ 모델 C: J = C × [φ₁, φ₂, JNET_opp_g1, JNET_opp_g2]")
        print(f"  │  C = [{CC[0,0]:+.4e}  {CC[0,1]:+.4e}  {CC[0,2]:+.4e}  {CC[0,3]:+.4e}]")
        print(f"  │      [{CC[1,0]:+.4e}  {CC[1,1]:+.4e}  {CC[1,2]:+.4e}  {CC[1,3]:+.4e}]")
        print(f"  │  R²: g1={r2_C1:.4f}  g2={r2_C2:.4f}")
        print(f"  └─")


if __name__ == "__main__":
    main()
