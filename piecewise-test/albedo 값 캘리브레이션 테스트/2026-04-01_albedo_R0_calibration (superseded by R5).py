"""
Albedo 최적값 역산: JNET0×2(GT neutron current)로부터 Marshak BC α 캘리브레이션.

목적:
  학습 가능 α의 초기값을 이론값(0.4692) 대신 데이터 기반 최적값으로 설정.
  논문 근거자료 확보.

방법:
  반사체 인접 연료 노드에서, GT 경계면 누설과 Marshak BC 수식을 대조:
    GT_leak = 2 × JNET0_reflector_face × A_face
    Marshak: leak = α × D/(D + α/2) × φ̄ × A_face
    → α_optimal 역산

세분화: 방향 3종 × 군 2개 = 6개 α
  - radial (XY 측면), bottom (Z 하부 K=2), top (Z 상부 K=21)

작성일: 2026-04-01
"""

import re
import sys
import os
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_nums(line):
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_for_albedo(nxs_path):
    """MAS_NXS에서 경계 노드의 DIF, FLX, JNET0 추출."""
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

    nodes = {}  # (pi, j, i) -> dict
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
        if k_nd < 1 or k_nd > 22 or i_nd < 0 or i_nd >= nx or j_nd < 0 or j_nd >= ny:
            continue
        if k_nd < 2 or k_nd > 21:
            # 반사체 노드도 저장 (경계 판단용)
            nodes[(k_nd - 2, j_nd, i_nd)] = {'fuel': False}
            continue
        pi = k_nd - 2
        g2_jt = col_map['JNET0-T'] - g2_off
        if len(g2) <= g2_jt:
            continue
        nodes[(pi, j_nd, i_nd)] = {
            'fuel': g1[col_map['NFS']] > 1e-10,
            'dif1': g1[col_map['DIF']], 'dif2': g2[col_map['DIF'] - g2_off],
            'flx1': g1[col_map['FLX']], 'flx2': g2[col_map['FLX'] - g2_off],
            'jn1': g1[col_map['JNET0-N']], 'js1': g1[col_map['JNET0-S']],
            'je1': g1[col_map['JNET0-E']], 'jw1': g1[col_map['JNET0-W']],
            'jb1': g1[col_map['JNET0-B']], 'jt1': g1[col_map['JNET0-T']],
            'jn2': g2[col_map['JNET0-N'] - g2_off], 'js2': g2[col_map['JNET0-S'] - g2_off],
            'je2': g2[col_map['JNET0-E'] - g2_off], 'jw2': g2[col_map['JNET0-W'] - g2_off],
            'jb2': g2[col_map['JNET0-B'] - g2_off], 'jt2': g2[col_map['JNET0-T'] - g2_off],
        }

    return nodes, zmesh, dx, dy


def solve_alpha(gt_leak_per_area, D, phi, h):
    """Marshak BC FD 형태에서 α 역산.

    gt_leak_per_area = 2 × JNET0 (per-unit-area GT neutron current)
    Marshak FD: leak_per_area = αD/(αh/2 + D) × φ̄
      (h = mesh size perpendicular to the face)

    역산:
      gt = αD/(αh/2 + D) × φ
      gt × (αh/2 + D) = αDφ
      gt×αh/2 + gt×D = αDφ
      gt×D = α(Dφ - gt×h/2)
      α = gt×D / (Dφ - gt×h/2)
    """
    gt = gt_leak_per_area
    denom = D * phi - gt * h / 2
    if abs(denom) < 1e-30:
        return np.nan
    alpha = gt * D / denom
    return alpha  # 음수 포함 — 물리적 해석을 위해 필터링 제거


def main():
    print("=" * 72)
    print("Albedo 최적값 역산: JNET0×2 기반 Marshak BC α 캘리브레이션")
    print("=" * 72)

    # 6개 α 수집: radial/bottom/top × g1/g2
    alpha_collections = {
        'radial_g1': [], 'radial_g2': [],
        'bottom_g1': [], 'bottom_g2': [],
        'top_g1': [], 'top_g2': [],
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

            nodes, zmesh, dx, dy = parse_nxs_for_albedo(nxs_path)

            for (pi, j, i), nd in nodes.items():
                if not nd.get('fuel', False):
                    continue

                dz = zmesh[pi + 1] if 0 <= pi < 20 else 10.0
                A_xz = dx * dz
                A_yz = dy * dz
                A_xy = dx * dy

                # 6면 이웃 검사 → 반사체면 식별
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

                    # 이 면이 반사체 인접 → α 역산
                    for g in [1, 2]:
                        D = nd[f'dif{g}']
                        phi = nd[f'flx{g}']
                        if D < 1e-10 or phi < 1e-10:
                            continue

                        # JNET0 × 2 = GT net current (per-unit-area)
                        jnet_key = f'j{face.lower()}{g}'
                        gt_per_area = 2.0 * nd[jnet_key]

                        # h = mesh size perpendicular to the face
                        if face in ('N', 'S'):
                            h = dy
                        elif face in ('E', 'W'):
                            h = dx
                        else:  # B, T
                            h = dz

                        alpha = solve_alpha(gt_per_area, D, phi, h)
                        if np.isnan(alpha):
                            continue

                        # 방향 분류
                        if face in ('N', 'S', 'E', 'W'):
                            direction = 'radial'
                        elif face == 'B':
                            direction = 'bottom'
                        else:
                            direction = 'top'

                        key = f'{direction}_g{g}'
                        alpha_collections[key].append(alpha)

    # ── 결과 출력 ──
    print("\n" + "=" * 72)
    print("Albedo 캘리브레이션 결과")
    print("=" * 72)
    print(f"\n  Marshak BC: leak = α × D/(D + α/2) × φ̄ × A_face")
    print(f"  이론값: α = 0.4692 (Marshak)")
    print(f"\n  {'방향/군':<16} {'N':>6} {'median':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*62}")

    results = {}
    for key in ['radial_g1', 'radial_g2', 'bottom_g1', 'bottom_g2', 'top_g1', 'top_g2']:
        vals = np.array(alpha_collections[key])
        if len(vals) == 0:
            print(f"  {key:<16} {'데이터 없음':>6}")
            continue
        # NaN/inf 제거
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            print(f"  {key:<16} {'유효값 없음':>6}")
            continue
        med = np.median(vals)
        n_neg = (vals < 0).sum()
        n_pos = (vals >= 0).sum()
        results[key] = {
            'n': len(vals), 'median': med, 'mean': vals.mean(),
            'std': vals.std(), 'min': vals.min(), 'max': vals.max(),
            'n_neg': n_neg, 'n_pos': n_pos,
        }
        print(f"  {key:<16} {len(vals):>6} {med:>8.4f} {vals.mean():>8.4f} "
              f"{vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}  "
              f"(양수:{n_pos} 음수:{n_neg})")

    # 이론값과 비교
    print(f"\n  이론값(Marshak) 대비:")
    for key, r in results.items():
        diff = r['median'] - 0.4692
        print(f"  {key:<16} median={r['median']:.4f}  "
              f"차이={diff:+.4f} ({diff/0.4692*100:+.1f}%)")

    # tf.Variable 초기값 제안
    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │ tf.Variable 초기값 제안                    │")
    print(f"  ├──────────────────────────────────────────┤")
    for key, r in results.items():
        print(f"  │ {key:<16} = {r['median']:.4f}                │")
    print(f"  └──────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
