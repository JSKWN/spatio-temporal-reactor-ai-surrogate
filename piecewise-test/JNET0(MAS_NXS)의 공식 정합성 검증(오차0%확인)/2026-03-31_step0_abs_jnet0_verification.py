"""
Step 0: ABS 정의 및 JNET0 부호 규약 역산 검증.

목적:
  - MAS_NXS ABS 컬럼이 Σ_c + Σ_f (가설A) vs Σ_c only (가설B) 중 어느 것인지 확인
  - JNET0 부호 규약 확인 (positive = leakage out vs positive = coordinate direction)
  - 방법: 내부 연료 노드에서 JNET0 밸런스 잔차를 4가지 조합으로 계산하여 최소 잔차 조합 채택

밸런스 방정식 (체적 적분):
  g=1: JNET_sum + Σ_r1 × φ₁ × V = (1/keff) × (νΣf₁×φ₁ + νΣf₂×φ₂) × V
  g=2: JNET_sum + Σ_a2 × φ₂ × V = Σ_s12 × φ₁ × V

가설 조합:
  A+pos: ABS = Σ_a = Σ_c+Σ_f, JNET positive=outward (leak term = +JNET_sum)
  A+neg: ABS = Σ_a = Σ_c+Σ_f, JNET positive=inward (leak term = -JNET_sum)
  B+pos: ABS = Σ_c only,       JNET positive=outward
  B+neg: ABS = Σ_c only,       JNET positive=inward

작성일: 2026-03-31
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
            # g1 fields
            'dif_g1': g1[col_map['DIF']],
            'abs_g1': g1[col_map['ABS']],
            'sca_g1': g1[col_map['SCA']],
            'fis_g1': g1[col_map['FIS']],
            'nfs_g1': g1[col_map['NFS']],
            'flx_g1': g1[col_map['FLX']],
            'jn_g1': g1[col_map['JNET0-N']],
            'jw_g1': g1[col_map['JNET0-W']],
            'je_g1': g1[col_map['JNET0-E']],
            'js_g1': g1[col_map['JNET0-S']],
            'jb_g1': g1[col_map['JNET0-B']],
            'jt_g1': g1[col_map['JNET0-T']],
        }
        # g2 fields
        if len(g2) > (col_map['JNET0-T'] - g2_offset):
            node['dif_g2'] = g2[col_map['DIF'] - g2_offset]
            node['abs_g2'] = g2[col_map['ABS'] - g2_offset]
            node['sca_g2'] = g2[col_map['SCA'] - g2_offset]
            node['fis_g2'] = g2[col_map['FIS'] - g2_offset]
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


def compute_balance(node, keff, dz, sign, abs_includes_fission, jnet_multiply_area=False):
    """단일 노드 밸런스 잔차 계산.

    Args:
        sign: +1 if JNET positive=outward (leak), -1 if positive=inward
        abs_includes_fission: True if ABS = Σ_c + Σ_f, False if ABS = Σ_c only
        jnet_multiply_area: True if JNET is per-unit-area (need to multiply by face area)
    """
    phi1 = node['flx_g1']
    phi2 = node['flx_g2']
    dx = 21.60780 / 2  # 10.80390 cm (node pitch = WIDE/ndivxy)
    dy = dx
    V = dx * dy * dz

    if jnet_multiply_area:
        # JNET is per-unit-area [n/cm²/s], multiply by face area
        A_ns = dx * dz  # N/S face area
        A_we = dy * dz  # W/E face area
        A_bt = dx * dy  # B/T face area
        jnet_sum_g1 = sign * (node['jn_g1'] * A_ns + node['js_g1'] * A_ns +
                              node['jw_g1'] * A_we + node['je_g1'] * A_we +
                              node['jb_g1'] * A_bt + node['jt_g1'] * A_bt)
        jnet_sum_g2 = sign * (node['jn_g2'] * A_ns + node['js_g2'] * A_ns +
                              node['jw_g2'] * A_we + node['je_g2'] * A_we +
                              node['jb_g2'] * A_bt + node['jt_g2'] * A_bt)
    else:
        # JNET is surface-integrated [n/s]
        jnet_sum_g1 = sign * (node['jn_g1'] + node['jw_g1'] + node['je_g1'] +
                              node['js_g1'] + node['jb_g1'] + node['jt_g1'])
        jnet_sum_g2 = sign * (node['jn_g2'] + node['jw_g2'] + node['je_g2'] +
                              node['js_g2'] + node['jb_g2'] + node['jt_g2'])

    # Removal
    if abs_includes_fission:
        sigma_r1 = node['abs_g1'] + node['sca_g1']
        sigma_a2 = node['abs_g2']
    else:
        sigma_r1 = node['abs_g1'] + node['fis_g1'] + node['sca_g1']
        sigma_a2 = node['abs_g2'] + node['fis_g2']

    removal_g1 = sigma_r1 * phi1 * V
    removal_g2 = sigma_a2 * phi2 * V

    # Source
    fission_src = (1.0 / keff) * (node['nfs_g1'] * phi1 + node['nfs_g2'] * phi2) * V
    scatter_src_g2 = node['sca_g1'] * phi1 * V

    # Residual: leak + removal - source = 0
    R_g1 = jnet_sum_g1 + removal_g1 - fission_src
    R_g2 = jnet_sum_g2 + removal_g2 - scatter_src_g2

    rel_g1 = abs(R_g1) / abs(removal_g1) * 100 if abs(removal_g1) > 1e-20 else float('nan')
    rel_g2 = abs(R_g2) / abs(removal_g2) * 100 if abs(removal_g2) > 1e-20 else float('nan')

    return R_g1, R_g2, rel_g1, rel_g2


def main():
    print("=" * 70)
    print("Step 0: ABS 정의 및 JNET0 부호 규약 역산 검증")
    print("=" * 70)

    # ── keff 추출 ──
    out_data = MasOutParser.parse(OUT_PATH)
    keff = out_data.keff
    print(f"\nkeff = {keff:.5f}")

    # ── MAS_NXS 파싱 ──
    nodes, nx, nxy, nz, wide, zmesh = parse_nxs_raw(NXS_PATH)
    print(f"NX={nx}, NXY={nxy}, NZ={nz}, WIDE={wide}")
    print(f"총 노드 수: {len(nodes)}")

    # ── 연료 내부 노드 선택 (K=5~18, 반사체/경계 제외) ──
    # I, J 범위 탐색
    fuel_nodes = [n for n in nodes
                  if 3 <= n['k'] <= 20  # 연료 + 여유
                  and n['fis_g1'] > 0 and n['flx_g1'] > 0
                  and 'flx_g2' in n]

    # I, J 범위 확인
    i_vals = sorted(set(n['i'] for n in fuel_nodes))
    j_vals = sorted(set(n['j'] for n in fuel_nodes))
    k_vals = sorted(set(n['k'] for n in fuel_nodes))
    print(f"\n연료 노드 I 범위: {i_vals[0]}~{i_vals[-1]} ({len(i_vals)}개)")
    print(f"연료 노드 J 범위: {j_vals[0]}~{j_vals[-1]} ({len(j_vals)}개)")
    print(f"연료 노드 K 범위: {k_vals[0]}~{k_vals[-1]} ({len(k_vals)}개)")
    print(f"연료 노드 총 수: {len(fuel_nodes)}")

    # 내부 노드만 선택 (I, J 양쪽 경계 1칸씩 제외, K도 2~21의 내부)
    i_inner = set(i_vals[1:-1]) if len(i_vals) > 2 else set(i_vals)
    j_inner = set(j_vals[1:-1]) if len(j_vals) > 2 else set(j_vals)
    k_inner = set(k_vals[1:-1]) if len(k_vals) > 2 else set(k_vals)

    inner_nodes = [n for n in fuel_nodes
                   if n['i'] in i_inner and n['j'] in j_inner and n['k'] in k_inner]
    print(f"내부 노드 수: {len(inner_nodes)}")

    # ── 샘플 노드 5개 선택 + 전체 통계 ──
    np.random.seed(42)
    sample_idx = np.random.choice(len(inner_nodes), size=min(5, len(inner_nodes)), replace=False)
    sample_nodes = [inner_nodes[i] for i in sample_idx]

    # ── 6가지 가설 조합 테스트 (면적 곱셈 가설 추가) ──
    # jnet_multiply_area: JNET이 per-unit-area [n/cm²/s]인지 여부
    hypotheses = [
        ("A+pos",    True,  +1, False, "ABS=Σ_c+Σ_f, JNET+=out, integrated"),
        ("A+neg",    True,  -1, False, "ABS=Σ_c+Σ_f, JNET+=in,  integrated"),
        ("A+pos+A",  True,  +1, True,  "ABS=Σ_c+Σ_f, JNET+=out, per-area"),
        ("A+neg+A",  True,  -1, True,  "ABS=Σ_c+Σ_f, JNET+=in,  per-area"),
        ("B+pos",    False, +1, False, "ABS=Σ_c only, JNET+=out, integrated"),
        ("B+neg",    False, -1, False, "ABS=Σ_c only, JNET+=in,  integrated"),
    ]

    print("\n" + "=" * 70)
    print("샘플 노드 5개 밸런스 잔차 (상대잔차 %)")
    print("=" * 70)

    for name, abs_inc_fis, sign, mul_area, desc in hypotheses:
        print(f"\n┌─ 가설 {name}: {desc}")
        for nd in sample_nodes:
            dz = zmesh[nd['k'] - 1]
            R1, R2, rel1, rel2 = compute_balance(nd, keff, dz, sign, abs_inc_fis, mul_area)
            print(f"│  I={nd['i']:2d} J={nd['j']:2d} K={nd['k']:2d}  "
                  f"g1: {rel1:8.3f}%  g2: {rel2:8.3f}%")
        print("└─")

    # ── 전체 내부 노드 통계 ──
    print("\n" + "=" * 70)
    print("전체 내부 노드 통계 (median 상대잔차 %)")
    print("=" * 70)
    print(f"{'가설':<10} {'설명':<40} {'g1 med':>8} {'g2 med':>8} {'g1 mean':>8} {'g2 mean':>8}")
    print("-" * 82)

    for name, abs_inc_fis, sign, mul_area, desc in hypotheses:
        rels_g1, rels_g2 = [], []
        for nd in inner_nodes:
            dz = zmesh[nd['k'] - 1]
            _, _, rel1, rel2 = compute_balance(nd, keff, dz, sign, abs_inc_fis, mul_area)
            if not np.isnan(rel1):
                rels_g1.append(rel1)
            if not np.isnan(rel2):
                rels_g2.append(rel2)

        med1 = np.median(rels_g1) if rels_g1 else float('nan')
        med2 = np.median(rels_g2) if rels_g2 else float('nan')
        mean1 = np.mean(rels_g1) if rels_g1 else float('nan')
        mean2 = np.mean(rels_g2) if rels_g2 else float('nan')
        print(f"{name:<10} {desc:<40} {med1:8.4f} {med2:8.4f} {mean1:8.4f} {mean2:8.4f}")

    # ── ndivxy 판정 ──
    print(f"\n=== ndivxy 판정 ===")
    n_asm = 9  # V-SMR 9×9
    # 연료 I 범위 너비
    fuel_i_range = i_vals[-1] - i_vals[0] + 1
    fuel_j_range = j_vals[-1] - j_vals[0] + 1
    print(f"연료 I 범위 너비: {fuel_i_range}, J 범위 너비: {fuel_j_range}")
    ndivxy_est = fuel_i_range / n_asm
    print(f"추정 ndivxy = {fuel_i_range}/{n_asm} = {ndivxy_est:.1f}")
    print(f"노드 피치 = WIDE / ndivxy = {wide} / {ndivxy_est:.0f} = {wide/ndivxy_est:.4f} cm")


if __name__ == "__main__":
    main()
