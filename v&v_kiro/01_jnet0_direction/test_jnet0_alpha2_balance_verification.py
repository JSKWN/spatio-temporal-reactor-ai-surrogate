"""
JNET0 α=2 검증: JNET0 = half net current 확인.

목적: α=2 적용 시 밸런스 잔차가 0.0002% 수준으로 수렴하는지 검증.

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


def compute_balance_alpha2(node, keff, dz):
    """α=2 적용한 밸런스 잔차 계산 (확정 가설: A+pos+A)."""
    phi1 = node['flx_g1']
    phi2 = node['flx_g2']
    dx = 21.60780 / 2  # 10.80390 cm
    dy = dx
    V = dx * dy * dz

    # JNET0 = half net current, per-unit-area [n/cm²/s]
    # leak = 2 × Σ(JNET0_face × A_face)
    A_ns = dx * dz
    A_we = dy * dz
    A_bt = dx * dy
    
    alpha = 2.0  # JNET0 scale factor
    jnet_sum_g1 = alpha * (node['jn_g1'] * A_ns + node['js_g1'] * A_ns +
                           node['jw_g1'] * A_we + node['je_g1'] * A_we +
                           node['jb_g1'] * A_bt + node['jt_g1'] * A_bt)
    jnet_sum_g2 = alpha * (node['jn_g2'] * A_ns + node['js_g2'] * A_ns +
                           node['jw_g2'] * A_we + node['je_g2'] * A_we +
                           node['jb_g2'] * A_bt + node['jt_g2'] * A_bt)

    # ABS = Σ_c + Σ_f (확정)
    sigma_r1 = node['abs_g1'] + node['sca_g1']
    sigma_a2 = node['abs_g2']

    removal_g1 = sigma_r1 * phi1 * V
    removal_g2 = sigma_a2 * phi2 * V

    # Source
    fission_src = (1.0 / keff) * (node['nfs_g1'] * phi1 + node['nfs_g2'] * phi2) * V
    scatter_src_g2 = node['sca_g1'] * phi1 * V

    # Residual
    R_g1 = jnet_sum_g1 + removal_g1 - fission_src
    R_g2 = jnet_sum_g2 + removal_g2 - scatter_src_g2

    rel_g1 = abs(R_g1) / abs(removal_g1) * 100 if abs(removal_g1) > 1e-20 else float('nan')
    rel_g2 = abs(R_g2) / abs(removal_g2) * 100 if abs(removal_g2) > 1e-20 else float('nan')

    return R_g1, R_g2, rel_g1, rel_g2


def main():
    # Open output file with UTF-8 encoding
    output_file = Path(__file__).parent / "test_jnet0_alpha2_balance_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        def p(text=""):
            print(text)
            f.write(text + "\n")
        
        p("=" * 70)
        p("JNET0 α=2 검증: JNET0 = half net current")
        p("=" * 70)

        # ── keff 추출 ──
        out_data = MasOutParser.parse(OUT_PATH)
        keff = out_data.keff
        p(f"\nkeff = {keff:.6f}")

        # ── MAS_NXS 파싱 ──
        nodes, nx, nxy, nz, wide, zmesh = parse_nxs_raw(NXS_PATH)
        p(f"WIDE = {wide} cm")
        p(f"총 노드 수: {len(nodes)}")

        # ── 연료 내부 노드 선택 ──
        fuel_nodes = [n for n in nodes
                      if 3 <= n['k'] <= 20
                      and n.get('abs_g1', 0) > 0 and n['flx_g1'] > 0
                      and 'flx_g2' in n]

        i_vals = sorted(set(n['i'] for n in fuel_nodes))
        j_vals = sorted(set(n['j'] for n in fuel_nodes))
        k_vals = sorted(set(n['k'] for n in fuel_nodes))

        i_inner = set(i_vals[1:-1]) if len(i_vals) > 2 else set(i_vals)
        j_inner = set(j_vals[1:-1]) if len(j_vals) > 2 else set(j_vals)
        k_inner = set(k_vals[1:-1]) if len(k_vals) > 2 else set(k_vals)

        inner_nodes = [n for n in fuel_nodes
                       if n['i'] in i_inner and n['j'] in j_inner and n['k'] in k_inner]
        p(f"내부 노드 수: {len(inner_nodes)}")

        # ── 샘플 노드 5개 ──
        np.random.seed(42)
        sample_idx = np.random.choice(len(inner_nodes), size=min(5, len(inner_nodes)), replace=False)
        sample_nodes = [inner_nodes[i] for i in sample_idx]

        p("\n" + "=" * 70)
        p("샘플 노드 5개 밸런스 잔차 (α=2.0)")
        p("=" * 70)
        for nd in sample_nodes:
            dz = zmesh[nd['k'] - 1]
            R1, R2, rel1, rel2 = compute_balance_alpha2(nd, keff, dz)
            p(f"I={nd['i']:2d} J={nd['j']:2d} K={nd['k']:2d}  "
              f"g1: {rel1:10.6f}%  g2: {rel2:10.6f}%")

        # ── 전체 내부 노드 통계 ──
        p("\n" + "=" * 70)
        p("전체 내부 노드 통계 (α=2.0)")
        p("=" * 70)

        rels_g1, rels_g2 = [], []
        for nd in inner_nodes:
            dz = zmesh[nd['k'] - 1]
            _, _, rel1, rel2 = compute_balance_alpha2(nd, keff, dz)
            if not np.isnan(rel1):
                rels_g1.append(rel1)
            if not np.isnan(rel2):
                rels_g2.append(rel2)

        p(f"\n  g1 (fast):")
        p(f"    N = {len(rels_g1)}")
        p(f"    median = {np.median(rels_g1):.6f}%")
        p(f"    mean   = {np.mean(rels_g1):.6f}%")
        p(f"    std    = {np.std(rels_g1):.6f}%")
        p(f"    max    = {np.max(rels_g1):.6f}%")

        p(f"\n  g2 (thermal):")
        p(f"    N = {len(rels_g2)}")
        p(f"    median = {np.median(rels_g2):.6f}%")
        p(f"    mean   = {np.mean(rels_g2):.6f}%")
        p(f"    std    = {np.std(rels_g2):.6f}%")
        p(f"    max    = {np.max(rels_g2):.6f}%")

        p("\n" + "=" * 70)
        p("결론")
        p("=" * 70)
        if np.median(rels_g1) < 0.001 and np.median(rels_g2) < 0.001:
            p("✅ α=2 적용 시 밸런스 잔차 < 0.001% → JNET0 = half net current 확정")
        else:
            p(f"⚠️  α=2 적용 시에도 잔차 > 0.001% (g1: {np.median(rels_g1):.4f}%, g2: {np.median(rels_g2):.4f}%)")


if __name__ == "__main__":
    main()
