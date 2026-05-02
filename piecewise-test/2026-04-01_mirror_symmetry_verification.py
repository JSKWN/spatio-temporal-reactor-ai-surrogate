"""
Mirror Symmetry 적용 시각화 검증.

목적: 11×11 LPD_BCH에서 quarter-core 추출 + mirror 대칭 매핑을 시각화.
      x=0, y=0 축 기준 REFLECT 대칭이 올바르게 적용되는지 확인.

작성일: 2026-04-01
"""

import sys, os
from pathlib import Path
import numpy as np

VSMR_ROOT = r"c:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following"
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess"))
sys.path.insert(0, os.path.join(VSMR_ROOT, "data_preprocess", "lf_preprocess"))
from lf_preprocess.mas_out_parser import MasOutParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
SAVE_DIR = Path(__file__).parent

# 11×11 LPD_BCH
ASSY = [
    ['o',  'o',  'o',  'R4', 'R2', 'R1', 'R2', 'R4', 'o',  'o',  'o' ],
    ['o',  'o',  'R6', 'R3', 'A3', 'B3', 'A3', 'R3', 'R6', 'o',  'o' ],
    ['o',  'R6', 'R5', 'A3', 'A2', 'A2', 'A2', 'A3', 'R5', 'R6', 'o' ],
    ['R4', 'R3', 'A3', 'B5', 'A3', 'A2', 'A3', 'B5', 'A3', 'R3', 'R4'],
    ['R2', 'A3', 'A2', 'A3', 'A2', 'A3', 'A2', 'A3', 'A2', 'A3', 'R2'],
    ['R1', 'B3', 'A2', 'A2', 'A3', 'A2', 'A3', 'A2', 'A2', 'B3', 'R1'],
    ['R2', 'A3', 'A2', 'A3', 'A2', 'A3', 'A2', 'A3', 'A2', 'A3', 'R2'],
    ['R4', 'R3', 'A3', 'B5', 'A3', 'A2', 'A3', 'B5', 'A3', 'R3', 'R4'],
    ['o',  'R6', 'R5', 'A3', 'A2', 'A2', 'A2', 'A3', 'R5', 'R6', 'o' ],
    ['o',  'o',  'R6', 'R3', 'A3', 'B3', 'A3', 'R3', 'R6', 'o',  'o' ],
    ['o',  'o',  'o',  'R4', 'R2', 'R1', 'R2', 'R4', 'o',  'o',  'o' ],
]

ASM_COLORS = {
    'A2': '#4ecdc4', 'A3': '#45b7d1', 'B3': '#f9ca24', 'B5': '#f0932b',
    'R1': '#dfe6e9', 'R2': '#dfe6e9', 'R3': '#b2bec3', 'R4': '#b2bec3',
    'R5': '#b2bec3', 'R6': '#b2bec3', 'o': '#ffffff',
}


def draw_cell(ax, x, y, name, highlight=None, alpha=1.0, fontsize=8):
    """하나의 어셈블리 셀을 그린다."""
    color = ASM_COLORS.get(name, '#ccc')
    ec = 'black' if name not in ('o',) else '#ddd'
    lw = 1.0 if name != 'o' else 0.3
    if highlight:
        ec = highlight
        lw = 2.5

    rect = plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                          facecolor=color, edgecolor=ec, lw=lw, alpha=alpha)
    ax.add_patch(rect)
    fc = 'black' if name not in ('o',) else '#ccc'
    ax.text(x, y, name, ha='center', va='center', fontsize=fontsize,
            color=fc, fontweight='bold' if name.startswith(('A', 'B')) else 'normal',
            alpha=alpha)


def main():
    print("=" * 78)
    print("Mirror Symmetry 적용 시각화")
    print("=" * 78)

    # Load flux for annotation
    lp_id, profile = "LP_0000", "t12_363_p50_power_lower"
    data_dir = WORKSPACE / lp_id / profile
    out = MasOutParser.parse(data_dir / f"MAS_OUT_{profile}_s0001_crs")
    phi = out.flux_3d  # (20, 9, 9, 2)
    z_mid = 10

    # ═══════════════════════════════════════════════════════════
    # Figure 1: 11×11 Full-core + Quarter 추출 + 대칭축
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # --- Panel A: 11×11 전체 노심 ---
    ax = axes[0]
    for j in range(11):
        for i in range(11):
            draw_cell(ax, i, 10 - j, ASSY[j][i], fontsize=7)

    # 대칭축
    ax.axhline(y=10 - 5, color='red', lw=3, ls='--', zorder=10)
    ax.axvline(x=5, color='blue', lw=3, ls='--', zorder=10)
    ax.text(5, 10 - 5 + 0.6, 'J symmetry axis (j=5)', color='red',
            ha='center', fontsize=8, fontweight='bold')
    ax.text(5 + 0.6, 10.5, 'I sym\n(i=5)', color='blue',
            ha='center', fontsize=7, fontweight='bold')

    # Quarter box
    rect = plt.Rectangle((4.55, -0.45), 5.9, 5.9, fill=False,
                          edgecolor='green', lw=4, ls='-', zorder=10)
    ax.add_patch(rect)
    ax.text(7.5, -0.8, 'Quarter-core (qy=0~4, qx=0~4)', color='green',
            ha='center', fontsize=9, fontweight='bold')

    # 4 quadrant labels
    ax.text(2, 8, 'Q2\n(mirror)', ha='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(8, 8, 'Q1\n(mirror)', ha='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(2, 2, 'Q3\n(mirror)', ha='center', fontsize=10, color='gray', alpha=0.7)
    ax.text(8, 2, 'Quarter\n(stored)', ha='center', fontsize=12, color='green', fontweight='bold')

    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.2, 11)
    ax.set_xticks(range(11))
    ax.set_xticklabels([f'i={i}' for i in range(11)], fontsize=6)
    ax.set_yticks(range(11))
    ax.set_yticklabels([f'j={10-i}' for i in range(11)], fontsize=6)
    ax.set_aspect('equal')
    ax.set_title('(A) 11×11 Full-core LPD_BCH\n+ Symmetry Axes + Quarter Region', fontsize=11)

    # --- Panel B: Quarter 5×5 + Mirror Neighbor 매핑 ---
    ax = axes[1]

    # Quarter 어셈블리 (qy=0~4, qx=0~4) = full (j=5~9, i=5~9)
    for qy in range(5):
        for qx in range(5):
            j, i = 5 + qy, 5 + qx
            name = ASSY[j][i]
            is_sym = (qy == 0) or (qx == 0)
            hl = '#e74c3c' if is_sym and name not in ('o', 'R3', 'R5', 'R6') else None
            draw_cell(ax, qx, 4 - qy, name, highlight=hl, fontsize=9)

    # Mirror neighbor 표시 (quarter 밖에 ghost cell로)
    # N면 mirror: qy=-1 → mirror qy=1 (실제 full j=4)
    for qx in range(5):
        j_nb, i_nb = 4, 5 + qx  # actual N neighbor in full-core
        name_nb = ASSY[j_nb][i_nb]
        j_mir, i_mir = 6, 5 + qx  # mirror = j=6 = qy=1
        name_mir = ASSY[j_mir][i_mir]

        # Draw ghost cell above qy=0
        draw_cell(ax, qx, 5, name_nb, alpha=0.4, fontsize=7)
        ax.text(qx, 5.55, f'j={j_nb}', ha='center', fontsize=5, color='gray')

        # Arrow: ghost → mirror source (qy=1)
        if name_nb not in ('o', 'R3', 'R4', 'R5', 'R6', 'R1', 'R2'):
            ax.annotate('', xy=(qx, 4 - 1 + 0.45), xytext=(qx, 5 - 0.45),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2, ls='--'))

    # W면 mirror: qx=-1 → mirror qx=1 (실제 full i=4)
    for qy in range(5):
        j_nb, i_nb = 5 + qy, 4  # actual W neighbor in full-core
        name_nb = ASSY[j_nb][i_nb]

        # Draw ghost cell left of qx=0
        draw_cell(ax, -1, 4 - qy, name_nb, alpha=0.4, fontsize=7)
        ax.text(-1, 4 - qy + 0.55, f'i={i_nb}', ha='center', fontsize=5, color='gray')

        # Arrow: ghost → mirror source (qx=1)
        if name_nb not in ('o', 'R3', 'R4', 'R5', 'R6', 'R1', 'R2'):
            ax.annotate('', xy=(1 - 0.45, 4 - qy), xytext=(-1 + 0.45, 4 - qy),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))

    # Symmetry boundary
    ax.plot([-0.5, -0.5], [-0.6, 5.6], 'r-', lw=4, label='W symmetry (qx=0 left)')
    ax.plot([-0.6, 4.6], [4.5, 4.5], 'b-', lw=4, label='N symmetry (qy=0 top)')

    # Labels
    ax.text(2, 6.2, 'Ghost cells (N neighbor)\n= full j=4, mirror of qy=1',
            ha='center', fontsize=8, color='blue', style='italic')
    ax.text(-2.5, 2, 'Ghost (W)\n= full i=4\nmirror of\nqx=1',
            ha='center', fontsize=7, color='red', style='italic')

    ax.set_xlim(-2.8, 5)
    ax.set_ylim(-0.8, 6.5)
    ax.set_xticks(range(-1, 5))
    ax.set_xticklabels(['ghost'] + [f'qx={i}' for i in range(5)], fontsize=7)
    ax.set_yticks(range(-1, 6))
    ax.set_yticklabels([''] + [f'qy={4-i}' for i in range(5)] + ['ghost'], fontsize=7)
    ax.set_aspect('equal')
    ax.set_title('(B) Quarter 5×5 + Mirror Ghost Cells\nREFLECT: N→qy=1, W→qx=1', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')

    # --- Panel C: 실제 flux로 mirror 검증 ---
    ax = axes[2]

    # 9×9 flux, z=10, g1
    # 9×9 index 4 = center (qy=0), 5~8 = quarter (qy=1~4)
    # mirror: index 3 (j=4) should equal index 5 (j=6)

    # Show flux values in quarter + ghost
    IS_FUEL = lambda j, i: ASSY[j][i] not in ('o', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6')

    for qy in range(5):
        for qx in range(5):
            j, i = 5 + qy, 5 + qx
            name = ASSY[j][i]
            color = ASM_COLORS.get(name, '#ccc')
            if IS_FUEL(j, i):
                # 9×9 index: j_9x9 = j-1 (since 9×9 skips j=0,j=10), i_9x9 = i-1
                j9 = j - 1
                i9 = i - 1
                val = phi[z_mid, j9, i9, 0]
                rect = plt.Rectangle((qx - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                     facecolor=color, edgecolor='black', lw=1)
                ax.add_patch(rect)
                ax.text(qx, 4 - qy + 0.15, name, ha='center', va='center', fontsize=7, fontweight='bold')
                ax.text(qx, 4 - qy - 0.2, f'{val:.2e}', ha='center', va='center', fontsize=5)
            else:
                rect = plt.Rectangle((qx - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                     facecolor='#eee', edgecolor='#999', lw=0.5)
                ax.add_patch(rect)
                ax.text(qx, 4 - qy, name, ha='center', va='center', fontsize=6, color='#999')

    # Ghost cells with actual flux from full 9×9 + comparison
    # N ghost (j=4 → 9×9 idx=3)
    for qx in range(5):
        j, i = 4, 5 + qx
        name = ASSY[j][i]
        if IS_FUEL(j, i):
            j9, i9 = j - 1, i - 1
            val_actual = phi[z_mid, j9, i9, 0]  # actual j=4

            # Mirror source: j=6 → 9×9 idx=5
            val_mirror = phi[z_mid, 5, i9, 0]

            color = ASM_COLORS.get(name, '#ccc')
            rect = plt.Rectangle((qx - 0.45, 5 - 0.45), 0.9, 0.9,
                                 facecolor=color, edgecolor='blue', lw=2, alpha=0.5)
            ax.add_patch(rect)
            ax.text(qx, 5 + 0.2, f'{name} (j=4)', ha='center', fontsize=6, color='blue')
            ax.text(qx, 5 - 0.15, f'{val_actual:.2e}', ha='center', fontsize=5, color='blue')

            # Check match
            match = abs(val_actual - val_mirror) < 1e6
            sym = '=' if match else '!='
            ax.text(qx, 4.6, f'{sym} qy=1', ha='center', fontsize=5,
                    color='green' if match else 'red', fontweight='bold')
        else:
            rect = plt.Rectangle((qx - 0.45, 5 - 0.45), 0.9, 0.9,
                                 facecolor='#f5f5f5', edgecolor='#ccc', lw=0.5, alpha=0.5)
            ax.add_patch(rect)
            ax.text(qx, 5, name, ha='center', fontsize=6, color='#bbb')

    # W ghost (i=4 → 9×9 idx=3)
    for qy in range(5):
        j, i = 5 + qy, 4
        name = ASSY[j][i]
        if IS_FUEL(j, i):
            j9, i9 = j - 1, i - 1
            val_actual = phi[z_mid, j9, i9, 0]

            # Mirror source: i=6 → 9×9 idx=5
            val_mirror = phi[z_mid, j9, 5, 0]

            color = ASM_COLORS.get(name, '#ccc')
            rect = plt.Rectangle((-1 - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                 facecolor=color, edgecolor='red', lw=2, alpha=0.5)
            ax.add_patch(rect)
            ax.text(-1, 4 - qy + 0.2, f'{name} (i=4)', ha='center', fontsize=6, color='red')
            ax.text(-1, 4 - qy - 0.15, f'{val_actual:.2e}', ha='center', fontsize=5, color='red')

            match = abs(val_actual - val_mirror) < 1e6
            sym = '=' if match else '!='
            ax.text(-0.4, 4 - qy, f'{sym}\nqx=1', ha='center', fontsize=5,
                    color='green' if match else 'red', fontweight='bold')
        else:
            rect = plt.Rectangle((-1 - 0.45, 4 - qy - 0.45), 0.9, 0.9,
                                 facecolor='#f5f5f5', edgecolor='#ccc', lw=0.5, alpha=0.5)
            ax.add_patch(rect)
            ax.text(-1, 4 - qy, name, ha='center', fontsize=6, color='#bbb')

    ax.plot([-0.5, -0.5], [-0.6, 5.6], 'r-', lw=4)
    ax.plot([-0.6, 4.6], [4.5, 4.5], 'b-', lw=4)

    ax.set_xlim(-2, 5)
    ax.set_ylim(-0.8, 6)
    ax.set_xticks(range(-1, 5))
    ax.set_xticklabels(['ghost\n(i=4)'] + [f'qx={i}\n(i={5+i})' for i in range(5)], fontsize=6)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'qy={4-i}\n(j={5+4-i})' for i in range(5)] + ['ghost\n(j=4)'], fontsize=6)
    ax.set_aspect('equal')
    ax.set_title(f'(C) Flux 검증: ghost = mirror of quarter\n{lp_id} step1, z={z_mid}, g1', fontsize=10)

    plt.suptitle('Mirror Symmetry: 11×11 LPD_BCH → Quarter 5×5 + REFLECT Ghost Cells',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = SAVE_DIR / "mirror_symmetry_verification.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # ── 콘솔 출력 ──
    print(f"\n{'='*78}")
    print("Mirror 매핑 확인 (Assembly)")
    print("=" * 78)

    print("\n  Quarter (j=5~9, i=5~9)의 대칭면 이웃:")
    print(f"  {'위치':>10s}  {'면':>4s}  {'full 이웃':>10s}  {'이웃 asm':>8s}  {'mirror':>10s}  {'mirror asm':>10s}  {'일치':>4s}")
    for qy in range(5):
        j = 5 + qy
        # W face
        j_nb, i_nb = j, 4
        j_mr, i_mr = j, 6  # mirror of i=4 about i=5
        print(f"  {'('+str(qy)+',0)':>10s}  {'W':>4s}  {'('+str(j_nb)+','+str(i_nb)+')':>10s}  {ASSY[j_nb][i_nb]:>8s}  {'('+str(j_mr)+','+str(i_mr)+')':>10s}  {ASSY[j_mr][i_mr]:>10s}  {'OK' if ASSY[j_nb][i_nb]==ASSY[j_mr][i_mr] else 'DIFF':>4s}")

    for qx in range(5):
        i = 5 + qx
        # N face
        j_nb, i_nb = 4, i
        j_mr, i_mr = 6, i
        print(f"  {'(0,'+str(qx)+')':>10s}  {'N':>4s}  {'('+str(j_nb)+','+str(i_nb)+')':>10s}  {ASSY[j_nb][i_nb]:>8s}  {'('+str(j_mr)+','+str(i_mr)+')':>10s}  {ASSY[j_mr][i_mr]:>10s}  {'OK' if ASSY[j_nb][i_nb]==ASSY[j_mr][i_mr] else 'DIFF':>4s}")

    print(f"\n  전체 mirror pair assembly 일치: ", end="")
    all_ok = True
    for j in range(5):
        for i in range(11):
            if ASSY[j][i] != ASSY[10-j][i]:
                all_ok = False
            if ASSY[j][i] != ASSY[j][10-i]:
                pass  # not necessarily J-mirror
    # Check J mirror: row j ↔ row 10-j
    j_ok = all(ASSY[j] == ASSY[10-j] for j in range(11) for _ in [0])
    # Actually check element-wise
    j_ok = all(ASSY[j][i] == ASSY[10-j][i] for j in range(11) for i in range(11))
    i_ok = all(ASSY[j][i] == ASSY[j][10-i] for j in range(11) for i in range(11))
    print(f"J-mirror={'OK' if j_ok else 'FAIL'}, I-mirror={'OK' if i_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
