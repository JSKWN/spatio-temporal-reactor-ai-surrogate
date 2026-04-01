"""
R4 반사체 인접면 식별 검증 + 시각화.

목적:
  R4 코드에서 사용하는 FACE_DIR 매핑으로 식별된 반사체 인접면이
  어셈블리 맵과 정확히 일치하는지 검증하고, 결과를 XY 맵으로 시각화.

매핑 (확정):
  N = -J (북쪽), S = +J (남쪽), E = +I (동쪽), W = -I (서쪽)

작성일: 2026-04-01
"""

import numpy as np

# ── 어셈블리 맵 (11×11) ──
ASM_MAP = np.array([r.split() for r in """o o o R4 R2 R1 R2 R4 o o o
o o R6 R3 A3 B3 A3 R3 R6 o o
o R6 R5 A3 A2 A2 A2 A3 R5 R6 o
R4 R3 A3 B5 A3 A2 A3 B5 A3 R3 R4
R2 A3 A2 A3 A2 A3 A2 A3 A2 A3 R2
R1 B3 A2 A2 A3 A2 A3 A2 A2 B3 R1
R2 A3 A2 A3 A2 A3 A2 A3 A2 A3 R2
R4 R3 A3 B5 A3 A2 A3 B5 A3 R3 R4
o R6 R5 A3 A2 A2 A2 A3 R5 R6 o
o o R6 R3 A3 B3 A3 R3 R6 o o
o o o R4 R2 R1 R2 R4 o o o""".strip().split('\n')])

IS_FUEL = np.array([[not c.startswith('R') and c != 'o' for c in r] for r in ASM_MAP])
IS_ORTHO = np.array([[c in ('R1', 'R2') for c in r] for r in ASM_MAP])
IS_DIAG = np.array([[c in ('R3', 'R4', 'R5', 'R6') for c in r] for r in ASM_MAP])

# ── R4 수정된 매핑 (확정) ──
FACE_DIR = {'N': (-1, 0), 'S': (+1, 0), 'E': (0, +1), 'W': (0, -1)}


def main():
    print("=" * 72)
    print("R4 반사체 인접면 식별 검증 + 시각화")
    print("매핑: N=-J(북), S=+J(남), E=+I(동), W=-I(서)")
    print("=" * 72)

    # ── 1. Full-core 어셈블리 단위 경계 식별 ──
    # 방법 A: 어셈블리 맵에서 직접 (정답)
    direct_boundaries = {}
    for aj in range(11):
        for ai in range(11):
            if not IS_FUEL[aj, ai]:
                continue
            faces = {}
            for face, (dj, di) in FACE_DIR.items():
                nj, ni = aj + dj, ai + di
                if 0 <= nj < 11 and 0 <= ni < 11 and not IS_FUEL[nj, ni]:
                    rtype = 'ortho' if IS_ORTHO[nj, ni] else 'diag' if IS_DIAG[nj, ni] else 'void'
                    faces[face] = (ASM_MAP[nj, ni], rtype)
            if faces:
                direct_boundaries[(aj, ai)] = faces

    # 방법 B: ndivxy=2 노드 격자에서 R4 코드 로직으로 (검증 대상)
    r4_boundaries = {}
    for j_nd in range(22):
        for i_nd in range(22):
            aj, ai = j_nd // 2, i_nd // 2
            if aj >= 11 or ai >= 11 or not IS_FUEL[aj, ai]:
                continue
            for face, (dj, di) in FACE_DIR.items():
                nj, ni = j_nd + dj, i_nd + di
                if nj < 0 or nj >= 22 or ni < 0 or ni >= 22:
                    continue
                naj, nai = nj // 2, ni // 2
                if naj >= 11 or nai >= 11:
                    continue
                if IS_FUEL[naj, nai]:
                    continue
                rtype = 'ortho' if IS_ORTHO[naj, nai] else 'diag' if IS_DIAG[naj, nai] else 'void'
                if (aj, ai) not in r4_boundaries:
                    r4_boundaries[(aj, ai)] = {}
                r4_boundaries[(aj, ai)][face] = (ASM_MAP[naj, nai], rtype)

    # ── 2. 비교 ──
    print("\n■ 방법 A (어셈블리 직접) vs 방법 B (R4 노드 로직) 비교:")
    all_match = True
    for key in sorted(set(list(direct_boundaries.keys()) + list(r4_boundaries.keys()))):
        d = direct_boundaries.get(key, {})
        r = r4_boundaries.get(key, {})
        match = d == r
        if not match:
            all_match = False
            print(f"  ✗ ({key[0]},{key[1]}) {ASM_MAP[key]}: 직접={d}, R4={r}")

    if all_match:
        print(f"  ✅ 전체 {len(direct_boundaries)}개 경계 어셈블리 완벽 일치!")

    # ── 3. Full-core 시각화 ──
    print(f"\n{'='*72}")
    print("Full-core 11×11 시각화")
    print("=" * 72)

    # 맵 1: 연료/반사체/경계 구분
    print("\n[맵 1] 연료/반사체 구분:")
    print("  ■ = 내부 연료 (CMFD only)")
    print("  □ = 반사체 인접 연료 (Albedo 필요)")
    print("  R = 반사체, · = 빈공간\n")

    print("         " + "  ".join(f"{i+1:2d}" for i in range(11)) + "  ← I (동→)")
    for j in range(11):
        row = f"  J={j+1:2d}   "
        for i in range(11):
            if IS_FUEL[j, i]:
                if (j, i) in direct_boundaries:
                    row += " □ "
                else:
                    row += " ■ "
            elif ASM_MAP[j, i] == 'o':
                row += " · "
            else:
                row += " R "
        suffix = " ← 북" if j == 0 else " ← 남" if j == 10 else ""
        print(row + suffix)
    print("  ↓ J (남→)")

    # 맵 2: 경계면 방향 표시
    print(f"\n[맵 2] 경계면 방향 (어떤 면이 반사체와 인접):")
    print("  기호: N/S/E/W = 해당 방향에 반사체 인접")
    print("  대문자 = ortho(R1,R2), 소문자 = diag(R3~R6)\n")

    print("         " + "  ".join(f"{i+1:2d}" for i in range(11)))
    for j in range(11):
        row = f"  J={j+1:2d}   "
        for i in range(11):
            if not IS_FUEL[j, i]:
                row += " ·  "
                continue
            faces = direct_boundaries.get((j, i), {})
            if not faces:
                row += " ■  "
                continue
            symbols = ""
            for face in ['N', 'S', 'E', 'W']:
                if face in faces:
                    rname, rtype = faces[face]
                    symbols += face.upper() if rtype == 'ortho' else face.lower()
            row += f"{symbols:4s}"
        print(row)

    # ── 4. Quarter-core 시각화 ──
    print(f"\n{'='*72}")
    print("Quarter-core (5×5) 시각화 — 우하단 (I=6~10, J=6~10)")
    print("=" * 72)

    print("\n[맵 3] Quarter-core 경계면:")
    print("         " + "  ".join(f"x={x}" for x in range(5)))
    for y in range(5):
        j_full = y + 5  # J=6~10 (0-based: 5~9)
        row = f"  y={y}    "
        for x in range(5):
            i_full = x + 5
            if not IS_FUEL[j_full, i_full]:
                row += " ·  "
                continue
            faces = direct_boundaries.get((j_full, i_full), {})
            if not faces:
                row += " ■  "
                continue
            symbols = ""
            for face in ['N', 'S', 'E', 'W']:
                if face in faces:
                    rname, rtype = faces[face]
                    symbols += face.upper() if rtype == 'ortho' else face.lower()
            row += f"{symbols:4s}"
        asm_names = " ".join(ASM_MAP[y + 5, x + 5] for x in range(5))
        print(row + f"  ({asm_names})")

    # 경계면 집계
    print(f"\n[집계]")
    ortho_count = 0
    diag_count = 0
    for (aj, ai), faces in direct_boundaries.items():
        for face, (rname, rtype) in faces.items():
            if rtype == 'ortho':
                ortho_count += 1
            elif rtype == 'diag':
                diag_count += 1

    print(f"  Full-core 경계 어셈블리: {len(direct_boundaries)}개")
    print(f"  Full-core 경계면: ortho {ortho_count}면, diag {diag_count}면, 합계 {ortho_count + diag_count}면")

    # Quarter-core
    qc_boundaries = {k: v for k, v in direct_boundaries.items() if k[0] >= 5 and k[1] >= 5}
    qc_ortho = sum(1 for faces in qc_boundaries.values() for _, (_, rt) in faces.items() if rt == 'ortho')
    qc_diag = sum(1 for faces in qc_boundaries.values() for _, (_, rt) in faces.items() if rt == 'diag')
    print(f"  Quarter-core 경계 어셈블리: {len(qc_boundaries)}개")
    print(f"  Quarter-core 경계면: ortho {qc_ortho}면, diag {qc_diag}면, 합계 {qc_ortho + qc_diag}면")


if __name__ == "__main__":
    main()
