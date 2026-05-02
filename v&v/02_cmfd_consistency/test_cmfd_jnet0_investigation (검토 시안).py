"""
⚠️ 본 파일 은 (검토 시안) — 45 줄 보조 스크립트, 결론 미제시.
최종 확정 결과 는 다음 자료 참조:
  - piecewise-test/JNET0(MAS_NXS)의 공식 정합성 검증(오차0%확인)/2026-03-31_step0_abs_jnet0_result.txt (A1)
  - piecewise-test/JNET0(MAS_NXS)의 공식 정합성 검증(오차0%확인)/2026-03-31_step0_abs_jnet0_verification.py (A2)
  - 분류 기록: project_fnc/v-smr_load_following/data_preprocess/archive/작성 내용 및 계획/2026-04-23 CMFD 모듈 검증 기록.md §3

CMFD vs JNET0 심층 조사: 오차 원인 분석.

목적:
  1. JNET0 부호 반전 테스트: JNET0×2 vs JNET0×(-2)
  2. 면적 스케일링 확인: J [n/cm²/s] vs J×A [n/s]
  3. 거리 계산 확인: h = dx/2 vs h = dx
  4. JNET0가 이미 면적 적분된 값인지 확인

작성일: 2026-04-02
"""

import re
import sys
from pathlib import Path
import numpy as np

# ─── 경로 설정 ───
sys.path.insert(0, str(Path(r"C:\Users\Administrator\Documents\GitHub\project_fnc\v-smr_load_following\data_preprocess")))

WORKSPACE = Path(r"D:\workspace_lf_20260326_40LP")
LP = "LP_0000"
PROFILE = "t12_363_p50_power_lower"
STEP = "s0001_crs"

NXS_PATH = WORKSPACE / LP / PROFILE / f"MAS_NXS_{PROFILE}_{STEP}"

NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?')


def parse_nums(line: str) -> list[float]:
    return [float(m.group()) for m in NUM_RE.finditer(line)]


def parse_nxs_full(nxs_path: Path):
    """MAS_NXS에서 DIF, FLX, JNET0 모두 추출."""
    lines = nxs_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    nx_nxy_nz = parse_nums(lines[1])
    nx, nxy, nz = int(nx_nxy_nz[0]), int(nx_nxy_nz[1]), int(nx_nxy_nz[2])
    wide = parse_nums(lines[3])[0]
    zmesh = parse_nums(lines[4])

    cols = lines[7].split()
    col_map = {name: idx for idx, name in enumerate(cols)}
