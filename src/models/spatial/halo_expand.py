"""Halo expand 함수 (SE-A4).

Quarter core (..., Z, N, N, C) → halo-expanded (..., Z, N+1, N+1, C).
대칭 유형 (mirror / rotation) 에 따라 halo cell 매핑이 다르다.
N×N 크기에 제한 없음 (현재 사용: N=5 → 6×6).

좌표계 (05a_symmetry_distinguishability_proof.md §2 참조):
    - Quarter (Q): (qy, qx) ∈ {0..N-1}². Q4 영역.
    - Halo (H):   (h, w) ∈ {0..N}². h = qy+1, w = qx+1.
        Inner cell: h ∈ {1..N}, w ∈ {1..N}
        Halo row:   h = 0, w ∈ {1..N}
        Halo col:   h ∈ {1..N}, w = 0
        Halo corner: (h, w) = (0, 0)

매핑 규칙 (05a §4 유도, 임의 N에 대해 성립):
    Mirror:
        halo(0, w) = inner(2, w)     — quarter[..., 1, :, :]  (2행 복사)
        halo(h, 0) = inner(h, 2)     — quarter[..., :, 1, :]  (2열 복사)
        halo(0, 0) = inner(2, 2)     — quarter[..., 1, 1, :]

    Rotation (90°):
        halo(0, w) = inner(w, 2)     — quarter[..., :, 1, :] 의 전치
        halo(h, 0) = inner(2, h)     — quarter[..., 1, :, :] 의 전치
        halo(0, 0) = inner(2, 2)     — quarter[..., 1, 1, :]  (corner 동일)

설계 근거: 05_symmetry_mode.md, 05a_symmetry_distinguishability_proof.md §4
작성일: 2026-04-15
"""

from __future__ import annotations

import tensorflow as tf


def halo_expand(quarter: tf.Tensor, sym_type: tf.Tensor) -> tf.Tensor:
    """Quarter core를 halo-expanded 격자로 변환.

    대칭축은 항상 Q4 경계에서 1칸 바깥에 위치하므로,
    halo cell은 항상 quarter 내부의 2번째 행/열 (index 1) 에 매핑된다.
    이 성질은 quarter 크기 N에 무관하게 성립한다.

    Args:
        quarter: (..., Z, N, N, C) — quarter core 데이터.
                 앞쪽 차원은 자유 (예: (B, Z, N, N, C) 또는 (B, T, Z, N, N, C)).
        sym_type: 스칼라 또는 (B,) int32 — 0=mirror, 1=rotation.

    Returns:
        (..., Z, N+1, N+1, C) — halo cell이 채워진 텐서.
    """
    rank = len(quarter.shape)

    # --- Mirror halo 매핑 ---
    # halo_row: inner(2, w) = quarter[..., 1, :, :]  shape (..., Z, 1, N, C)
    # halo_col: inner(h, 2) = quarter[..., :, 1, :]  shape (..., Z, N, 1, C)
    # corner:   inner(2, 2) = quarter[..., 1, 1, :]  shape (..., Z, 1, 1, C)
    mirror_row = quarter[..., 1:2, :, :]
    mirror_col = quarter[..., :, 1:2, :]
    mirror_corner = quarter[..., 1:2, 1:2, :]

    # --- Rotation halo 매핑 ---
    # halo_row: inner(w, 2) = quarter[..., :, 1, :] 의 H↔W 전치
    #   quarter[..., :, 1, :] → (..., Z, N, C) → expand → (..., Z, N, 1, C)
    #   전치 → (..., Z, 1, N, C)
    rot_row_src = quarter[..., :, 1:2, :]          # (..., Z, N, 1, C)
    perm = list(range(rank))
    perm[-3], perm[-2] = perm[-2], perm[-3]        # H축 ↔ W축 교환
    rot_row = tf.transpose(rot_row_src, perm=perm)  # (..., Z, 1, N, C)

    # halo_col: inner(2, h) = quarter[..., 1, :, :] 의 H↔W 전치
    #   quarter[..., 1, :, :] → (..., Z, 1, N, C) → 전치 → (..., Z, N, 1, C)
    rot_col_src = quarter[..., 1:2, :, :]          # (..., Z, 1, N, C)
    rot_col = tf.transpose(rot_col_src, perm=perm)  # (..., Z, N, 1, C)

    # corner: inner(2, 2) — mirror와 동일
    rot_corner = mirror_corner

    # --- sym_type 분기 ---
    is_mirror = tf.equal(sym_type, 0)

    if len(is_mirror.shape) == 0:
        # 스칼라 sym_type
        halo_row = tf.cond(is_mirror,
                           lambda: mirror_row, lambda: rot_row)
        halo_col = tf.cond(is_mirror,
                           lambda: mirror_col, lambda: rot_col)
        halo_corner = tf.cond(is_mirror,
                              lambda: mirror_corner, lambda: rot_corner)
    else:
        # 배치 sym_type (B,) — per-sample 선택
        cond = is_mirror
        for _ in range(rank - 1):
            cond = tf.expand_dims(cond, axis=-1)
        halo_row = tf.where(cond, mirror_row, rot_row)
        halo_col = tf.where(cond, mirror_col, rot_col)
        halo_corner = tf.where(cond, mirror_corner, rot_corner)

    # --- 조립: (N+1, N+1) 격자 ---
    # 상단 행: [corner(1,1)] + [halo_row(1,N)] → (..., Z, 1, N+1, C)
    top_row = tf.concat([halo_corner, halo_row], axis=-2)

    # 하단 N행: [halo_col(N,1)] + [quarter(N,N)] → (..., Z, N, N+1, C)
    bottom_rows = tf.concat([halo_col, quarter], axis=-2)

    # 전체: top + bottom → (..., Z, N+1, N+1, C)
    halo_grid = tf.concat([top_row, bottom_rows], axis=-3)

    return halo_grid
