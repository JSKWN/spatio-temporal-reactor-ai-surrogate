"""SE-A4 V&V — halo_expand function verification.

Target: halo_expand (mirror/rotation symmetry mapping)
Grid: (Z, N, N, C) -> (Z, N+1, N+1, C), tested with N=5
Date: 2026-04-15

Includes V1: text-based grid visualization of symmetry mapping.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf
from src.models.spatial.halo_expand import halo_expand

print("=" * 70)
print("SE-A4 V&V — halo_expand (mirror / rotation)")
print("=" * 70)

all_pass = True

# ================================================================
# Test data: identifiable values for visual verification
# Use single channel, z=1 for clarity. Value = qy*10 + qx
# Quarter (5x5):
#   [ 0  1  2  3  4]
#   [10 11 12 13 14]
#   [20 21 22 23 24]
#   [30 31 32 33 34]
#   [40 41 42 43 44]
# ================================================================
vals = np.array([[qy * 10 + qx for qx in range(5)] for qy in range(5)], dtype=np.float32)
quarter = tf.constant(vals.reshape(1, 1, 5, 5, 1))  # (B=1, Z=1, 5, 5, C=1)

hm = halo_expand(quarter, tf.constant(0))  # mirror
hr = halo_expand(quarter, tf.constant(1))  # rotation

hm_2d = hm[0, 0, :, :, 0].numpy().astype(int)
hr_2d = hr[0, 0, :, :, 0].numpy().astype(int)
q_2d = vals.astype(int)

# ================================================================
# G1: Shape
# ================================================================
print("\n--- G1: Shape ---")
B, Z, C = 2, 20, 21
q_full = tf.random.normal((B, Z, 5, 5, C))
hm_full = halo_expand(q_full, tf.constant(0))
hr_full = halo_expand(q_full, tf.constant(1))
ok = hm_full.shape.as_list() == [B, Z, 6, 6, C]
print(f"  mirror: ({B},{Z},5,5,{C}) -> {hm_full.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok
ok = hr_full.shape.as_list() == [B, Z, 6, 6, C]
print(f"  rotation: ({B},{Z},5,5,{C}) -> {hr_full.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# V1: Symmetry mapping visualization
# ================================================================
print("\n--- V1: Symmetry Mapping Visualization ---")

print("\n  Quarter (5x5) — value = qy*10 + qx:")
print("  " + "-" * 26)
for qy in range(5):
    row = " ".join(f"{q_2d[qy, qx]:3d}" for qx in range(5))
    print(f"  | {row} |")
print("  " + "-" * 26)

print("\n  Mirror halo (6x6):")
print("  h\\w  " + "".join(f" w={w} " for w in range(6)))
print("  " + "-" * 38)
for h in range(6):
    tag = "HALO" if h == 0 else f"q{h-1}  "
    row = " ".join(f"{hm_2d[h, w]:4d}" for w in range(6))
    print(f"  {tag}| {row} |")
print("  " + "-" * 38)

print("\n  Mirror mapping rule:")
print("    halo(0, w) = inner(2, w) = quarter[1, :]")
print("    halo(h, 0) = inner(h, 2) = quarter[:, 1]")
print("    halo(0, 0) = inner(2, 2) = quarter[1, 1]")

print("\n  Rotation halo (6x6):")
print("  h\\w  " + "".join(f" w={w} " for w in range(6)))
print("  " + "-" * 38)
for h in range(6):
    tag = "HALO" if h == 0 else f"q{h-1}  "
    row = " ".join(f"{hr_2d[h, w]:4d}" for w in range(6))
    print(f"  {tag}| {row} |")
print("  " + "-" * 38)

print("\n  Rotation mapping rule:")
print("    halo(0, w) = inner(w, 2) = quarter[w-1, 1]  (transpose)")
print("    halo(h, 0) = inner(2, h) = quarter[1, h-1]  (transpose)")
print("    halo(0, 0) = inner(2, 2) = quarter[1, 1]")

print("\n  Mirror vs Rotation difference (halo cells only):")
print(f"    halo row (h=0): mirror={list(hm_2d[0, 1:])} vs rotation={list(hr_2d[0, 1:])}")
print(f"    halo col (w=0): mirror={list(hm_2d[1:, 0])} vs rotation={list(hr_2d[1:, 0])}")
print(f"    corner (0,0):   mirror={hm_2d[0,0]} vs rotation={hr_2d[0,0]} (same)")

# ================================================================
# C1: Mirror mapping correctness
# ================================================================
print("\n--- C1: Mirror Mapping ---")

# halo_row(0, w=1..5) = inner(2, w) = quarter[1, :]
ok = np.array_equal(hm_2d[0, 1:], q_2d[1, :])
print(f"  halo_row = quarter[1,:] = {list(q_2d[1,:])}: {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# halo_col(h=1..5, 0) = inner(h, 2) = quarter[:, 1]
ok = np.array_equal(hm_2d[1:, 0], q_2d[:, 1])
print(f"  halo_col = quarter[:,1] = {list(q_2d[:,1])}: {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# corner(0,0) = inner(2,2) = quarter[1,1]
ok = hm_2d[0, 0] == q_2d[1, 1]
print(f"  corner = quarter[1,1] = {q_2d[1,1]}: {'PASS' if ok else 'FAIL'}")
all_pass &= bool(ok)

# ================================================================
# C2: Rotation mapping correctness
# ================================================================
print("\n--- C2: Rotation Mapping ---")

# halo_row(0, w=1..5) = inner(w, 2) = quarter[w-1, 1] -> transposed
expected_rot_row = [q_2d[w - 1, 1] for w in range(1, 6)]
ok = list(hr_2d[0, 1:]) == expected_rot_row
print(f"  halo_row = [q[w-1,1]] = {expected_rot_row}: {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# halo_col(h=1..5, 0) = inner(2, h) = quarter[1, h-1] -> transposed
expected_rot_col = [q_2d[1, h - 1] for h in range(1, 6)]
ok = list(hr_2d[1:, 0]) == expected_rot_col
print(f"  halo_col = [q[1,h-1]] = {expected_rot_col}: {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# corner(0,0) = inner(2,2) = quarter[1,1]
ok = hr_2d[0, 0] == q_2d[1, 1]
print(f"  corner = quarter[1,1] = {q_2d[1,1]}: {'PASS' if ok else 'FAIL'}")
all_pass &= bool(ok)

# ================================================================
# C3: Inner cell preservation
# ================================================================
print("\n--- C3: Inner Cell Preservation ---")
ok_m = np.array_equal(hm_2d[1:, 1:], q_2d)
ok_r = np.array_equal(hr_2d[1:, 1:], q_2d)
print(f"  mirror halo[1:,1:] == quarter: {'PASS' if ok_m else 'FAIL'}")
print(f"  rotation halo[1:,1:] == quarter: {'PASS' if ok_r else 'FAIL'}")
all_pass &= ok_m and ok_r

# ================================================================
# C4: Mirror != Rotation
# ================================================================
print("\n--- C4: Mirror != Rotation ---")
diff = not np.array_equal(hm_2d, hr_2d)
print(f"  mirror != rotation (asymmetric data): {'PASS' if diff else 'FAIL'}")
all_pass &= diff

# same inner, different halo
inner_same = np.array_equal(hm_2d[1:, 1:], hr_2d[1:, 1:])
halo_diff = not np.array_equal(hm_2d[0, :], hr_2d[0, :])
print(f"  inner cells identical: {'PASS' if inner_same else 'FAIL'}")
print(f"  halo row differs: {'PASS' if halo_diff else 'FAIL'}")
all_pass &= inner_same and halo_diff

# ================================================================
# C5: Batch per-sample selection
# ================================================================
print("\n--- C5: Batch Per-sample Selection ---")
quarter_batch = tf.concat([quarter, quarter], axis=0)  # (2, 1, 5, 5, 1)
hb = halo_expand(quarter_batch, tf.constant([0, 1]))
hb_s0 = hb[0, 0, :, :, 0].numpy().astype(int)
hb_s1 = hb[1, 0, :, :, 0].numpy().astype(int)
ok_s0 = np.array_equal(hb_s0, hm_2d)
ok_s1 = np.array_equal(hb_s1, hr_2d)
print(f"  sample 0 (sym=0) matches mirror: {'PASS' if ok_s0 else 'FAIL'}")
print(f"  sample 1 (sym=1) matches rotation: {'PASS' if ok_s1 else 'FAIL'}")
all_pass &= ok_s0 and ok_s1

# ================================================================
print("\n" + "=" * 70)
print(f"SE-A4 V&V: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
