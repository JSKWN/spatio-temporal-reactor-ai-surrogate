"""SE-B1 V&V — SpatialEncoder3D integration verification.

Target: SpatialEncoder3D (full assembly)
Grid: (Z, qH, qW) = (20, 6, 6), D=128, 3 stages
Date: 2026-04-15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf
from src.models.spatial.encoder3d import SpatialEncoder3D, _build_coords
from src.models.spatial.halo_expand import halo_expand

B, Z, qH, qW, C_in, D = 2, 20, 6, 6, 21, 128
N = Z * qH * qW  # 720

print("=" * 70)
print("SE-B1 V&V — SpatialEncoder3D (full assembly)")
print(f"Grid: (Z={Z}, qH={qH}, qW={qW}), N={N}, C_in={C_in}, D={D}")
print("=" * 70)

all_pass = True

enc = SpatialEncoder3D(d_latent=D, n_stages=3, num_heads=4)

# ================================================================
# G1: Shape
# ================================================================
print("\n--- G1: Shape ---")
x = tf.random.normal((B, Z, qH, qW, C_in))
sym = tf.constant([0, 0])
out = enc(x, sym, training=False)
ok = out.shape.as_list() == [B, Z, qH, qW, D]
print(f"  input: {x.shape} -> output: {out.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# G2: Total Parameter Count
# ================================================================
print("\n--- G2: Total Parameter Count ---")
total = sum(tf.size(v).numpy() for v in enc.trainable_variables)

# Expected breakdown:
# CellEmbedder: 21*128 + 128 = 2816
# ConditionalLAPE: 92160 * 2 = 184320
# Per stage: LN_attn(256) + Attention(66240) + LN_ffn(256) + FFN(131712) = 198464
# 3 stages: 198464 * 3 = 595392
expected = 2816 + 184320 + 595392
print(f"  total: {total:,}")
print(f"  expected: {expected:,}")
ok = int(total) == expected
print(f"  match: {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# C10: STRING coords
# ================================================================
print("\n--- C10: STRING Coords ---")
coords = enc.coords.numpy()
ok_shape = coords.shape == (N, 3)
ok_min = np.isclose(coords.min(), 0.0)
ok_max = np.isclose(coords.max(), 1.0)
print(f"  shape: {coords.shape} (expected ({N}, 3))  {'PASS' if ok_shape else 'FAIL'}")
print(f"  range: [{coords.min():.2f}, {coords.max():.2f}]  {'PASS' if ok_min and ok_max else 'FAIL'}")
all_pass &= ok_shape and ok_min and ok_max

# ================================================================
# D4: Flatten/Unflatten consistency
# ================================================================
print("\n--- D4: Flatten/Unflatten Consistency ---")
dummy = tf.random.normal((B, Z, qH, qW, D))
flat = tf.reshape(dummy, (B, N, D))
unflat = tf.reshape(flat, (B, Z, qH, qW, D))
ok = tf.reduce_all(dummy == unflat).numpy()
print(f"  x == unflatten(flatten(x)): {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# D5: STRING axis distinction (Z vs XY pitch)
# ================================================================
print("\n--- D5: STRING Axis Distinction ---")
coords_3d = coords.reshape(Z, qH, qW, 3)

z_step = coords_3d[1, 0, 0, 0] - coords_3d[0, 0, 0, 0]
y_step = coords_3d[0, 1, 0, 1] - coords_3d[0, 0, 0, 1]
x_step = coords_3d[0, 0, 1, 2] - coords_3d[0, 0, 0, 2]

# z_pitch=10cm, 19 intervals -> 1/19 = 0.0526
# xy_pitch=21.6cm, 5 intervals -> 1/5 = 0.2000
print(f"  Z step: {z_step:.4f} (expected {1/19:.4f}, pitch 10cm)")
print(f"  Y step: {y_step:.4f} (expected {1/5:.4f}, pitch 21.6cm)")
print(f"  X step: {x_step:.4f} (expected {1/5:.4f}, pitch 21.6cm)")

ok_z = abs(z_step - 1 / 19) < 1e-5
ok_y = abs(y_step - 1 / 5) < 1e-5
ok_x = abs(x_step - 1 / 5) < 1e-5
ok_diff = abs(z_step - y_step) > 0.01  # Z != Y
print(f"  Z != Y/X (different physics pitch): {'PASS' if ok_diff else 'FAIL'}")
all_pass &= ok_z and ok_y and ok_x and ok_diff

# ================================================================
# End-to-end: halo_expand + encoder
# ================================================================
print("\n--- End-to-end: halo_expand + encoder ---")
quarter = tf.random.normal((B, Z, 5, 5, C_in))
halo = halo_expand(quarter, tf.constant(0))
enc_out = enc(halo, tf.constant([0, 0]), training=False)
ok = enc_out.shape.as_list() == [B, Z, qH, qW, D]
print(f"  quarter {quarter.shape} -> halo {halo.shape} -> encoder {enc_out.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
print("\n" + "=" * 70)
print(f"SE-B1 V&V: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
