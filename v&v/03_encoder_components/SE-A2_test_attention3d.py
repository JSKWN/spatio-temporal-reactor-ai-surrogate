"""SE-A2 V&V — attention3d.py component verification.

Target: STRINGRelativePE3D, FullAttention3D
Grid: (Z, qH, qW) = (20, 6, 6), N=720 (halo expanded)
Date: 2026-04-15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf
from src.models.spatial.attention3d import STRINGRelativePE3D, FullAttention3D
from src.models.spatial.encoder3d import _build_coords

B, Z, qH, qW, D = 2, 20, 6, 6, 128
N = Z * qH * qW  # 720
num_heads = 4
head_dim = D // num_heads  # 32
n_blocks = head_dim // 2   # 16

print("=" * 70)
print("SE-A2 V&V — attention3d.py (STRINGRelativePE3D, FullAttention3D)")
print(f"Grid: (Z={Z}, qH={qH}, qW={qW}), N={N}, D={D}, H={num_heads}")
print("=" * 70)

all_pass = True

# --- Coords ---
coords = tf.constant(_build_coords(Z, qH, qW), dtype=tf.float32)

# ================================================================
# G1 + G2: STRINGRelativePE3D shape + params
# ================================================================
print("\n--- STRINGRelativePE3D ---")

string_pe = STRINGRelativePE3D(num_heads=num_heads, head_dim=head_dim)
q = tf.random.normal((B, num_heads, N, head_dim))
k = tf.random.normal((B, num_heads, N, head_dim))
q_rot, k_rot = string_pe(q, k, coords)

ok = q_rot.shape.as_list() == [B, num_heads, N, head_dim]
print(f"  G1 q_rot shape: {q_rot.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

ok = k_rot.shape.as_list() == [B, num_heads, N, head_dim]
print(f"  G1 k_rot shape: {k_rot.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

string_params = sum(tf.size(v).numpy() for v in string_pe.trainable_variables)
expected_params = num_heads * n_blocks * 3  # theta_z + theta_y + theta_x
ok = int(string_params) == expected_params
print(f"  G2 params: {string_params} (expected {expected_params})  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# L2 norm preservation (rotation isometry)
# ================================================================
print("\n--- L2 Norm Preservation ---")

q_norm = tf.norm(q, axis=-1)       # (B, H, N)
q_rot_norm = tf.norm(q_rot, axis=-1)
max_diff = tf.reduce_max(tf.abs(q_norm - q_rot_norm)).numpy()
ok = max_diff < 1e-5
print(f"  max|norm(q) - norm(q_rot)| = {max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# Translation invariance
# ================================================================
print("\n--- Translation Invariance ---")

attn = FullAttention3D(d_latent=D, num_heads=num_heads)
x = tf.random.normal((B, N, D))

# Normal coords
y = attn(x, coords, training=False)

# Shifted coords (+0.1)
coords_shifted = coords + 0.1
y_shifted = attn(x, coords_shifted, training=False)

max_shift_diff = tf.reduce_max(tf.abs(y - y_shifted)).numpy()
ok = max_shift_diff < 1e-5
print(f"  max|y - y_shifted| = {max_shift_diff:.2e}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# G1 + G2: FullAttention3D shape + params
# ================================================================
print("\n--- FullAttention3D ---")

ok = y.shape.as_list() == [B, N, D]
print(f"  G1 output shape: {y.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

attn_params = sum(tf.size(v).numpy() for v in attn.trainable_variables)
# Q/K/V proj: 3 * (D*D + D) = 3 * 16512 = 49536
# Out proj: D*D + D = 16512
# STRING: 192
expected_attn = 3 * (D * D + D) + (D * D + D) + expected_params
ok = int(attn_params) == expected_attn
print(f"  G2 params: {attn_params:,} (expected {expected_attn:,})  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# Coords validation
# ================================================================
print("\n--- Coords Validation ---")

coords_np = coords.numpy()
ok_shape = coords_np.shape == (N, 3)
ok_min = np.isclose(coords_np.min(), 0.0)
ok_max = np.isclose(coords_np.max(), 1.0)
print(f"  coords shape: {coords_np.shape}  {'PASS' if ok_shape else 'FAIL'}")
print(f"  coords range: [{coords_np.min():.2f}, {coords_np.max():.2f}]  {'PASS' if ok_min and ok_max else 'FAIL'}")
all_pass &= ok_shape and ok_min and ok_max

# Z vs XY spacing ratio
z_coords = coords_np[:, 0].reshape(Z, qH, qW)
y_coords = coords_np[:, 1].reshape(Z, qH, qW)
z_step = z_coords[1, 0, 0] - z_coords[0, 0, 0]
y_step = y_coords[0, 1, 0] - y_coords[0, 0, 0]
# z_pitch=10cm over 19 intervals -> step = 1/19
# xy_pitch=21.6cm over 5 intervals -> step = 1/5
print(f"  Z step: {z_step:.4f} (expected {1/19:.4f})")
print(f"  Y step: {y_step:.4f} (expected {1/5:.4f})")
ok_ratio = abs(z_step - 1/19) < 1e-5 and abs(y_step - 1/5) < 1e-5
print(f"  Step ratio reflects physics pitch: {'PASS' if ok_ratio else 'FAIL'}")
all_pass &= ok_ratio

# ================================================================
print("\n" + "=" * 70)
print(f"SE-A2 V&V: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
