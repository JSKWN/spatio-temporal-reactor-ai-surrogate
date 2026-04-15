"""SE-A3 V&V — ConditionalLAPE3D component verification.

Target: ConditionalLAPE3D (symmetry-conditioned position embedding)
Grid: (Z, qH, qW) = (20, 6, 6), D=128
Date: 2026-04-15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
from src.models.spatial.layers3d import ConditionalLAPE3D, CellEmbedder

B, Z, qH, qW, D = 2, 20, 6, 6, 128

print("=" * 70)
print("SE-A3 V&V — ConditionalLAPE3D")
print(f"Grid: (Z={Z}, qH={qH}, qW={qW}), D={D}")
print("=" * 70)

all_pass = True
lape = ConditionalLAPE3D(z_dim=Z, qh_dim=qH, qw_dim=qW, d_latent=D)
x = tf.random.normal((B, Z, qH, qW, D))

# ================================================================
# G1: Shape
# ================================================================
print("\n--- G1: Shape ---")
out0 = lape(x, tf.constant([0, 0]))
ok = out0.shape.as_list() == [B, Z, qH, qW, D]
print(f"  sym=[0,0]: {x.shape} -> {out0.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

out1 = lape(x, tf.constant([1, 1]))
ok = out1.shape.as_list() == [B, Z, qH, qW, D]
print(f"  sym=[1,1]: {x.shape} -> {out1.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# G2: Parameter Count
# ================================================================
print("\n--- G2: Parameter Count ---")
params = sum(tf.size(v).numpy() for v in lape.trainable_variables)
expected = Z * qH * qW * D * 2  # mirror + rotation
ok = int(params) == expected
print(f"  params: {params:,} (expected {expected:,})  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# C6: Table selection
# ================================================================
print("\n--- C6: Table Selection ---")
# mirror != rotation output
diff = not tf.reduce_all(out0 == out1).numpy()
print(f"  sym=0 vs sym=1 output differs: {'PASS' if diff else 'FAIL'}")
all_pass &= diff

# ================================================================
# C7: Gradient isolation
# ================================================================
print("\n--- C7: Gradient Isolation ---")

# sym=0 -> only mirror gets gradient
with tf.GradientTape() as tape:
    o = lape(x, tf.constant([0, 0]))
    loss = tf.reduce_mean(o)
grads = tape.gradient(loss, lape.trainable_variables)

for v, g in zip(lape.trainable_variables, grads):
    has_grad = g is not None and tf.reduce_any(g != 0).numpy()
    if 'mirror' in v.name:
        ok = has_grad
        print(f"  sym=0: {v.name} gradient exists: {'PASS' if ok else 'FAIL'}")
    elif 'rotation' in v.name:
        ok = not has_grad
        print(f"  sym=0: {v.name} gradient = 0: {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

# sym=1 -> only rotation gets gradient
with tf.GradientTape() as tape:
    o = lape(x, tf.constant([1, 1]))
    loss = tf.reduce_mean(o)
grads = tape.gradient(loss, lape.trainable_variables)

for v, g in zip(lape.trainable_variables, grads):
    has_grad = g is not None and tf.reduce_any(g != 0).numpy()
    if 'rotation' in v.name:
        ok = has_grad
        print(f"  sym=1: {v.name} gradient exists: {'PASS' if ok else 'FAIL'}")
    elif 'mirror' in v.name:
        ok = not has_grad
        print(f"  sym=1: {v.name} gradient = 0: {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

# ================================================================
# C8: Per-sample selection
# ================================================================
print("\n--- C8: Per-sample Selection ---")
out_mix = lape(x, tf.constant([0, 1]))
s0_ok = tf.reduce_all(out_mix[0] == out0[0]).numpy()
s1_ok = tf.reduce_all(out_mix[1] == out1[1]).numpy()
print(f"  sample 0 (sym=0) matches mirror: {'PASS' if s0_ok else 'FAIL'}")
print(f"  sample 1 (sym=1) matches rotation: {'PASS' if s1_ok else 'FAIL'}")
all_pass &= s0_ok and s1_ok

# ================================================================
# C9: get_norm_map()
# ================================================================
print("\n--- C9: get_norm_map() ---")
norms = lape.get_norm_map()
ok_keys = set(norms.keys()) == {'mirror', 'rotation'}
ok_mirror = norms['mirror'].shape.as_list() == [Z, qH, qW]
ok_rot = norms['rotation'].shape.as_list() == [Z, qH, qW]
ok_dtype = norms['mirror'].dtype == tf.float32 and norms['rotation'].dtype == tf.float32
print(f"  keys: {list(norms.keys())}  {'PASS' if ok_keys else 'FAIL'}")
print(f"  mirror shape: {norms['mirror'].shape}  {'PASS' if ok_mirror else 'FAIL'}")
print(f"  rotation shape: {norms['rotation'].shape}  {'PASS' if ok_rot else 'FAIL'}")
print(f"  dtype: float32  {'PASS' if ok_dtype else 'FAIL'}")
all_pass &= ok_keys and ok_mirror and ok_rot and ok_dtype

# ================================================================
# D1: Conditional LAPE effect (not ignored)
# ================================================================
print("\n--- D1: Conditional LAPE Effect ---")
l2_diff = tf.norm(out0 - out1).numpy()
l2_out = tf.norm(out0).numpy()
ratio = l2_diff / l2_out
print(f"  L2(mirror_out - rotation_out) = {l2_diff:.4f}")
print(f"  L2(mirror_out) = {l2_out:.4f}")
print(f"  ratio = {ratio:.4f}")
ok = ratio > 0.001  # 0.1% threshold
print(f"  LAPE not ignored (ratio > 0.1%): {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# D2: LAPE scale vs CellEmbedder scale
# ================================================================
print("\n--- D2: LAPE Scale vs CellEmbedder Scale ---")
ce = CellEmbedder(d_latent=D)
x_raw = tf.random.normal((B, Z, qH, qW, 21))
ce_out = ce(x_raw)
ce_std = tf.math.reduce_std(ce_out).numpy()
lape_std_m = tf.math.reduce_std(lape.lape_mirror).numpy()
lape_std_r = tf.math.reduce_std(lape.lape_rotation).numpy()
scale_ratio = max(lape_std_m, lape_std_r) / ce_std
print(f"  CellEmbedder output std: {ce_std:.4f}")
print(f"  LAPE mirror std: {lape_std_m:.4f}")
print(f"  LAPE rotation std: {lape_std_r:.4f}")
print(f"  scale ratio (LAPE/CellEmb): {scale_ratio:.4f}")
ok = scale_ratio < 10.0  # 10x threshold
print(f"  LAPE does not dominate (ratio < 10x): {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
print("\n" + "=" * 70)
print(f"SE-A3 V&V: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
