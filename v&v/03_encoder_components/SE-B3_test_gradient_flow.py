"""SE-B3 V&V — Gradient flow + domain-specific threat verification.

Target: SpatialEncoder3D full pipeline
Tests: G3 (gradient flow), G4 (dead activation), G5 (numerical stability),
       D3 (halo attention bias)
Date: 2026-04-15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf
from src.models.spatial.encoder3d import SpatialEncoder3D
from src.models.spatial.halo_expand import halo_expand

B, Z, qH, qW, C_in, D = 2, 20, 6, 6, 21, 128
N = Z * qH * qW

print("=" * 70)
print("SE-B3 V&V — Gradient Flow + Domain Threats")
print("=" * 70)

all_pass = True

enc = SpatialEncoder3D(d_latent=D, n_stages=3, num_heads=4)
x = tf.random.normal((B, Z, qH, qW, C_in))
sym = tf.constant([0, 0])

# ================================================================
# G3: Gradient Flow — all variables have gradient, no vanishing/exploding
# ================================================================
print("\n--- G3: Gradient Flow ---")

with tf.GradientTape() as tape:
    out = enc(x, sym, training=True)
    loss = tf.reduce_mean(out ** 2)

grads = tape.gradient(loss, enc.trainable_variables)

no_grad = []
grad_norms = []
for v, g in zip(enc.trainable_variables, grads):
    if g is None or not tf.reduce_any(g != 0).numpy():
        # rotation LAPE is expected to have no gradient when sym=0
        if 'rotation' not in v.name:
            no_grad.append(v.name)
    else:
        gn = tf.norm(g).numpy()
        grad_norms.append((v.name, gn))

if no_grad:
    print(f"  WARNING: {len(no_grad)} unexpected variables without gradient:")
    for n in no_grad:
        print(f"    - {n}")
    all_pass = False
else:
    print(f"  All variables have gradient (rotation LAPE excluded as expected)")

# Vanishing / Exploding check
if grad_norms:
    norms = [gn for _, gn in grad_norms]
    min_gn = min(norms)
    max_gn = max(norms)
    print(f"  Gradient norm range: [{min_gn:.2e}, {max_gn:.2e}]")
    ok_vanish = min_gn > 1e-12
    ok_explode = max_gn < 1e6
    print(f"  No vanishing (min > 1e-12): {'PASS' if ok_vanish else 'FAIL'}")
    print(f"  No exploding (max < 1e6): {'PASS' if ok_explode else 'FAIL'}")
    all_pass &= ok_vanish and ok_explode

    # Top/bottom 3 by gradient norm
    sorted_gn = sorted(grad_norms, key=lambda x: x[1])
    print(f"\n  Smallest gradient norms:")
    for name, gn in sorted_gn[:3]:
        short = name.split('/')[-1]
        print(f"    {short}: {gn:.2e}")
    print(f"  Largest gradient norms:")
    for name, gn in sorted_gn[-3:]:
        short = name.split('/')[-1]
        print(f"    {short}: {gn:.2e}")

# ================================================================
# G4: Dead Activation — GELU output zero ratio
# ================================================================
print("\n--- G4: Dead Activation (GELU) ---")

# Hook into FFN GELU outputs
# Run forward pass and check intermediate activation
h = enc.cell_embedder(x)
h = enc.lape(h, sym)
h = tf.reshape(h, (B, N, D))

for i, stage in enumerate(enc.stages):
    h_ln = stage.ln_attn(h)
    h_attn = stage.attention(h_ln, enc.coords, training=False)
    h = h + h_attn

    h_ln2 = stage.ln_ffn(h)
    # Access GELU output via FFN internals
    ffn_hidden = stage.ffn.dense_in(h_ln2)
    ffn_activated = stage.ffn.act(ffn_hidden)

    total = tf.size(ffn_activated).numpy()
    zeros = tf.reduce_sum(tf.cast(ffn_activated == 0.0, tf.int32)).numpy()
    ratio = zeros / total * 100

    h_ffn = stage.ffn(h_ln2, training=False)
    h = h + h_ffn

    ok = ratio < 50.0  # 50% threshold
    print(f"  Stage {i}: GELU zero ratio = {ratio:.1f}% ({zeros}/{total})  {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

# ================================================================
# G5: Numerical Stability
# ================================================================
print("\n--- G5: Numerical Stability ---")

# Normal random input
out_normal = enc(x, sym, training=False)
ok_nan = not tf.reduce_any(tf.math.is_nan(out_normal)).numpy()
ok_inf = not tf.reduce_any(tf.math.is_inf(out_normal)).numpy()
print(f"  Normal input: NaN={not ok_nan}, Inf={not ok_inf}  {'PASS' if ok_nan and ok_inf else 'FAIL'}")
all_pass &= ok_nan and ok_inf

# Large input (x10)
x_large = x * 10.0
out_large = enc(x_large, sym, training=False)
ok_nan = not tf.reduce_any(tf.math.is_nan(out_large)).numpy()
ok_inf = not tf.reduce_any(tf.math.is_inf(out_large)).numpy()
print(f"  Large input (x10): NaN={not ok_nan}, Inf={not ok_inf}  {'PASS' if ok_nan and ok_inf else 'FAIL'}")
all_pass &= ok_nan and ok_inf

# Zero input
x_zero = tf.zeros_like(x)
out_zero = enc(x_zero, sym, training=False)
ok_nan = not tf.reduce_any(tf.math.is_nan(out_zero)).numpy()
ok_inf = not tf.reduce_any(tf.math.is_inf(out_zero)).numpy()
print(f"  Zero input: NaN={not ok_nan}, Inf={not ok_inf}  {'PASS' if ok_nan and ok_inf else 'FAIL'}")
all_pass &= ok_nan and ok_inf

# ================================================================
# D3: Halo Attention Bias
# ================================================================
print("\n--- D3: Halo Attention Bias ---")

# Extract attention weights from stage 0
h_check = enc.cell_embedder(x)
h_check = enc.lape(h_check, sym)
h_check = tf.reshape(h_check, (B, N, D))
h_ln = enc.stages[0].ln_attn(h_check)

attn = enc.stages[0].attention
Q = attn.q_proj(h_ln)
K = attn.k_proj(h_ln)
V = attn.v_proj(h_ln)
Q = attn._split_heads(Q)
K = attn._split_heads(K)
Q, K = attn.string_pe(Q, K, enc.coords)
scale = attn.scale
logits = tf.matmul(Q, K, transpose_b=True) * scale  # (B, H, N, N)
weights = tf.nn.softmax(logits, axis=-1)  # (B, H, N, N)

# Average attention received by each token (column-wise mean)
attn_received = tf.reduce_mean(weights, axis=[0, 1, 2]).numpy()  # (N,)
attn_map = attn_received.reshape(Z, qH, qW)

# Halo cells: h=0 (row 0) or w=0 (col 0)
halo_mask = np.zeros((qH, qW), dtype=bool)
halo_mask[0, :] = True
halo_mask[:, 0] = True
inner_mask = ~halo_mask

# Average per z-level
halo_attn_avg = attn_map[:, halo_mask].mean()
inner_attn_avg = attn_map[:, inner_mask].mean()
ratio = halo_attn_avg / inner_attn_avg

print(f"  Avg attention received — halo: {halo_attn_avg:.6f}, inner: {inner_attn_avg:.6f}")
print(f"  Ratio (halo/inner): {ratio:.4f}")
ok = ratio < 3.0  # halo should not get 3x more attention
print(f"  Halo not dominating (ratio < 3.0): {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
print("\n" + "=" * 70)
print(f"SE-B3 V&V: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
