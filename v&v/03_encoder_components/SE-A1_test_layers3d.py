"""SE-A1 V&V — layers3d.py 컴포넌트 단독 검증.

대상: CellEmbedder, FFN3D, LearnedAbsolutePE3D
Grid: (Z, qH, qW) = (20, 6, 6) — halo expand 후 기준
작성일: 2026-04-15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
from src.models.spatial.layers3d import CellEmbedder, FFN3D, LearnedAbsolutePE3D

B, Z, qH, qW, C_in, D = 2, 20, 6, 6, 21, 128
N = Z * qH * qW  # 720

print("=" * 70)
print("SE-A1 V&V — layers3d.py (CellEmbedder, FFN3D, LearnedAbsolutePE3D)")
print(f"Grid: (Z={Z}, qH={qH}, qW={qW}), N={N}, C_in={C_in}, D={D}")
print("=" * 70)

all_pass = True

# ================================================================
# G1: Shape 검증
# ================================================================
print("\n--- G1: Shape 검증 ---")

# CellEmbedder
ce = CellEmbedder(d_latent=D)
x_ce = tf.random.normal((B, Z, qH, qW, C_in))
y_ce = ce(x_ce)
ok = y_ce.shape.as_list() == [B, Z, qH, qW, D]
print(f"  CellEmbedder: {x_ce.shape} → {y_ce.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# LearnedAbsolutePE3D
lape = LearnedAbsolutePE3D(z_dim=Z, qh_dim=qH, qw_dim=qW, d_latent=D)
x_lape = tf.random.normal((B, Z, qH, qW, D))
y_lape = lape(x_lape)
ok = y_lape.shape.as_list() == [B, Z, qH, qW, D]
print(f"  LearnedAbsolutePE3D: {x_lape.shape} → {y_lape.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# FFN3D
ffn = FFN3D(d_latent=D, expand_ratio=4)
x_ffn = tf.random.normal((B, N, D))
y_ffn = ffn(x_ffn)
ok = y_ffn.shape.as_list() == [B, N, D]
print(f"  FFN3D: {x_ffn.shape} → {y_ffn.shape}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# get_norm_map
norm_map = lape.get_norm_map()
ok = norm_map.shape.as_list() == [Z, qH, qW] and norm_map.dtype == tf.float32
print(f"  get_norm_map(): shape={norm_map.shape}, dtype={norm_map.dtype}  {'PASS' if ok else 'FAIL'}")
all_pass &= ok

# ================================================================
# G2: Parameter Count
# ================================================================
print("\n--- G2: Parameter Count ---")

tests = [
    ("CellEmbedder", ce, C_in * D + D),
    ("LearnedAbsolutePE3D", lape, Z * qH * qW * D),
    ("FFN3D (expand=4)", ffn, (D * D * 4 + D * 4) + (D * 4 * D + D)),
]

for name, layer, expected in tests:
    measured = sum(tf.size(v).numpy() for v in layer.trainable_variables)
    ok = int(measured) == expected
    print(f"  {name}: 측정={measured:,}, 수식={expected:,}  {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

# ================================================================
# 결과
# ================================================================
print("\n" + "=" * 70)
print(f"SE-A1 V&V 결과: {'ALL PASS' if all_pass else 'FAIL'}")
print("=" * 70)
