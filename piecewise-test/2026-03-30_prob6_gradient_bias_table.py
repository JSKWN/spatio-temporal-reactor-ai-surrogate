"""
Prob-6 검증: relativebiastable gradient 단절 테스트

목적:
  RelativeAttention3D의 relative_bias_table에 대한 gradient가
  build()에서 tf.gather_nd를 실행하면 끊기고,
  call()에서 실행하면 정상 흐름을 확인한다.

테스트:
  1. BrokenBiasAttention: build()에서 gather_nd 실행 (기존 방식) → gradient 없음
  2. FixedBiasAttention: call()에서 gather_nd 실행 (수정 방식) → gradient 있음
  3. 시각화: 두 방식의 gradient 존재 여부 + norm 비교 bar chart

실행:
  cd spatio-temporal-reactor-ai-surrogate
  python piecewise-test/2026-03-30_prob6_gradient_bias_table.py

작성일: 2026-03-30
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 공통: 3D lookup indices 생성 (상수)
# ─────────────────────────────────────────────
def compute_lookup_indices(D, H, W, num_heads):
    """
    3D 상대 좌표 → bias table 인덱스.
    relative_bias_table shape: (num_heads, 2D-1, 2H-1, 2W-1)
    lookup_indices shape: (num_heads, N, N, 4)  where N = D*H*W
    """
    N = D * H * W
    coords = np.stack(np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    ), axis=-1).reshape(N, 3)  # (N, 3)

    # 상대 좌표: (N, N, 3)
    rel = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    rel[:, :, 0] += D - 1  # offset to positive
    rel[:, :, 1] += H - 1
    rel[:, :, 2] += W - 1

    # (num_heads, N, N, 4) — [head_idx, z_rel, y_rel, x_rel]
    indices = np.zeros((num_heads, N, N, 4), dtype=np.int32)
    for h in range(num_heads):
        indices[h, :, :, 0] = h
        indices[h, :, :, 1:] = rel

    return tf.constant(indices)


# ─────────────────────────────────────────────
# 1. BrokenBiasAttention: build()에서 gather_nd (기존 — gradient 끊김)
# ─────────────────────────────────────────────
class BrokenBiasAttention(Layer):
    """기존 방식: build()에서 tf.gather_nd 실행 → reindexed_bias가 frozen tensor."""

    def __init__(self, hidden_size=32, head_size=16, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size

    def build(self, input_shape):
        B, D, H, W, C = input_shape
        N = D * H * W

        self.wq = Dense(self.hidden_size, use_bias=False)
        self.wk = Dense(self.hidden_size, use_bias=False)
        self.wv = Dense(self.hidden_size, use_bias=False)
        self.out_proj = Dense(self.hidden_size, use_bias=False)

        # 학습 가능한 bias table
        self.relative_bias_table = self.add_weight(
            name='relative_bias_table',
            shape=(self.num_heads, 2*D-1, 2*H-1, 2*W-1),
            initializer='zeros',
            trainable=True,
        )

        # ⚠️ build()에서 gather_nd 실행 → frozen tensor
        lookup = compute_lookup_indices(D, H, W, self.num_heads)
        self.reindexed_bias = tf.gather_nd(self.relative_bias_table, lookup)
        # shape: (num_heads, N, N)

        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        D, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        N = D * H * W

        x_flat = tf.reshape(x, [B, N, C])
        Q = self.wq(x_flat)
        K = self.wk(x_flat)
        V = self.wv(x_flat)

        Q = tf.reshape(Q, [B, N, self.num_heads, self.head_size])
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, heads, N, d_k)
        K = tf.reshape(K, [B, N, self.num_heads, self.head_size])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, [B, N, self.num_heads, self.head_size])
        V = tf.transpose(V, [0, 2, 1, 3])

        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.head_size, tf.float32)
        )
        # ⚠️ self.reindexed_bias는 build()에서 계산된 frozen tensor
        scores += self.reindexed_bias

        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, V)  # (B, heads, N, d_k)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.hidden_size])
        out = self.out_proj(out)
        return tf.reshape(out, [B, D, H, W, self.hidden_size])


# ─────────────────────────────────────────────
# 2. FixedBiasAttention: call()에서 gather_nd (수정 — gradient 정상)
# ─────────────────────────────────────────────
class FixedBiasAttention(Layer):
    """수정 방식: call()에서 tf.gather_nd 실행 → gradient 정상 흐름."""

    def __init__(self, hidden_size=32, head_size=16, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size

    def build(self, input_shape):
        B, D, H, W, C = input_shape

        self.wq = Dense(self.hidden_size, use_bias=False)
        self.wk = Dense(self.hidden_size, use_bias=False)
        self.wv = Dense(self.hidden_size, use_bias=False)
        self.out_proj = Dense(self.hidden_size, use_bias=False)

        # 학습 가능한 bias table
        self.relative_bias_table = self.add_weight(
            name='relative_bias_table',
            shape=(self.num_heads, 2*D-1, 2*H-1, 2*W-1),
            initializer='zeros',
            trainable=True,
        )

        # ✅ lookup_indices만 build()에서 계산 (상수)
        self.lookup_indices = compute_lookup_indices(D, H, W, self.num_heads)

        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        D, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        N = D * H * W

        x_flat = tf.reshape(x, [B, N, C])
        Q = self.wq(x_flat)
        K = self.wk(x_flat)
        V = self.wv(x_flat)

        Q = tf.reshape(Q, [B, N, self.num_heads, self.head_size])
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, [B, N, self.num_heads, self.head_size])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, [B, N, self.num_heads, self.head_size])
        V = tf.transpose(V, [0, 2, 1, 3])

        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.head_size, tf.float32)
        )
        # ✅ call() 안에서 gather_nd → gradient 경로 유지
        reindexed_bias = tf.gather_nd(self.relative_bias_table, self.lookup_indices)
        scores += reindexed_bias

        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.hidden_size])
        out = self.out_proj(out)
        return tf.reshape(out, [B, D, H, W, self.hidden_size])


# ─────────────────────────────────────────────
# 테스트 함수
# ─────────────────────────────────────────────
def test_gradient(layer_class, name, input_shape=(2, 4, 3, 3, 8)):
    """layer에 대해 gradient를 계산하고 각 변수별 gradient norm 반환."""
    layer = layer_class(hidden_size=32, head_size=16, name=name)
    x = tf.random.normal(input_shape)

    with tf.GradientTape() as tape:
        out = layer(x)
        loss = tf.reduce_mean(out ** 2)

    grads = tape.gradient(loss, layer.trainable_variables)

    results = {}
    for var, grad in zip(layer.trainable_variables, grads):
        short_name = var.name.split('/')[-1].replace(':0', '')
        if grad is not None:
            results[short_name] = float(tf.norm(grad).numpy())
        else:
            results[short_name] = None  # gradient 없음!

    return results


def train_and_track(layer_class, name, input_shape=(2, 4, 3, 3, 8), n_steps=20, lr=0.01):
    """
    n_steps만큼 학습하면서 매 스텝:
    - relative_bias_table의 L2 norm 변화 추적
    - loss 변화 추적
    """
    layer = layer_class(hidden_size=32, head_size=16, name=name)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    # 고정 입력/타겟 (매 스텝 동일)
    x = tf.random.normal(input_shape, seed=42)
    target = tf.random.normal(input_shape[:4] + (32,), seed=99)

    bias_norms = []
    losses = []

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = tf.reduce_mean((out - target) ** 2)

        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))

        # bias_table norm 기록
        for var in layer.trainable_variables:
            if 'relative_bias_table' in var.name:
                bias_norms.append(float(tf.norm(var).numpy()))
                break

        losses.append(float(loss.numpy()))

    return bias_norms, losses


def main():
    print("=" * 70)
    print("Prob-6 검증: relative_bias_table이 학습되는가?")
    print("=" * 70)

    input_shape = (2, 4, 3, 3, 8)
    n_steps = 20
    print(f"\n입력 shape: {input_shape}")
    print(f"학습 스텝: {n_steps}, optimizer: SGD(lr=0.01)")
    print(f"N = D*H*W = {input_shape[1]*input_shape[2]*input_shape[3]} tokens\n")

    # ─── 학습 시뮬레이션 ───
    print("[1/2] 기존 방식 (build gather) 학습 중...")
    broken_norms, broken_losses = train_and_track(
        BrokenBiasAttention, 'broken', input_shape, n_steps
    )
    print("[2/2] 수정 방식 (call gather) 학습 중...")
    fixed_norms, fixed_losses = train_and_track(
        FixedBiasAttention, 'fixed', input_shape, n_steps
    )

    # ─── 콘솔 출력 ───
    print("\n" + "-" * 70)
    print(f"{'step':<6} {'기존 bias_table norm':<25} {'수정 bias_table norm':<25} {'기존 loss':<15} {'수정 loss':<15}")
    print("-" * 70)
    for i in range(n_steps):
        print(f"{i:<6} {broken_norms[i]:<25.6f} {fixed_norms[i]:<25.6f} {broken_losses[i]:<15.6f} {fixed_losses[i]:<15.6f}")

    broken_changed = abs(broken_norms[-1] - broken_norms[0])
    fixed_changed = abs(fixed_norms[-1] - fixed_norms[0])

    print("-" * 70)
    print(f"\nbias_table norm 변화량:")
    print(f"  기존: {broken_norms[0]:.6f} -> {broken_norms[-1]:.6f} (delta = {broken_changed:.6f})")
    print(f"  수정: {fixed_norms[0]:.6f} -> {fixed_norms[-1]:.6f} (delta = {fixed_changed:.6f})")
    print()

    if broken_changed < 1e-10 and fixed_changed > 1e-6:
        print("  [PASS] 기존 방식: bias_table 값 변화 없음 (gradient 단절 -> 학습 불가)")
        print("  [PASS] 수정 방식: bias_table 값 변화 있음 (gradient 정상 -> 학습 진행)")
    print()

    # ─── 시각화 ───
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    steps = range(n_steps)

    # 차트 1: relative_bias_table norm 변화
    axes[0].plot(steps, broken_norms, 'o-', color='#e74c3c', linewidth=2,
                 markersize=5, label='broken (build gather)')
    axes[0].plot(steps, fixed_norms, 's-', color='#2ecc71', linewidth=2,
                 markersize=5, label='fixed (call gather)')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('relative_bias_table L2 Norm')
    axes[0].set_title('relative_bias_table weight norm per step\n'
                      '(broken: no change = not learning)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 주석
    axes[0].annotate('gradient X -> frozen',
                     xy=(n_steps-1, broken_norms[-1]),
                     xytext=(n_steps*0.4, broken_norms[-1] + (max(fixed_norms) - min(broken_norms))*0.3),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9, color='red')
    axes[0].annotate('gradient O -> learning',
                     xy=(n_steps-1, fixed_norms[-1]),
                     xytext=(n_steps*0.15, fixed_norms[-1] - (max(fixed_norms) - min(broken_norms))*0.2),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                     fontsize=9, color='green')

    # 차트 2: Loss 변화
    axes[1].plot(steps, broken_losses, 'o-', color='#e74c3c', linewidth=2,
                 markersize=5, label='broken (build gather)')
    axes[1].plot(steps, fixed_losses, 's-', color='#2ecc71', linewidth=2,
                 markersize=5, label='fixed (call gather)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Loss per step\n'
                      '(both decrease, but fixed uses bias_table too)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, '2026-03-30_prob6_gradient_result.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"result: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
