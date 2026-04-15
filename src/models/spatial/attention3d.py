"""공간 인코더 attention 모듈 (SE-A2).

두 클래스:
    STRINGRelativePE3D    — Block-diagonal STRING (axis-independent frequency, RoPE-3D)
    FullAttention3D       — Multi-head full attention with STRING applied to Q, K

설계 결정 (2026-04-08, plan: breezy-herding-meadow.md):

(1) STRING 구현 — Block-diagonal 2x2 + axis-independent frequency
    - head_dim/2개의 2x2 회전 블록 (RoPE 표준 패턴).
    - 각 head, 각 블록, 각 축 (Z/Y/X)이 *독립* frequency 학습. 총 freq 수:
        n_freq = num_heads × (head_dim / 2) × 3   (예: 4 × 16 × 3 = 192)
    - 교환 조건 자동 보장 (block-diagonal은 본질적으로 commute).
    - dense STRING 대비 표현력 손실 없음 (commutativity-strict dense는 unitary 변환에
      의해 block-diagonal로 환원되므로 wrap된 W_Q/W_K가 그 변환을 흡수).
    - 좌표는 정규화된 물리 cm (0~1 범위)로 입력. 비균일 메시 확장 시 인터페이스 변경 없음.

    회전 적용 식 (block i, axis-independent freq):
        angle_i(r_z, r_y, r_x) = θ_z[i]·r_z + θ_y[i]·r_y + θ_x[i]·r_x

        Q'[..., 2i  ] = Q[..., 2i  ]·cos(angle_i) - Q[..., 2i+1]·sin(angle_i)
        Q'[..., 2i+1] = Q[..., 2i  ]·sin(angle_i) + Q[..., 2i+1]·cos(angle_i)

    Frequency init: log-spaced base frequencies (RoPE 표준).
        θ[i] = base ^ (-2i / head_dim)   for i in 0..head_dim/2-1
    이후 학습 가능 변수로 fine-tuning. base = 10000 (RoPE 표준).

(2) FullAttention3D
    - (B, N, D) flat sequence input. 본 인코더에서는 N = Z·qH·qW (halo expand 후 720 = 20·6·6).
    - Multi-head: D -> num_heads × head_dim, head별 attention 후 concat -> output proj.
    - STRING은 Q, K에만 적용 (V는 회전 안 받음). Attention score = softmax(Q'·K'^T / sqrt(d)).
    - QK-Norm은 미적용 (사용자 결정 2026-04-08, 학습 안정성 이슈 발생 시 재검토).

작성일: 2026-04-08
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import layers


class STRINGRelativePE3D(layers.Layer):
    """Block-diagonal STRING with axis-independent frequencies (3D RoPE).

    head_dim/2 개의 2x2 회전 블록을 Q, K에 적용. 각 블록은 (Z, Y, X) 축별로
    독립 학습 frequency를 가짐 → 모델이 축 종류를 구별 가능.

    Inputs to call():
        q:        (B, num_heads, N, head_dim)
        k:        (B, num_heads, N, head_dim)
        coords:   (N, 3)  — 정규화된 (z, y, x) 좌표 (0~1 범위)

    Outputs:
        q_rot, k_rot: (B, num_heads, N, head_dim)

    Args:
        num_heads:   attention head 수
        head_dim:    각 head의 차원. 짝수여야 함 (2x2 block 분할).
        base:        log-spaced frequency 초기화의 base (RoPE 표준 10000)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        base: float = 10000.0,
        name: str = "string_pe3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for block-diagonal STRING, got {head_dim}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_blocks = head_dim // 2
        self.base = base

    def build(self, input_shape):
        # Log-spaced base frequencies (RoPE standard):
        #   θ[i] = base ^ (-2i / head_dim),  i = 0..n_blocks-1
        # 작은 i = 큰 freq (빠른 회전, 단거리 민감), 큰 i = 작은 freq (느린 회전, 장거리)
        i_arr = tf.cast(tf.range(self.n_blocks), tf.float32)
        base_freq = tf.pow(self.base, -2.0 * i_arr / float(self.head_dim))
        # shape: (n_blocks,)

        # Per-head, per-axis 학습 가능 frequency.
        # init: 모든 head, 모든 축이 같은 base_freq로 시작.
        # shape: (num_heads, n_blocks)
        init = tf.tile(base_freq[tf.newaxis, :], [self.num_heads, 1])

        self.theta_z = self.add_weight(
            name="theta_z",
            shape=(self.num_heads, self.n_blocks),
            initializer=tf.keras.initializers.Constant(init.numpy()),
            trainable=True,
        )
        self.theta_y = self.add_weight(
            name="theta_y",
            shape=(self.num_heads, self.n_blocks),
            initializer=tf.keras.initializers.Constant(init.numpy()),
            trainable=True,
        )
        self.theta_x = self.add_weight(
            name="theta_x",
            shape=(self.num_heads, self.n_blocks),
            initializer=tf.keras.initializers.Constant(init.numpy()),
            trainable=True,
        )
        super().build(input_shape)

    def _compute_angles(self, coords: tf.Tensor) -> tf.Tensor:
        """좌표에서 회전 각도 계산.

        Args:
            coords: (N, 3) — (z, y, x) 정규화 좌표

        Returns:
            angles: (num_heads, N, n_blocks)
                angles[h, n, i] = θ_z[h, i]·r_z[n] + θ_y[h, i]·r_y[n] + θ_x[h, i]·r_x[n]
        """
        # coords[:, 0] = z, [:, 1] = y, [:, 2] = x
        rz = coords[:, 0:1]  # (N, 1)
        ry = coords[:, 1:2]  # (N, 1)
        rx = coords[:, 2:3]  # (N, 1)

        # theta_*: (num_heads, n_blocks) → (num_heads, 1, n_blocks)
        theta_z = self.theta_z[:, tf.newaxis, :]
        theta_y = self.theta_y[:, tf.newaxis, :]
        theta_x = self.theta_x[:, tf.newaxis, :]

        # rz, ry, rx: (N, 1) → (1, N, 1)
        rz = rz[tf.newaxis, ...]
        ry = ry[tf.newaxis, ...]
        rx = rx[tf.newaxis, ...]

        # broadcast: (num_heads, N, n_blocks)
        angles = theta_z * rz + theta_y * ry + theta_x * rx
        return angles

    def _apply_rotation(self, x: tf.Tensor, angles: tf.Tensor) -> tf.Tensor:
        """Block-diagonal 2x2 회전 적용.

        Args:
            x:       (B, num_heads, N, head_dim)
            angles:  (num_heads, N, n_blocks)

        Returns:
            x_rot:   (B, num_heads, N, head_dim)
        """
        # x를 (..., n_blocks, 2)로 reshape — 마지막 축의 인접 2개씩 묶어 2D 평면화
        shape = tf.shape(x)
        B, H, N = shape[0], shape[1], shape[2]
        x_pairs = tf.reshape(x, (B, H, N, self.n_blocks, 2))
        x0 = x_pairs[..., 0]  # (B, H, N, n_blocks)
        x1 = x_pairs[..., 1]  # (B, H, N, n_blocks)

        cos = tf.cos(angles)  # (H, N, n_blocks)
        sin = tf.sin(angles)
        # broadcast over batch: (1, H, N, n_blocks)
        cos = cos[tf.newaxis, ...]
        sin = sin[tf.newaxis, ...]

        # 2x2 회전:
        #   [x0']   [cos  -sin] [x0]
        #   [x1'] = [sin   cos] [x1]
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos

        # stack 후 다시 head_dim으로 펼침
        x_rot_pairs = tf.stack([x0_rot, x1_rot], axis=-1)  # (B, H, N, n_blocks, 2)
        x_rot = tf.reshape(x_rot_pairs, (B, H, N, self.head_dim))
        return x_rot

    def call(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        coords: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        angles = self._compute_angles(coords)  # (H, N, n_blocks)
        q_rot = self._apply_rotation(q, angles)
        k_rot = self._apply_rotation(k, angles)
        return q_rot, k_rot

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "base": self.base,
            }
        )
        return config


class FullAttention3D(layers.Layer):
    """Multi-head full attention with STRING (Q, K rotated by relative position).

    입력: (B, N, D) flat sequence
        - 본 인코더에서 N = Z·qH·qW (halo expand 후 720 = 20·6·6)
        - 호출자가 사전에 (B, Z, qH, qW, D)에서 reshape 필요
    추가 입력: coords (N, 3) — 정규화 물리 좌표

    구조:
        Q = q_proj(x), K = k_proj(x), V = v_proj(x)        # 각 (B, N, num_heads*head_dim)
        Q, K -> reshape (B, num_heads, N, head_dim)
        Q, K = STRING(Q, K, coords)                          # block-diag rotation
        attn_logits = Q @ K^T / sqrt(head_dim)              # (B, num_heads, N, N)
        attn_weights = softmax(attn_logits, axis=-1)
        out = attn_weights @ V                               # (B, num_heads, N, head_dim)
        out -> reshape (B, N, D) -> output projection
    """

    def __init__(
        self,
        d_latent: int,
        num_heads: int,
        head_dim: int = None,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        rope_base: float = 10000.0,
        name: str = "full_attention3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if head_dim is None:
            assert d_latent % num_heads == 0, (
                f"d_latent ({d_latent}) must be divisible by num_heads ({num_heads})"
            )
            head_dim = d_latent // num_heads
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.attention_dropout = attention_dropout
        self.proj_dropout = proj_dropout

        self.q_proj = layers.Dense(self.inner_dim, use_bias=True, name="q_proj")
        self.k_proj = layers.Dense(self.inner_dim, use_bias=True, name="k_proj")
        self.v_proj = layers.Dense(self.inner_dim, use_bias=True, name="v_proj")
        self.out_proj = layers.Dense(d_latent, use_bias=True, name="out_proj")

        self.string_pe = STRINGRelativePE3D(
            num_heads=num_heads,
            head_dim=head_dim,
            base=rope_base,
        )

        self.attn_dropout = (
            layers.Dropout(attention_dropout) if attention_dropout > 0 else None
        )
        self.out_dropout = (
            layers.Dropout(proj_dropout) if proj_dropout > 0 else None
        )

        self.scale = 1.0 / math.sqrt(float(head_dim))

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """(B, N, inner_dim) -> (B, num_heads, N, head_dim)"""
        shape = tf.shape(x)
        B, N = shape[0], shape[1]
        x = tf.reshape(x, (B, N, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        """(B, num_heads, N, head_dim) -> (B, N, inner_dim)"""
        shape = tf.shape(x)
        B, _, N = shape[0], shape[1], shape[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (B, N, self.inner_dim))

    def call(
        self,
        x: tf.Tensor,
        coords: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # x: (B, N, D),  coords: (N, 3)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = self._split_heads(Q)  # (B, H, N, head_dim)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # STRING relative position encoding
        Q, K = self.string_pe(Q, K, coords)

        # Scaled dot-product attention
        # logits: (B, H, N, N)
        attn_logits = tf.matmul(Q, K, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # attended: (B, H, N, head_dim)
        attended = tf.matmul(attn_weights, V)
        attended = self._merge_heads(attended)  # (B, N, inner_dim)

        out = self.out_proj(attended)  # (B, N, D)
        if self.out_dropout is not None:
            out = self.out_dropout(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_latent": self.d_latent,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "attention_dropout": self.attention_dropout,
                "proj_dropout": self.proj_dropout,
                "rope_base": self.string_pe.base,
            }
        )
        return config
