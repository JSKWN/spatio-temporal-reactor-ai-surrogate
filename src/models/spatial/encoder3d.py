"""공간 인코더 전체 조립 (SE-B1).

SpatialEncoder3D: halo expand 이후의 (B, Z, qH, qW, C_in) 입력을 받아
(B, Z, qH, qW, D) 잠재 표현을 출력.

파이프라인:
    CellEmbedder(C_in → D)
    → ConditionalLAPE3D(sym_type) 또는 LearnedAbsolutePE3D
    → reshape (B, N, D)   where N = Z * qH * qW
    → [Pre-LN → FullAttention3D(STRING) → residual
       → Pre-LN → FFN3D → residual] × n_stages
    → reshape (B, Z, qH, qW, D)

STRING 좌표:
    격자가 고정이므로 build() 시 정규화 물리 좌표 (N, 3) 를 pre-compute.
    Z축 10cm 간격, XY축 21.6cm pitch 기준으로 0~1 정규화.
    향후 비균일 메시 (반사체 포함) 시 외부 주입으로 전환 가능.

설계 근거:
    - 03_attention_and_position.md: Full Attention × 3, D=128, H=4, LAPE+STRING
    - 2026-04-14 Conditional LAPE 적용 검토.md: ConditionalLAPE3D
    - 05_symmetry_mode.md: halo (6,6) all-the-way

작성일: 2026-04-15
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .layers3d import CellEmbedder, ConditionalLAPE3D, FFN3D, LearnedAbsolutePE3D
from .attention3d import FullAttention3D


def _build_coords(
    z_dim: int,
    qh_dim: int,
    qw_dim: int,
    z_pitch_cm: float = 10.0,
    xy_pitch_cm: float = 21.6,
) -> np.ndarray:
    """격자의 정규화 물리 좌표를 생성.

    각 축의 물리 좌표를 0~1로 정규화하여 (N, 3) 배열로 반환.
    N = z_dim * qh_dim * qw_dim.

    Args:
        z_dim:       Z 격자 크기 (예: 20)
        qh_dim:      H 격자 크기 (예: 6, halo 포함)
        qw_dim:      W 격자 크기 (예: 6, halo 포함)
        z_pitch_cm:  Z축 간격 (cm). 기본 10.0
        xy_pitch_cm: XY축 간격 (cm). 기본 21.6

    Returns:
        coords: (N, 3) float32 — (z, y, x) 정규화 좌표 (0~1 범위)
    """
    # 물리 좌표 (cm)
    z_coords = np.arange(z_dim, dtype=np.float32) * z_pitch_cm
    y_coords = np.arange(qh_dim, dtype=np.float32) * xy_pitch_cm
    x_coords = np.arange(qw_dim, dtype=np.float32) * xy_pitch_cm

    # 0~1 정규화
    if z_dim > 1:
        z_coords = z_coords / z_coords[-1]
    if qh_dim > 1:
        y_coords = y_coords / y_coords[-1]
    if qw_dim > 1:
        x_coords = x_coords / x_coords[-1]

    # meshgrid → flatten → (N, 3)
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
    return coords


class _EncoderStage(layers.Layer):
    """Pre-LN Attention + FFN 1개 stage.

    x → LN → Attention → +residual → LN → FFN → +residual
    """

    def __init__(
        self,
        d_latent: int,
        num_heads: int,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        name: str = "encoder_stage",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ln_attn = layers.LayerNormalization(epsilon=1e-5, name="ln_attn")
        self.attention = FullAttention3D(
            d_latent=d_latent,
            num_heads=num_heads,
            attention_dropout=dropout,
            proj_dropout=dropout,
            rope_base=rope_base,
            name="attention",
        )
        self.ln_ffn = layers.LayerNormalization(epsilon=1e-5, name="ln_ffn")
        self.ffn = FFN3D(
            d_latent=d_latent,
            expand_ratio=ffn_expand,
            dropout=dropout,
            name="ffn",
        )

    def call(
        self,
        x: tf.Tensor,
        coords: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            x:      (B, N, D)
            coords: (N, 3) — 정규화 물리 좌표 (STRING용)
        """
        # Pre-LN + Attention + residual
        h = self.ln_attn(x)
        h = self.attention(h, coords, training=training)
        x = x + h

        # Pre-LN + FFN + residual
        h = self.ln_ffn(x)
        h = self.ffn(h, training=training)
        x = x + h

        return x


class SpatialEncoder3D(layers.Layer):
    """공간 인코더 전체.

    입력:  (B, Z, qH, qW, C_in)  — halo expand 후 형상 (예: B, 20, 6, 6, 21)
    출력:  (B, Z, qH, qW, D)     — 잠재 표현 (예: B, 20, 6, 6, 128)

    Args:
        d_latent:     잠재 차원 D (예: 128)
        n_stages:     attention stage 수 (예: 3)
        num_heads:    attention head 수 (예: 4)
        ffn_expand:   FFN 확장 비율 (예: 4)
        z_dim:        Z 격자 크기 (예: 20)
        qh_dim:       H 격자 크기 (예: 6, halo 포함)
        qw_dim:       W 격자 크기 (예: 6, halo 포함)
        lape_type:    'conditional' | 'single' — LAPE 유형 선택
        z_pitch_cm:   Z축 물리 간격 (cm). STRING 좌표용
        xy_pitch_cm:  XY축 물리 간격 (cm). STRING 좌표용
        dropout:      attention/FFN dropout 비율
        rope_base:    STRING base 주파수
        init_scale:   LAPE 초기화 stddev
    """

    def __init__(
        self,
        d_latent: int,
        n_stages: int,
        num_heads: int,
        ffn_expand: int = 4,
        z_dim: int = 20,
        qh_dim: int = 6,
        qw_dim: int = 6,
        lape_type: str = "conditional",
        z_pitch_cm: float = 10.0,
        xy_pitch_cm: float = 21.6,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        init_scale: float = 0.02,
        name: str = "spatial_encoder3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.d_latent = d_latent
        self.z_dim = z_dim
        self.qh_dim = qh_dim
        self.qw_dim = qw_dim
        self.n_tokens = z_dim * qh_dim * qw_dim
        self.lape_type = lape_type

        # 정규화 물리 좌표 pre-compute (STRING용, 학습 불가 상수)
        coords_np = _build_coords(z_dim, qh_dim, qw_dim, z_pitch_cm, xy_pitch_cm)
        self.coords = tf.constant(coords_np, dtype=tf.float32)  # (N, 3)

        # CellEmbedder: C_in → D
        self.cell_embedder = CellEmbedder(d_latent=d_latent, name="cell_embedder")

        # LAPE
        if lape_type == "conditional":
            self.lape = ConditionalLAPE3D(
                z_dim=z_dim,
                qh_dim=qh_dim,
                qw_dim=qw_dim,
                d_latent=d_latent,
                init_scale=init_scale,
                name="cond_lape",
            )
        elif lape_type == "single":
            self.lape = LearnedAbsolutePE3D(
                z_dim=z_dim,
                qh_dim=qh_dim,
                qw_dim=qw_dim,
                d_latent=d_latent,
                init_scale=init_scale,
                name="lape",
            )
        else:
            raise ValueError(
                f"lape_type must be 'conditional' or 'single', got '{lape_type}'"
            )

        # Attention + FFN stages
        self.stages = [
            _EncoderStage(
                d_latent=d_latent,
                num_heads=num_heads,
                ffn_expand=ffn_expand,
                dropout=dropout,
                rope_base=rope_base,
                name=f"stage_{i}",
            )
            for i in range(n_stages)
        ]

    def call(
        self,
        x: tf.Tensor,
        sym_type: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            x:        (B, Z, qH, qW, C_in) — halo expand 완료 입력
            sym_type: (B,) int32 — 0=mirror, 1=rotation.
                      lape_type='single' 이면 사용하지 않으나 인터페이스 통일을 위해 받음.
            training: bool

        Returns:
            (B, Z, qH, qW, D) — 잠재 표현
        """
        B = tf.shape(x)[0]

        # CellEmbedder: (B, Z, qH, qW, C_in) → (B, Z, qH, qW, D)
        h = self.cell_embedder(x)

        # LAPE: (B, Z, qH, qW, D) → (B, Z, qH, qW, D)
        if self.lape_type == "conditional":
            h = self.lape(h, sym_type)
        else:
            h = self.lape(h)

        # Flatten spatial: (B, Z, qH, qW, D) → (B, N, D)
        h = tf.reshape(h, (B, self.n_tokens, self.d_latent))

        # Attention stages (STRING coords는 내부 상수 사용)
        for stage in self.stages:
            h = stage(h, self.coords, training=training)

        # Reshape back: (B, N, D) → (B, Z, qH, qW, D)
        h = tf.reshape(
            h, (B, self.z_dim, self.qh_dim, self.qw_dim, self.d_latent)
        )

        return h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_latent": self.d_latent,
                "n_stages": len(self.stages),
                "num_heads": self.stages[0].attention.num_heads
                if self.stages
                else 4,
                "z_dim": self.z_dim,
                "qh_dim": self.qh_dim,
                "qw_dim": self.qw_dim,
                "lape_type": self.lape_type,
            }
        )
        return config
