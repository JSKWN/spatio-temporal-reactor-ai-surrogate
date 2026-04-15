"""공간 인코더 기본 layer 모듈 (SE-A1).

네 클래스:
    CellEmbedder          — Conv3D(1,1,1) per-cell channel projection (C_in -> D)
    FFN3D                 — Pre-LN block의 token-wise 비선형 feed-forward (D -> D*r -> D)
    LearnedAbsolutePE3D   — (Z, qH, qW, D) trainable absolute position embedding
    ConditionalLAPE3D     — 대칭 유형별 조건부 LAPE (mirror/rotation 2개 텐서, sym_type 분기)

설계 결정 (2026-04-08, plan: breezy-herding-meadow.md):

(1) CellEmbedder
    - 한 cell의 21채널 raw 데이터(state 10 + xs_fuel 10 + rod 1)를 D차원 latent 벡터로
      끌어올리는 입력 적응 layer. spatial mixing 없음 (Conv3D 1×1×1 = Dense와 동치).
    - 모든 spatial mixing은 후속 FullAttention3D가 담당 (ViT patch embedding 패턴).
    - 5D 텐서 형식 유지를 위해 Dense 대신 Conv3D(1,1,1) 사용. 후속 LAPE가 5D를 기대.

(2) FFN3D
    - Transformer 표준의 token-wise feed-forward (Vaswani et al. 2017 이래).
    - Attention은 본질적으로 token 간 *선형* 가중합 → token 내부 *비선형* 변환은 FFN 담당.
    - Attention이 모은 정보를 GELU로 비선형 처리하여 표현력 확보.
    - CellEmbedder와의 차이:
        * CellEmbedder: 인코더 진입부 1회, 21→D 입력 적응
        * FFN3D: 각 attention block 안 3회, D→D*expand→D 비선형 처리

(3) LearnedAbsolutePE3D
    - 500개 cell (Z×qH×qW = 20×5×5) × D=128 = 64,000개 학습 가능 스칼라.
    - build()에서 (20, 5, 5, 128) variable을 RandomNormal(stddev=0.02)로 초기화.
    - call()에서 입력에 broadcast 합산: x + embedding[None, ...]
    - L_diffusion 손실에서 backprop된 gradient가 위치별 임베딩을 update →
      boundary 셀과 interior 셀이 서로 다른 임베딩으로 진화 (의도된 BC 식별 학습).
    - STRING(상대 PE)과 동시 사용. 두 PE는 서로 다른 자리에 들어가 간섭 없음.
      Dufter et al. 2022 (Computational Linguistics 48(3):733):
        "absolute and relative position embeddings are complementary ...
         the combination outperforms either alone"
    - 학습 종료 후 분석 hook: get_norm_map() → (Z, qH, qW) 위치별 L2 노름.
      boundary가 interior보다 큰 노름이면 LAPE가 BC 구조를 흡수했다는 신호.

작성일: 2026-04-08
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class CellEmbedder(layers.Layer):
    """Per-cell channel projection (Conv3D 1x1x1).

    한 cell의 raw 채널을 D차원 latent 벡터로 변환하는 입력 적응 layer.
    spatial mixing 없음. 채널 차원만 projection.

    Conv3D(1,1,1)은 Dense layer와 수학적으로 동치. 5D 텐서 형식 유지를 위해
    Conv3D 형식 채택 (후속 LAPE/Attention이 5D를 기대).

    입력: (B, Z, qH, qW, C_in)
        - C_in = 21 (state 10 + xs_fuel 10 + rod_map 1) 가정. 단, 임의 C_in 허용.
    출력: (B, Z, qH, qW, d_latent)
    """

    def __init__(self, d_latent: int, name: str = "cell_embedder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_latent = d_latent
        self.proj = layers.Conv3D(
            filters=d_latent,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding="valid",
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="proj",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"d_latent": self.d_latent})
        return config


class FFN3D(layers.Layer):
    """Pre-LN attention block의 token-wise 비선형 feed-forward.

    Transformer 표준 구성: Dense expand → GELU → Dense contract.
    Attention(token간 선형 결합) 직후 token 내부에서 비선형 변환을 수행.

    구조: Dense(D -> D*expand) -> GELU -> dropout -> Dense(D*expand -> D) -> dropout

    입력/출력: (B, N, D) 또는 (B, ..., D) — 마지막 축에만 작용 (token-wise).
    """

    def __init__(
        self,
        d_latent: int,
        expand_ratio: int = 4,
        dropout: float = 0.0,
        name: str = "ffn3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.d_latent = d_latent
        self.expand_ratio = expand_ratio
        self.dropout = dropout

        d_hidden = d_latent * expand_ratio
        self.dense_in = layers.Dense(
            d_hidden,
            activation=None,
            kernel_initializer="he_normal",
            name="dense_in",
        )
        self.act = layers.Activation("gelu", name="gelu")
        self.dropout_in = (
            layers.Dropout(dropout, name="dropout_in") if dropout > 0 else None
        )
        self.dense_out = layers.Dense(
            d_latent,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="dense_out",
        )
        self.dropout_out = (
            layers.Dropout(dropout, name="dropout_out") if dropout > 0 else None
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.dense_in(x)
        h = self.act(h)
        if self.dropout_in is not None:
            h = self.dropout_in(h, training=training)
        h = self.dense_out(h)
        if self.dropout_out is not None:
            h = self.dropout_out(h, training=training)
        return h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_latent": self.d_latent,
                "expand_ratio": self.expand_ratio,
                "dropout": self.dropout,
            }
        )
        return config


class LearnedAbsolutePE3D(layers.Layer):
    """3D 학습 가능 절대 위치 임베딩 (Learned Absolute Position Embedding).

    형태: (Z, qH, qW, D) trainable variable.
        예: (20, 5, 5, 128) → 20·5·5·128 = 64,000개 학습 가능 스칼라.
        500개의 cell이 각자 D차원 정체성 벡터를 가짐.

    초기화: RandomNormal(mean=0, stddev=0.02). 학습 시작 시 모든 임베딩이
        작은 랜덤 벡터에서 시작하여, L_diffusion 손실 backprop으로 위치별 차이가
        진화함. boundary 셀에서 손실 잔차가 크면 그 위치의 임베딩이 interior와
        다른 방향으로 update됨 (의도된 BC 식별 학습).

    forward: 입력에 broadcast 합산.
        x(B, Z, qH, qW, D) + embedding[None, ...](1, Z, qH, qW, D) = 출력

    STRING(상대 PE)과 동시 사용:
        - STRING은 attention 점수에 상대 offset bias 주입 (번역 불변)
        - LAPE는 token feature에 위치별 정체성 추가 (번역 비불변)
        - 서로 다른 자리에 들어가 간섭 없음
        Dufter et al. 2022 Computational Linguistics 48(3):733 —
            "absolute and relative position embeddings are complementary"

    학습 종료 후 분석:
        get_norm_map() → tf.Tensor (Z, qH, qW), float32
            위치별 L2 노름. boundary 셀이 interior보다 큰 노름이면
            "LAPE가 BC 구조를 흡수했다" 는 신호.

    Args:
        z_dim:        Z 격자 크기 (예: 20)
        qh_dim:       quarter H 격자 크기 (예: 5)
        qw_dim:       quarter W 격자 크기 (예: 5)
        d_latent:     임베딩 차원 D (예: 128)
        init_scale:   RandomNormal stddev (기본 0.02, ViT 표준)
    """

    def __init__(
        self,
        z_dim: int,
        qh_dim: int,
        qw_dim: int,
        d_latent: int,
        init_scale: float = 0.02,
        name: str = "lape3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.z_dim = z_dim
        self.qh_dim = qh_dim
        self.qw_dim = qw_dim
        self.d_latent = d_latent
        self.init_scale = init_scale

    def build(self, input_shape):
        # input_shape: (B, Z, qH, qW, D)
        # 64K (예시) 학습 가능 스칼라를 단일 variable로 보유
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.z_dim, self.qh_dim, self.qw_dim, self.d_latent),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.init_scale
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (B, Z, qH, qW, D)
        # broadcast: (1, Z, qH, qW, D) + (B, Z, qH, qW, D) = (B, Z, qH, qW, D)
        return x + self.embedding[tf.newaxis, ...]

    def get_norm_map(self) -> tf.Tensor:
        """위치별 L2 노름 — 학습 종료 후 분석용 hook.

        Returns:
            tf.Tensor of shape (Z, qH, qW), dtype float32.
            각 (z, qy, qx) 위치의 d차원 임베딩 벡터의 L2 노름.

        Usage:
            norm_map = encoder.lape.get_norm_map().numpy()  # (20, 5, 5)
            np.save("lape_norm_map.npy", norm_map)
        """
        return tf.norm(self.embedding, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "z_dim": self.z_dim,
                "qh_dim": self.qh_dim,
                "qw_dim": self.qw_dim,
                "d_latent": self.d_latent,
                "init_scale": self.init_scale,
            }
        )
        return config


class ConditionalLAPE3D(layers.Layer):
    """대칭 유형별 조건부 3D 절대 위치 임베딩.

    mirror / rotation 두 개의 trainable 텐서를 보유하고,
    sym_type (0=mirror, 1=rotation) 에 따라 하나를 선택하여 입력에 add.
    BERT의 segment embedding과 동일 원리 (2-class 조건부 위치 임베딩 분기).

    각 텐서의 형상: (Z, qH, qW, D).
        예: (20, 6, 6, 128) → 720개 cell × 128차원 = 92,160개 학습 가능 스칼라.
        두 테이블 합계: 184,320개.

    Gradient 흐름:
        tf.where에 의해 선택된 텐서만 계산 그래프에 포함된다.
        선택된 텐서에는 모든 물리량 loss (L_data, L_data_halo, L_diff_rel 등)
        의 gradient가 흐른다. 선택되지 않은 텐서는 gradient = 0.

    설계 근거: 2026-04-14 Conditional LAPE 적용 검토.md

    Args:
        z_dim:        Z 격자 크기 (예: 20)
        qh_dim:       quarter H 격자 크기 (예: 6, halo 포함)
        qw_dim:       quarter W 격자 크기 (예: 6, halo 포함)
        d_latent:     임베딩 차원 D (예: 128)
        init_scale:   RandomNormal stddev (기본 0.02, ViT 표준)
    """

    def __init__(
        self,
        z_dim: int,
        qh_dim: int,
        qw_dim: int,
        d_latent: int,
        init_scale: float = 0.02,
        name: str = "cond_lape3d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.z_dim = z_dim
        self.qh_dim = qh_dim
        self.qw_dim = qw_dim
        self.d_latent = d_latent
        self.init_scale = init_scale

    def build(self, input_shape):
        shape = (self.z_dim, self.qh_dim, self.qw_dim, self.d_latent)

        # 각 테이블에 다른 seed를 부여하여 초기값이 다르게 생성되도록 함.
        # Keras의 unseeded RandomNormal은 동일 build() 내에서 동일 값을 반환하는 문제가 있음.
        self.lape_mirror = self.add_weight(
            name="lape_mirror",
            shape=shape,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.init_scale, seed=42
            ),
            trainable=True,
        )
        self.lape_rotation = self.add_weight(
            name="lape_rotation",
            shape=shape,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.init_scale, seed=137
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, sym_type: tf.Tensor) -> tf.Tensor:
        """입력에 대칭 유형별 위치 임베딩을 add.

        Args:
            x:        (B, Z, qH, qW, D) — CellEmbedder 출력.
            sym_type: (B,) int32 — 0=mirror, 1=rotation.

        Returns:
            (B, Z, qH, qW, D) — 위치 임베딩이 더해진 텐서.
        """
        # sym_type (B,) → (B, 1, 1, 1, 1) 로 확장하여 broadcasting
        cond = tf.equal(sym_type[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], 0)
        lape = tf.where(
            cond,
            self.lape_mirror[tf.newaxis, ...],
            self.lape_rotation[tf.newaxis, ...],
        )
        return x + lape

    def get_norm_map(self) -> dict:
        """대칭 유형별 위치 L2 노름 맵 — 학습 종료 후 분석용.

        Returns:
            dict with keys 'mirror', 'rotation'.
            각 값은 tf.Tensor of shape (Z, qH, qW), dtype float32.

        Usage:
            norms = encoder.cond_lape.get_norm_map()
            mirror_map = norms['mirror'].numpy()   # (20, 6, 6)
            rot_map = norms['rotation'].numpy()     # (20, 6, 6)
        """
        return {
            "mirror": tf.norm(self.lape_mirror, axis=-1),
            "rotation": tf.norm(self.lape_rotation, axis=-1),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "z_dim": self.z_dim,
                "qh_dim": self.qh_dim,
                "qw_dim": self.qw_dim,
                "d_latent": self.d_latent,
                "init_scale": self.init_scale,
            }
        )
        return config
