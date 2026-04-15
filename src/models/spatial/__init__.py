"""공간 인코더 패키지 (Spatial Encoder).

v-SMR load-following surrogate의 인코더-Mamba-디코더 파이프라인 중 인코더부.

주요 모듈:
    layers3d        — CellEmbedder, FFN3D, LearnedAbsolutePE3D, ConditionalLAPE3D
    attention3d     — STRINGRelativePE3D, FullAttention3D
    halo_expand     — halo_expand() 함수 (quarter → halo 격자 변환)
    encoder3d       — SpatialEncoder3D (전체 조립)

기획 근거: implementation_plans/공간인코더 구현 계획/
"""
