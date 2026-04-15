# 03 Encoder Components — V&V

> 공간 인코더(Spatial Encoder) 컴포넌트별 기능 테스트 보고서.
> 본 폴더는 각 SE 단계 코드 작성 후 검증 결과를 기록.

## 보고서 목록

| 단계 | 대상 파일 | 테스트 스크립트 | 출력 | 상태 |
|---|---|---|---|:---:|
| SE-A1 | `layers3d.py` | `SE-A1_test_layers3d.py` | `*_output.txt` | ALL PASS |
| SE-A2 | `attention3d.py` | `SE-A2_test_attention3d.py` | `*_output.txt` | ALL PASS |
| SE-A3 | `layers3d.py` (ConditionalLAPE3D) | `SE-A3_test_conditional_lape.py` | `*_output.txt` | ALL PASS |
| SE-A4 | `halo_expand.py` | `SE-A4_test_halo_expand.py` | `*_output.txt` | ALL PASS |
| SE-B1 | `encoder3d.py` | `SE-B1_test_encoder3d.py` | `*_output.txt` | ALL PASS |
| SE-B3 | 전체 검증 (gradient flow 등) | `SE-B3_test_gradient_flow.py` | `*_output.txt` | ALL PASS (초기) |

## 검증 항목 상세

### SE-A1: layers3d.py (CellEmbedder, FFN3D, LearnedAbsolutePE3D)

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-A1-T1 | Shape 검증 (6×6 halo 격자 기준) | PASS |
| SE-A1-T2 | Parameter count (수식 vs 측정) | PASS |

### SE-A2: attention3d.py (STRINGRelativePE3D, FullAttention3D)

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-A2-T1 | Shape 검증 (N=720) | PASS |
| SE-A2-T2 | Parameter count | PASS |
| SE-A2-T3 | STRING L2 norm 보존 (회전 isometry) | PASS (9.54e-7) |
| SE-A2-T4 | STRING 이동 불변성 (translation invariance) | PASS (2.68e-7) |
| SE-A2-T5 | Coords 형상 및 범위 [0,1] | PASS |
| SE-A2-T6 | Z/XY 축 간격이 물리 pitch 비율 반영 | PASS |

### SE-A3: ConditionalLAPE3D

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-A3-T1 | Shape 검증 | PASS |
| SE-A3-T2 | Parameter count (184,320) | PASS |
| SE-A3-T3 | 테이블 선택 정확성 (sym=0→mirror, sym=1→rotation) | PASS |
| SE-A3-T4 | Gradient 격리 (선택되지 않은 테이블 gradient=0) | PASS |
| SE-A3-T5 | Per-sample 분기 (batch 내 혼합 sym_type) | PASS |
| SE-A3-T6 | get_norm_map() 형상/dtype | PASS |
| SE-A3-T7 | LAPE 스케일 vs CellEmbedder 스케일 (초기 비율) | PASS (0.014) |
| SE-A3-T8 | Conditional LAPE 효과 (mirror vs rotation 출력 차이) | **미결** (학습 후) |
| SE-A3-T9 | LAPE 스케일 지배 여부 (학습 후 재확인) | **미결** (학습 후) |
| SE-A3-T10 | LAPE norm map 분석 (boundary vs interior 패턴) | **미결** (학습 후) |

### SE-A4: halo_expand

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-A4-T1 | Shape 검증 (5,5)→(6,6) | PASS |
| SE-A4-T2 | Mirror 매핑 정확성 | PASS |
| SE-A4-T3 | Rotation 매핑 정확성 | PASS |
| SE-A4-T4 | Inner cell 보존 | PASS |
| SE-A4-T5 | Mirror ≠ Rotation (비대칭 데이터) | PASS |
| SE-A4-T6 | Batch per-sample 선택 | PASS |
| SE-A4-T7 | 대칭 매핑 시각화 (텍스트 격자) | PASS |

### SE-B1: encoder3d.py (SpatialEncoder3D 통합)

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-B1-T1 | Shape 검증 (B,20,6,6,21)→(B,20,6,6,128) | PASS |
| SE-B1-T2 | 전체 파라미터 수 (782,528) | PASS |
| SE-B1-T3 | STRING coords 형상 및 범위 | PASS |
| SE-B1-T4 | Flatten/Unflatten 역변환 일관성 | PASS |
| SE-B1-T5 | STRING 축 구별 (Z vs XY pitch 비율) | PASS |
| SE-B1-T6 | End-to-end (halo_expand → encoder) | PASS |

### SE-B3: 전체 검증

| ID | 항목 | 상태 |
|---|---|:---:|
| SE-B3-T1 | Gradient flow (전 변수 gradient 존재) | PASS (초기) |
| SE-B3-T2 | Gradient norm 범위 (소실/폭발 없음) | PASS ([6.07e-4, 3.99]) |
| SE-B3-T3 | Dead activation (GELU zero 비율) | PASS (0.0%) |
| SE-B3-T4 | 수치 안정성 (정상/대입력/영입력) | PASS |
| SE-B3-T5 | Halo attention 편향 (halo/inner 가중치 비율) | PASS (1.006) |
| SE-B3-T6 | Gradient flow 재확인 (학습 중/후) | **미결** (학습 중 모니터링) |
| SE-B3-T7 | Dead activation 재확인 (학습 후) | **미결** (학습 후) |
| SE-B3-T8 | Halo attention 편향 재확인 (학습 후) | **미결** (학습 후) |

## V&V 중 발견된 버그

| 단계 | 내용 | 수정 |
|---|---|---|
| SE-A3 | Keras unseeded RandomNormal이 동일 build() 내에서 동일 값 생성 → mirror/rotation 테이블 초기값 동일 | seed=42 / seed=137 분리 적용 |

## Deprecated

| 파일 | 비고 |
|---|---|
| `(deprecated) SE-A1_layers3d_report.md` | 2026-04-08 초판, (5,5) 격자 기준 |
| `(deprecated) SE-A2_attention3d_report.md` | 2026-04-08 초판, N=500 기준 |

## 관련 문서

- 인코더 구현 계획: `implementation_plans/공간인코더 구현 계획/2026-03-30 모델 구현 계획(공간 인코더).md`
- Conditional LAPE 설계: `implementation_plans/공간인코더 구현 계획/2026-04-14 Conditional LAPE 적용 검토.md`
