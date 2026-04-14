# 03 Encoder Components — V&V

> 공간 인코더(Spatial Encoder) 컴포넌트별 단독 기능 테스트 보고서.
> 본 폴더는 각 SE 단계 코드 작성 후 sanity check 결과를 기록.

## 목적

인코더 코드 작성 단계별로 *단독 컴포넌트* 의 동작을 확인:
- 텐서 형상 변환 정확성
- 파라미터 수의 수식 검증
- 핵심 수학적 성질 (회전 보존, 이동 불변성 등)
- 의도한 forward 동작 오류 없음

전체 인코더 조립 후 학습 검증(gradient flow, 단일 샘플 과적합, xs_fuel sensitivity 등)은
별도 단계(SE-B3)에서 `tests/encoder/`에 unit test로 구현.

## 보고서 목록

| 단계 | 대상 파일 | 보고서 | 상태 |
|---|---|---|:---:|
| SE-A1 | `src/models/spatial/layers3d.py` | [`SE-A1_layers3d_report.md`](SE-A1_layers3d_report.md) | ✅ |
| SE-A2 | `src/models/spatial/attention3d.py` | [`SE-A2_attention3d_report.md`](SE-A2_attention3d_report.md) | ✅ |
| SE-B1 | `src/models/spatial/encoder3d.py` | (예정) | ⏳ |
| SE-B2 | `configs/model.yaml` | — (config만, 별도 보고서 없음) | — |
| SE-B3 | `tests/encoder/*` | (예정) | ⏳ |

## 검증 컴포넌트 요약

| 컴포넌트 | 위치 | 핵심 검증 |
|---|---|---|
| `CellEmbedder` | layers3d | shape, params (2,816) |
| `FFN3D` | layers3d | shape, params (131,712) |
| `LearnedAbsolutePE3D` | layers3d | shape, params (64,000), `get_norm_map()` 동작 |
| `STRINGRelativePE3D` | attention3d | shape, params (192), **L2 norm 보존** (~9.5e-7), **이동 불변성** (~5.1e-7) |
| `FullAttention3D` | attention3d | shape, params (66,240) |

## 관련 문서

- 인코더 구현 plan: `C:\Users\Administrator\.claude\plans\breezy-herding-meadow.md`
- 인코더 미결정 항목 정리: `implementation_plans/공간인코더 구현 계획/2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md`
- STRING 위치 인코딩 검토: `implementation_plans/공간인코더 구현 계획/2026-04-04 3D 위치 인코딩 기법 검토.md`
