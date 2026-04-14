# 02. CellEmbedder — Conv3D(1,1,1), 21 → 128 채용 이유

> **결정**: Conv3D(kernel=(1,1,1)) per-cell projection. 21 → 128. spatial mixing 없음.
> **결정 일자**: 2026-04-08

---

## 1. 핵심 변경

- 이전: Conv3D(15,5,5) 등 채널 + 공간 정보를 함께 압축하는 Stem (~2.89M, 전체의 84%)
- 신규: 순수 cell-단위 임베딩 layer로 단순화. 공간 정보 처리는 후속 Full Attention이 담당
- 인코더 전체 방향: pure Transformer 기반 + 상대 위치(STRING) + 절대 위치(LAPE)

---

## 2. 채택 이유

### 2.1. Stem이 공간 정보를 사전 압축할 필요 없음
- 후속 Full Attention이 첫 block부터 모든 cell 간 mixing 수행
- conv 기반 사전 mixing은 attention과 역할 중복
- 고정 conv kernel이 학습 가능한 attention 패턴을 제약

### 2.2. CNN zero-padding이 quarter 격자에서 비물리적
- 3×3 이상 kernel은 격자 경계에서 zero-padding 적용
- qy=0/qx=0 면은 진짜 경계가 아니라 대칭축. 그 너머에 fuel cell 존재
- conv가 그 위치에 0을 채우면 모델이 "fuel cell이 없다" 고 학습
- BoundaryPad 패턴(외곽 zero, 내부 reflect)으로 우회 시도했으나 외곽 zero의 비물리성으로 폐기 이력 존재
- Conv3D(1,1,1)은 이웃을 보지 않으므로 padding 자체가 불필요 → 문제 없음

---

## 3. 다른 옵션 거부 사유

| 옵션 | 거부 사유 |
|---|---|
| Conv3D(3,3,3) | §2.1, §2.2 모두 해당 |
| Dense per-cell | Conv3D(1,1,1)과 수학적 동치. 5D 텐서 형식 유지를 위해 Conv3D 형식 채택 |
| 기존 Conv3D(15,5,5) | §2.1, §2.2 모두 해당 + 파라미터 비대 |

---

## 4. 파라미터

- Conv3D(1,1,1) 21→128: `21·128 + 128 = 2,816`
- 인코더 전체 0.66M의 0.4%
- 이전 Stem 2.89M 대비 1000× 감소

---

## 5. 클래스명

- "Stem" 은 일반적으로 spatial mixing 포함 의미
- 본 layer는 cell 단위 임베딩만 담당 → `CellEmbedder` 사용 (사용자 제안)

---

## 6. 전제

- 입력 21채널이 모두 O(1) 스케일로 정규화되어 있어야 함
- per-cell linear projection이라 채널 간 스케일 차이가 크면 학습 초기 불균형 위험
- 데이터 전처리에서 zscore/log_zscore 적용 확인 필요
- SE-B3 단계 xs_fuel sensitivity test로 점검

---

## 7. 결정 과정

- 2026-04-07: Stem 구조 잠정 미결
- 2026-04-08 사용자 의문: "STEM은 왜 하는데? 굳이 필요있나"
- BoundaryPad 폐기 이력이 conv 기반 Stem 거부의 핵심 근거
- 사용자 결정: "(3,3,3)이 아니라 (1,1,1)로 셀마다 해라"
- 사용자: "CellEmbedder를 Conv3D로 유지하지? 어차피 기능은 Dense와 동일하잖아" → Conv3D(1,1,1) 형식 확정

---

## 8. 참고

- 이전 모델 분석: `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md` §2.1
- BoundaryPad 폐기: `참고 파일/공간 인코더 구현 참고자료(+기존 코드 정보)/기존 코드(custom voxel attention) 정보 및 구조 차용 고찰(260402).md`
- 기능 검증: `v&v/03_encoder_components/SE-A1_layers3d_report.md`
