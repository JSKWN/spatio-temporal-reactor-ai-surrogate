# 01. 입력 격자 — Quarter 5×5 채용 이유

> **결정**: 데이터 입력 격자는 Quarter 5×5×20 (총 500 cell). Full-core 직접 입력은 채택하지 않음. **인코더 진입 전에 halo expand로 (6,6) (총 720 cell)로 확장하여 인코더/Mamba/디코더가 처리. 최종 출력은 다시 (5,5) crop.**
> **결정 일자**: 2026-04-08 (Quarter 5×5 데이터 입력 결정 + halo expand 결정 동시)
> **개정일**: 2026-04-09 — 단계별 형상 구분 명시. §1.5 신설 (입력/처리/출력 단계 형상). 본 문서의 "Quarter 5×5 채용"은 *데이터 입력 격자* 결정이며, *인코더 처리 격자* 결정 (halo (6,6)) 은 `05_symmetry_mode.md` 참조

---

## 1. 검토 옵션

본 절은 **데이터 입력 격자** 결정에 한정. 인코더 처리 형상은 §1.5 와 `05_symmetry_mode.md` 참조.

| # | 옵션 | 격자 형상 | 노드 수 |
|---|---|---|---:|
| (A) | Quarter (채택) | (Z=20, qH=5, qW=5) | 500 |
| (B) | Full fuel core | (Z=20, H=9, W=9) | 1620 |
| (C) | Geometric full core | (Z=20, H=11, W=11) | 2420 |

> Half-core는 Y/X 한쪽 대칭만 풀어주는 어중간한 옵션 → 검토 제외.

---

## 1.5. 입력 단계 vs 인코더 처리 단계 vs 최종 출력 단계 형상 구분 (개정 2026-04-09)

본 문서의 "Quarter 5×5 채용"은 *데이터 입력 격자* 결정이며, 모델 내부의 모든 컴포넌트가 (5,5) 로 처리한다는 의미가 아님. `05_symmetry_mode.md` (2026-04-08) 결정으로 **인코더/Mamba/디코더는 모두 halo (6,6) 로 처리**, 최종 출력만 (5,5) 로 crop.

| 단계 | 형상 | 의미 | 결정 출처 |
|---|---|---|---|
| **데이터 입력** | (B, T, 20, 5, 5, 21) | 데이터셋에서 로드되는 quarter core 형상 | **본 문서 (01)** |
| **halo expand 후** | (B, T, 20, 6, 6, 21) | 입력 함수 `halo_expand()` 적용 후, 인코더 진입 전 | `05_symmetry_mode` |
| **인코더 처리** | (B, 20, 6, 6, 128) | CellEmbedder ~ attention stage 모두 (6,6) | `05_symmetry_mode` |
| **Mamba 처리** | (6,6) latent 그대로 통과 | 시간축 처리, 공간 차원 유지 | `05_symmetry_mode` |
| **디코더 처리** | (B, 20, 6, 6, 128) → (B, 20, 6, 6, C_out) | 디코더도 (6,6) feature space | `05_symmetry_mode` |
| **L_diff_rel 입력** | (B, 20, 6, 6, C_out) | 디코더 출력 그대로, 합산 도메인은 5×5 inner cell | `Physical Loss 통합 레퍼런스 §3.6` |
| **최종 외부 출력** | (B, 20, 5, 5, C_out) | crop 1회 (외부 인터페이스 한 곳에서만) | `05_symmetry_mode` |

이 단계 구분의 의미:
- **본 문서의 결정 (Quarter 5×5)** 은 데이터 입력 형상에 한정. Full-core (1620) 직접 입력 대비 ~75% 정보 중복 회피, attention FLOPs 10.5× 감소
- **05_symmetry_mode 결정 (halo 6,6)** 은 *데이터 입력에서 인코더 입력으로의 변환* 단계. quarter symmetry에 의한 boundary 정보 손실을 halo cell로 직접 전달 — 본 문서 §4.1의 "STRING/LAPE/L_diffusion 으로 대칭 정보 흡수" 가설을 더 강한 방식으로 진화
- **두 결정은 모순 없이 공존**: 데이터 입력은 quarter, 인코더 처리는 halo expand. 데이터 효율성과 ML 학습 신호 강도의 양립

---

## 2. 핵심 우려 — Quarter symmetry로 인한 정보 손실

- Quarter는 1/4 잘라낸 격자
- 원점 부근(qy=0, qx=0) cell이 full-core에서는 사방 fuel cell로 둘러싸여 있음
- Quarter에서는 대칭 너머 이웃 정보가 격자에서 잘려나감
- 인코더가 이 사실을 인지 못하면 원점 부근 cell의 latent에 정보 손실 가능
- 추가: 현재 데이터 기준으로는 mirror 형태로 보이지만, 실제로는 rotational symmetry(90° 회전 시 동일)일 가능성도 있음. 어느 쪽이든 격자 너머 fuel cell 존재 정보가 latent에 반영되어야 함

---

## 3. 두 가지 선택지

### 3.1. Full-core 확장 — 인코더가 진짜 이웃을 직접 봄
### 3.2. Quarter 유지 + 다른 메커니즘으로 대칭 정보 전달 (채택)

---

## 4. 채택 이유

### 4.1. 대칭 정보 전달 메커니즘 (08에서 진화)

본 문서 작성 시점 (2026-04-08 초안) 의 가설:
- **STRING**: 상대 거리 기반 attention 인코딩 → 원점 부근 cell의 상대 위치 분포 차이 반영
- **LAPE**: 학습 가능 절대 위치 임베딩 → 원점 부근 cell에 다른 정체성 학습 (`05_position_encoding_lape.md`)
- **L_diffusion 손실**: 디코더 출력에 대칭 reflection을 명시 계산. 잔차 손실 → backprop으로 인코더에 학습 신호 전달

**05_symmetry_mode (2026-04-08, 본 문서 같은 날 후속 결정) 에서의 진화**:
- 위 가설은 모두 **간접적** 메커니즘 (PE 학습, loss backprop) 에 의존
- 더 직접적 방식: **halo expand로 sym 매핑된 cell 값을 인코더 입력 단계에서 명시적으로 부여**
- 이로써 인코더의 boundary cell이 halo cell을 attention input으로 *직접* 봄 → 대칭 inductive bias가 attention 처리에 자동 흡수
- 또한 halo cell에 L_data_halo (λ=0.3) 를 부과하여 비물리적 발산 차단 (`Physical Loss 통합 레퍼런스 §3.6`)
- **§4.1 가설 (STRING/LAPE/L_diffusion 만으로 충분) 은 폐기되지 않음** — 여전히 작동하나, halo expand가 더 강하고 직접적인 신호를 추가 제공

진화 후의 대칭 정보 전달 메커니즘 (4중 신호):
1. **halo expand** (08) — 입력 단계에서 sym 매핑된 cell을 텐서에 직접 추가
2. **STRING** — halo cell 좌표를 (6,6) 격자에 자연 통합, 상대 거리 attention
3. **LAPE** — halo cell 위치에도 학습 가능 변수 부여 (LAPE 변수 16K → 23K, +7K 1.07%)
4. **L_data_halo** — halo cell에 직접 supervision (λ=0.3)
5. **L_diff_rel** — Sobolev regularizer + Albedo BC 학습 (`Physical Loss 통합 레퍼런스 §3.6`)

### 4.2. Full-core 확장의 연산량 부담

| 항목 | Quarter (500) | **Halo (720)** | Full-core (1620) |
|---|:---:|:---:|:---:|
| Attention FLOPs (N² scaling) | 1× (기준) | **2.07×** | 10.5× |
| Mamba sequence 길이 | 1× | 1.44× | 3.24× |
| 학습 step 시간 | 1× | 약간 증가 | 3~5× 추정 |

- 4분면이 대칭이라 Full-core에는 ~75% 정보 중복인데도 attention의 N² 비용은 그대로
- **Halo (720)** 는 Full-core (1620) 대비 여전히 **5배 가벼움**. Quarter (500) 대비 2.07× 증가는 절대치 trivial (현재 격자 크기 기준)
- Halo expand의 연산 비용은 Full-core 대비 충분히 감수 가능 → 데이터 효율성 (Quarter) 과 ML 학습 신호 강도 (Halo) 양립

---

## 5. 후속 검토 조건 (개정 2026-04-09)

본 절은 본 문서 작성 시점의 가설 (PE만으로 대칭 정보 흡수 가능) 에 기반한 후속 검토 조건이었음. 05_symmetry_mode 결정 (halo expand) 으로 더 강한 신호가 추가되었으므로 본 절의 우려는 상당 부분 해소.

원래 우려:
- 학습 후 원점 부근 cell의 L_diff 잔차가 다른 cell 대비 현저히 높을 경우
- → STRING/LAPE만으로 대칭 정보 흡수 부족 신호

08 결정 후 후속 검토 순서 (조정):
1. **halo cell 학습 모니터링**: M1 (symmetry violation), M2 (halo phi error) — `compressed-wandering-stroustrup.md` Phase 4
2. halo expand + L_data_halo 로도 부족하면 → BC mask 채널 추가 검토
3. Full-core 재고는 마지막 옵션

---

## 6. 결정 과정

- 2026-04-08: 사용자 의문 "1/4 노심을 이용하는데에 따른 인코딩 문제가 없는가, 모델이 1/4만 보면서 BC는 디코더가 쓰는 게 정합성 있는가"
- 사용자 지적: "crop이 center row, col 포함하므로 self-loop는 자기 자신의 대칭" → Laplacian PE self-loop 폐기
- LAPE 정체와 역할 정리 후 BC mask 미사용 + Quarter 유지 합의
- 사용자 응답: "Quarter 5×5 유지 (Recommended)"

---

## 7. 참고

- 검토 과정: `2026-04-07 인코더 미결정 항목 및 Attention 가중치 비교.md`
- **인코더 처리 격자 결정 (halo 6,6)**: 본 폴더 `05_symmetry_mode.md`
- LAPE 도입 근거: 본 폴더 `05_position_encoding_lape.md`
- LAPE/STRING 적용 위치 + 학계 표준: `2026-04-04 3D 위치 인코딩 기법 검토.md` §3.6
- BC mask 미사용: 본 폴더 `06_omitted_options.md`
- L_data_halo + L_diff_rel: `physical loss 개선 계획/2026-03-30 Physical Loss 통합 레퍼런스.md` §3.6
- ML 위협 검토 + 문서 일관성 정정 plan: `C:\Users\Administrator\.claude\plans\compressed-wandering-stroustrup.md`
