# 적용 가능 기술 검토 — 인덱스

> **작성일**: 2026-04-15
> **목적**: 원자로 노심 시공간 대리모델의 시계열 처리를 위한 기술 스택 포괄 조사
> **범위**: 학술 월드모델, SSM 변종, 자율주행/로보틱스 역공학, 산업 PDE 대리모델
> **대체**: 기존 `2026-04-15 시계열모델 적용 기법 검토.md` → 본 폴더로 대체

---

## 우리 아키텍처 요약

```
[halo_expand] → [Spatial Encoder] → [Temporal Model] → [Spatial Decoder] → [crop]
                   (ViT, 완료)        (Mamba, 미착수)     (미착수)
```

- **입력**: (B, 20, 6, 6, 21) — state(10) + xs_fuel(10) + rod_map(1)
- **인코더 출력**: (B, 20, 6, 6, 128) — 720 cell token × 128차원
- **시간 모델**: cell-wise 독립 Mamba, CRS 575-step 자기회귀 / Branch 단일시점
- **제어 주입**: rod_map → 인코더 21ch, p_load → Mamba concat + Decoder AdaLN-Zero
- **핵심 특징**: Xe/I-135 마르코프 성질, Branch GT (oracle imagination), 물리 제약 Loss

---

## 기능 분류 체계

각 소스 문서는 아래 6개 기능 축으로 분석한다. 해당 없는 기능은 생략.

| 기능 | 설명 | 우리 현재 설계 |
|------|------|---------------|
| **시간 모델링** | 상태 h(t)를 다음 시점으로 전이하는 메커니즘 | Mamba S6, cell-wise 독립 |
| **상태 표현** | 잠재 공간의 구조와 차원 | 720 cell token × 128D (결정적) |
| **제어/조건 주입** | 외부 제어 변수를 모델에 넣는 방식 | concat(Mamba) + AdaLN(Decoder) |
| **시공간 결합** | 공간과 시간 정보의 결합 방식 | 분리 (공간=인코더 attn, 시간=Mamba) |
| **롤아웃 안정화** | 자기회귀 누적 오차 대응 | pushforward trick + scheduled sampling |
| **분기/다중경로** | 동일 시점에서 여러 미래를 평가하는 방식 | Branch Mamba 우회 (시간맥락 없음) |

---

## 소스 문서 목록

### 1. 학술 월드모델 계열

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 01 | [DreamerV3](01_DreamerV3.md) | Hafner et al. (ICLR 2024 / Nature 2025) | 완전공개 | RSSM, Block GRU, Cat(32×32), imagination |
| 02 | [Drama](02_Drama.md) | Wang et al. (ICLR 2025) | 완전공개 | Mamba-2 MBRL, InferenceParams, prefix burn-in |
| 03 | [TokenWM](03_TokenWM.md) | Zhai et al. (ICLR 2025 Workshop) | 완전공개 | 토큰화 잠재 상태, FiLM, Memory bank |
| 04 | [Dreamer-CDP](04_Dreamer-CDP.md) | Mohammadi et al. (2026) | 완전공개 | 재구성 없는 세계모델, 잠재 예측 loss |
| 05 | [V-JEPA 2](05_V-JEPA_2.md) | Meta AI (2025) | 완전공개 | 동결 인코더, action-conditioned predictor |

### 2. SSM / 시퀀스 모델 변종

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 06 | [Mamba 계열](06_Mamba_계열.md) | Gu, Dao et al. (2023-2026) | 완전공개 | S4→S6→SSD→Mamba-3, 이산화, 복소 상태 |
| 07 | [MNO 신경연산자](07_MNO_신경연산자.md) | Cheng et al. (JCP 2025) | 완전공개 | Mamba + PDE 대리모델, 90% 오차 감소 |
| 08 | [Neural ODE / Latent ODE](08_Neural_ODE_Latent_ODE.md) | Chen et al. (NeurIPS 2018) + 후속 | 완전공개 | 연속시간 모델, adjoint method |

### 3. 자율주행 세계모델

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 09 | [Tesla FSD](09_Tesla_FSD.md) | Tesla AI (2021-2025) | 역공학추론 | Occupancy Network, Neural World Simulator |
| 10 | [Waymo](10_Waymo.md) | Waymo Research (2023-2025) | 부분공개 | UniSim, 시뮬레이션 기반 검증 |
| 11 | [NVIDIA Cosmos](11_NVIDIA_Cosmos.md) | NVIDIA (2025) | 부분공개 | DiT 잠재 확산, Predict/Transfer/Reason |

### 4. 로보틱스 행동모델

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 12 | [GR00T N1](12_GR00T_N1.md) | NVIDIA (2025) | 부분공개 | VLA dual system, DiT 행동 생성 |
| 13 | [RT-2 / VLA](13_RT-2_VLA.md) | Google DeepMind (2023-2025) | 부분공개 | Vision-Language-Action, 웹 스케일 사전학습 |
| 14 | [π₀ Diffusion Policy](14_pi0_Diffusion_Policy.md) | Physical Intelligence (2024-2025) | 부분공개 | Flow matching, action chunking |

### 5. 산업 PDE / 물리 대리모델

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 15 | [NVIDIA Modulus](15_NVIDIA_Modulus.md) | NVIDIA (2021-2025) | 완전공개 | PINN/FNO/DeepONet 통합 프레임워크 |
| 16 | [FNO 스펙트럼](16_FNO_스펙트럼.md) | Li et al. (ICLR 2021) + 후속 | 완전공개 | Fourier Neural Operator, 해상도 불변 |
| 17 | [Genie 3](17_Genie_3.md) | Google DeepMind (2025) | 부분공개 | 자기회귀 세계 생성, Latent Action Model |

### 6. 아키텍처 패턴 조사

| # | 파일 | 소스 | 공개 수준 | 핵심 키워드 |
|---|------|------|----------|-----------|
| 20 | [기상/물리 ML 아키텍처 패턴](20_기상_물리_ML_아키텍처_패턴.md) | GraphCast, FourCastNet, GenCast, ClimaX 등 | 완전공개 | 인코더-프로세서-디코더 비대칭 설계, 경량 디코더 |
| 21 | [Mamba+Attention 하이브리드](21_Mamba_Attention_하이브리드.md) | Jamba, Nemotron-H, Hymba, OLMo 등 (2024-2026) | 완전공개 | SSM:Attn 3:1~10:1, 도메인 횡단 검증, GDN |

### 7. 종합 분석

| # | 파일 | 내용 |
|---|------|------|
| 18 | [기능별 교차비교](18_기능별_교차비교.md) | 전 소스에서 추출한 기법을 기능 축별로 교차 비교 |
| 19 | [우리 아키텍처 적용 로드맵](19_우리_아키텍처_적용_로드맵.md) | 최종 채택 후보, 구현 우선순위, 리스크 평가 |

---

## 각 문서의 표준 구조

모든 소스 문서는 다음 섹션으로 구성된다:

1. **시스템 개요** — 아키텍처 다이어그램, 해결 문제, 입출력
2. **기능별 상세 분석** — 6개 기능 축 중 해당하는 항목만 기술
3. **역공학 추론** — 비공개 소스에만 해당. 확신도 표기 (높음/중간/낮음)
4. **우리 문제 적용성 평가** — 기능별 O/△/X + 근거 + 구현 난이도
5. **핵심 차용 후보** — 즉시 적용 / 수정 후 적용 / 부적합 분류

---

## 기존 문서의 6개 핵심 발견 (재평가 대상)

기존 `2026-04-15 시계열모델 적용 기법 검토.md`의 발견을 본 폴더의 상세 분석으로 재검증한다:

1. ~~우리는 이미 TokenWM 아키텍처를 구현하고 있었다~~ → `03_TokenWM.md`에서 재평가
2. ~~Drama의 prefix burn-in이 Phase 1→2 전환의 핵심~~ → `02_Drama.md`에서 재평가
3. ~~Dreamer-CDP의 잠재 예측 loss가 Mamba 학습을 가속~~ → `04_Dreamer-CDP.md`에서 재평가
4. ~~V-JEPA 2의 frozen-encoder 패턴이 2-Phase를 정당화~~ → `05_V-JEPA_2.md`에서 재평가
5. ~~Branch GT는 oracle imagination~~ → `18_기능별_교차비교.md`에서 재평가
6. ~~Mamba-2가 실용적 선택, Mamba-3는 향후~~ → `06_Mamba_계열.md`에서 재평가
