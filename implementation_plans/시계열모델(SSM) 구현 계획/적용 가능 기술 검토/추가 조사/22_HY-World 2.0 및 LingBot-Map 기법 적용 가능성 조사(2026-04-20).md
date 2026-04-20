# HY-World 2.0 및 LingBot-Map 기법 적용 가능성 조사

> **작성일**: 2026-04-19
> **성격**: 외부 장기 예측 기법의 적용 가능성 매핑 보고서 (후보 식별까지만, 구현 채택 결정은 보류)
> **전제 설계**: `2026-04-15 모델 구현 계획(시계열).md` — C안 확정, Mamba-3 + Attention 2:1, N=2 Blocks
> **본 문서의 위치**: `2026-04-19 이후 참고 문헌/` 폴더. 확정 계획이 아닌 참고 자료
> **비고**: 본 조사는 현재 확정 아키텍처의 변경을 **제안하지 않음**. 외부 기법과의 대응 관계를 정리하고, 향후 필요 시 참조 가능한 대체/보강 후보 목록을 남김

---

## 1. 조사 배경 및 범위

### 1.1 배경

- CRS 575 step 장기 rollout 에서 hidden state drift 누적 우려가 이미 `2026-04-09 SSM 기반 시계열 모델 설계.md` §1.3 에 기재됨
- 이에 대응하는 기 채택 장치: pushforward, prefix burn-in, delta prediction, physical loss
- 외부 동향에서 장기 시점 예측 및 streaming 상태 관리에 집중한 두 모델이 2026년 초 공개됨
  - HY-World 2.0 (Tencent Hunyuan)
  - LingBot-Map (streaming 3D reconstruction)
- 이들이 제시하는 기법이 현재 설계의 장기 일관성 문제에 대한 추가 장치로 유효한지 검토 필요

### 1.2 범위

- **포함**: 두 외부 모델의 구성요소 중 현재 시공간 프로세서 (Mamba-3 + Attention) 와 구조적 대응이 성립하는 부분의 식별
- **포함**: 대체 또는 업그레이드 후보를 Tier 로 분류 및 우선순위 부여
- **제외**: 실제 구현 결정, 세부 설계 (조건 토큰 차원, cross-attention head 수 등)
- **제외**: 구현에 따른 성능 실험

---

## 2. 용어 주의 — "SSM" 약어 충돌

- 본 문서에는 **이름이 같지만 의미가 완전히 다른 두 SSM** 이 등장함. 혼동 방지를 위해 표기 약속을 먼저 제시
- 우리 계획 폴더 이름 `시계열모델(SSM)` 도 전자 의미임

| 표기 | 풀네임 | 맥락 | 의미 |
|---|---|---|---|
| **SSM (우리)** | State-Space Model | Mamba, Linear RNN 계열 | 연속 상태공간 수식을 신경망으로 근사하는 시계열 모델. h(t) 라는 hidden state 를 재귀로 갱신 |
| **SSM++ (HY-World)** | Spatial-Stereo Memory | 컴퓨터 비전 | 스테레오(양안) 시점 기반으로 과거 프레임의 공간 정보를 저장하고 조건부로 꺼내 쓰는 메모리 뱅크 |

- 두 SSM 은 기술 계보가 다름. 이름이 같은 것은 우연
- 이하 본 문서는 HY-World 개념을 가리킬 때 **"SSM++ (Spatial-Stereo Memory)"** 로 풀어 씀

---

## 3. 현재 시공간 프로세서 아키텍처 요약

| 구성요소 | 현재 설계 | 참고 |
|---|---|---|
| 시간 모델 | Mamba-3 단독 누적, h(t) 단일 텐서 | 시계열 계획 §5.3 |
| 공간 결합 | Self-Attention (시점별 stateless) | 시계열 계획 §5.4 |
| 조건 주입 | p_load 를 Mamba 입력에 concat | 시계열 계획 §5.5 |
| 장기 일관성 대응 | Pushforward, burn-in, delta prediction | 2026-04-09 §1.3 |
| Branch 지원 | h(t) detach snapshot (Drama 패턴) | 2026-04-16 학습방법론 §4 |
| 학습 샘플링 | 575 step 균등 parallel scan | 시계열 계획 §3.3 |

### 3.1 h(t) 의 본질 재확인

- Mamba hidden state 는 시점별 누적 압축된 **단일 텐서**
- T step 진행 후 모델이 "기억" 하는 것은 h(T) 하나
- 과거 h(1), h(2), … 를 별도 보관하지 않음
- 재귀형 모델의 본질적 특성이며, 장기 drift 는 이 압축 구조의 부작용

---

## 4. HY-World 2.0 의 정체와 원 맥락

### 4.1 HY-World 2.0 이 무엇인가

- **하는 일**: 텍스트/이미지 → 사용자가 시점을 옮기며 탐색 가능한 **3D 가상 공간 (world)** 을 생성
- 영상 생성 모델 (video generation) 과 3D 재구성 기술의 결합
- **4 단계 파이프라인** 으로 구성
  1. 파노라마 이미지 생성 (360도 뷰)
  2. 가상 카메라의 이동 경로 계획 (trajectory planning)
  3. 메모리 기반 뷰 확장 (이미 생성한 시점 정보를 참조해 새 각도 렌더링)
  4. 3D 기하 재구성 (여러 시점 → 3D 공간)
- 시계열 관련 핵심은 3 단계(뷰 확장) 와 4 단계(재구성 = WorldStereo 2.0) 에 몰려 있음

### 4.2 "frame" 이란 무엇이며 "축소" 는 어떤 의미인가

- **frame**: 3D world 생성 과정에서 중간 산출물로 만들어지는 **한 시점에서 본 2D 이미지**. 영화/애니메이션의 프레임과 동일 개념
- 가상 카메라가 경로를 따라 이동하면서 수백~수천 frame 이 생성됨
- "keyframe latent space" 의 "축소" 는 **차원 축소가 아니라 프레임 개수 축소**
  - 수백 frame 을 모두 신경망으로 처리하면 비용이 큼
  - 정보 밀도가 높은 소수의 frame (= keyframe) 만 선별해 latent 로 인코딩
  - 나머지 frame 은 keyframe 을 기준으로 보간/참조
  - 애니메이션에서 "주요 장면만 그리고 사이는 자동 채움" 과 같은 발상

### 4.3 구성요소 대응표 (원 맥락 포함)

| HY-World 개념 | 원 맥락에서의 역할 | 우리 원자로 모델과의 대응 | 상태 |
|---|---|---|---|
| **Keyframe latent space** | 가상 카메라 경로상의 수많은 2D frame 중 정보량 큰 소수만 선별해 latent 인코딩. 계산량·메모리 축소. | 우리는 5 분 간격으로 575 step 물리 상태가 주어짐. "버릴 수 있는 frame" 이란 개념이 성립하지 않음 (물리적 상태를 임의로 누락 불가) | 개념 부적합 |
| **Global-Geometric Memory (GGM)** | 지금까지 재구성한 3D 공간의 기하 정보 (점·면 집합 = point cloud) 를 전역 메모리로 유지. 새 시점 생성 시 참조. | 우리 Attention 이 매 시점 720 cell 공간을 재결합하는 것과 **형식만 유사**. 다만 우리는 시점간 공간 메모리를 별도 유지하지 않음 | 부분 대응 |
| **SSM++ (Spatial-Stereo Memory)** | 과거 시점 frame 전체를 참조하지 않고 **현재와 유사한 과거 시점의 일부만 선택적으로 불러와** feature 를 보강. retrieval 방식의 메모리 뱅크. | 현재 h(t) 는 모든 과거가 균등 압축된 단일 텐서. 선택적 참조 기능 없음 | 미반영 |
| **Camera pose 7D 토큰 conditioning** | 카메라 자세를 7 차원 벡터 (위치 3D + 회전 쿼터니언 4D) 로 인코딩하여 **별도 토큰**으로 feature 에 주입. | 우리 p_load 는 Mamba 입력에 단순 concat. "조건을 별도 토큰으로" 라는 구조가 없음 | 약한 형태만 존재 |

### 4.4 이식 가치 평가

- **유효**: SSM++ (Spatial-Stereo Memory) 의 "선택적 retrieval" 철학, camera pose 토큰화 철학
- **부적합**: keyframe latent (우리 물리 시계열에는 frame 을 개수 기준으로 축소할 여지 없음)
- **이식 불가**: 파노라마 생성, trajectory planning, camera ray 기하 (원자로 격자 동역학에 대응 없음)

---

## 5. LingBot-Map 의 정체와 원 맥락

### 5.1 LingBot-Map 이 무엇인가

- **하는 일**: 로봇/드론에 달린 카메라가 움직이는 동안, 실시간으로 **3D 공간을 점진적으로 복원**
- 핵심 키워드는 **streaming 3D reconstruction**
  - streaming: 전체 영상을 미리 보지 않고 프레임이 들어오는 대로 순차 처리
  - 우리 Mamba 자기회귀 추론과 구조적으로 동일한 사고 (과거만 보고 현재 판단)
- 사용하는 핵심 모듈: **Geometric Context Transformer**
  - 3D 기하 문맥 (geometry) 을 attention 의 입력 구성에 명시적으로 반영
  - attention 의 입력 문맥을 **세 종류로 분해** 하는 것이 LingBot-Map 의 고유 디자인

### 5.2 세 종류 문맥의 의미

| 문맥 | 원 맥락에서의 의미 | 기능 |
|---|---|---|
| **Anchor context** | 세계 좌표계의 **고정 기준점**. 카메라가 어디로 움직여도 변하지 않는 "세상의 원점" 에 해당하는 latent. | 전역 좌표 일관성 보장. drift 되지 않는 고정 기준 |
| **Pose-reference window** | 가장 최근 몇 프레임의 카메라 자세 (pose) + 주변 local feature 를 **고해상도로 직접** 유지. | 최근 local 정보는 압축하지 않고 그대로 보존 → 미세 변화 추적 |
| **Trajectory memory** | 지금까지의 카메라 **이동 궤적 (trajectory)** 전체를 **압축 latent** 로 저장. | 아주 옛날 시점 정보도 적은 비용으로 참조 가능. 장기 drift 보정 |

- 이 세 요소가 합쳐지면 streaming 상태를 compact 하게 유지하면서도 long-range correction 가능
- 우리 용어로 번역하면 "전역 기준 + 최근 local + 장기 압축 기억" 의 3 단 메모리

### 5.3 구성요소 대응표 (원 맥락 포함)

| LingBot 개념 | 원 맥락에서의 역할 | 우리 원자로 모델과의 대응 | 상태 |
|---|---|---|---|
| **Anchor context** | 세계 좌표계 원점 역할. 카메라 이동 중에도 변하지 않는 기준 latent. | 현재 아키텍처에는 **없음**. p_load 만 매 step concat 될 뿐 "구간 기준" 이나 "초기 정상상태 기준" latent 를 유지하는 장치 없음 | 미반영 |
| **Pose-reference window** | 최근 8~32 프레임의 카메라 자세/local feature 를 별도 stack 으로 고해상도 유지. | Mamba h(t) 가 최근 정보를 강하게 반영하지만 내부에 **압축된 형태**로만 존재. 별도 stack 으로 꺼낼 수 없음 | 압축되어 분리 불가 |
| **Trajectory memory** | 과거 이동 궤적 전체를 압축 latent 로 저장. | Mamba h(t) 가 형식상 이 역할을 겸하고 있음. 다만 LingBot 은 trajectory 를 "메모리 형태로 꺼내서 참조" 하는 반면, 우리는 h(t) 가 내부 재귀에만 쓰임 | 부분 대응, drift 우려 그대로 |

### 5.4 TokenWM Memory Bank 와의 관계 (기 검토, 미도입)

- 우리 `적용 가능 기술 검토/03_TokenWM.md` 에 Memory Bank Cross-Attention 기법이 **이미 검토됨**
- 현재 설계는 TokenWM 의 720 cell × 128D 토큰 구조만 채택하고 **Memory Bank 자체는 미도입**
- TokenWM Memory Bank, LingBot trajectory memory, HY-World SSM++ 는 **사실상 같은 계열의 기법** (과거 정보를 메모리로 저장하고 필요 시 retrieval)
- 재검토 가치 있음

---

## 6. 대체 또는 업그레이드 후보 (Tier 분류)

### 6.1 Tier 1 — 구조 변경 작고 효과 큼

#### 6.1.1 후보 ① Condition 토큰화 (p_load / sym_type / load regime)

- **현재 제약**
  - p_load 를 Mamba 입력에 단순 concat → Attention block 은 간접 전달만 받음
  - sym_type 은 인코더 LAPE 에 반영되어 Processor 에서 재활용 불가
  - load regime (정출력 / 감발 / 증발 등) 은 명시적으로 입력되지 않음
- **대체안**
  - p_load(t), sym_type, load regime id 를 별도 condition 토큰 1~3 개로 인코딩
  - Attention block 입력에 prepend → (B, 720, 128) 이 (B, 720+K, 128) 이 됨
  - HY-World 7D camera pose 토큰 → 우리 조건 벡터로 번역
  - LingBot anchor context "전역 기준 상태" 역할 겸함
- **구현 난이도**: 낮음
- **기대 효과**: p_load 급변 구간 공간 응답 패턴 더 명시적 학습, Branch 예측 시 load regime 구분 강화
- **리스크**: K 개 토큰이 720 cell 토큰 attention weight 를 희석 가능. 튜닝 필요

#### 6.1.2 후보 ② Event memory (소규모 참조 메모리 뱅크)

- **현재 제약**
  - h(t) 누적 압축이므로 575 step 후반에 초기 급변 구간 정보 drift 가능
  - Branch 분기 시 h(t) 외 참조 근거 없음
- **대체안**
  - 소규모 memory slot (M=8~16) 유지
  - 저장 trigger 후보
    - rod_map 변화량이 임계 초과한 시점 직후 latent
    - p_load slope 부호 전환 시점
    - Branch 분기 직전 시점
  - Attention block 내부에 cross-attention 1 층 추가하여 현재 토큰이 memory 참조
- **구현 난이도**: 중
- **기대 효과**: 장기 drift 보정, 이벤트 직후 상태를 직접 참조하여 일관성 개선
- **리스크**: trigger 휴리스틱이 도메인 지식 의존. 잘못 설계 시 의미 없는 정보만 축적
- **재사용 가능 자산**: `적용 가능 기술 검토/03_TokenWM.md` Memory Bank 설계

### 6.2 Tier 2 — 구조 변경 중간, 효과 검증 필요

#### 6.2.1 후보 ③ Anchor 토큰 (구간 기준 latent)

- 현재 load segment (정출력 유지 구간 / 감발 구간 등) 시작점 latent 를 별도 보관
- Attention block 에서 global anchor 로 참조
- 근거: LingBot anchor context 직접 이식
- 난이도: 중. "segment 시작" 정의 필요
- 후보 ① 과 결합 가능: condition token 중 하나를 anchor latent 로 대체

#### 6.2.2 후보 ④ Reference window (최근 N step latent stack)

- h(t) 에 모든 과거를 압축하는 대신 **최근 8~32 step latent 를 별도 stack**
- Attention block 이 window 를 직접 참조 (또는 cross-attention)
- 근거: LingBot pose-reference window
- 난이도: 중~상. Mamba parallel scan 과의 통합이 설계 문제
- **주의**: Mamba 가 이미 최근 정보를 강하게 반영하므로 reference window 가 정보 중복일 가능성. 실험 전 기대 이익 낮게 평가

### 6.3 Tier 3 — 학습 전략 영역 (구조 변경 불필요)

#### 6.3.1 후보 ⑤ 이벤트 기반 샘플링 커리큘럼

- HY-World keyframe 철학의 "약한 형태" 이식
- 정출력 plateau 구간은 sparse 샘플링, rod/부하 급변 구간은 dense 샘플링
- 학습 루프 단계에서만 구현. 모델 구조 변경 없음
- 난이도: 낮음. 데이터 로더 샘플링 weight 조정
- Tier 1, 2 와 독립적으로 병행 가능

---

## 7. 적용 불가 또는 이식 의미 없는 부분 (명시적 배제)

| 외부 기법 | 배제 사유 |
|---|---|
| HY-World 파노라마 생성 / trajectory planning | 시계열 예측이 아니라 콘텐츠 생성 파이프라인 |
| HY-World keyframe 프레임 개수 축소 | 5 분 간격 물리 시계열은 "버릴 수 있는 프레임" 개념 부적합 |
| LingBot 3D 좌표 grounding | 원자로 격자는 고정 토폴로지. 좌표 기반 grounding 불필요 |
| HY-World camera ray 기하 conditioning | 원자로 시뮬레이션에 기하학적 대응물 없음 |

---

## 8. 기존 확정 아키텍처와의 정합성 체크

- **h(t) detach + Drama 패턴**: Event memory (②) 와 충돌하지 않음. Event memory 는 추가 채널, h(t) 누적은 그대로 유지
- **Mamba-3 복소 상태**: Condition 토큰 (①) 은 Attention block 에만 추가되므로 Mamba 구조 영향 없음
- **2-Phase 학습**: Condition 토큰은 Phase 1 (공간 사전학습) 단계에서도 바로 활용 가능 (p_load 는 이미 인코더 입력에 포함됨)
- **Physical Loss**: 본 문서의 어떤 후보도 Loss 구조를 변경하지 않음
- **잔차 예측 디코더 (Conv3D 1×1×1)**: 본 문서의 어떤 후보도 디코더를 변경하지 않음

---

## 9. 종합 권장 및 후속 판단 보류 메모

### 9.1 우선순위

| 우선순위 | 후보 | 근거 | 기대 가치 |
|---|---|---|---|
| 1 | 후보 ① Condition 토큰화 | 구조 변경 최소, 미검토 영역, 설계 자연스러움 | 높음 |
| 2 | 후보 ② Event memory | drift 문제 직접 대응, TokenWM 기존 검토 확장 | 중~높음 |
| 3 | 후보 ⑤ 이벤트 샘플링 커리큘럼 | 모델 구조 불변, 학습 전략 보강 | 중 |
| 4 | 후보 ③ Anchor 토큰 | 후보 ① 에 흡수 가능 | 중 |
| 5 | 후보 ④ Reference window | Mamba 와 기능 중복 가능, 구조 비용 큼 | 불확실 |

### 9.2 판단 보류 상태 명시

- 본 문서는 **후보 식별까지만** 담음
- 실제 구현 채택은 별도 시점에 별도 문서에서 결정
- 현재 확정 아키텍처 (`2026-04-15 모델 구현 계획(시계열).md`) 는 본 문서를 이유로 **수정되지 않음**
- 학습 계획 폴더, 인코더 계획 문서, Physical Loss 계획 문서 모두 영향 없음

### 9.3 후속 문서 작성 시 필요 작업 (참고용 메모)

- 후보 ① 채택 시
  - condition token 차원 결정 (K=1 혹은 K=3)
  - condition token 의 학습 가능한 encoder 설계
  - Attention block 입력 확장에 따른 position encoding 영향 검토
- 후보 ② 채택 시
  - trigger 휴리스틱 정의 (임계값, slope 판정 방식)
  - memory slot 갱신 정책 (FIFO / 학습 가능 gating)
  - cross-attention head 수 및 위치
  - `03_TokenWM.md` Memory Bank 구현 참조
- 후보 ⑤ 채택 시
  - 이벤트 구간 정의 기준
  - 데이터 로더 샘플링 weight 식

---

## 부록 A. 참고 문헌

### A.1 외부 기법 원 출처

- HY-World 2.0: https://github.com/Tencent-Hunyuan/HY-World-2.0
- LingBot-Map: https://huggingface.co/robbyant/lingbot-map

### A.2 본 문서와 관련된 내부 문서

- `2026-04-15 모델 구현 계획(시계열).md` — 현재 확정 아키텍처 (변경 대상 아님)
- `2026-04-09 SSM 기반 시계열 모델 설계.md` §1.3 — 장기 drift 우려 최초 기재
- `2026-04-16 시계열 아키텍처 학습방법론 재검증.md`
- `2026-04-16 시계열 학습방법론 검토.md` §4 — Drama h(t) detach 패턴
- `적용 가능 기술 검토/02_Drama.md` — h(t) 관리 및 InferenceParams snapshot
- `적용 가능 기술 검토/03_TokenWM.md` — Memory Bank Cross-Attention (기 검토, 미도입)
- `적용 가능 기술 검토/06_Mamba_계열.md` — Mamba-3 복소 상태 근거
- `적용 가능 기술 검토/21_Mamba_Attention_하이브리드.md` — 하이브리드 비율 근거
