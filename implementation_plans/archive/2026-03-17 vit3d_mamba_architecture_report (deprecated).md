> [deprecated 2026-03-30] Part1(ViT3D) → 폐기. Part2(Mamba SSM) → `2026-03-30 Mamba SSM 수학 레퍼런스.md`. Z=22→Z=20 오류.

# 공간 인코더(ViT3D) + 시간 모델(Mamba) 연동 구조 및 데이터 전처리 설계서

> SMART 원자로 부하추종 운전 서로게이트 모델

| 항목 | 내용 |
|---|---|
| 대상 원자로 | SMART (365 MWth) |
| 노심 구조 | 22(축방향) x 5 x 5 (1/4 대칭), N_node = 550 |
| 시나리오 | 20개, 576 스텝/시나리오 (48h, 5분 간격) |
| 봉 위치 | 31개/스텝 (CRS 1 + Branch 30) |
| 프레임워크 | TensorFlow 2 |
| GPU | NVIDIA RTX A6000 x 3 (48GB each) |
| 작성일 | 2026-03-17 |

본 문서는 코딩 에이전트가 데이터 전처리 파이프라인 및 모델 학습 코드를 구현할 때 참조하는 설계 사양서입니다. 공간 인코더(CustomMaxViT3D)와 시간 모델(Mamba)의 역할 분담, 데이터 흐름, HDF5 스키마, 그리고 Dataloader 조립 로직을 상세히 기술합니다.

---

## 목차

1. [핵심 개념: 왜 2-Stage 구조인가](#1-핵심-개념-왜-2-stage-구조인가)
2. [전체 아키텍처 상세](#2-전체-아키텍처-상세)
3. [공간 인코더 상세](#3-공간-인코더-spatialencoder-상세)
4. [Mamba 시간 모델 상세](#4-mamba-시간-모델-상세)
5. [데이터 전처리 설계](#5-데이터-전처리-설계)
6. [Dataloader 조립 로직](#6-dataloader-조립-로직-tfdatadataset)
7. [전처리 파이프라인](#7-전처리-파이프라인-raw-data--hdf5)
8. [학습 및 추론 흐름](#8-학습-및-추론-흐름)
9. [구현 체크리스트](#9-구현-체크리스트-코딩-에이전트용)
10. [미결 사항 및 결정 필요 항목](#10-미결-사항-및-결정-필요-항목)
11. [국부적 공간 효과(Local Spatial Effect) 포착 한계 분석](#11-국부적-공간-효과local-spatial-effect-포착-한계-분석)

---

## 1. 핵심 개념: 왜 2-Stage 구조인가

### 1.1 Mamba의 본질

Mamba는 **1D 시퀀스 모델**입니다. 매 시점 `t`에서 하나의 벡터 `x(t)`를 받아 내부 hidden state `h(t)`를 갱신하며, `h(t)`는 과거 이력 전체의 압축된 요약본 역할을 합니다.

```
h(t) = A * h(t-1) + B * x(t)
y(t) = C * h(t)
```

여기서 `x(t)`는 **벡터 하나**입니다. `[22, 5, 5, C]` 같은 3D 텐서를 직접 입력받는 구조가 아닙니다.

따라서 노심의 3D 공간 분포(중성자속, 온도, 밀도 등)를 Mamba에 직접 넣을 수 없으며, **3D 공간 정보를 벡터로 요약(임베딩)하는 별도의 공간 인코더가 반드시 필요**합니다.

### 1.2 역할 분담

| 구분 | 모듈 | 역할 | 입력 형태 | 출력 형태 |
|---|---|---|---|---|
| 공간 인코더 | CustomMaxViT3D (기존 구현 재활용) | 이 시점의 노심 상태가 공간적으로 어떤 분포인가? | `[22, 5, 5, C_in]` | `z(t) : [D_model]` |
| 시간 모델 | Mamba | 노심 상태가 시간에 따라 어떻게 변해왔는가? | `z(1),...,z(T) : [T, D_model]` | `h(1),...,h(T) : [T, D_model]` |
| 예측 헤드 | MLP + Reshape | h(t)와 query로부터 3D 물리량 예측 | `h(t) + delta_z(t,p)` | 출력 물리량 `[N_node]` |

공간 인코더는 매 타임스텝에 독립적으로 적용(가중치 공유)되며, Mamba는 공간 인코더가 출력한 요약 벡터의 시계열만 처리합니다. 즉, Mamba에게 3D 공간 구조를 이해시키는 것이 아니라, **이미 공간 정보가 요약된 벡터들의 시간적 변화만 추적**하게 합니다.

---

## 2. 전체 아키텍처 상세

### 2.1 전체 데이터 흐름 (학습 시)

**Stage 1: 공간 인코딩 (매 스텝 t, 가중치 공유, 576회)**

```
input_crs(t) = concat([                         총 16채널
    xs_fuel      [22,5,5,10],   고정 거시단면적
    Xe(t)        [22,5,5, 1],   Xe-135 수밀도
    Sm(t)        [22,5,5, 1],   Sm-149 수밀도
    T_cool(t)    [22,5,5, 1],   냉각재 온도
    rho_cool(t)  [22,5,5, 1],   냉각재 밀도
    power_prev(t)[22,5,5, 1],   이전 스텝 critical 출력
    rod_map_crs(t)[22,5,5,1]    CRS 제어봉 위치
], axis=-1) = [22, 5, 5, 16]

z_crs(t) = SpatialEncoder(input_crs(t))  -->  [D_model]
```

**Stage 2: 시간 스캔 (Mamba, 시나리오 전체)**

```
z_crs(1), z_crs(2), ..., z_crs(576)
  + pload(1), pload(2), ..., pload(576)   (FiLM 또는 concat conditioning)
  --> Mamba Block x N_layers
  --> h(1), h(2), ..., h(576)             각 h(t): 시간 이력이 압축된 상태 벡터
```

**Stage 3: 예측 헤드 (Critical + 31-way Branch)**

```
[Critical 예측 (p=0)]
h(t) --> CriticalHead --> Xe(t+1), Sm(t+1), power, T_cool, T_fuel, rho_cool, keff

[Branch 예측 (p=0~30)]
delta_rod(t,p)  = rod_branch(t,p) - rod_crs(t)           [22,5,5,1]
delta_z(t,p)    = RodDeltaEncoder(delta_rod(t,p))         [D_delta]   경량 3D Conv
concat(h(t), delta_z(t,p), pload_embed)
  --> BranchHead --> power, T_cool, T_fuel, rho_cool, keff
```

### 2.2 Branch Query 처리: 차분(Delta) 인코딩

CRS 입력과 Branch 입력의 차이는 오직 `rod_map`뿐입니다. `xs_fuel`, `Xe`, `Sm`, `T_cool`, `rho_cool`, `power_prev`는 동일 시점에서 전부 동일합니다(Frozen Xenon).

따라서 Branch query를 처리하는 가장 효율적이고 물리적으로 정확한 방식은 **차분(delta) 인코딩**입니다:

| 항목 | 설명 |
|---|---|
| 물리적 의미 | Branch는 CRS 상태에서 봉 위치만 달리한 '섭동(perturbation)' 계산. `delta_rod`가 섭동의 크기와 공간적 위치를 정확히 표현 |
| 효율성 | SpatialEncoder(무거운 MaxViT3D)는 CRS 1회만 호출. RodDeltaEncoder(경량 Conv3D 2~3층)만 31회 추가 호출 |
| 구조적 장점 | `p=0`(CRS offset)이면 `delta_rod=0` → CRS 예측과 자동 일관. `h(t)`에 이미 공간 정보가 인코딩되어 있으므로, delta만 추가하면 충분 |

**대안 비교:**

| 방식 | 설명 | 비고 |
|---|---|---|
| Option 1 (풀 인코딩 31회) | 정확하지만 31×576 = 17,856회 MaxViT3D 호출 필요 | 비효율 |
| Option 2 (rod만 별도 인코딩) | 공간 상호작용 일부 누락 | 부정확 |
| **Option 3 (Delta)** | **효율 + 정확도 균형 최적** | **권장** |

---

## 3. 공간 인코더 (SpatialEncoder) 상세

### 3.1 기존 CustomMaxViT3D 재활용

기존에 단일 스냅샷 예측용으로 구현된 CustomMaxViT3D의 구성요소를 거의 그대로 재활용합니다.

| 기존 모듈 | 재활용 | 용도 |
|---|---|---|
| BoundaryPad | 그대로 | 1/4 대칭 경계 반사 패딩 |
| SuperpositioningNeighborCells | 그대로 | 인접 셀 정보 중첩 (공간 context 확장) |
| Stem3D (Conv3D+BN+GELU) | 그대로 | 초기 특성 추출 |
| MBConv3D | 그대로 | 지역적 특성 추출 (depthwise separable 3D) |
| BlockAttention3D (local) | 그대로 | 지역 복셀 어텐션 (인접 노드 상호작용) |
| GridAttention3D (global sparse) | 그대로 | 전역 희소 어텐션 (원거리 공간 상관) |
| RelativeAttention3D | 그대로 | 3D 상대 위치 편향 |
| AxialSE | 그대로 | 축방향 Squeeze-Excitation |
| FFN_3D | 그대로 | 비선형 변환 |
| **final_conv** | **수정 필요** | **기존: P3D 출력 → 수정: Global Pool + Linear → z(t)** |

### 3.2 수정 사항

**(1) final_conv 교체:**
기존 `final_conv`는 `[B,22,5,5]` P3D 출력용. 이를 `GlobalAveragePooling3D + Linear(D_model)`로 교체하여 `z(t)` 벡터 출력.

**(2) 입력 채널 수 변경:**
기존 10ch (XS 9 + rod 1) → **16ch** (XS 10 + Xe 1 + Sm 1 + T_cool 1 + rho_cool 1 + power_prev 1 + rod_map 1). Stem3D의 첫 Conv3D `in_channels`만 수정.

**(3) 가중치 공유:**
모든 타임스텝(576개)에서 동일한 SpatialEncoder 인스턴스 적용.

### 3.3 RodDeltaEncoder (경량 Branch Query 인코더)

```python
RodDeltaEncoder( delta_rod [22,5,5,1] )
  --> Conv3D(1,  32, kernel=3, pad='same') + BN + ReLU
  --> Conv3D(32, 64, kernel=3, pad='same') + BN + ReLU
  --> GlobalAveragePool3D
  --> Linear(64, D_delta)
  --> delta_z : [D_delta]

delta_rod = rod_map_branch(t,p) - rod_map_crs(t)
```

대부분의 노드에서 `0`이며, 봉 이동 영향 노드만 비영값 → **sparse 특성**

---

## 4. Mamba 시간 모델 상세

### 4.1 입력 형태

Mamba는 공간 인코더의 출력 시퀀스를 입력받습니다:

```
입력: [z_crs(1), z_crs(2), ..., z_crs(576)]   shape: [B, T=576, D_model]
pload conditioning: pload(t)를 FiLM 또는 concat으로 각 스텝에 적용
출력: [h(1), h(2), ..., h(576)]                shape: [B, T=576, D_model]
```

### 4.2 Mamba 블록 구조

| 구성 요소 | 설명 | 역할 |
|---|---|---|
| Linear Projection | D_model → D_inner (expansion) | 차원 확장 |
| 1D Conv | kernel=4, causal padding | 지역 시간 패턴 포착 |
| SiLU Activation | Swish 게이팅 | 비선형성 도입 |
| SSM (Selective Scan) | A, B, C, delta가 입력 의존 | 시간 동역학 추적의 핵심 |
| Gate + Output Proj | D_inner → D_model | 잔차 연결과 결합 |

### 4.3 선택 메커니즘과 원자로 물리의 정합성

| 물리 현상 | Mamba 반응 | 정합성 |
|---|---|---|
| Xe 과도현상 (pload 급변 시) | delta(t) 증가 → h(t) 급격 갱신 | 높음: Xe 시간 상수(~9h)의 비선형 과도를 추적 |
| 정상 운전 (pload 일정, Xe 평형) | delta(t) 감소 → h(t) 느리게 변화 | 적합: 평형 접근 시 이전 상태를 오래 기억하는 것이 물리적으로 정확 |
| 봉 이동 이벤트 (CRS 위치 변화) | B(t), C(t) 입력 의존 변화 | 적합: 봉 이동에 따른 중성자속 재분포를 h(t)에 반영 |
| 출력 분포의 비선형성 (power peaking) | Mamba 자체는 선형 recurrence → 비선형은 예측 헤드에서 | 보완 필요: 예측 헤드(MLP)의 비선형 층으로 대응 |

### 4.4 Mamba-2 (SSD) 적용 고려

Mamba-2는 상태 차원(N)을 16에서 64~256으로 확대하면서 학습 속도를 **2~8배 개선**했습니다. 550개 노드의 다양한 공간 패턴이 `z(t)` 벡터에 압축되어 있으므로, 더 큰 상태 공간이 이 정보를 풍부하게 인코딩하는 데 유리합니다. A6000 3대 환경에서 학습 속도 향상도 실질적 이점입니다.

> **주의:** 공식 Mamba 구현은 PyTorch + Triton 기반입니다. TensorFlow 2에서는 selective scan의 커스텀 구현이 필요하며, 이것이 가장 큰 기술적 도전 과제입니다. 대안으로 `tf.while_loop` 기반 SSM 구현 또는 JAX 변환을 고려할 수 있습니다.

---

## 5. 데이터 전처리 설계

### 5.1 기존 계획 해설 (첨부 md 파일)

기존 계획(`data_preproc_package_plan.md`)은 시나리오 1개를 통째로 HDF5에 저장하는 방식입니다. 즉, 576 타임스텝 × 31 봉위치의 모든 물리량을 시나리오 단위로 묶어 저장합니다. 이는 Mamba의 시퀀스 처리 특성상 올바른 접근입니다 — Mamba는 시나리오 전체(576스텝)를 하나의 시퀀스로 스캔하므로, **시나리오 단위 저장이 학습 데이터 로딩에 가장 자연스럽습니다.**

기존 HDF5 스키마는 Mamba에 직접 flatten된 벡터를 넣는 구조로 설계되었습니다. 2-Stage 구조에서는 공간 인코더가 3D 형태를 필요로 하므로, **저장 형태를 3D로 유지하거나 Dataloader에서 reshape하는 방식으로 수정**합니다.

### 5.2 HDF5 스키마 (2-Stage 구조 반영)

기존 스키마 대비 주요 변경점:
1. N_node(550) 대신 3D shape(22,5,5) 저장 옵션 추가
2. 공간 인코더 입력 조립을 위한 채널 분리 유지
3. Branch rod_map의 delta 사전 계산 저장 추가

| 데이터셋 | Shape | 용도 | 비고 |
|---|---|---|---|
| `fixed/xs_fuel` | `[22, 5, 5, 10]` | 거시단면적 (고정) | 전 시나리오 공통, LP당 1회 |
| `critical_xe` | `[576, 22, 5, 5]` | Xe-135 수밀도 | 시간 진행 (CRS 전용) |
| `critical_sm` | `[576, 22, 5, 5]` | Sm-149 수밀도 | 시간 진행 (CRS 전용) |
| `query_rod_map_3d` | `[576, 31, 22, 5, 5]` | 제어봉 3D 삽입 분율 | index 0 = CRS |
| `query_pload` | `[576]` | 목표 출력 수준 | 스칼라 시계열 |
| `result_power` | `[576, 31, 22, 5, 5]` | Q_abs_power_sharing | Critical(idx 0) + Branch |
| `result_tcool` | `[576, 31, 22, 5, 5]` | 냉각재 온도 | Critical + Branch |
| `result_tfuel` | `[576, 31, 22, 5, 5]` | 연료 온도 | Critical + Branch |
| `result_rhocool` | `[576, 31, 22, 5, 5]` | 냉각재 밀도 | Critical + Branch |
| `result_keff` | `[576, 31]` | keff 임계성 | 스칼라 |
| `result_ao` | `[576, 31]` | 축방향 출력 편차 | 검증용 |

**설계 선택: flatten([N_node]) vs 3D([22,5,5])**

기존 계획은 flatten으로 저장하고 학습 시 reshape하는 방식이었습니다. 2-Stage 구조에서는 공간 인코더가 3D 입력을 받으므로, 처음부터 3D shape로 저장하면 Dataloader에서 reshape 비용을 절약합니다. 다만, 용량은 동일하므로 둘 다 가능합니다. 어느 방식이든 Dataloader에서 최종 텐서를 조립하는 로직만 달라집니다.

### 5.3 용량 추정

| 항목 | Shape | 1 시나리오 (float32) | 20 시나리오 |
|---|---|---|---|
| Xe + Sm | `[576, 22, 5, 5] x 2` | 2.4 MB | 48 MB |
| query_rod_map_3d | `[576, 31, 22, 5, 5]` | 39.3 MB | 786 MB |
| result (4종) | `[576, 31, 22, 5, 5] x 4` | 157.2 MB | 3.1 GB |
| keff + ao | `[576, 31] x 2` | 0.14 MB | 2.8 MB |
| **합계** | | **~199 MB** | **~4.0 GB** |
| gzip 압축 후 | | **~65 MB** | **~1.3 GB** |

---

## 6. Dataloader 조립 로직 (tf.data.Dataset)

### 6.1 학습 시 배치 구성

Mamba는 시퀀스 전체를 한 번에 스캔하므로, **1 배치 = 1 시나리오(576 스텝)**입니다.

```python
def load_scenario(scenario_key):
    # HDF5에서 로드
    xs_fuel  = h5['fixed/xs_fuel']              # [22,5,5,10]
    xe       = h5[f'{key}/critical_xe']          # [576,22,5,5]
    sm       = h5[f'{key}/critical_sm']          # [576,22,5,5]
    rod_maps = h5[f'{key}/query_rod_map_3d']     # [576,31,22,5,5]
    pload    = h5[f'{key}/query_pload']          # [576]
    power    = h5[f'{key}/result_power']         # [576,31,22,5,5]
    tcool    = h5[f'{key}/result_tcool']         # [576,31,22,5,5]
    ...
    # 공간 인코더 입력 조립 (매 스텝 t)
    for t in range(576):
        power_prev_t = power[t-1, 0] if t > 0 else zeros   # critical 이전 출력
        input_3d = concat([
            xs_fuel,                          # [22,5,5,10]
            xe[t,:,:,:,None],                 # [22,5,5, 1]
            sm[t,:,:,:,None],                 # [22,5,5, 1]
            tcool[t-1,0,:,:,:,None],          # [22,5,5, 1]
            rhocool[t-1,0,:,:,:,None],        # [22,5,5, 1]
            power_prev_t[:,:,:,None],         # [22,5,5, 1]
            rod_maps[t,0,:,:,:,None],         # [22,5,5, 1]  CRS rod
        ], axis=-1)                           # --> [22,5,5,16]
```

### 6.2 Branch Delta 계산

```python
# 사전 계산 (전처리 시 또는 Dataloader 내에서)
delta_rod_maps = rod_maps[:, :, :, :, :] - rod_maps[:, 0:1, :, :, :]
# shape: [576, 31, 22, 5, 5]
# delta_rod_maps[:, 0, :, :, :] == 0  (CRS 자기 자신과의 차이)
# 이를 HDF5에 사전 저장하면 학습 시 계산 비용 절약
# dataset에 'delta_rod_map_3d' [576, 31, 22, 5, 5] 추가 저장 권장
```

### 6.3 정규화 적용 시점

HDF5에는 **raw 값을 저장**하고, Dataloader에서 런타임 정규화를 적용합니다. `norm_stats`(mean, std 등)는 학습 시나리오 14개에서만 계산하여 metadata에 저장합니다.

| 채널 | 정규화 방법 | 이유 |
|---|---|---|
| XS (nu_sigma_f 등) | log10 + Z-score | E-02~E-05 스케일, 로그 변환 필요 |
| Xe, Sm | log10 + Z-score | E-09~E-07 스케일 |
| T_cool | Z-score | 303~330 °C, 선형 스케일 |
| rho_cool | Z-score | 0.66~0.72 g/cc, 선형 스케일 |
| rod_map | MinMax (0~1) | 이미 0~1 범위 |
| pload | 그대로 (0~1) | 이미 정규화됨 |
| power_sharing | log10 + Z-score | 상대 비율 |
| keff | (k−1.0) × 1e5 [pcm] | 임계성 편차 |

---

## 7. 전처리 파이프라인 (Raw Data → HDF5)

### 7.1 원자료 구조

```
시나리오 수 : 20개
총 시간     : 48시간
간격        : 5분
스텝        : 576
봉 위치     : 31개/스텝 (CRS 1 + Branch 30)
노드        : 22(축) x 11 x 11(반경) --> 1/4 대칭 크롭 --> 22 x 5 x 5
```

### 7.2 파이프라인 흐름 (시나리오 1개)

| 단계 | 모듈 | 입력 | 출력 | 횟수 |
|---|---|---|---|---|
| 1. 파일 스캔 | `path_resolver` | `rawdata_dir, config` | 파일 경로 목록 | 1회/시나리오 |
| 2a. CRS 파싱 | `mas_out_parser` `mas_inp_parser` `mas_sum_parser` | MAS_OUT, MAS_INP, MAS_SUM 파일 | P3D, Xe, Sm, T_cool, T_fuel, rho_cool, pload, keff, AO | 576회 |
| 2b. Branch 파싱 | 동일 파서 | Branch 파일 × 30 | P3D, T_cool, T_fuel, rho_cool, keff, AO | 576 × 30회 |
| 2c. 공간 변환 | `spatial_transform` | (11,11) 격자 | (5,5) 1/4 대칭 | 파싱 즉시 |
| 2d. Rod Map 생성 | `rod_map_builder` | bank% + loading pattern + pload + P_design | `[22,5,5]` rod_map | 576 × 31회 |
| 2e. Q_abs 변환 | `step_assembler` | power_sharing | Q_abs `[22,5,5]` | 576 × 31회 |
| 3. 시계열 스택 | `dataset_builder` | 576개 스텝 데이터 | `[576, ...]` 배열 | 1회 |
| 4. HDF5 기록 | `dataset_builder` | stacked 배열 | gzip 압축 HDF5 | 1회 |
| 5. 정규화 통계 | `normalizer` | 학습 14개 시나리오 | `norm_stats` dict | 전체 후 1회 |

### 7.3 핵심 전처리 규칙

1. **블록 선택**: CRS 파일은 마지막 블록(`_2`=출력), Branch 파일은 `_1` 사용
2. **Xe/Sm**: CRS에서만 추출. Branch에서는 추출하지 않음 (Frozen Xenon)
3. **critical index**: 31-way 배열의 index 0 = CRS (offset=0). `r0_00p`는 무시하고 CRS 값 사용
4. **(11,11) → (5,5)**: 파싱 직후 크롭 적용. 이후 모든 연산은 크롭된 shape 기준
5. **Q_abs**: `= power_sharing × (pload × P_design) / N_fuel`
6. **Rod Map**: `bank%(R1,R2,R3) + core loading pattern → 3D [22,5,5]` 삽입 분율
7. **정규화**: HDF5에 raw 값 저장, `norm_stats`는 별도 저장, 적용은 런타임(Dataloader)

---

## 8. 학습 및 추론 흐름

### 8.1 학습 (Teacher Forcing)

```
1 epoch = 14 train 시나리오 순회
1 batch = 1 시나리오 = 576 스텝
```

```python
for scenario in train_scenarios:
    # Stage 1: 공간 인코딩 (576회, 병렬화 가능)
    z_seq = [SpatialEncoder(input_3d(t)) for t in range(576)]
    z_seq = stack(z_seq)                        # [576, D_model]

    # Stage 2: Mamba 시간 스캔
    h_seq = Mamba(z_seq, pload_seq)             # [576, D_model]

    # Stage 3: 예측
    for t in range(576):
        # Critical 예측
        crit_pred     = CriticalHead(h_seq[t])
        loss_crit    += loss_fn(crit_pred, targets_crit[t])

        # Branch 예측 (31-way)
        for p in range(31):
            dz        = RodDeltaEncoder(delta_rod[t,p])
            br_pred   = BranchHead(h_seq[t], dz, pload[t])
            loss_branch += loss_fn(br_pred, targets_branch[t,p])
```

### 8.2 추론 (자기회귀)

```python
# 워밍업: 과거 N_warmup 스텝의 실측 입력으로 h(t) 초기화
for t in range(N_warmup):
    z = SpatialEncoder(ground_truth_input(t))
    h = Mamba.step(z, pload[t])                 # 1스텝씩 recurrent

# 자기회귀 예측
for t in range(N_warmup, 576):
    # Critical 예측
    crit_pred = CriticalHead(h)                 # Xe, Sm, power, T_cool, ...

    # Branch 예측 (31-way)
    for p in range(31):
        dz      = RodDeltaEncoder(delta_rod[t,p])
        br_pred = BranchHead(h, dz, pload[t])

    # 자기회귀: 예측값으로 다음 입력 구성
    next_input = build_input(
        xs_fuel,
        crit_pred.Xe,    crit_pred.Sm,
        crit_pred.T_cool, crit_pred.rho_cool,
        crit_pred.power, rod_map_crs[t+1]
    )
    z = SpatialEncoder(next_input)
    h = Mamba.step(z, pload[t+1])
```

> **N_warmup 권장**: 100~120 스텝 (Xe-135 시정수 ~9h = ~108 스텝). 워밍업 구간에서는 실측 데이터를 사용하므로 오차 누적이 없습니다. 워밍업 후부터 예측값을 재입력하는 자기회귀가 시작됩니다.

---

## 9. 구현 체크리스트 (코딩 에이전트용)

### Phase A: 전처리 파이프라인

| # | 작업 | 상태 | 비고 |
|---|---|---|---|
| A1 | `mas_out_parser`: `_parse_power_sharing_3d()` | stub | `$P3D_N` 블록 |
| A2 | `mas_out_parser`: `_parse_xe_dist_3d()` | stub | `$XESM3D_N` 첫줄=Xe |
| A3 | `mas_out_parser`: `_parse_sm_dist_3d()` | stub | 둘째줄=Sm |
| A4 | `mas_out_parser`: `_parse_fb3d()` 신규 | 누락 | `$FB3D_N` → T_cool, T_fuel, rho_cool |
| A5 | 블록 선택 로직 (CRS=`_2`, Branch=`_1`) | 미구현 | 파서 docstring에 명시 |
| A6 | `mas_inp_parser`: `_parse_power_load()` | stub | |
| A7 | `mas_inp_parser`: `_parse_rod_positions()` | stub | bank% 추출 |
| A8 | `mas_sum_parser`: AO 추출 | 누락 | MAS_SUM에서 AO 파싱 |
| B1 | `spatial_transform`: (11,11)→(5,5) 크롭 | 신규 | 1/4 대칭 |
| B5 | `rod_map_builder`: bank% → 3D `[22,5,5]` | 신규 | core loading pattern 필요 |
| C2 | `dataset_builder`: HDF5 스키마 재설계 | 재설계 | 3D shape 저장 |
| C6 | `delta_rod_map` 사전 계산 저장 | 신규 | `[576,31,22,5,5]` |
| D1 | `normalizer`: 채널별 통계 계산 | 신규 | 학습 시나리오 14개 기준 |

### Phase B: 모델 구현

| # | 작업 | 비고 |
|---|---|---|
| M1 | SpatialEncoder: CustomMaxViT3D에서 `final_conv` → GlobalPool + Linear, 입력 16ch | 수정 |
| M2 | RodDeltaEncoder: Conv3D ×2 + Pool + Linear 구현 | 경량, D_delta 출력 |
| M3 | Mamba Block 구현 (TF2 또는 JAX) | selective scan 커스텀 필요 |
| M4 | CriticalHead: `h(t)` → Xe, Sm, power, T_cool, T_fuel, rho_cool, keff | N_node 출력 |
| M5 | BranchHead: `h(t)` + `delta_z` + `pload` → power, T_cool, T_fuel, rho_cool, keff | N_node 출력 |
| M6 | pload conditioning: FiLM 또는 concat 방식 결정 | Mamba 입력에 적용 |
| M7 | Dataloader: HDF5 → `tf.data.Dataset` 변환 | 공간 인코더 입력 조립 |
| M8 | Loss 함수: 다중 타겟 가중 MSE + 물리 제약 | Q_abs 합산 제약 포함 |

---

## 10. 미결 사항 및 결정 필요 항목

| # | 항목 | 선택지 | 영향 |
|---|---|---|---|
| 1 | 3D 저장 vs Flatten 저장 | A: `[22,5,5]` 그대로 저장 / B: `[550]` flatten 저장 | Dataloader reshape 비용 vs HDF5 chunk 최적화 |
| 2 | 축방향 반사체 plane (K=1,22) | A: 22 plane 유지 / B: 20 plane (반사체 제외) | N_node: 550 vs 500 / 반사체 정보 활용 여부 |
| 3 | pload conditioning 방식 | A: FiLM (gamma, beta) / B: Concat + Linear | 모델 복잡도 vs 성능 |
| 4 | D_model 크기 | 256, 512, 1024 등 | 표현력 vs 메모리/속도 |
| 5 | Mamba 레이어 수 | 2, 4, 6 등 | 시간 동역학 학습 깊이 |
| 6 | TF2 vs PyTorch/JAX | A: TF2 커스텀 SSM / B: JAX/Flax (mamba_ssm) / C: PyTorch 전환 | 구현 난이도 결정적 영향 |
| 7 | 학습 전략 | A: End-to-end 동시 학습 / B: 2-stage (인코더 사전학습 → Mamba fine-tune) | 학습 안정성 vs 효율 |
| 8 | N_warmup 최적화 | 100, 120, 150 등 | Xe 시정수 ~108스텝 기준 |

이상의 설계서는 공간 인코더(CustomMaxViT3D)와 시간 모델(Mamba)의 역할 분담, 데이터 흐름, 전처리 파이프라인을 정의합니다. 코딩 에이전트는 본 문서의 HDF5 스키마, Dataloader 조립 로직, 그리고 구현 체크리스트를 참조하여 구현을 진행할 수 있습니다.

---

## 11. 국부적 공간 효과(Local Spatial Effect) 포착 한계 분석

> **본 섹션은 현재 아키텍처의 핵심 한계를 비판적으로 분석하고, 개선 방향을 제시하는 추가 분석 섹션입니다.**

### 11.1 문제의 본질: GlobalAveragePooling의 정보 병목

#### 핵심 질문

제어봉 삽입 이벤트는 그 효과가 **극히 국소적(local)**일 수 있습니다. 22×5×5 = 550개 노드로 구성된 3D 노심 격자에서, 특정 제어봉 한 개의 삽입은 해당 봉 주변의 수십 개 노드에만 즉각적인 영향을 미칩니다. 그러나 현재 아키텍처에서 SpatialEncoder의 최종 단계는 `GlobalAveragePooling3D`로서, 550개 노드의 3D 특성 맵(feature map) 전체를 **단 하나의 D_model 차원 벡터** `z(t)`로 압축합니다.

이 설계 선택이 Mamba가 국부적 공간 섭동을 감지하는 능력에 미치는 영향을 정량적으로 분석합니다.

#### 정보 희석 메커니즘

GlobalAveragePooling3D의 수학적 정의는 다음과 같습니다:

```
GlobalAvgPool(F) = (1/N_node) × Σ_{k,i,j} F[k, i, j, :]
               = (1/550) × Σ_{k=1}^{22} Σ_{i=1}^{5} Σ_{j=1}^{5} F[k, i, j, :]
```

여기서 `F ∈ ℝ^{22×5×5×C}` 는 GlobalAvgPool 직전의 특성 맵이며, 출력 `z(t) ∈ ℝ^{D_model}` 은 공간 전체의 단순 산술 평균(Linear으로 투영 전)에서 출발합니다.

**정보 희석 계수 계산:**

```
영향 노드 수      : N_affected  ≈ 20  (제어봉 1개 삽입 시 영향권, 축방향 슬라이스 ~1~2개)
전체 노드 수      : N_total     = 550
희석 계수         : α = N_total / N_affected = 550 / 20 ≈ 27.5

국소 출력 변화    : ΔP_local    = -30%   (영향 노드에서의 출력 감소)
평균 출력 변화    : ΔP_avg      = ΔP_local × (N_affected / N_total)
                               = -30% × (20/550)
                               ≈ -1.09%
```

즉, **국소 출력이 30% 감소하더라도 GlobalAvgPool을 거친 후 Mamba가 "보는" 신호는 고작 ~1.1% 변화에 불과**합니다.

#### 구체적 수치 예시

R2 제어봉을 예로 들겠습니다. R2 봉은 1/4 대칭 격자에서 (i=2, j=2) 반경 위치에 배치되어 있다고 가정합니다.

```
[시나리오 가정]
- R2 봉이 t=200 스텝에서 50% 추가 삽입됨
- 영향 노드: 축방향 10~16 (7개 plane), 반경 (2,2) 및 인접 2개 = 약 21노드
- 해당 노드의 출력 sharing이 평균 25% 감소

[GlobalAvgPool 전/후 비교]
단계                            신호 크기
--------------------------------------------
영향 노드의 feature 활성화 변화  크고 명확 (ΔF ≈ 25%)
GlobalAvgPool 후 z(t) 변화      ΔP_avg = 25% × (21/550) ≈ 0.95%
Linear 투영 후 z(t) 성분 변화   D_model 차원에 분산되어 더욱 희석
Mamba h(t) 갱신 강도            매우 약함 (delta(t)도 작게 계산됨)
```

#### Xe 과도현상과의 근본적 차이

이 문제는 Xe 과도현상과 비교할 때 그 심각성이 더욱 부각됩니다:

| 현상 | 영향 노드 수 | GlobalAvgPool 신호 강도 | Mamba 감지 용이성 |
|---|---|---|---|
| Xe 과도 (pload 급변) | ~550개 (전체) | 강함 (~5~20%) | 높음 |
| 봉 이동 (CRS 대폭 삽입) | ~50~100개 | 보통 (~2~5%) | 중간 |
| 봉 미세 조정 (부분 삽입) | ~10~30개 | **약함 (~0.3~1%)** | **낮음** |
| 단일 Branch 봉 차이 | ~5~20개 | **매우 약함 (~0.1~0.5%)** | **매우 낮음** |

#### 4.3절 정합성 표의 과대평가 문제

4절의 정합성 표에서 "봉 이동 이벤트 → B(t),C(t) 변화 → 적합"이라고 명시된 부분은 **과도하게 낙관적**입니다. 정확히 말하면:

- **노심 전체 keff 변화** (봉 삽입으로 인한 반응도 변화): GlobalAvgPool에서 비교적 보존됨 → "적합"
- **국부 power tilt 및 핫스팟 위치 변화**: GlobalAvgPool에서 대폭 희석 → **"조건부 적합" 또는 "우려"**

이 두 가지를 명확히 구분하지 않으면 모델 성능에 대한 잘못된 기대가 형성됩니다.

---

### 11.2 현재 구조에서의 완화 요소

비관적 분석만으로는 충분하지 않습니다. GlobalAvgPool의 희석 문제를 부분적으로 완화하는 요소들도 존재합니다.

#### 완화 요소 1: CustomMaxViT3D는 단순 공간 평균이 아님

GlobalAvgPool이 적용되는 것은 MaxViT3D의 **최종 출력 특성 맵** `F ∈ ℝ^{22×5×5×C_feat}`에 대해서입니다. 이 특성 맵이 이미 풍부한 공간 처리를 거쳤다는 점이 중요합니다.

```
입력 [22,5,5,16]
    ↓ BoundaryPad + SuperpositioningNeighborCells  → 공간 컨텍스트 확장
    ↓ Stem3D (Conv3D + BN + GELU)                 → 초기 특성 추출
    ↓ MBConv3D (depthwise separable)               → 국부 패턴 (수용 영역 확장)
    ↓ BlockAttention3D (local attention)           → 인접 노드 간 상호작용
    ↓ GridAttention3D (global sparse)              → 원거리 공간 상관 학습
    ↓ AxialSE (축방향 Squeeze-Excitation)          → 축방향 패턴 강조
    ↓ FFN_3D                                       → 비선형 변환
    ↓ F[22, 5, 5, C_feat]                          ← GlobalAvgPool 전 상태
    ↓ GlobalAveragePooling3D
    ↓ z(t) [D_model]
```

**BlockAttention3D와 GridAttention3D의 효과:**

`BlockAttention3D`는 국부 수용 영역 내의 노드들 간 어텐션을 계산하므로, 제어봉 삽입 노드의 활성화 변화가 **인접 노드들의 특성 맵에도 전파**됩니다. 이론적으로는 단일 노드의 이벤트가 주변 3×3×3 = 27개 노드로 퍼져, GlobalAvgPool에서의 실질적 영향 범위가 증가합니다.

```
[어텐션 전파 효과 추정]
원시 영향 노드: 20개
BlockAttention3D 전파 후 영향 노드: ~20 × (1 + attn_spread)
  attn_spread ≈ 0.5~2.0 (학습된 어텐션 패턴에 따라 다름)
실질 희석 계수: 550 / (20 × 1.5~3.0) ≈ 9~18

즉, 원시 희석(27.5배) → 어텐션 보완 후 희석(9~18배) 가능
```

그러나 이것도 여전히 상당한 희석입니다.

#### 완화 요소 2: D_model 차원의 역할 분담

`z(t)`는 스칼라가 아니라 `D_model`(예: 512 또는 1024) 차원 벡터입니다. 충분히 학습이 이루어지면, 각 차원이 다른 공간 패턴을 전담하도록 특화될 수 있습니다.

```
[가설적 차원 전담 분포 (D_model=512 예시)]
dim  0~127:  전역적 노심 상태 (전체 출력 수준, 평균 온도)
dim 128~255: 축방향 출력 분포 패턴 (AO, 상/하부 출력 비율)
dim 256~383: 반경 방향 출력 분포 (내부/외부 링 비율)
dim 384~447: R1 봉 영역 효과 (국부 신호)
dim 448~511: R2/R3 봉 영역 효과 (국부 신호)
```

이 가설이 현실화되려면:
1. D_model이 충분히 커야 함 (최소 256 이상, 권장 512~1024)
2. 학습 데이터의 봉 이동 다양성이 충분해야 함 (20개 시나리오가 충분한지 검토 필요)
3. Loss 함수가 국부 정확도를 충분히 패널라이즈해야 함

#### 완화 요소 3: RodDeltaEncoder의 국부 정보 보완 (Branch 한정)

Branch 예측 경로에서는 `RodDeltaEncoder`가 `delta_rod [22,5,5,1]`를 직접 처리합니다:

```python
delta_z(t,p) = RodDeltaEncoder(delta_rod(t,p))
             = RodDeltaEncoder(rod_branch(t,p) - rod_crs(t))
```

`delta_rod`는 GlobalAvgPool을 **거치지 않습니다**. 즉:
- 봉 삽입 노드 `(k,i,j)`에서 `delta_rod[k,i,j] ≠ 0`이고, 그 정확한 공간 위치가 `delta_z`에 인코딩됩니다.
- BranchHead는 `concat(h(t), delta_z(t,p))` 를 입력으로 받으므로, 국부 봉 위치 정보가 **직접** 예측에 활용됩니다.

이것은 Branch 예측에 있어서 GlobalAvgPool 한계를 실질적으로 완화하는 설계입니다.

```
[Branch 예측 정보 흐름]
h(t)      [D_model]:  시간 이력 + 전역 공간 요약 (GlobalAvgPool 경유, 희석 있음)
delta_z(t,p) [D_delta]:  봉 섭동의 정확한 공간 위치 (GlobalAvgPool 미경유, 희석 없음)
→ BranchHead가 두 정보를 결합 → 국부 효과 예측 가능

단, 이 보완은 Branch 예측에만 적용됨. CRS Critical 예측은 h(t)만 사용.
```

---

### 11.3 그럼에도 남는 근본적 한계

완화 요소들을 고려하더라도, 현재 구조에는 원리적으로 해결되지 않는 한계들이 있습니다.

#### 한계 1: CRS Critical 예측에는 보완 경로 없음

이것이 가장 심각한 한계입니다.

```
[CRS Critical 예측 정보 흐름]
input_crs(t) [22,5,5,16]
    → SpatialEncoder
    → GlobalAveragePooling3D   ←── 여기서 국부 정보 희석
    → z(t) [D_model]
    → Mamba
    → h(t) [D_model]
    → CriticalHead
    → Xe(t+1), Sm(t+1), power[22,5,5], T_cool[22,5,5], keff
```

CRS 예측에서 `rod_map_crs(t)`는 `input_crs(t)`의 일부로 SpatialEncoder에 입력되지만, `GlobalAveragePooling3D`를 거치면서 봉의 정확한 공간 위치 정보가 희석됩니다.

**특히 위험한 시나리오:**

```
시나리오: CRS 봉이 t=250에서 60%→30% 부분 인출 (상부 영역에서)

물리적 실제:
  - 상부 축방향(k=16~22) 출력 증가 약 +15%
  - 하부 축방향(k=1~8)  변화 거의 없음
  - AO(축방향 출력 편차)가 크게 변화 → 핵안전 제한치 관련

GlobalAvgPool 후 z(t) 변화:
  - 전체 평균 출력 변화: +15% × (7 plane / 22 plane) ≈ +4.8%
  - AO 패턴의 공간 구조는 희석 → z(t)에서 "전체 출력 증가" 신호만 남음
  - CriticalHead가 AO 정확히 재현하려면 D_model의 특화된 차원에 의존해야 함
```

#### 한계 2: 자기회귀 추론 시 오차 누적의 폭발성

Teacher Forcing 학습에서는 모든 `t`에서 실측 입력을 사용하므로 오차 누적이 없습니다. 그러나 자기회귀 추론에서는 다음 구조가 만들어집니다:

```
t=200: CRS 봉 이동 이벤트 발생
  → z(200) 에서 국부 신호 희석 → h(200) 에서 불완전 반영
  → CriticalHead가 power_3d(201) 예측 시 실제보다 더 균일하게 예측
  → 예측된 power_3d(201)로 input_crs(201) 구성
  → SpatialEncoder(input_crs(201)) 에서 z(201) 계산
  → 이 z(201)도 실제 국부 패턴이 아닌 예측된 균일 패턴 기반
  → 오차가 t=202, 203, ... 으로 누적

[오차 누적 모식도]
t=200 → 국부 오차 δ_local ≈ 10%  (AO 예측 오차)
t=210 → δ_local × growth_rate^10 ← 안정적이면 감소, 불안정이면 증가
t=300 → Xe 피드백과 결합 시 비선형 폭발 가능
t=576 → 최악의 경우 수십 % 오차
```

이 오차 누적 문제는 `N_warmup`을 아무리 길게 설정해도 완전히 해결되지 않습니다. Xe 시정수(~9시간 = 108 스텝)보다 긴 워밍업을 제공해도, 워밍업 종료 후 첫 번째 봉 이동 이벤트에서 국부 오차가 발생하면 이후 전체 예측에 전파됩니다.

#### 한계 3: Power Peaking Factor 정확도 — 핵안전의 핵심

원자로 안전해석에서 가장 중요한 지표 중 하나는 **Power Peaking Factor(PPF)** 또는 **핫채널계수(Hot Channel Factor)**입니다:

```
PPF(t) = max_{k,i,j}[ power[k,i,j,t] ] / avg[ power[:,:,:,t] ]
```

이 지표는 정의상 국부 최대값에 의존하므로, GlobalAvgPool 이후의 전역 평균 신호만으로는 정확히 예측할 수 없습니다.

**정보 이론적 하계(Lower Bound):**

```python
# GlobalAvgPool이 일어나는 순간 다음 정보가 소실됨
I_lost = H(power_local | z(t))   # z(t)가 주어졌을 때 local power의 불확실성

# 이 소실된 정보를 CriticalHead의 MLP로 복원해야 함
# MLP는 학습 데이터의 통계적 패턴에서 "전형적인 국부 분포"를 학습하지만,
# 학습 분포를 벗어난 특이한 국부 패턴은 원리적으로 재현 불가

# 예시:
z_A(t) ≈ z_B(t)  (전역 평균이 비슷한 두 상태)
但: power_local_A[top_zone] >> power_local_B[top_zone]  (국부는 완전히 다름)
→ CriticalHead 입장에서 A와 B를 구분할 수 없음
```

#### 한계 4: Teacher Forcing vs 자기회귀 간의 트레이닝-추론 불일치

학습 시 Teacher Forcing에서는:
- `input_crs(t)` = 실측 ground truth 사용
- `z(t)` = 실제 노심 상태에서 추출된 정보

추론 시 자기회귀에서는:
- `input_crs(t)` = 모델 예측값 사용
- `z(t)` = 예측된(불완전한) 노심 상태에서 추출된 정보

이 **Train-Inference Mismatch**(또는 Exposure Bias)는 시퀀스 모델의 일반적 문제이지만, 국부 공간 효과 맥락에서 특히 심화됩니다:

- 학습 시에는 `power_prev(t-1)` 이 실측값이므로 국부 패턴이 정확히 인코딩됨
- 추론 시에는 `power_prev(t-1)` 이 예측값이고, 국부 패턴이 이미 희석된 상태
- 이 희석이 다음 스텝의 `input_crs(t)`에서도 희석된 `z(t)`를 만들어 복리로 누적

**Scheduled Sampling** 등의 기법으로 완화 가능하지만, 현재 설계에 포함되어 있지 않습니다.

---

### 11.4 대안 아키텍처 제안

GlobalAvgPool의 정보 병목을 해결하기 위한 4가지 구체적 대안을 제시합니다.

---

#### Option A: 공간 토큰 시퀀스 (Spatial Token Sequence)

**핵심 아이디어:** GlobalAvgPool → 단일 z(t) 대신, MaxViT3D 출력에서 **K개의 공간 토큰**을 유지하고, Mamba에 시공간 결합 시퀀스를 입력합니다.

```
[현재 구조]
각 t마다: [22,5,5,C] → GlobalAvgPool → z(t) [D_model]
Mamba 입력: z(1), z(2), ..., z(576)     길이 = 576

[Option A 구조]
각 t마다: [22,5,5,C] → SpatialTokenizer → z_1(t), z_2(t), ..., z_K(t)  각 [D_model]
Mamba 입력: z_1(1),...,z_K(1), z_1(2),...,z_K(2), ..., z_K(576)
            길이 = K × 576
```

**K 선택 옵션:**

| K 값 | 의미 | 총 시퀀스 길이 | 메모리 부담 |
|---|---|---|---|
| K=22 | 축방향 1 plane = 1 토큰 (5×5 radial avgpool) | 22 × 576 = 12,672 | 22배 증가 |
| K=5 | 축방향 블록 5개 (4~5 plane 묶음) | 5 × 576 = 2,880 | 5배 증가 |
| K=25 | 반경 방향 5×5 위치 각 1 토큰 (22 axial avgpool) | 25 × 576 = 14,400 | 25배 증가 |
| K=110 | 축방향 2 plane + 반경 방향 5×5 = (11 axial groups) × (5×5) / 25 | 110 × 576 = 63,360 | 110배 증가 |

**Mamba의 선형 복잡도 특성상** 시퀀스 길이 증가는 Transformer처럼 O(L²)이 아닌 O(L)이므로, K=22 정도(총 12,672 토큰)는 실용적입니다.

```python
# Option A 의사코드 (K=22, 축방향 평균 토큰)
class SpatialTokenizerAxial(tf.keras.layers.Layer):
    def call(self, F):
        # F: [B, 22, 5, 5, C_feat]
        # 각 축방향 plane을 1개 토큰으로 (radial avgpool)
        tokens = tf.reduce_mean(F, axis=[2, 3])  # [B, 22, C_feat]
        tokens = self.linear_proj(tokens)         # [B, 22, D_model]
        return tokens  # K=22 공간 토큰

# 시퀀스 구성
# z_tokens(t): [B, K, D_model] for each t
# all_tokens: concat over t → [B, K×T, D_model] = [B, 22×576, D_model]
all_tokens = tf.reshape(
    tf.stack([z_tokens_t for t in range(576)], axis=2),  # [B, K, T, D_model]
    [B, 22*576, D_model]
)

# Mamba 처리
h_all = MambaBlocks(all_tokens)  # [B, K×T, D_model]
h_per_step = tf.reshape(h_all, [B, 22, 576, D_model])
# h(t) = h_per_step[:, :, t, :]  →  [B, 22, D_model]  (축방향 분해된 시간 상태)
```

**장점:**
- 국부 공간 정보가 Mamba의 hidden state에 명시적으로 보존됨
- Vision Mamba(Vim)의 공간 패치 시퀀스 처리 방식과 동일한 철학
- Mamba 선형 복잡도 덕분에 긴 시퀀스에도 효율적

**단점:**
- Mamba 메모리 소비 K배 증가 (A6000 48GB × 3에서도 K≤22 수준 권장)
- CriticalHead, BranchHead 입력이 `[K × D_model]`으로 확장되어 아키텍처 전면 수정 필요
- 시퀀스 내 공간-시간 순서 정의 필요 (어떤 토큰 정렬 방식이 최적인지 불명확)

---

#### Option B: 다중 해상도 풀링 (Multi-Resolution Pooling)

**핵심 아이디어:** 단일 GlobalAvgPool 대신, **세 가지 해상도**의 풀링 결과를 계층적으로 결합합니다.

```
F [22, 5, 5, C_feat]
    ├── GlobalAvgPool3D          → z_global(t) [D/2]        전역 상태
    ├── AxialAvgPool(axis=2,3)  → z_axial(t)  [22, D/4]    축방향 profile 보존
    └── RegionalAvgPool          → z_regional(t) [4, D/4]   4 사분면 반경 구조

z_axial_flat   = Flatten(z_axial)   → [22 × D/4]
z_regional_flat = Flatten(z_regional) → [4 × D/4]

z(t) = concat([z_global, z_axial_flat, z_regional_flat])
     = [D/2 + 22×D/4 + 4×D/4]
     = [D/2 + 26×D/4]
     = D × (0.5 + 6.5)
     = 7 × D  (D/2 + D/4×26 = D×(0.5+6.5))
```

**구체적 차원 예시 (D_model=512):**

```
z_global(t)  : [256]             = 전체 출력 수준, keff, 평균 온도
z_axial(t)   : [22 × 128] = [2816]  = 각 축방향 위치의 공간 요약
z_regional(t): [4 × 128]  = [512]   = 4 사분면의 반경 방향 요약

z(t) 총 차원  : 256 + 2816 + 512 = 3584

Mamba 입력: [B, 576, 3584]
```

이 방식의 핵심은 `z_axial(t)`에서 **축방향 AO(Axial Offset)** 정보가 명시적으로 보존된다는 것입니다. 제어봉의 축방향 삽입 위치가 `z_axial`의 해당 plane 차원에 직접 반영됩니다.

```python
# Option B 의사코드
class MultiResPooling(tf.keras.layers.Layer):
    def __init__(self, D_model):
        super().__init__()
        self.global_pool  = GlobalAveragePooling3D()
        self.proj_global  = Dense(D_model // 2)
        self.proj_axial   = Dense(D_model // 4)  # 각 축방향 plane에 적용
        self.proj_regional = Dense(D_model // 4) # 각 사분면에 적용

    def call(self, F):
        # F: [B, 22, 5, 5, C_feat]
        # 전역
        z_global = self.proj_global(self.global_pool(F))  # [B, D/2]

        # 축방향 (radial avgpool 후 각 plane 독립 처리)
        F_axial = tf.reduce_mean(F, axis=[2, 3])          # [B, 22, C_feat]
        z_axial = self.proj_axial(F_axial)                # [B, 22, D/4]
        z_axial_flat = tf.reshape(z_axial, [B, 22 * (D_model//4)])

        # 반경 사분면 (4 quadrants: i∈{0..2}, j∈{0..2} 구조에서 사분면 분리)
        # 5x5 격자의 4 사분면: [0:3, 0:3], [0:3, 2:5], [2:5, 0:3], [2:5, 2:5]
        q1 = tf.reduce_mean(F[:, :, 0:3, 0:3, :], axis=[1,2,3])  # [B, C_feat]
        q2 = tf.reduce_mean(F[:, :, 0:3, 2:5, :], axis=[1,2,3])
        q3 = tf.reduce_mean(F[:, :, 2:5, 0:3, :], axis=[1,2,3])
        q4 = tf.reduce_mean(F[:, :, 2:5, 2:5, :], axis=[1,2,3])
        z_regions = tf.stack([q1,q2,q3,q4], axis=1)              # [B, 4, C_feat]
        z_regional = self.proj_regional(z_regions)               # [B, 4, D/4]
        z_regional_flat = tf.reshape(z_regional, [B, 4 * (D_model//4)])

        return tf.concat([z_global, z_axial_flat, z_regional_flat], axis=-1)
```

**장점:**
- 현재 Mamba 아키텍처를 거의 그대로 유지 (시퀀스 길이 576 유지)
- `z(t)`의 차원이 커지지만 구조 변경은 minimal
- AO, 반경 방향 분포 등 핵심 물리 지표가 명시적으로 보존됨

**단점:**
- z(t) 차원이 7배 증가 → Mamba 내부 파라미터 수 증가
- A6000 메모리 내에서 가능하지만, D_model 설정에 주의 필요

---

#### Option C: 공간 기억 보조 경로 (Spatial Memory Bypass)

**핵심 아이디어:** GlobalAvgPool → z(t) → Mamba 흐름은 유지하되, MaxViT3D의 **풀 특성 맵** `F(t)`를 별도 경로로 저장하고, 예측 헤드에서 직접 참조합니다.

```
[현재 구조]
F(t) [22,5,5,C] → GlobalAvgPool → z(t) → Mamba → h(t) → CriticalHead → 예측

[Option C 구조]
F(t) [22,5,5,C] → GlobalAvgPool → z(t) → Mamba → h(t) ──────────────────┐
                │                                                           ↓
                └──── 직접 저장 ──────────────────── F(t) ──→ CriticalHead(h(t), F(t))
```

**CriticalHead (Option C 버전):**

```python
class CriticalHeadWithSpatialMemory(tf.keras.layers.Layer):
    def __init__(self, D_model, C_feat):
        super().__init__()
        # Cross-attention: h(t)가 query, F(t)가 key/value
        self.cross_attn = CrossAttention3D(D_model, C_feat)
        self.mlp = MLP([D_model * 2, D_model, N_node])

    def call(self, h_t, F_t):
        # h_t: [B, D_model]  (Mamba 시간 상태)
        # F_t: [B, 22, 5, 5, C_feat]  (공간 특성 맵)

        # h_t가 F_t에서 관련 공간 정보를 추출
        spatial_context = self.cross_attn(
            query=h_t[:, None, :],            # [B, 1, D_model]
            key_value=F_t                     # [B, 22*5*5, C_feat]
        )  # [B, 1, D_model]

        # 시간 상태 + 공간 맥락 결합
        combined = tf.concat([h_t, tf.squeeze(spatial_context, 1)], axis=-1)
        return self.mlp(combined)  # [B, N_node × n_outputs]
```

**장점:**
- Mamba 시퀀스 길이 유지 (가장 작은 구조 변경)
- CriticalHead가 필요시 공간 맵을 "참조"하므로 국부 정보 복원 가능
- Mamba는 시간 동역학만, Head는 공간 세부정보까지 담당하는 깔끔한 역할 분리

**단점:**
- `F(t)` 저장 필요: `[22,5,5,C_feat]` × 576 스텝 × 배치
  - 예: C_feat=256이면 `22×5×5×256×576 × 4byte ≈ 820 MB/시나리오` → GPU 메모리 부담
- Cross-attention 구현 복잡도 증가
- F(t)를 모든 스텝에서 메모리에 유지해야 하므로 gradient checkpointing 필요 가능성

**메모리 절감 방안:**

```python
# F(t) 전체를 저장하는 대신, 압축된 버전 저장
F_compressed(t) = AxialAttentionCompress(F(t))  # [22, 5, 5, C_feat] → [22, C_small]
# C_small = 32~64로 압축 시: 22×64×576×4byte ≈ 3.2 MB/시나리오 → 현실적
```

---

#### Option D: 현재 구조 유지 + 정밀 모니터링 (Pragmatic Approach)

**핵심 아이디어:** 1차 구현에서는 현재 GlobalAvgPool 구조를 유지하고, 국부 정확도를 집중적으로 모니터링하여 개선 필요성을 **데이터 기반으로 판단**합니다.

**근거:**

1. D_model=512~1024은 550개 노드의 패턴 정보를 담기에 충분히 큰 공간일 수 있습니다.
2. MaxViT3D의 어텐션 메커니즘이 이미 국부 이벤트를 특성 맵에 "증폭"시킬 수 있습니다.
3. 20개 시나리오 × 576 스텝 × 31 봉 위치 = 357,120개의 학습 샘플은 패턴 학습에 충분할 수 있습니다.

**필수 추가 검증 지표:**

```python
# 현재 loss_total에 추가해야 할 모니터링 지표
def compute_local_accuracy_metrics(y_pred_power, y_true_power):
    """
    y_pred_power, y_true_power: [B, 22, 5, 5]
    """
    # 1. Per-node RMSE
    node_rmse = tf.sqrt(tf.reduce_mean(
        tf.square(y_pred_power - y_true_power), axis=0
    ))  # [22, 5, 5]

    # 2. Power Peaking Factor RMSE
    ppf_pred = tf.reduce_max(y_pred_power, axis=[1,2,3]) / \
               tf.reduce_mean(y_pred_power, axis=[1,2,3])
    ppf_true = tf.reduce_max(y_true_power, axis=[1,2,3]) / \
               tf.reduce_mean(y_true_power, axis=[1,2,3])
    ppf_rmse = tf.sqrt(tf.reduce_mean(tf.square(ppf_pred - ppf_true)))

    # 3. Axial Offset RMSE
    power_upper = tf.reduce_mean(y_pred_power[:, 12:, :, :], axis=[1,2,3])
    power_lower = tf.reduce_mean(y_pred_power[:, :11, :, :], axis=[1,2,3])
    ao_pred = (power_upper - power_lower) / (power_upper + power_lower)

    power_upper_t = tf.reduce_mean(y_true_power[:, 12:, :, :], axis=[1,2,3])
    power_lower_t = tf.reduce_mean(y_true_power[:, :11, :, :], axis=[1,2,3])
    ao_true = (power_upper_t - power_lower_t) / (power_upper_t + power_lower_t)
    ao_rmse = tf.sqrt(tf.reduce_mean(tf.square(ao_pred - ao_true)))

    # 4. Hot spot error (최대 출력 노드 오차)
    hotspot_pred = tf.reduce_max(y_pred_power, axis=[1,2,3])
    hotspot_true = tf.reduce_max(y_true_power, axis=[1,2,3])
    hotspot_rmse = tf.sqrt(tf.reduce_mean(tf.square(hotspot_pred - hotspot_true)))

    return {
        'node_rmse_map': node_rmse,          # [22, 5, 5] - 공간 오차 분포
        'ppf_rmse': ppf_rmse,                # 스칼라
        'ao_rmse': ao_rmse,                  # 스칼라
        'hotspot_rmse': hotspot_rmse,        # 스칼라
        'max_node_rmse': tf.reduce_max(node_rmse),  # 가장 나쁜 노드
    }
```

**판단 기준 (제안):**

| 지표 | 허용 기준 | 초과 시 대응 |
|---|---|---|
| Per-node RMSE (평균) | < 1.5% (상대 오차) | Option A 또는 B 검토 |
| Per-node RMSE (최대) | < 5% (가장 나쁜 노드) | Option A 또는 B 검토 |
| PPF RMSE | < 2% | Option A, B, 또는 C 검토 |
| AO RMSE | < 0.02 (절대값) | Option B (축방향 토큰 보존) 우선 |
| Hotspot RMSE | < 3% | Option C (공간 기억 보조) 검토 |

---

### 11.5 권장사항

현재 개발 단계와 가용 자원을 고려한 단계적 권장사항입니다.

#### 1단계: 현재 구조로 1차 구현 진행 (즉시 실행)

```
권장: Option D (현재 구조 유지 + 모니터링)
이유:
  - 구현 복잡도 최소화 → 빠른 프로토타이핑
  - GlobalAvgPool이 실제로 얼마나 문제가 되는지 실험적으로 확인 필요
  - D_model=512 이상으로 설정하면 어느 정도 완화 가능성 있음
```

**필수 설정값:**

| 하이퍼파라미터 | 최솟값 | 권장값 | 이유 |
|---|---|---|---|
| D_model | 256 | 512~1024 | 550개 노드 패턴을 1벡터에 담으려면 차원이 넉넉해야 함 |
| N_mamba_layers | 2 | 4~6 | 국부 이벤트의 시간 동역학을 충분히 추적하기 위해 깊은 SSM 필요 |
| D_delta | 64 | 128 | Branch 봉 섭동의 공간 세부정보 인코딩 |

#### 2단계: 검증 후 피드백 기반 판단 (1차 학습 완료 후)

```
검증 프로토콜:
1. 전체 20 시나리오에 대해 자기회귀 추론 실행
2. 위 정의된 local_accuracy_metrics 계산
3. 판단 기준표와 비교

분기 A: 모든 지표 허용 기준 충족
  → 현재 구조 확정. 논문/보고서에 GlobalAvgPool 충분성 실험적 입증 기술

분기 B: AO RMSE 또는 per-node RMSE 기준 초과
  → Option B (다중 해상도 풀링) 먼저 시도
  → z_axial [22, D/4] 추가만으로도 AO 문제 대부분 해결 가능성

분기 C: PPF RMSE 또는 Hotspot RMSE 기준 초과
  → Option A (공간 토큰, K=22) 또는 Option C (공간 기억 보조) 적용
  → A: 근본적 해결이지만 아키텍처 전면 수정 필요
  → C: 현재 구조와 가장 호환적이나 메모리 관리 필요
```

#### 3단계: 장기 개선 방향

```
A6000 × 3 환경 기준 실현 가능한 최종 아키텍처:

[공간 인코더]
  CustomMaxViT3D → MultiResPooling (Option B)
  z(t) = [z_global [256], z_axial [22×128], z_regional [4×128]]
  총 차원: 3584

[시간 모델]
  Mamba-2 (SSD) × 4~6 layers
  입력: z(t) [3584] + pload FiLM conditioning
  D_model_internal: 256~512 (내부 SSM 상태)

[예측 헤드]
  CriticalHead: h(t) [D_model] → Xe, Sm, power, T_cool, keff
  BranchHead: h(t) + delta_z [D_delta] + pload_embed → power, T_cool, keff

[검증 지표]
  loss_total + per-node RMSE + PPF RMSE + AO RMSE + Hotspot RMSE
```

---

### 11.6 정합성 표 수정 (4.3절 보완)

4.3절의 정합성 표를 국부 공간 효과 관점에서 재평가하여 수정합니다.

| 물리 현상 | Mamba 반응 | 기존 정합성 평가 | **국부 효과 고려한 수정 평가** |
|---|---|---|---|
| Xe 과도현상 (pload 급변) | delta(t) 증가 → h(t) 급격 갱신 | 높음: Xe 시간 상수(~9h)의 비선형 과도 추적 | **영향 없음**: Xe는 노심 전체에 동시적으로 변화함. GlobalAvgPool이 전체 신호를 그대로 전달. 정합성 "높음" 유지 |
| 정상 운전 (pload 일정, Xe 평형) | delta(t) 감소 → h(t) 느리게 변화 | 적합: 평형 접근 시 이전 상태를 오래 기억 | **영향 없음**: 전체적으로 균일한 상태 변화이므로 GlobalAvgPool 희석 문제 없음. "적합" 유지 |
| 봉 이동 이벤트 (CRS 위치 변화) | B(t),C(t) 변화 | 적합: 봉 이동에 따른 중성자속 재분포를 h(t)에 반영 | **조건부 적합 (수정)**: 노심 전체 keff 변화 및 전역 반응도 변화는 GlobalAvgPool에서 포착 가능. 그러나 **국부 power tilt, AO 변화, 핫스팟 이동**은 GlobalAvgPool에서 희석 (희석 계수 ~9~28배). h(t)에 국부 공간 구조가 충분히 반영된다고 보장할 수 없음 |
| 출력 분포 비선형성 (power peaking) | 선형 recurrence → 비선형은 예측 헤드에서 | 보완 필요: 예측 헤드(MLP)의 비선형 층으로 대응 | **우려 가중 (수정)**: GlobalAvgPool 이후 z(t)에서 이미 국부 peak 정보가 약화된 상태에서, MLP가 peak 분포를 복원해야 함. 이는 원리적 정보 손실 → D_model 충분히 크게 설정하고 실험적 검증 필수 |
| **Branch 봉 위치 차이 (Branch 전용)** | **delta_rod → delta_z → BranchHead에 직접 전달** | *(기존 표에 미포함)* | **적합 (추가)**: RodDeltaEncoder가 GlobalAvgPool을 **거치지 않고** 봉 위치 차이를 직접 인코딩. Branch 예측에서 국부 효과 포착 능력은 현재 구조에서도 충분함. Branch 예측의 국부 정확도 > CRS Critical 예측의 국부 정확도 예상 |
| **축방향 출력 편차 (AO) 변화** | *(기존 표에 미포함)* | *(기존 표에 미포함)* | **우려 (추가)**: 봉 삽입 깊이 변화에 따른 AO 변화는 축방향 출력 분포의 국부적 변화. GlobalAvgPool에서 축방향 정보 희석. Option B(축방향 토큰 보존)로 개선 가능 |

**수정된 정합성 요약:**

```
Xe 과도현상   : ★★★★★  (완전 적합 - 영향 없음)
정상 운전     : ★★★★★  (완전 적합 - 영향 없음)
봉 이동(전역) : ★★★☆☆  (조건부 적합 - 전역 반응도는 O, 국부 분포는 △)
power peaking : ★★☆☆☆  (우려 - 원리적 정보 손실)
Branch 예측   : ★★★★☆  (대체로 적합 - delta_z 보완)
AO 변화       : ★★☆☆☆  (우려 - 축방향 희석)
```

**실용적 결론:**

현재 아키텍처는 **Xe 과도현상 추적** 및 **전역 반응도 변화**에는 우수한 성능을 기대할 수 있습니다. 그러나 **국부 출력 분포**, **Power Peaking Factor**, **축방향 출력 편차(AO)** 정확도는 GlobalAvgPool로 인한 원리적 한계가 있으며, 이는 실험적 검증을 통해서만 그 심각성을 정확히 평가할 수 있습니다. 1차 구현을 현재 구조로 진행하되, 위에서 정의한 국부 정확도 지표를 반드시 모니터링하고, 허용 기준 미달 시 Option A~C 중 적절한 대안으로 마이그레이션하는 단계적 전략을 권장합니다.

---

*문서 끝*

> **버전 이력**
>
> | 버전 | 날짜 | 변경 내용 |
> |---|---|---|
> | 1.0 | 2026-03-17 | 초안 작성 (섹션 1~10: 아키텍처 및 전처리 설계) |
> | 1.1 | 2026-03-17 | 섹션 11 추가: 국부적 공간 효과 포착 한계 분석 및 대안 아키텍처 제안 |
