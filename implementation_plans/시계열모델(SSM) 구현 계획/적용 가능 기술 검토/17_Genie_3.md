# Genie 3 — 자기회귀 세계 생성

> **출처**: Google DeepMind (2025). *Genie 3.* [Blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
> **공개 수준**: 부분공개 (블로그 공개, 논문/코드 비공개)
> **우리 아키텍처 맥락**: 대규모 자기회귀 세계 생성 모델. 병렬 다중 분기 미지원 → 우리 Branch 구조와 대조

---

## 1. 시스템 개요

### 직관

Genie 3는 비디오 프레임을 자기회귀적으로 생성하여 "가상 세계"를 실시간으로 만든다. 사용자의 행동(키보드/마우스)에 반응하여 720p 24fps 비디오를 생성한다.

### 아키텍처 (추론)

```
[이전 프레임들 + 행동] → [Autoregressive Transformer]
    → [Latent Action Model: 8-dim 잠재 행동 추론]
    → [다음 프레임 토큰 생성]
    → [Video Decoder] → [720p 프레임]
```

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 — 자기회귀 Transformer

- 프레임 단위 자기회귀 생성
- 이전 프레임들의 컨텍스트를 참조하여 다음 프레임 예측
- 장기 일관성: 수백 프레임에 걸쳐 장면 일관성 유지

### 2.2 상태 표현 — Latent Action Model (LAM)

**Genie 시리즈의 고유 기법**: 행동 라벨 없이도 "잠재 행동"을 자동 추론:
```
두 연속 프레임 (f_t, f_{t+1}) → LAM Encoder → z_action ∈ R^8
```
- 프레임 간 차이에서 "무엇이 바뀌었는지"를 8차원으로 압축
- 명시적 행동 라벨(조이스틱 입력 등) 없이 학습 가능

**우리 문제와의 대조**: 우리는 제어 입력(rod_map, p_load)이 명시적으로 주어지므로 LAM이 불필요. 그러나 LAM의 "두 시점 간 변화를 잠재 벡터로 압축"하는 아이디어는 잠재 예측 loss(CDP)와 관련.

### 2.3 분기/다중경로 — 미지원

**Genie 3의 한계**: 병렬 다중 미래 분기를 **지원하지 않는다**. 순차적으로 단일 궤적만 생성.

우리 문제와의 핵심 차이:
| Genie 3 | 우리 |
|---|---|
| 단일 궤적 순차 생성 | **29개 Branch 병렬 평가** |
| 행동에 대한 GT 없음 | **Branch GT 존재** |

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 |
|------|----------|------|
| 자기회귀 프레임 생성 | O (이미 유사) | 우리 575-step 자기회귀와 동일 개념 |
| Latent Action Model | X | 명시적 제어 입력이 있으므로 불필요 |
| 720p 실시간 생성 | X | 우리는 오프라인 학습. 실시간 생성 불필요 |
| 장기 일관성 유지 | △ | 수백 프레임 일관성 기법은 575 step 안정성에 참고 가능. 구체 기법 미공개 |

### 핵심 차용
- **없음**: Genie 3는 비디오 생성에 특화되어 있으며, 우리 문제에 직접 적용 가능한 기법이 제한적. 자기회귀 세계모델의 규모 확장 가능성을 보여주는 사례로서의 의의.

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| Genie 3 | DeepMind. 2025. [Blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) |
| Genie 2 | DeepMind. 2024. [Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) |
