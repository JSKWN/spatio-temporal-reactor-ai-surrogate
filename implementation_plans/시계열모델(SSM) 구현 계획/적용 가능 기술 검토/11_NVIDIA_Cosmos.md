# NVIDIA Cosmos — World Foundation Model

> **출처**: NVIDIA (2025). *Cosmos World Foundation Model Platform.* [arXiv:2501.03575](https://arxiv.org/abs/2501.03575). Cosmos Policy: [arXiv:2601.16163](https://arxiv.org/abs/2601.16163)
> **공개 수준**: 부분공개 (논문 + 일부 코드 공개)
> **우리 아키텍처 맥락**: DiT 기반 잠재 확산 모델. **Latent Frame Injection**과 **3D-factorized RoPE**가 우리 조건 주입 및 위치 인코딩과 관련

---

## 1. 시스템 개요

### 직관

Cosmos는 "물리 세계의 동작을 이해하는 기초 모델"이다. 2000만 시간의 비디오로 사전학습한 후, 특정 응용(자율주행, 로봇)에 LoRA로 미세조정한다. 핵심은 **비디오 토크나이저 + DiT(Diffusion Transformer)**의 조합.

### 아키텍처 (7B/14B 모델)

```
[비디오 프레임 시퀀스] → [Tokenizer: 8×8×8 시공간 압축, 잠재 차원 16]
    → [3D Patchification: (1,2,2) 비중첩 패치]
    → [DiT Blocks ×28~36]
        각 블록: Self-Attention → Cross-Attention(T5 텍스트) → FFN
        AdaLN으로 diffusion timestep 조건
        3D-factorized RoPE
    → [Detokenizer] → [예측 비디오]
```

| 구성 | 7B | 14B |
|------|-----|------|
| Layers | 28 | 36 |
| Model dim | 4,096 | 5,120 |
| FFN hidden | 16,384 | 20,480 |
| Attention heads | 32 | 40 |

---

## 2. 기능별 상세 분석

### 2.1 시간 모델링 — Latent Diffusion

Cosmos는 자기회귀가 아닌 **확산(Diffusion)** 기반 시간 예측:
- 미래 프레임을 노이즈에서 점진적으로 복원
- EDM(Elucidated Diffusion Model) 공식화
- Video2World 모드: 과거 비디오 + 텍스트 → 미래 비디오 예측

**우리 문제와의 비교**: 확산 모델은 **다중 모드(multimodal) 예측**에 강하지만, 우리 문제는 결정론적(동일 입력 → 동일 출력). 확산의 이점이 제한적이며, 자기회귀 Mamba가 더 적합.

### 2.2 상태 표현 — 8×8×8 시공간 토크나이저

**직관**: 비디오의 시간 8프레임 × 높이 8 × 너비 8 = 512배 압축. 잠재 차원 16.

우리 문제와의 대응:
- Cosmos: (T, H, W) → (T/8, H/8, W/8, 16) — 시공간 동시 압축
- 우리: (20, 6, 6, 21) → (20, 6, 6, 128) — 공간은 보존, 채널만 변환
- 차이: Cosmos는 공간 해상도를 줄이지만, 우리는 720 cell을 모두 보존 (cell별 물리량이 중요)

### 2.3 제어/조건 주입 — Latent Frame Injection

**이것이 Cosmos에서 가장 주목할 기법이다.**

Cosmos Policy(arXiv:2601.16163)에서 로봇 제어에 적용한 방법:

```
1. 비디오 VAE로 이미지 시퀀스를 잠재 프레임으로 변환
2. 빈 placeholder 잠재 프레임을 삽입 (행동/상태 채널용)
3. Placeholder를 덮어씀:
   - 로봇 관절 상태 q_t → [-1, +1] 정규화 → 잠재 프레임 형태로 복제
   - 행동 청크 (K × d_act) → 동일하게 잠재 프레임 형태로 복제
4. 덮어쓴 잠재 프레임이 비디오 잠재 프레임과 동일한 DiT에서 처리
5. 추론 시 행동 잠재 프레임을 디코딩하여 연속 행동으로 복원
```

**핵심**: **아키텍처 수정 없이** 제어 신호를 주입. 동일한 DiT가 비디오와 행동을 동시에 처리.

**우리 문제에서의 시사점**:
- 현재: p_load를 Mamba에 concat, 디코더에 AdaLN으로 주입
- Cosmos 패턴 적용 시: p_load를 "잠재 프레임"으로 변환하여 720 cell token과 함께 Mamba에 입력하는 방안. 그러나 p_load는 스칼라이므로 concat이 더 자연스러움
- **적용 가능한 경우**: 향후 다중 제어 입력(p_load + 붕소 농도 + 유량 등)이 추가될 때, 각각을 별도 "조건 토큰"으로 Mamba에 주입하는 방안

### 2.4 위치 인코딩 — 3D-Factorized RoPE

```
Feature 차원을 3등분: [temporal, height, width]
각 축에 독립적인 RoPE 주파수 적용
```

- FPS 인식: 프레임 레이트에 따라 시간 축 주파수 스케일링
- 학습 가능한 절대 위치 임베딩도 병행 (RoPE + Absolute PE)

**우리 인코더와의 직접 대응**: 우리 STRING은 **3D RoPE**의 일반화이다:
- Cosmos: (T, H, W) 3축 factorized RoPE
- 우리 STRING: (Z, Y, X) 3축 독립 주파수, 블록 대각 회전
- **동일한 원리**를 서로 다른 물리 공간에 적용. Cosmos가 이를 대규모(7B+)에서 검증

### 2.5 효율화 — AdaLN-LoRA

```
기존 AdaLN: scale, shift, gate = Linear(conditioning)    ← 전체 파라미터
AdaLN-LoRA: scale, shift, gate = LoRA(conditioning)      ← rank-256 분해
```

- 36% 파라미터 감소, 성능 유지
- 조건(diffusion timestep)에 대한 변조를 저랭크로 근사

**우리 문제에서의 시사점**: 디코더의 AdaLN-Zero에서 p_load 조건 변조를 LoRA로 압축 가능. 단, 우리 규모(수 M params)에서 36% 절감의 절대 효과는 제한적.

---

## 3. 우리 문제 적용성 평가

| 기법 | 적용 가능 | 근거 | 구현 난이도 |
|------|----------|------|-----------|
| 확산(Diffusion) 시간 예측 | X | 결정론적 시뮬레이터 → 확산의 다중 모드 이점 불필요 | - |
| **3D-factorized RoPE** | O (이미 채택) | STRING과 동일 원리. Cosmos가 대규모 검증 | - |
| **Latent Frame Injection** | △ | 다중 제어 입력 시 참고. 현재 p_load concat으로 충분 | 중 |
| AdaLN-LoRA | △ | 디코더 AdaLN 경량화 가능. 소규모에서 효과 제한 | 하 |
| 8×8×8 토크나이저 | X | 우리는 공간 해상도 보존 필수 (cell별 물리량) | - |
| T5 텍스트 cross-attention | X | 텍스트 조건 없음 | - |

---

## 4. 핵심 차용 후보

### 이미 반영
- **3D RoPE (STRING)**: Cosmos가 대규모(7B)에서 검증. 우리 인코더에서 동일 원리 사용 중

### 수정 후 적용 가능
- **조건 토큰 주입 (Latent Frame Injection 변형)**: 향후 다중 제어 입력 확장 시, 각 제어 변수를 독립 토큰으로 Mamba에 주입하는 패턴
- **AdaLN-LoRA**: 디코더 AdaLN 경량화. 절대 효과보다는 개념적 참고

### 부적합
- **확산 기반 시간 예측**: 우리 문제는 결정론적. 확산의 샘플링 비용(다단계 denoising)이 불필요한 오버헤드
- **시공간 동시 압축**: cell별 물리량 보존이 필수이므로 공간 해상도 압축 부적합

---

## 참고 문헌

| 모델 | 참조 |
|------|------|
| Cosmos | NVIDIA. 2025. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575) |
| Cosmos Policy | NVIDIA. 2026. [arXiv:2601.16163](https://arxiv.org/abs/2601.16163) |
