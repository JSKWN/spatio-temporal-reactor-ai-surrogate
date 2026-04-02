# L_diffusion Assembly-Level Implementation V&V

## 목적

L_diffusion physical loss의 집합체 수준 구현을 위한 검증 및 확인(Verification & Validation) 작업.

## 검증 항목

### 1. JNET0 방향 규약 명확화 ✅ 완료
- [x] JNET0 외향(outward) 방향 정의 확인 → positive = outward 확정
- [x] 전류 연속 조건 검증 (node_A.face + node_B.opposite_face = 0) → 모든 면 0.000e+00 완벽 성립
- [x] I/J 인덱스 매핑 확인 (I=East, J=South) → J 증가 = 남쪽(South) 확정
- [x] JNET0 = J_net/2 스케일 팩터 검증 → α=2 적용 시 median 0.000147% 잔차
- [x] ABS 정의 확정 → ABS = Σ_c + Σ_f (총 흡수 거시단면적)
- [x] JNET0 단위 확정 → per-unit-area [n/cm²/s]
- 검증 보고서: `v&v_kiro/01_jnet0_direction/verification_report.md`

### 2. CMFD 방향 규약 정합성
- [ ] CMFD 면 전류 공식 방향 확인 (J = D̃ × (φ_center - φ_neighbor) / h)
- [ ] 조화평균 확산계수 적용 확인
- [ ] 6면 일관된 외향 규약 적용 확인
- [ ] Mirror CMFD 대칭면 처리 확인

### 3. 집합체 수준 밸런스 계산
- [ ] 2군 밸런스 잔차 공식 검증
- [ ] 단면적 정의 확인 (Σ_r1, Σ_a2, D_g)
- [ ] 집합체 피치 21.608cm 적용 확인
- [ ] 면적/체적 계산 확인

### 4. 반사체 경계면 Albedo BC
- [ ] Marshak BC 공식 적용 확인
- [ ] 학습 가능 파라미터 12개 정의 확인
- [ ] R5 캘리브레이션 초기값 적용 확인
- [ ] alpha 범위 제약 [0.01, 2.0] 확인

### 5. 대칭면 Mirror CMFD
- [ ] 대칭 노드 매핑 확인 (qy=-1→qy=1, qx=-1→qx=1)
- [ ] Mirror CMFD 누설 계산 확인
- [ ] MAS_OUT flux_3d 대칭 조건 검증

### 6-10. 기타
- [ ] L_diffusion 손실 함수 구현 확인
- [ ] XS 일관성 (MAS_XSL vs MAS_NXS)
- [ ] 검증 테스트 스크립트 실행
- [ ] 문서 통합 및 정리
- [ ] Parser/Pretty Printer round-trip 검증

## 디렉토리 구조

```
v&v_kiro/
├── README.md                    # 이 파일
├── 01_jnet0_direction/          # JNET0 방향 규약 검증
├── 02_cmfd_consistency/         # CMFD 방향 규약 정합성
├── 03_assembly_balance/         # 집합체 밸런스 계산
├── 04_albedo_bc/                # Albedo 경계조건
├── 05_mirror_symmetry/          # Mirror 대칭 처리
└── reports/                     # 종합 보고서
```
