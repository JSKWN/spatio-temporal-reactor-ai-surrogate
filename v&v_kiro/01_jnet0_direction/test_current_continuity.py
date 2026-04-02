"""
JNET0 전류 연속 조건 검증 테스트

목적: 인접 노드 쌍에서 outward 전류의 합이 0인지 확인 (입자 보존 법칙)

검증 공식:
  (I,J).E + (I+1,J).W = 0  (동서 면)
  (I,J).S + (I,J+1).N = 0  (남북 면, J 증가 = 남쪽)
  (I,J).N + (I,J-1).S = 0  (남북 면, J 감소 = 북쪽)

기대 결과: 모든 내부 연료-연료 면에서 0.000e+00 (부동소수점 정밀도 내)
"""

import numpy as np
from typing import Dict, List, Tuple


def verify_current_continuity(nodes: List[Dict], group: str = 'g1') -> Dict:
    """
    전류 연속 조건 검증.
    
    Args:
        nodes: 노드 데이터 리스트 (각 노드는 i, j, k, jn_g1, js_g1, je_g1, jw_g1 등 포함)
        group: 'g1' 또는 'g2'
    
    Returns:
        검증 결과 딕셔너리 (EW_errors, NS_errors, 통계)
    """
    # 노드를 (i,j,k) 키로 인덱싱
    node_map = {(n['i'], n['j'], n['k']): n for n in nodes}
    
    ew_errors = []  # E/W 면 전류 연속 오차
    ns_errors = []  # N/S 면 전류 연속 오차
    
    for node in nodes:
        i, j, k = node['i'], node['j'], node['k']
        
        # E/W 면 검증: (I,J).E + (I+1,J).W = 0
        neighbor_e = node_map.get((i+1, j, k))
        if neighbor_e:
            je = node[f'je_{group}']
            jw_nb = neighbor_e[f'jw_{group}']
            error = je + jw_nb
            ew_errors.append(abs(error))
        
        # N/S 면 검증: (I,J).S + (I,J+1).N = 0  (J 증가 = 남쪽)
        neighbor_s = node_map.get((i, j+1, k))
        if neighbor_s:
            js = node[f'js_{group}']
            jn_nb = neighbor_s[f'jn_{group}']
            error = js + jn_nb
            ns_errors.append(abs(error))
    
    return {
        'ew_median': np.median(ew_errors) if ew_errors else np.nan,
        'ew_max': np.max(ew_errors) if ew_errors else np.nan,
        'ew_count': len(ew_errors),
        'ns_median': np.median(ns_errors) if ns_errors else np.nan,
        'ns_max': np.max(ns_errors) if ns_errors else np.nan,
        'ns_count': len(ns_errors),
    }


def verify_jnet0_scale_factor(nodes: List[Dict], keff: float, zmesh: List[float],
                               group: str = 'g1') -> Tuple[float, float]:
    """
    JNET0 스케일 팩터 α 역산.
    
    각 노드에서 밸런스를 만족하는 α를 계산하여 median 값 반환.
    α=2이면 JNET0 = J_net/2 규약 확정.
    
    Args:
        nodes: 노드 데이터
        keff: 유효증배계수
        zmesh: 축방향 메시 크기 [cm]
        group: 'g1' 또는 'g2'
    
    Returns:
        (median_alpha, std_alpha)
    """
    dx = dy = 21.60780 / 2  # 10.80390 cm
    A_ns = dx * zmesh[0]  # 간단히 첫 zmesh 사용
    A_we = dy * zmesh[0]
    A_bt = dx * dy
    
    alphas = []
    
    for node in nodes:
        k = node['k']
        if k < 2 or k > 21:  # 반사체 제외
            continue
        
        dz = zmesh[k - 1]
        V = dx * dy * dz
        
        phi1 = node['flx_g1']
        phi2 = node.get('flx_g2', 0)
        
        # JNET 합 (α=1 기준)
        jnet_sum = (node[f'jn_{group}'] * A_ns + node[f'js_{group}'] * A_ns +
                    node[f'jw_{group}'] * A_we + node[f'je_{group}'] * A_we +
                    node[f'jb_{group}'] * A_bt + node[f'jt_{group}'] * A_bt)
        
        # Removal & Source
        sigma_r1 = node['abs_g1'] + node['sca_g1']
        sigma_a2 = node.get('abs_g2', 0)
        
        if group == 'g1':
            removal = sigma_r1 * phi1 * V
            source = (1.0 / keff) * (node['nfs_g1'] * phi1 + node.get('nfs_g2', 0) * phi2) * V
        else:
            removal = sigma_a2 * phi2 * V
            source = node['sca_g1'] * phi1 * V
        
        # α 역산: α × jnet_sum + removal = source
        # α = (source - removal) / jnet_sum
        if abs(jnet_sum) > 1e-10:
            alpha = (source - removal) / jnet_sum
            alphas.append(alpha)
    
    return np.median(alphas), np.std(alphas)


# ============================================================
# 예상 결과 (기존 테스트 결과 기반)
# ============================================================
"""
전류 연속 조건 (매핑 수정 후):
  E/W 면: 0.000e+00  ✓
  N/S 면: 0.000e+00  ✓

JNET0 스케일 팩터:
  median α = 2.0000
  std α = 0.03

→ JNET0 = J_net/2 확정
→ 밸런스 계산 시 ×2 필요
"""


if __name__ == "__main__":
    print("=" * 70)
    print("JNET0 전류 연속 및 스케일 팩터 검증")
    print("=" * 70)
    print("\n이 스크립트는 검증 로직의 템플릿입니다.")
    print("실제 실행을 위해서는 MAS_NXS 파일 파싱 코드가 필요합니다.")
    print("\n기존 검증 결과:")
    print("  - 전류 연속: E/W, N/S 모든 면에서 0.000e+00")
    print("  - 스케일 팩터: median α = 2.0000, std = 0.03")
    print("  - 결론: JNET0 = J_net/2 (half net current)")
