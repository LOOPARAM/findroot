import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar, brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BalancedRootFinder:
    def __init__(self, domain_range=(-50, 50), step=0.5):
        self.domain_range = domain_range
        self.step = step
        self.initial_guesses = np.arange(domain_range[0], domain_range[1] + step, step)
        
    def f(self, x, a, b, c):
        """f(x) = e^x * x / (x^4 + bx^2 - cx - 5) - a"""
        if x == 0:
            return np.inf
        try:
            denominator = x**4 + b*x**2 - c*x - 5
            if abs(denominator) < 1e-12:  # 분모가 0에 가까우면
                return np.inf
            return np.exp(x) * x / denominator - a
        except:
            return np.inf
    
    def find_roots(self, a, b, c, tolerance=1e-8, max_roots=20):
        """주어진 a, b, c에 대해 모든 근을 찾기"""
        roots = []
        
        # fsolve를 사용한 근찾기
        for guess in self.initial_guesses:
            try:
                # fsolve 사용
                root_val = fsolve(lambda x: self.f(x, a, b, c), guess, full_output=True)
                if root_val[2] == 1:  # 수렴 성공
                    x_root = root_val[0][0]
                    
                    # 실제로 근인지 확인
                    if abs(self.f(x_root, a, b, c)) < tolerance:
                        # 중복 제거 (이미 찾은 근과 너무 가까우면 제외)
                        if not any(abs(x_root - r) < tolerance for r in roots):
                            # 정의역 내에 있는지 확인
                            if self.domain_range[0] <= x_root <= self.domain_range[1]:
                                roots.append(x_root)
                                
                                # 최대 근 개수 제한
                                if len(roots) >= max_roots:
                                    break
            except:
                continue
        
        return sorted(roots)
    
    def generate_balanced_dataset(self, n_samples=10000, param_range=(-100, 100), 
                                 target_distribution=None, max_attempts_per_target=10000):
        """균등한 근 개수 분포를 가진 데이터셋 생성"""
        
        # 기본 타겟 분포 (0~4개 근을 균등하게)
        if target_distribution is None:
            target_distribution = {
                0: n_samples // 4,  # 25%
                1: n_samples // 4,  # 25%
                2: n_samples // 4,  # 25%
                3: n_samples - 3 * (n_samples // 4),  # 나머지 (25%)
                4: 1  # 1%
            }
        
        print(f"균등 분포 데이터셋 생성 시작: {n_samples}개 샘플")
        print(f"매개변수 범위: a, b, c ∈ [{param_range[0]}, {param_range[1]}]")
        print(f"정의역: x ∈ [{self.domain_range[0]}, {self.domain_range[1]}]")
        print("타겟 분포:")
        for num_roots, count in target_distribution.items():
            print(f"  {num_roots}개 근: {count}개 샘플 ({count/n_samples*100:.1f}%)")
        print("-" * 60)
        
        dataset = []
        current_counts = {k: 0 for k in target_distribution.keys()}
        total_attempts = 0
        
        # 각 근 개수별로 목표 달성까지 생성
        for target_roots in target_distribution.keys():
            target_count = target_distribution[target_roots]
            attempts = 0
            
            print(f"\n{target_roots}개 근을 가진 샘플 {target_count}개 생성 중...")
            
            while current_counts[target_roots] < target_count and attempts < max_attempts_per_target:
                # a, b, c 랜덤 생성
                a = np.random.uniform(param_range[0], param_range[1])
                b = np.random.uniform(param_range[0], param_range[1])
                c = np.random.uniform(param_range[0], param_range[1])
                
                # 근 찾기
                roots = self.find_roots(a, b, c)
                num_roots = len(roots)
                
                # 목표 근 개수와 일치하면 저장
                if num_roots == target_roots:
                    sample = {
                        'a': a,
                        'b': b,
                        'c': c,
                        'num_roots': num_roots,
                        'roots': roots
                    }
                    dataset.append(sample)
                    current_counts[target_roots] += 1
                    
                    # 진행상황 출력
                    if current_counts[target_roots] % max(1, target_count // 10) == 0:
                        progress = current_counts[target_roots] / target_count * 100
                        print(f"  {target_roots}개 근: {current_counts[target_roots]}/{target_count} ({progress:.1f}%)")
                        print(f"    마지막 샘플: a={a:.2f}, b={b:.2f}, c={c:.2f}")
                        if roots:
                            print(f"    근: {[f'{r:.3f}' for r in roots[:5]]}")
                
                attempts += 1
                total_attempts += 1
                
                # 진행상황 모니터링
                if attempts % 1000 == 0:
                    success_rate = current_counts[target_roots] / attempts * 100 if attempts > 0 else 0
                    print(f"    시도 {attempts}회, 성공률: {success_rate:.1f}%")
            
            if current_counts[target_roots] < target_count:
                print(f"  경고: {target_roots}개 근 목표({target_count})를 달성하지 못함. 현재: {current_counts[target_roots]}개")
        
        print(f"\n총 시도 횟수: {total_attempts}")
        print("실제 생성된 분포:")
        for num_roots, count in current_counts.items():
            percentage = count / sum(current_counts.values()) * 100 if sum(current_counts.values()) > 0 else 0
            print(f"  {num_roots}개 근: {count}개 샘플 ({percentage:.1f}%)")
        
        return dataset
    
    def generate_uniform_balanced_dataset(self, n_samples=10000, param_range=(-100, 100)):
        """동일한 매개변수 범위에서 균등 분포 생성"""
        print(f"균등 분포 데이터셋 생성 시작: {n_samples}개 샘플")
        print(f"모든 매개변수 동일 범위: a, b, c ∈ [{param_range[0]}, {param_range[1]}]")
        print("-" * 60)
        
        dataset = []
        target_counts = {
            0: n_samples // 4,  # 25%
            1: n_samples // 4,  # 25%
            2: n_samples // 4,  # 25%
            3: n_samples - 3 * (n_samples // 4),  # 나머지 (25%)
            4: 1  # 1%
        }
        
        for target_roots in range(5):  # 0~4개 근
            target_per_category = target_counts[target_roots]
            current_count = 0
            attempts = 0
            max_attempts = target_per_category * 100  # 충분한 시도 횟수
            
            print(f"{target_roots}개 근 샘플 {target_per_category}개 생성 중...")
            
            while current_count < target_per_category and attempts < max_attempts:
                # 동일한 범위에서 매개변수 생성
                a = np.random.uniform(param_range[0], param_range[1])
                b = np.random.uniform(param_range[0], param_range[1])
                c = np.random.uniform(param_range[0], param_range[1])
                
                roots = self.find_roots(a, b, c)
                num_roots = len(roots)
                
                if num_roots == target_roots:
                    sample = {
                        'a': a,
                        'b': b,
                        'c': c,
                        'num_roots': num_roots,
                        'roots': roots
                    }
                    dataset.append(sample)
                    current_count += 1
                    
                    if current_count % max(1, target_per_category // 5) == 0:
                        progress = current_count / target_per_category * 100
                        print(f"  진행: {current_count}/{target_per_category} ({progress:.1f}%)")
                
                attempts += 1
                
                # 진행상황 중간 체크
                if attempts % 5000 == 0:
                    success_rate = current_count / attempts * 100 if attempts > 0 else 0
                    print(f"    시도 {attempts}회, 현재 성공률: {success_rate:.2f}%")
            
            success_rate = current_count / attempts * 100 if attempts > 0 else 0
            print(f"  완료: {current_count}/{target_per_category}개 생성, 최종 성공률: {success_rate:.2f}%")
        
        return dataset
    
    def analyze_dataset(self, dataset):
        """데이터셋 분석"""
        print("=" * 60)
        print("데이터셋 분석 결과")
        print("=" * 60)
        
        num_roots_distribution = {}
        all_roots = []
        
        for sample in dataset:
            num_roots = sample['num_roots']
            if num_roots in num_roots_distribution:
                num_roots_distribution[num_roots] += 1
            else:
                num_roots_distribution[num_roots] = 1
            
            all_roots.extend(sample['roots'])
        
        print(f"총 샘플 수: {len(dataset)}")
        print(f"근 개수 분포:")
        for num_roots in sorted(num_roots_distribution.keys()):
            count = num_roots_distribution[num_roots]
            percentage = count / len(dataset) * 100
            print(f"  {num_roots}개 근: {count}개 샘플 ({percentage:.1f}%)")
        
        if all_roots:
            print(f"\n전체 근 통계:")
            print(f"  총 근 개수: {len(all_roots)}")
            print(f"  근 범위: [{min(all_roots):.3f}, {max(all_roots):.3f}]")
            print(f"  근 평균: {np.mean(all_roots):.3f}")
            print(f"  근 표준편차: {np.std(all_roots):.3f}")
        
        return num_roots_distribution
    
    def save_dataset(self, dataset, filename='balanced_neural_network_dataset.csv'):
        """데이터셋을 CSV로 저장"""
        # 가장 많은 근을 가진 샘플의 근 개수 찾기
        max_roots = max(len(sample['roots']) for sample in dataset) if dataset else 0
        
        # DataFrame 생성용 데이터
        data_for_df = []
        
        for sample in dataset:
            row = {
                'a': sample['a'],
                'b': sample['b'], 
                'c': sample['c'],
                'num_roots': sample['num_roots']
            }
            
            # 각 근을 별도 컬럼으로 저장
            for i in range(max_roots):
                if i < len(sample['roots']):
                    row[f'root_{i+1}'] = sample['roots'][i]
                else:
                    row[f'root_{i+1}'] = np.nan
            
            data_for_df.append(row)
        
        df = pd.DataFrame(data_for_df)
        df.to_csv(filename, index=False)
        print(f"\n데이터셋이 '{filename}'에 저장되었습니다.")
        print(f"DataFrame 크기: {df.shape}")
        return df

# 메인 실행
if __name__ == "__main__":
    # BalancedRootFinder 객체 생성
    root_finder = BalancedRootFinder(domain_range=(-50, 50), step=1)
    
    print("=== 균등 분포 데이터셋 생성 ===")
    print("동일한 매개변수 범위에서 각 근 개수를 균등하게 생성합니다.")
    
    # 균등 분포 생성
    dataset = root_finder.generate_uniform_balanced_dataset(n_samples=5000, param_range=(-100, 100))
    
    # 분석
    distribution = root_finder.analyze_dataset(dataset)
    
    # 저장
    df = root_finder.save_dataset(dataset, 'uniform_balanced_dataset.csv')
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 근 개수 분포 히스토그램
    plt.subplot(2, 3, 1)
    root_counts = [sample['num_roots'] for sample in dataset]
    plt.hist(root_counts, bins=range(max(root_counts)+2), alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('근의 개수')
    plt.ylabel('빈도')
    plt.title('균등한 근 개수 분포')
    plt.grid(True, alpha=0.3)
    
    # 매개변수 분포
    plt.subplot(2, 3, 2)
    a_values = [sample['a'] for sample in dataset]
    b_values = [sample['b'] for sample in dataset]
    c_values = [sample['c'] for sample in dataset]
    
    plt.hist(a_values, alpha=0.6, label='a', bins=30, color='red')
    plt.hist(b_values, alpha=0.6, label='b', bins=30, color='green')
    plt.hist(c_values, alpha=0.6, label='c', bins=30, color='blue')
    plt.xlabel('매개변수 값')
    plt.ylabel('빈도')
    plt.title('매개변수 분포 (동일 범위)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 근의 분포
    plt.subplot(2, 3, 3)
    all_roots = []
    for sample in dataset:
        all_roots.extend(sample['roots'])
    
    if all_roots:
        plt.hist(all_roots, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.xlabel('근의 값')
        plt.ylabel('빈도')
        plt.title('모든 근의 분포')
        plt.grid(True, alpha=0.3)
    
    # 근 개수별 매개변수 분포 (a vs b)
    for i, target_roots in enumerate(range(3)):  # 0, 1, 2개 근만 표시
        plt.subplot(2, 3, 4+i)
        samples_with_target = [s for s in dataset if s['num_roots'] == target_roots]
        if samples_with_target:
            a_vals = [s['a'] for s in samples_with_target]
            b_vals = [s['b'] for s in samples_with_target]
            plt.scatter(a_vals, b_vals, alpha=0.6, s=10)
            plt.xlabel('매개변수 a')
            plt.ylabel('매개변수 b')
            plt.title(f'{target_roots}개 근 - a vs b')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n균등 분포 데이터셋 생성 완료!")
    print(f"모든 매개변수가 동일한 범위 [-100, 100]에서 생성되었습니다.")
    
    # 샘플 데이터 출력
    print("\n샘플 데이터 (각 근 개수별 1개씩):")
    for num_roots in sorted(set(s['num_roots'] for s in dataset)):
        sample = next(s for s in dataset if s['num_roots'] == num_roots)
        print(f"{num_roots}개 근: a={sample['a']:.3f}, b={sample['b']:.3f}, c={sample['c']:.3f}")
        if sample['roots']:
            print(f"  근: {[f'{r:.3f}' for r in sample['roots']]}")
        else:
            print(f"  근: 없음")