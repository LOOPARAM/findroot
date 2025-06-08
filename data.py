import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar, brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class RootFinder:
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
    
    def generate_dataset(self, n_samples=10000, param_range=(-100, 100)):
        """데이터셋 생성"""
        print(f"데이터셋 생성 시작: {n_samples}개 샘플")
        print(f"매개변수 범위: a, b, c ∈ [{param_range[0]}, {param_range[1]}]")
        print(f"정의역: x ∈ [{self.domain_range[0]}, {self.domain_range[1]}]")
        print("-" * 60)
        
        dataset = []
        
        for i in range(n_samples):
            # a, b, c 랜덤 생성
            a = np.random.uniform(param_range[0], param_range[1])
            b = np.random.uniform(param_range[0], param_range[1])
            c = np.random.uniform(param_range[0], param_range[1])
            
            # 근 찾기
            roots = self.find_roots(a, b, c)
            
            # 결과 저장
            sample = {
                'a': a,
                'b': b,
                'c': c,
                'num_roots': len(roots),
                'roots': roots
            }
            dataset.append(sample)
            
            # 진행상황 출력
            if (i + 1) % 100 == 0:
                print(f"진행상황: {i + 1}/{n_samples} ({(i + 1)/n_samples*100:.1f}%)")
                if roots:
                    print(f"  마지막 샘플: a={a:.2f}, b={b:.2f}, c={c:.2f}")
                    print(f"  근 개수: {len(roots)}, 근: {[f'{r:.3f}' for r in roots[:5]]}")
                else:
                    print(f"  마지막 샘플: a={a:.2f}, b={b:.2f}, c={c:.2f}, 근 없음")
                print()
        
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
    
    def save_dataset(self, dataset, filename='neural_network_dataset.csv'):
        """데이터셋을 CSV로 저장"""
        # 가장 많은 근을 가진 샘플의 근 개수 찾기
        max_roots = max(len(sample['roots']) for sample in dataset)
        
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
    # RootFinder 객체 생성
    root_finder = RootFinder(domain_range=(-50, 50), step=1)
    
    # 작은 테스트 먼저
    print("테스트: 몇 개 샘플로 시작")
    test_dataset = root_finder.generate_dataset(n_samples=50, param_range=(-10, 10))
    root_finder.analyze_dataset(test_dataset)
    
    print("\n" + "="*60)
    print("본격적인 데이터셋 생성")
    
    # 실제 데이터셋 생성
    dataset = root_finder.generate_dataset(n_samples=10000, param_range=(-100, 100))
    
    # 분석
    distribution = root_finder.analyze_dataset(dataset)
    
    # 저장
    df = root_finder.save_dataset(dataset)
    
    # 샘플 데이터 출력
    print("\n샘플 데이터 (처음 5개):")
    print(df.head())
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # 근 개수 분포 히스토그램
    plt.subplot(2, 2, 1)
    root_counts = [sample['num_roots'] for sample in dataset]
    plt.hist(root_counts, bins=range(max(root_counts)+2), alpha=0.7, edgecolor='black')
    plt.xlabel('근의 개수')
    plt.ylabel('빈도')
    plt.title('근 개수 분포')
    plt.grid(True, alpha=0.3)
    
    # 매개변수 분포
    plt.subplot(2, 2, 2)
    a_values = [sample['a'] for sample in dataset]
    b_values = [sample['b'] for sample in dataset]
    c_values = [sample['c'] for sample in dataset]
    
    plt.hist(a_values, alpha=0.5, label='a', bins=30)
    plt.hist(b_values, alpha=0.5, label='b', bins=30)
    plt.hist(c_values, alpha=0.5, label='c', bins=30)
    plt.xlabel('매개변수 값')
    plt.ylabel('빈도')
    plt.title('매개변수 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 근의 분포
    plt.subplot(2, 2, 3)
    all_roots = []
    for sample in dataset:
        all_roots.extend(sample['roots'])
    
    if all_roots:
        plt.hist(all_roots, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('근의 값')
        plt.ylabel('빈도')
        plt.title('모든 근의 분포')
        plt.grid(True, alpha=0.3)
    
    # 매개변수와 근 개수의 관계
    plt.subplot(2, 2, 4)
    a_vals = [sample['a'] for sample in dataset]
    num_roots = [sample['num_roots'] for sample in dataset]
    plt.scatter(a_vals, num_roots, alpha=0.5, s=1)
    plt.xlabel('매개변수 a')
    plt.ylabel('근의 개수')
    plt.title('매개변수 a와 근 개수의 관계')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n데이터셋 생성 완료!")
    print(f"이제 이 데이터로 신경망을 훈련시킬 수 있습니다.")