import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
import time
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 가능하면 GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RootPredictorNet(nn.Module):
    """근 예측 신경망 - 최대 4개 근까지 예측"""
    def __init__(self, input_dim=3, hidden_dims=[128, 256, 512, 512, 256, 128], max_roots=3):
        super(RootPredictorNet, self).__init__()
        self.max_roots = max_roots
        
        # 공통 백본 네트워크
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1 if hidden_dim <= 256 else 0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 근의 개수 예측 헤드 (0개부터 4개까지)
        self.num_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, max_roots + 1),  # 0, 1, 2, 3, 4개 근
            nn.Softmax(dim=1)
        )
        
        # 근의 값 예측 헤드
        self.values_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, max_roots)  # 최대 4개 근의 값
        )
    
    def forward(self, x):
        # 공통 특성 추출
        features = self.backbone(x)
        
        # 두 개의 헤드로 예측
        num_roots = self.num_head(features)
        root_values = self.values_head(features)
        
        return num_roots, root_values

class SpeedComparator:
    def __init__(self, model_path='best_root_model10.pth', csv_file_path='neural_network_dataset5.csv'):
        self.model_path = model_path
        self.csv_file_path = csv_file_path
        self.model = None
        self.scaler = StandardScaler()
        self.device = device
        self.max_roots = 3
        self.load_model()
    
    def load_model(self):
        """학습된 모델과 스케일러 로드"""
        try:
            # 모델 구조 생성
            self.model = RootPredictorNet(max_roots=self.max_roots).to(self.device)
            
            # 학습된 가중치 로드
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            # 스케일러 학습 (원본 데이터에서)
            if self.csv_file_path:
                try:
                    df = pd.read_csv(self.csv_file_path)
                    X = df[['a', 'b', 'c']].values
                    self.scaler.fit(X)
                    print("✅ 모델과 스케일러 로드 완료")
                except Exception as e:
                    print(f"⚠️ CSV 파일을 찾을 수 없어 기본 스케일러 사용: {e}")
                    # 기본적인 스케일러 설정 (대략적인 범위 기반)
                    self.scaler.mean_ = np.array([0.0, 0.0, 0.0])
                    self.scaler.scale_ = np.array([2.0, 2.0, 2.0])
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("먼저 모델을 학습해주세요!")
            self.model = None
    
    def f(self, x, a, b, c):
        """f(x) = e^x * x / (x^4 + bx^2 - cx - 5) - a"""
        if abs(x) < 1e-12:  # x가 0에 가까우면
            return np.inf
        try:
            denominator = x**4 + b*x**2 - c*x - 5
            if abs(denominator) < 1e-10:  # 분모가 0에 가까우면
                return np.inf
            return np.exp(x) * x / denominator - a
        except:
            return np.inf
    
    def traditional_root_finding(self, a, b, c, domain_range=(-10, 10), step=0.5, tolerance=1e-6):
        """전통적인 방식: 정의역을 쫙 훑으면서 근 찾기"""
        roots = []
        initial_guesses = np.arange(domain_range[0], domain_range[1] + step, step)
        
        for guess in initial_guesses:
            try:
                root_val = fsolve(lambda x: self.f(x, a, b, c), guess, full_output=True, xtol=tolerance)
                if root_val[2] == 1:  # 수렴 성공
                    x_root = root_val[0][0]
                    
                    # 실제로 근인지 확인
                    if abs(self.f(x_root, a, b, c)) < tolerance:
                        # 중복 제거
                        if not any(abs(x_root - r) < tolerance for r in roots):
                            if domain_range[0] <= x_root <= domain_range[1]:
                                roots.append(x_root)
            except:
                continue
        
        return sorted(roots)
    
    def model_guided_root_finding(self, a, b, c, tolerance=1e-6, backup_range=(-10, 10)):
        """모델 예측값을 초기값으로 사용하는 방식"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 1. 모델로 근 예측
        X_input = self.scaler.transform([[a, b, c]])
        X_tensor = torch.FloatTensor(X_input).to(self.device)
        
        with torch.no_grad():
            pred_num, pred_values = self.model(X_tensor)
            pred_num = pred_num.cpu().numpy()
            pred_values = pred_values.cpu().numpy()
        
        # 2. 예측된 근의 개수와 값 얻기
        predicted_num_roots = np.argmax(pred_num[0])
        
        roots = []
        
        if predicted_num_roots > 0:
            # 3. 예측된 근들을 초기값으로 사용하여 정확한 근 찾기
            predicted_roots = pred_values[0][:predicted_num_roots]
            
            for initial_guess in predicted_roots:
                try:
                    root_val = fsolve(lambda x: self.f(x, a, b, c), initial_guess, 
                                    full_output=True, xtol=tolerance)
                    if root_val[2] == 1:  # 수렴 성공
                        x_root = root_val[0][0]
                        
                        # 실제로 근인지 확인
                        if abs(self.f(x_root, a, b, c)) < tolerance:
                            # 중복 제거
                            if not any(abs(x_root - r) < tolerance for r in roots):
                                roots.append(x_root)
                except:
                    continue
        
        # 4. 모델이 놓친 근이 있을 수 있으므로, 약간의 백업 검색 수행
        # (전체 범위보다는 훨씬 적은 포인트로)
        backup_guesses = np.linspace(backup_range[0], backup_range[1], 20)  # 20개 포인트만
        
        for guess in backup_guesses:
            try:
                root_val = fsolve(lambda x: self.f(x, a, b, c), guess, 
                                full_output=True, xtol=tolerance)
                if root_val[2] == 1:  # 수렴 성공
                    x_root = root_val[0][0]
                    
                    if abs(self.f(x_root, a, b, c)) < tolerance:
                        # 중복 제거
                        if not any(abs(x_root - r) < tolerance for r in roots):
                            if backup_range[0] <= x_root <= backup_range[1]:
                                roots.append(x_root)
            except:
                continue
        
        return sorted(roots)
    
    def compare_single_case(self, a, b, c, domain_range=(-10, 10), step=0.5, num_runs=10):
        """단일 케이스에 대한 속도 비교"""
        print(f"\n{'='*60}")
        print(f"속도 비교 테스트: a={a}, b={b}, c={c}")
        print(f"도메인 범위: {domain_range}, 스텝: {step}")
        print(f"반복 횟수: {num_runs}")
        print(f"{'='*60}")
        
        # 전통적인 방식 시간 측정
        traditional_times = []
        traditional_results = None
        
        for i in range(num_runs):
            start_time = time.time()
            roots = self.traditional_root_finding(a, b, c, domain_range, step)
            end_time = time.time()
            
            traditional_times.append(end_time - start_time)
            if i == 0:  # 첫 번째 결과 저장
                traditional_results = roots
        
        # 모델 가이드 방식 시간 측정
        model_times = []
        model_results = None
        
        if self.model is not None:
            for i in range(num_runs):
                start_time = time.time()
                roots = self.model_guided_root_finding(a, b, c, backup_range=domain_range)
                end_time = time.time()
                
                model_times.append(end_time - start_time)
                if i == 0:  # 첫 번째 결과 저장
                    model_results = roots
        
        # 결과 출력
        traditional_avg = np.mean(traditional_times) * 1000  # ms로 변환
        traditional_std = np.std(traditional_times) * 1000
        
        print(f"\n🔍 전통적인 방식 (전체 도메인 스캔):")
        print(f"  평균 시간: {traditional_avg:.2f} ± {traditional_std:.2f} ms")
        print(f"  찾은 근: {len(traditional_results)}개 - {[f'{r:.4f}' for r in traditional_results]}")
        
        if self.model is not None and model_times:
            model_avg = np.mean(model_times) * 1000  # ms로 변환
            model_std = np.std(model_times) * 1000
            
            print(f"\n🤖 모델 가이드 방식:")
            print(f"  평균 시간: {model_avg:.2f} ± {model_std:.2f} ms")
            print(f"  찾은 근: {len(model_results)}개 - {[f'{r:.4f}' for r in model_results]}")
            
            # 속도 향상 계산
            speedup = traditional_avg / model_avg if model_avg > 0 else float('inf')
            print(f"\n📊 성능 비교:")
            print(f"  속도 향상: {speedup:.2f}x")
            print(f"  시간 단축: {((traditional_avg - model_avg) / traditional_avg * 100):.1f}%")
            
            # 정확도 비교
            if len(traditional_results) == len(model_results):
                if len(traditional_results) == 0:
                    print(f"  정확도: ✅ 완벽 (둘 다 근 없음)")
                else:
                    errors = [abs(t - m) for t, m in zip(sorted(traditional_results), sorted(model_results))]
                    avg_error = np.mean(errors)
                    print(f"  정확도: 평균 오차 {avg_error:.6f}")
                    if avg_error < 1e-4:
                        print(f"           ✅ 매우 정확!")
                    elif avg_error < 1e-2:
                        print(f"           ✅ 정확")
                    else:
                        print(f"           ⚠️ 약간 부정확")
            else:
                print(f"  정확도: ❌ 근 개수 불일치 (전통: {len(traditional_results)}, 모델: {len(model_results)})")
        
        return {
            'traditional_time': traditional_avg,
            'model_time': model_avg if self.model is not None else None,
            'speedup': speedup if self.model is not None else None,
            'traditional_roots': traditional_results,
            'model_roots': model_results if self.model is not None else None
        }
    
    def run_comprehensive_benchmark(self, test_cases=None, domain_range=(-10, 10), step=0.5, num_runs=5):
        """종합적인 벤치마크 테스트"""
        if test_cases is None:
            # 다양한 난이도의 테스트 케이스들
            test_cases = [
                # 간단한 케이스들
                (1.0, 2.0, 3.0),
                (-0.5, 1.5, -2.0),
                (0.8, -1.0, 0.5),
                
                # 중간 난이도
                (2.5, 0.0, 1.0),
                (-1.2, 3.0, -1.5),
                (0.3, -0.8, 2.2),
                
                # 복잡한 케이스들
                (1.8, 1.2, -0.7),
                (-0.9, -1.5, 1.8),
                (3.0, -2.0, 4.0),
                (0.0, 1.0, 0.0),

                (1.0, 2.0, 3.0),
                (-5.5, 15.2, -20.8),
                (8.3, -12.7, 5.9),
                (25.0, 0.0, 18.5),
                (-18.7, 22.3, 35.1),
                (7.4, 33.8, -14.2),
                (42.6, -19.5, 0.0),
                (-28.9, 6.7, 41.3),
                (16.8, -7.4, 29.7),
                (0.0, 38.2, -23.5),
                (-31.4, 45.7, 12.8),
                (39.3, 21.6, -8.9),
                (4.7, -26.3, 37.4),
                (-15.1, 9.8, -34.6),
                (27.9, 13.2, -46.7),
                (-22.8, -17.3, 26.5),
                (11.4, 48.9, -7.1),
                (35.7, -31.2, 19.8),
                (-9.6, 32.4, -51.3),
                (24.1, -4.8, 33.9)
            ]
        
        print(f"🚀 종합 벤치마크 테스트 시작!")
        print(f"테스트 케이스: {len(test_cases)}개")
        print(f"각 케이스당 반복: {num_runs}회")
        print(f"도메인 범위: {domain_range}, 스텝: {step}")
        print(f"{'='*80}")
        
        results = []
        total_traditional_time = 0
        total_model_time = 0
        speedup_list = []
        accuracy_count = 0
        
        for i, (a, b, c) in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] 테스트: a={a}, b={b}, c={c}")
            
            result = self.compare_single_case(a, b, c, domain_range, step, num_runs)
            results.append(result)
            
            total_traditional_time += result['traditional_time']
            if result['model_time'] is not None:
                total_model_time += result['model_time']
                if result['speedup'] is not None:
                    speedup_list.append(result['speedup'])
            
            # 정확도 체크
            if (result['traditional_roots'] is not None and 
                result['model_roots'] is not None and
                len(result['traditional_roots']) == len(result['model_roots'])):
                accuracy_count += 1
        
        # 전체 결과 요약
        print(f"\n{'='*80}")
        print(f"📊 종합 벤치마크 결과 요약")
        print(f"{'='*80}")
        
        print(f"전체 테스트 케이스: {len(test_cases)}개")
        print(f"총 전통적 방식 시간: {total_traditional_time:.2f} ms")
        
        if total_model_time > 0:
            print(f"총 모델 가이드 시간: {total_model_time:.2f} ms")
            overall_speedup = total_traditional_time / total_model_time
            print(f"전체 평균 속도 향상: {overall_speedup:.2f}x")
            print(f"전체 시간 단축: {((total_traditional_time - total_model_time) / total_traditional_time * 100):.1f}%")
            
            if speedup_list:
                print(f"개별 속도 향상 통계:")
                print(f"  최소: {min(speedup_list):.2f}x")
                print(f"  최대: {max(speedup_list):.2f}x")
                print(f"  평균: {np.mean(speedup_list):.2f}x")
                print(f"  표준편차: {np.std(speedup_list):.2f}")
        
        print(f"근 개수 정확도: {accuracy_count}/{len(test_cases)} ({accuracy_count/len(test_cases)*100:.1f}%)")
        
        # 결과 시각화
        self.plot_benchmark_results(results, test_cases)
        
        return results
    
    def plot_benchmark_results(self, results, test_cases):
        """벤치마크 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 시간 비교 바 차트
        ax1 = axes[0, 0]
        traditional_times = [r['traditional_time'] for r in results]
        model_times = [r['model_time'] for r in results if r['model_time'] is not None]
        
        x = np.arange(len(test_cases))
        width = 0.35
        
        ax1.bar(x - width/2, traditional_times, width, label='Traditional Method', alpha=0.8, color='red')
        if model_times:
            ax1.bar(x + width/2, model_times, width, label='Model-Guided Method', alpha=0.8, color='blue')
        
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time Comparison by Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{i+1}' for i in range(len(test_cases))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 속도 향상 비율
        ax2 = axes[0, 1]
        speedups = [r['speedup'] for r in results if r['speedup'] is not None]
        if speedups:
            ax2.bar(range(len(speedups)), speedups, alpha=0.8, color='green')
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
            ax2.set_xlabel('Test Case')
            ax2.set_ylabel('Speedup (x)')
            ax2.set_title('Speed Improvement Ratio')
            ax2.set_xticks(range(len(speedups)))
            ax2.set_xticklabels([f'{i+1}' for i in range(len(speedups))])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 시간 분포 히스토그램
        ax3 = axes[1, 0]
        ax3.hist(traditional_times, alpha=0.7, label='Traditional Method', bins=10, color='red')
        if model_times:
            ax3.hist(model_times, alpha=0.7, label='Model-Guided Method', bins=10, color='blue')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Execution Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 누적 시간 비교
        ax4 = axes[1, 1]
        cumulative_traditional = np.cumsum(traditional_times)
        if model_times:
            cumulative_model = np.cumsum(model_times)
            ax4.plot(range(1, len(cumulative_traditional)+1), cumulative_traditional, 
                    'o-', label='Traditional Method', color='red')
            ax4.plot(range(1, len(cumulative_model)+1), cumulative_model, 
                    'o-', label='Model-Guided Method', color='blue')
        else:
            ax4.plot(range(1, len(cumulative_traditional)+1), cumulative_traditional, 
                    'o-', label='Traditional Method', color='red')
        
        ax4.set_xlabel('Test Case Number')
        ax4.set_ylabel('Cumulative Time (ms)')
        ax4.set_title('Cumulative Execution Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_domain_step_impact(self, a=1.0, b=2.0, c=3.0, num_runs=3):
        """도메인 범위와 스텝 크기가 속도에 미치는 영향 분석"""
        print(f"\n{'='*60}")
        print(f"도메인/스텝 크기 영향 분석")
        print(f"테스트 케이스: a={a}, b={b}, c={c}")
        print(f"{'='*60}")
        
        # 다양한 도메인 범위와 스텝 크기
        domain_ranges = [(-5, 5), (-10, 10), (-20, 20), (-50, 50)]
        steps = [1.0, 0.5, 0.25, 0.1]
        
        results = []
        
        for domain_range in domain_ranges:
            for step in steps:
                print(f"\n테스트: 도메인 {domain_range}, 스텝 {step}")
                
                # 전체 초기값 개수 계산
                total_guesses = len(np.arange(domain_range[0], domain_range[1] + step, step))
                print(f"  초기값 개수: {total_guesses}")
                
                # 시간 측정
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    roots = self.traditional_root_finding(a, b, c, domain_range, step)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times) * 1000  # ms
                std_time = np.std(times) * 1000
                
                print(f"  평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
                print(f"  찾은 근: {len(roots)}개")
                
                results.append({
                    'domain_range': domain_range,
                    'step': step,
                    'total_guesses': total_guesses,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'num_roots': len(roots)
                })
        
        # 결과 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 초기값 개수 vs 시간
        ax1 = axes[0, 0]
        guess_counts = [r['total_guesses'] for r in results]
        times = [r['avg_time'] for r in results]
        ax1.scatter(guess_counts, times, alpha=0.7, s=50)
        ax1.set_xlabel('Number of Initial Guesses')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_title('Initial Guesses vs Execution Time')
        ax1.grid(True, alpha=0.3)
        
        # 도메인 크기별 시간
        ax2 = axes[0, 1]
        domain_sizes = [(r[1] - r[0]) for r in domain_ranges]
        for step in steps:
            step_times = [r['avg_time'] for r in results if r['step'] == step]
            ax2.plot(domain_sizes, step_times, 'o-', label=f'Step={step}', linewidth=2)
        ax2.set_xlabel('Domain Size')
        ax2.set_ylabel('Average Time (ms)')
        ax2.set_title('Execution Time by Domain Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 스텝 크기별 시간
        ax3 = axes[1, 0]
        for domain_range in domain_ranges:
            range_times = [r['avg_time'] for r in results if r['domain_range'] == domain_range]
            ax3.plot(steps, range_times, 'o-', label=f'Domain={domain_range}', linewidth=2)
        ax3.set_xlabel('Step Size')
        ax3.set_ylabel('Average Time (ms)')
        ax3.set_title('Execution Time by Step Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 효율성 (시간/초기값개수)
        ax4 = axes[1, 1]
        efficiency = [r['avg_time'] / r['total_guesses'] for r in results]
        ax4.scatter(guess_counts, efficiency, alpha=0.7, s=50, c=times, cmap='viridis')
        ax4.set_xlabel('Number of Initial Guesses')
        ax4.set_ylabel('Efficiency (ms/guess)')
        ax4.set_title('Computational Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results

# 메인 실행 함수
def main():
    """메인 함수"""
    print("⚡ Root Finding Speed Comparison Tool")
    print("=" * 60)
    
    # 비교기 초기화
    comparator = SpeedComparator()
    
    if comparator.model is None:
        print("⚠️  모델이 로드되지 않았습니다. 전통적 방식만 테스트합니다.")
    
    while True:
        print("\n🎮 메뉴:")
        print("1. 단일 케이스 속도 비교")
        print("2. 종합 벤치마크 테스트")
        print("3. 도메인/스텝 크기 영향 분석")
        print("4. 커스텀 테스트 케이스")
        print("5. 종료")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        if choice == '1':
            try:
                print("\n단일 케이스 속도 비교")
                a = float(input("a 값: "))
                b = float(input("b 값: "))
                c = float(input("c 값: "))
                
                # 옵션 설정
                domain_input = input("도메인 범위 (기본: -10,10): ").strip()
                if domain_input:
                    domain_parts = [float(x.strip()) for x in domain_input.split(',')]
                    domain_range = (domain_parts[0], domain_parts[1])
                else:
                    domain_range = (-10, 10)
                
                step_input = input("스텝 크기 (기본: 0.5): ").strip()
                step = float(step_input) if step_input else 0.5
                
                runs_input = input("반복 횟수 (기본: 10): ").strip()
                num_runs = int(runs_input) if runs_input else 10
                
                comparator.compare_single_case(a, b, c, domain_range, step, num_runs)
                
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요!")
                
        elif choice == '2':
            print("\n종합 벤치마크 테스트")
            
            # 옵션 설정
            domain_input = input("도메인 범위 (기본: -50,50): ").strip()
            if domain_input:
                domain_parts = [float(x.strip()) for x in domain_input.split(',')]
                domain_range = (domain_parts[0], domain_parts[1])
            else:
                domain_range = (-50, 50)
            
            step_input = input("스텝 크기 (기본: 0.5): ").strip()
            step = float(step_input) if step_input else 0.5
            
            runs_input = input("반복 횟수 (기본: 5): ").strip()
            num_runs = int(runs_input) if runs_input else 5
            
            # 커스텀 테스트 케이스 사용 여부
            custom_input = input("커스텀 테스트 케이스를 사용하시겠습니까? (y/n): ").strip().lower()
            
            test_cases = None
            if custom_input == 'y':
                test_cases = []
                print("테스트 케이스를 입력하세요 (종료하려면 빈 줄 입력):")
                while True:
                    case_input = input("a,b,c: ").strip()
                    if not case_input:
                        break
                    try:
                        a, b, c = [float(x.strip()) for x in case_input.split(',')]
                        test_cases.append((a, b, c))
                        print(f"  추가됨: ({a}, {b}, {c})")
                    except:
                        print("❌ 올바른 형식으로 입력해주세요! (예: 1.0, 2.0, 3.0)")
                
                if not test_cases:
                    print("❌ 테스트 케이스가 없습니다. 기본 케이스를 사용합니다.")
                    test_cases = None
            
            comparator.run_comprehensive_benchmark(test_cases, domain_range, step, num_runs)
            
        elif choice == '3':
            try:
                print("\n도메인/스텝 크기 영향 분석")
                a = float(input("a 값 (기본: 1.0): ") or "1.0")
                b = float(input("b 값 (기본: 2.0): ") or "2.0")
                c = float(input("c 값 (기본: 3.0): ") or "3.0")
                
                runs_input = input("반복 횟수 (기본: 3): ").strip()
                num_runs = int(runs_input) if runs_input else 3
                
                comparator.analyze_domain_step_impact(a, b, c, num_runs)
                
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요!")
                
        elif choice == '4':
            print("\n커스텀 테스트 케이스")
            try:
                # 여러 케이스 입력
                test_cases = []
                print("여러 테스트 케이스를 입력하세요 (종료하려면 빈 줄 입력):")
                while True:
                    case_input = input(f"케이스 {len(test_cases)+1} (a,b,c): ").strip()
                    if not case_input:
                        break
                    try:
                        a, b, c = [float(x.strip()) for x in case_input.split(',')]
                        test_cases.append((a, b, c))
                        print(f"  추가됨: ({a}, {b}, {c})")
                    except:
                        print("❌ 올바른 형식으로 입력해주세요! (예: 1.0, 2.0, 3.0)")
                
                if not test_cases:
                    print("❌ 테스트 케이스가 없습니다.")
                    continue
                
                # 옵션 설정
                domain_input = input("도메인 범위 (기본: -50,50): ").strip()
                if domain_input:
                    domain_parts = [float(x.strip()) for x in domain_input.split(',')]
                    domain_range = (domain_parts[0], domain_parts[1])
                else:
                    domain_range = (-50, 50)
                
                step_input = input("스텝 크기 (기본: 0.5): ").strip()
                step = float(step_input) if step_input else 0.5
                
                runs_input = input("반복 횟수 (기본: 5): ").strip()
                num_runs = int(runs_input) if runs_input else 5
                
                comparator.run_comprehensive_benchmark(test_cases, domain_range, step, num_runs)
                
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요!")
                
        elif choice == '5':
            print("👋 프로그램 종료!")
            break
            
        else:
            print("❌ 올바른 선택지를 입력해주세요!")

if __name__ == "__main__":
    main()