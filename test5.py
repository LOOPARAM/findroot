import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
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

class RootTester:
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
    
    def calculate_actual_roots(self, a, b, c, domain_range=(-10, 10), step=0.5, tolerance=1e-6):
        """실제 근을 수치적으로 계산하는 함수"""
        def f(x, a, b, c):
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
        
        roots = []
        initial_guesses = np.arange(domain_range[0], domain_range[1] + step, step)
        
        for guess in initial_guesses:
            try:
                root_val = fsolve(lambda x: f(x, a, b, c), guess, full_output=True, xtol=tolerance)
                if root_val[2] == 1:  # 수렴 성공
                    x_root = root_val[0][0]
                    
                    # 실제로 근인지 확인
                    if abs(f(x_root, a, b, c)) < tolerance:
                        # 중복 제거
                        if not any(abs(x_root - r) < tolerance for r in roots):
                            if domain_range[0] <= x_root <= domain_range[1]:
                                roots.append(x_root)
            except:
                continue
        
        return sorted(roots)
    
    def predict_roots(self, a, b, c):
        """근 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 입력 정규화
        X_input = self.scaler.transform([[a, b, c]])
        X_tensor = torch.FloatTensor(X_input).to(self.device)
        
        with torch.no_grad():
            pred_num, pred_values = self.model(X_tensor)
            
            # CPU로 이동 후 numpy 변환
            pred_num = pred_num.cpu().numpy()
            pred_values = pred_values.cpu().numpy()
        
        # 근의 개수 결정
        predicted_num_roots = np.argmax(pred_num[0])
        predicted_probs = pred_num[0]
        
        # 해당 개수만큼 근의 값 반환
        if predicted_num_roots == 0:
            return [], predicted_probs
        else:
            roots = pred_values[0][:predicted_num_roots].tolist()
            return roots, predicted_probs
    
    def test_single_case(self, a, b, c, show_plot=False):
        """단일 케이스 테스트"""
        print(f"\n{'='*60}")
        print(f"테스트 케이스: a={a}, b={b}, c={c}")
        print(f"방정식: e^x * x / (x^4 + {b}x^2 - {c}x - 5) - {a} = 0")
        print(f"{'='*60}")
        
        try:
            # 실제 근 계산
            print("🔍 실제 근 계산 중...")
            actual_roots = self.calculate_actual_roots(a, b, c)
            
            # 신경망 예측
            print("🤖 신경망 예측 중...")
            predicted_roots, probs = self.predict_roots(a, b, c)
            
            # 결과 출력
            print(f"\n📊 결과:")
            print(f"  실제 근 ({len(actual_roots)}개): {[f'{r:.4f}' for r in actual_roots]}")
            print(f"  예측 근 ({len(predicted_roots)}개): {[f'{r:.4f}' for r in predicted_roots]}")
            print(f"  예측 확률: {dict(zip(['0개', '1개', '2개', '3개', '4개'], [f'{p:.3f}' for p in probs]))}")
            
            # 정확도 평가
            self.evaluate_accuracy(actual_roots, predicted_roots)
            
            # 그래프 그리기
            if show_plot:
                self.plot_function_and_roots(a, b, c, actual_roots, predicted_roots)
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    
    def evaluate_accuracy(self, actual_roots, predicted_roots):
        """정확도 평가"""
        print(f"\n📈 정확도 평가:")
        
        if len(actual_roots) == len(predicted_roots):
            if len(actual_roots) == 0:
                print("  ✅ 완벽! (실근 없음을 정확히 예측)")
            else:
                actual_sorted = sorted(actual_roots)
                predicted_sorted = sorted(predicted_roots)
                errors = [abs(a - p) for a, p in zip(actual_sorted, predicted_sorted)]
                avg_error = np.mean(errors)
                max_error = max(errors)
                
                print(f"  평균 오차: {avg_error:.4f}")
                print(f"  최대 오차: {max_error:.4f}")
                
                if avg_error < 0.1:
                    print("  ✅ 매우 정확한 예측!")
                elif avg_error < 0.5:
                    print("  ✅ 정확한 예측")
                elif avg_error < 1.0:
                    print("  ⚠️ 약간 부정확")
                else:
                    print("  ❌ 부정확한 예측")
        else:
            print(f"  ❌ 근의 개수 불일치 (실제: {len(actual_roots)}, 예측: {len(predicted_roots)})")
    
    def plot_function_and_roots(self, a, b, c, actual_roots, predicted_roots):
        """함수와 근들을 시각화"""
        def f(x, a, b, c):
            """원래 함수"""
            try:
                denominator = x**4 + b*x**2 - c*x - 5
                if abs(denominator) < 1e-10:
                    return np.inf
                return np.exp(x) * x / denominator - a
            except:
                return np.inf
        
        # x 범위 설정
        x_range = max(10, max([abs(r) for r in actual_roots + predicted_roots] + [5]) * 1.5)
        x = np.linspace(-x_range, x_range, 2000)
        
        # y 값 계산
        y = []
        for xi in x:
            yi = f(xi, a, b, c)
            if abs(yi) > 50:  # 너무 큰 값은 자르기
                yi = np.sign(yi) * 50
            y.append(yi)
        
        # 그래프 그리기
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = e^x·x/(x^4+{b}x^2-{c}x-5) - {a}')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # 실제 근 표시
        if actual_roots:
            plt.scatter(actual_roots, [0]*len(actual_roots), 
                       color='red', s=100, marker='o', label=f'실제 근 ({len(actual_roots)}개)', zorder=5)
        
        # 예측 근 표시
        if predicted_roots:
            plt.scatter(predicted_roots, [0]*len(predicted_roots), 
                       color='green', s=100, marker='x', label=f'예측 근 ({len(predicted_roots)}개)', zorder=5)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'함수 그래프와 근의 비교\na={a}, b={b}, c={c}')
        plt.legend()
        plt.ylim(-10, 10)
        plt.show()
    
    def run_multiple_tests(self, test_cases=None, show_plots=False):
        """여러 테스트 케이스 실행"""
        if test_cases is None:
            # 기본 테스트 케이스들
            test_cases = [
                (1.0, 2.0, 3.0),
                (-0.5, 1.5, -2.0),
                (0.8, -1.0, 0.5),
                (2.5, 0.0, 1.0),
                (-1.2, 3.0, -1.5),
                (0.3, -0.8, 2.2),
                (1.8, 1.2, -0.7),
                (-0.9, -1.5, 1.8),
                (0.0, 1.0, 0.0),
                (3.0, -2.0, 4.0)
            ]
        
        print(f"🚀 {len(test_cases)}개 테스트 케이스 실행 시작!")
        
        correct_count = 0
        total_error = 0
        error_count = 0
        
        for i, (a, b, c) in enumerate(test_cases):
            try:
                print(f"\n[{i+1}/{len(test_cases)}] ", end="")
                
                # 실제 근 계산
                actual_roots = self.calculate_actual_roots(a, b, c)
                
                # 신경망 예측
                predicted_roots, probs = self.predict_roots(a, b, c)
                
                print(f"a={a}, b={b}, c={c}")
                print(f"  실제: {len(actual_roots)}개 {[f'{r:.3f}' for r in actual_roots]}")
                print(f"  예측: {len(predicted_roots)}개 {[f'{r:.3f}' for r in predicted_roots]}")
                
                # 정확도 체크
                if len(actual_roots) == len(predicted_roots):
                    correct_count += 1
                    if len(actual_roots) > 0:
                        actual_sorted = sorted(actual_roots)
                        predicted_sorted = sorted(predicted_roots)
                        error = np.mean([abs(a - p) for a, p in zip(actual_sorted, predicted_sorted)])
                        total_error += error
                        error_count += 1
                        print(f"  ✅ 정확! 평균오차: {error:.4f}")
                    else:
                        print(f"  ✅ 정확! (근 없음)")
                else:
                    print(f"  ❌ 근 개수 불일치")
                
                # 그래프 표시 (요청시)
                if show_plots:
                    self.plot_function_and_roots(a, b, c, actual_roots, predicted_roots)
                    
            except Exception as e:
                print(f"  ❌ 오류: {e}")
        
        # 전체 결과 요약
        print(f"\n{'='*60}")
        print(f"📊 전체 테스트 결과 요약")
        print(f"{'='*60}")
        print(f"전체 테스트: {len(test_cases)}개")
        print(f"근 개수 정확도: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.1f}%)")
        if error_count > 0:
            print(f"평균 근 값 오차: {total_error/error_count:.4f}")
        print(f"{'='*60}")
    
    def interactive_test(self):
        """대화형 테스트"""
        print("🎯 대화형 테스트 모드")
        print("방정식: e^x * x / (x^4 + bx^2 - cx - 5) - a = 0")
        print("종료하려면 'q' 입력")
        
        while True:
            try:
                print("\n" + "-"*40)
                user_input = input("a, b, c 값을 입력하세요 (예: 1.0, 2.0, 3.0): ").strip()
                
                if user_input.lower() == 'q':
                    print("👋 테스트 종료!")
                    break
                
                # 입력 파싱
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != 3:
                    print("❌ a, b, c 세 개의 값을 입력해주세요!")
                    continue
                
                a, b, c = values
                
                # 그래프 표시 여부 확인
                show_plot = input("그래프를 보시겠습니까? (y/n): ").strip().lower() == 'y'
                
                # 테스트 실행
                self.test_single_case(a, b, c, show_plot=show_plot)
                
            except ValueError:
                print("❌ 올바른 숫자 형식으로 입력해주세요!")
            except KeyboardInterrupt:
                print("\n👋 테스트 종료!")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

# 메인 실행 함수
def main():
    """메인 테스트 함수"""
    print("🧪 Neural Network Root Predictor Tester")
    print("=" * 50)
    
    # 테스터 초기화
    tester = RootTester()
    
    if tester.model is None:
        print("모델을 먼저 학습해주세요!")
        return
    
    while True:
        print("\n🎮 테스트 메뉴:")
        print("1. 기본 테스트 케이스 실행 (그래프 없이)")
        print("2. 기본 테스트 케이스 실행 (그래프 포함)")
        print("3. 단일 케이스 테스트")
        print("4. 대화형 테스트")
        print("5. 종료")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        if choice == '1':
            tester.run_multiple_tests(show_plots=False)
            
        elif choice == '2':
            tester.run_multiple_tests(show_plots=True)
            
        elif choice == '3':
            try:
                print("\n단일 케이스 테스트")
                a = float(input("a 값: "))
                b = float(input("b 값: "))
                c = float(input("c 값: "))
                show_plot = input("그래프 표시? (y/n): ").strip().lower() == 'y'
                tester.test_single_case(a, b, c, show_plot=show_plot)
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요!")
                
        elif choice == '4':
            tester.interactive_test()
            
        elif choice == '5':
            print("👋 프로그램 종료!")
            break
            
        else:
            print("❌ 올바른 선택지를 입력해주세요!")

if __name__ == "__main__":
    main()