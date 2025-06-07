import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class PolynomialRootNet(nn.Module):
    def __init__(self, input_size=6, output_size=10, hidden_size=512, num_layers=4, dropout=0.3):
        super(PolynomialRootNet, self).__init__()
        
        layers = []
        current_size = input_size
        
        # 첫 번째 층
        layers.extend([
            nn.Linear(current_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout)
        ])
        current_size = hidden_size
        
        # 중간 층들
        for i in range(num_layers - 2):
            next_size = hidden_size // (2 ** (i + 1))
            next_size = max(next_size, 64)  # 최소 64개 뉴런
            
            layers.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU(),
                nn.BatchNorm1d(next_size),
                nn.Dropout(dropout * 0.8)  # 점진적으로 드롭아웃 감소
            ])
            current_size = next_size
        
        # 출력층
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class EnhancedPolynomialTester:
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        강화된 모델 테스터
        
        Args:
            model_path: 저장된 모델 파일 경로
            device: 'auto', 'cpu', 'cuda'
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        print(f"모델 로드 완료: {model_path}")
        print(f"사용 디바이스: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        if model_path.endswith('.jit'):
            # TorchScript 모델
            model = torch.jit.load(model_path, map_location=self.device)
        else:
            # PyTorch 또는 Pickle 모델
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
            
            # 모델 아키텍처 정보 추출
            arch_info = checkpoint.get('model_architecture', {})
            model = PolynomialRootNet(
                input_size=arch_info.get('input_size', 6),
                output_size=arch_info.get('output_size', 10),
                hidden_size=arch_info.get('hidden_size', 512),
                num_layers=arch_info.get('num_layers', 4)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        return model
    
    def predict_roots(self, coefficients: List[float]) -> Tuple[List[complex], np.ndarray]:
        """근 예측"""
        if len(coefficients) != 6:
            raise ValueError("5차 다항식의 계수 6개가 필요합니다.")
        
        # 텐서로 변환
        coeffs_tensor = torch.FloatTensor(coefficients).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_roots = self.model(coeffs_tensor)
            pred_roots = pred_roots.cpu().numpy()[0]
        
        # 복소수 형태로 변환
        complex_roots = []
        for i in range(5):
            imag = pred_roots[i*2]
            real = pred_roots[i*2 + 1]
            complex_roots.append(complex(real, imag))
        
        return complex_roots, pred_roots
    
    def verify_roots_numpy(self, coefficients: List[float], roots: List[complex]) -> List[complex]:
        """NumPy를 사용한 방정식 검증 (더 정확함)"""
        a, b, c, d, e, f = coefficients
        verification_results = []
        
        for root in roots:
            # f(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f
            result = (a * root**5 + b * root**4 + c * root**3 + 
                     d * root**2 + e * root + f)
            verification_results.append(result)
        
        return verification_results
    
    def get_known_polynomial_roots(self) -> List[Tuple[List[float], List[complex]]]:
        """알려진 근을 가진 테스트 다항식들"""
        test_cases = []
        
        # 1. 간단한 예제: (x-1)(x-2)(x-3)(x-4)(x-5) = 0
        # 전개하면: x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120 = 0
        coeffs1 = [1, -15, 85, -225, 274, -120]
        roots1 = [1+0j, 2+0j, 3+0j, 4+0j, 5+0j]
        test_cases.append((coeffs1, roots1))
        
        # 2. 복소수 근 포함: (x-1)(x-i)(x+i)(x-2)(x-3) = 0
        # (x-1)(x^2+1)(x-2)(x-3) = (x-1)(x-2)(x-3)(x^2+1)
        # = (x^3-6x^2+11x-6)(x^2+1) = x^5 - 6x^4 + 12x^3 - 6x^2 + 11x - 6
        coeffs2 = [1, -6, 12, -6, 11, -6]
        roots2 = [1+0j, 0+1j, 0-1j, 2+0j, 3+0j]
        test_cases.append((coeffs2, roots2))
        
        # 3. 더 간단한 예제: x^5 - 1 = 0 (5차 단위근)
        coeffs3 = [1, 0, 0, 0, 0, -1]
        roots3 = []
        for k in range(5):
            angle = 2 * np.pi * k / 5
            root = complex(np.cos(angle), np.sin(angle))
            roots3.append(root)
        test_cases.append((coeffs3, roots3))
        
        # 4. x^5 = 0 (5중근 0)
        coeffs4 = [1, 0, 0, 0, 0, 0]
        roots4 = [0+0j, 0+0j, 0+0j, 0+0j, 0+0j]
        test_cases.append((coeffs4, roots4))
        
        # 5. (x-1)^5 = 0 (5중근 1)
        # 전개: x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 = 0
        coeffs5 = [1, -5, 10, -10, 5, -1]
        roots5 = [1+0j, 1+0j, 1+0j, 1+0j, 1+0j]
        test_cases.append((coeffs5, roots5))
        
        return test_cases
    
    def test_known_polynomials(self):
        """알려진 근을 가진 다항식들로 테스트"""
        print(f"\n{'='*60}")
        print("알려진 근을 가진 다항식 테스트")
        print(f"{'='*60}")
        
        test_cases = self.get_known_polynomial_roots()
        
        for i, (coeffs, true_roots) in enumerate(test_cases):
            print(f"\n--- 테스트 케이스 {i+1} ---")
            
            # 방정식 출력
            self._print_equation(coeffs)
            
            # 예측
            pred_roots, raw_pred = self.predict_roots(coeffs)
            
            # 실제 근 출력
            print("실제 근들:")
            for j, root in enumerate(true_roots):
                if abs(root.imag) < 1e-10:
                    print(f"  근 {j+1}: {root.real:.6f}")
                else:
                    print(f"  근 {j+1}: {root.real:.6f} + {root.imag:.6f}i")
            
            # 예측 근 출력
            print("예측 근들:")
            for j, root in enumerate(pred_roots):
                if abs(root.imag) < 1e-6:
                    print(f"  근 {j+1}: {root.real:.6f}")
                else:
                    print(f"  근 {j+1}: {root.real:.6f} + {root.imag:.6f}i")
            
            # 오차 계산 (순서를 맞춰서)
            errors = self._calculate_root_errors(pred_roots, true_roots)
            print(f"\n근별 최소 오차:")
            for j, error in enumerate(errors):
                print(f"  근 {j+1}: {error:.6f}")
            
            avg_error = np.mean(errors)
            print(f"평균 오차: {avg_error:.6f}")
            
            # 방정식 검증
            verification = self.verify_roots_numpy(coeffs, pred_roots)
            print(f"\n방정식 검증:")
            for j, result in enumerate(verification):
                magnitude = abs(result)
                print(f"  f(근 {j+1}) = {result:.6f}, |f(x)| = {magnitude:.6f}")
            
            avg_verification = np.mean([abs(v) for v in verification])
            print(f"평균 |f(x)|: {avg_verification:.6f}")
            
            # 성능 평가
            if avg_error < 0.1:
                print("✅ 성능: 우수")
            elif avg_error < 1.0:
                print("⚠️ 성능: 보통")
            else:
                print("❌ 성능: 나쁨")
    
    def _calculate_root_errors(self, pred_roots: List[complex], true_roots: List[complex]) -> List[float]:
        """근들 사이의 최소 오차 계산 (순서 무관)"""
        errors = []
        used_true_indices = set()
        
        for pred_root in pred_roots:
            min_error = float('inf')
            best_true_idx = -1
            
            for true_idx, true_root in enumerate(true_roots):
                if true_idx in used_true_indices:
                    continue
                
                error = abs(pred_root - true_root)
                if error < min_error:
                    min_error = error
                    best_true_idx = true_idx
            
            if best_true_idx != -1:
                used_true_indices.add(best_true_idx)
                errors.append(min_error)
            else:
                errors.append(float('inf'))
        
        return errors
    
    def _print_equation(self, coefficients: List[float]):
        """방정식을 보기 좋게 출력"""
        a, b, c, d, e, f = coefficients
        terms = []
        
        if a != 0:
            if a == 1:
                terms.append("x^5")
            elif a == -1:
                terms.append("-x^5")
            else:
                terms.append(f"{a:.3f}x^5")
        
        if b != 0:
            if b > 0 and terms:
                terms.append(f" + {b:.3f}x^4")
            elif b < 0:
                terms.append(f" - {abs(b):.3f}x^4")
            else:
                terms.append(f"{b:.3f}x^4")
        
        if c != 0:
            if c > 0 and terms:
                terms.append(f" + {c:.3f}x^3")
            elif c < 0:
                terms.append(f" - {abs(c):.3f}x^3")
            else:
                terms.append(f"{c:.3f}x^3")
        
        if d != 0:
            if d > 0 and terms:
                terms.append(f" + {d:.3f}x^2")
            elif d < 0:
                terms.append(f" - {abs(d):.3f}x^2")
            else:
                terms.append(f"{d:.3f}x^2")
        
        if e != 0:
            if e > 0 and terms:
                terms.append(f" + {e:.3f}x")
            elif e < 0:
                terms.append(f" - {abs(e):.3f}x")
            else:
                terms.append(f"{e:.3f}x")
        
        if f != 0:
            if f > 0 and terms:
                terms.append(f" + {f:.3f}")
            elif f < 0:
                terms.append(f" - {abs(f):.3f}")
            else:
                terms.append(f"{f:.3f}")
        
        equation = "".join(terms) + " = 0"
        print(f"방정식: {equation}")
    
    def analyze_model_behavior(self):
        """모델 동작 분석"""
        print(f"\n{'='*60}")
        print("모델 동작 분석")
        print(f"{'='*60}")
        
        # 1. 모델 정보
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"모델 파라미터:")
        print(f"  총 파라미터: {total_params:,}")
        print(f"  훈련 가능한 파라미터: {trainable_params:,}")
        
        # 2. 입력 크기별 출력 확인
        print(f"\n입력 크기별 출력 테스트:")
        test_inputs = [
            [1, 0, 0, 0, 0, 0],  # x^5
            [0, 1, 0, 0, 0, 0],  # x^4
            [0, 0, 1, 0, 0, 0],  # x^3
            [0, 0, 0, 1, 0, 0],  # x^2
            [0, 0, 0, 0, 1, 0],  # x
            [0, 0, 0, 0, 0, 1],  # 상수
        ]
        
        for i, input_coeffs in enumerate(test_inputs):
            pred_roots, _ = self.predict_roots(input_coeffs)
            print(f"  입력 {i+1} {input_coeffs}: 출력 범위 [{min(abs(r) for r in pred_roots):.3f}, {max(abs(r) for r in pred_roots):.3f}]")
        
        # 3. 모델의 안정성 테스트 (같은 입력에 대한 일관성)
        print(f"\n모델 안정성 테스트:")
        test_coeffs = [1, -2, 3, -1, 2, -1]
        results = []
        
        for _ in range(5):
            pred_roots, _ = self.predict_roots(test_coeffs)
            results.append(pred_roots)
        
        # 결과 일관성 확인
        max_variation = 0
        for i in range(5):  # 각 근에 대해
            values = [results[j][i] for j in range(5)]  # 5번의 예측에서 i번째 근
            variations = [abs(values[k] - values[0]) for k in range(1, 5)]
            max_var = max(variations) if variations else 0
            max_variation = max(max_variation, max_var)
        
        print(f"  최대 예측 변동: {max_variation:.6f}")
        if max_variation < 1e-6:
            print("  ✅ 모델이 안정적입니다")
        else:
            print("  ⚠️ 모델에 약간의 불안정성이 있습니다")
    
    def quick_performance_test(self):
        """빠른 성능 테스트"""
        print(f"\n{'='*60}")
        print("빠른 성능 테스트")
        print(f"{'='*60}")
        
        # 간단한 테스트 케이스들
        simple_tests = [
            # [계수들, 설명]
            ([1, 0, 0, 0, 0, -1], "x^5 - 1 = 0 (5차 단위근)"),
            ([1, 0, 0, 0, 0, 0], "x^5 = 0 (5중근 0)"),
            ([1, -5, 10, -10, 5, -1], "(x-1)^5 = 0 (5중근 1)"),
            ([1, -15, 85, -225, 274, -120], "(x-1)(x-2)(x-3)(x-4)(x-5) = 0"),
        ]
        
        total_score = 0
        max_score = len(simple_tests)
        
        for i, (coeffs, description) in enumerate(simple_tests):
            print(f"\n테스트 {i+1}: {description}")
            
            pred_roots, _ = self.predict_roots(coeffs)
            verification = self.verify_roots_numpy(coeffs, pred_roots)
            avg_verification = np.mean([abs(v) for v in verification])
            
            print(f"  평균 |f(x)|: {avg_verification:.6f}")
            
            # 점수 계산 (낮을수록 좋음)
            if avg_verification < 0.01:
                score = 1.0
                grade = "A"
            elif avg_verification < 0.1:
                score = 0.8
                grade = "B"
            elif avg_verification < 1.0:
                score = 0.6
                grade = "C"
            elif avg_verification < 10.0:
                score = 0.4
                grade = "D"
            else:
                score = 0.0
                grade = "F"
            
            total_score += score
            print(f"  점수: {grade} ({score:.1f}/1.0)")
        
        final_score = (total_score / max_score) * 100
        print(f"\n총 점수: {final_score:.1f}/100")
        
        if final_score >= 90:
            print("🎉 모델 성능: 우수")
        elif final_score >= 70:
            print("👍 모델 성능: 양호")
        elif final_score >= 50:
            print("⚠️ 모델 성능: 보통")
        else:
            print("❌ 모델 성능: 개선 필요")
    
    def interactive_test(self):
        """대화형 테스트"""
        print(f"\n{'='*60}")
        print("대화형 다항식 근 예측 테스트")
        print("5차 다항식: ax^5 + bx^4 + cx^3 + dx^2 + ex + f = 0")
        print("'quit' 또는 'exit'을 입력하면 종료됩니다.")
        print("'known'을 입력하면 알려진 테스트 케이스 목록을 봅니다.")
        print(f"{'='*60}")
        
        known_cases = self.get_known_polynomial_roots()
        
        while True:
            try:
                print("\n다항식 계수를 입력하세요 (a b c d e f):")
                user_input = input("계수들 (공백으로 구분): ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("테스트를 종료합니다.")
                    break
                
                if user_input.lower() == 'known':
                    print("\n알려진 테스트 케이스들:")
                    descriptions = [
                        "x^5 - 1 = 0 (5차 단위근)",
                        "복소수 근 포함 케이스",
                        "(x-1)(x-2)(x-3)(x-4)(x-5) = 0",
                        "x^5 = 0 (5중근 0)",
                        "(x-1)^5 = 0 (5중근 1)"
                    ]
                    for i, (coeffs, desc) in enumerate(zip([case[0] for case in known_cases], descriptions)):
                        print(f"  {i+1}. {desc}")
                        print(f"     계수: {' '.join(map(str, coeffs))}")
                    continue
                
                # 계수 파싱
                coeffs = list(map(float, user_input.split()))
                
                if len(coeffs) != 6:
                    print("6개의 계수가 필요합니다. 다시 입력해주세요.")
                    continue
                
                # 예측 수행
                print(f"\n--- 예측 결과 ---")
                self._print_equation(coeffs)
                
                pred_roots, _ = self.predict_roots(coeffs)
                
                print("예측된 근들:")
                for i, root in enumerate(pred_roots):
                    if abs(root.imag) < 1e-6:
                        print(f"  근 {i+1}: {root.real:.6f}")
                    else:
                        print(f"  근 {i+1}: {root.real:.6f} + {root.imag:.6f}i")
                
                # 방정식 검증
                verification = self.verify_roots_numpy(coeffs, pred_roots)
                print(f"\n방정식 검증:")
                for i, result in enumerate(verification):
                    magnitude = abs(result)
                    print(f"  f(근 {i+1}) = {result:.6f}, |f(x)| = {magnitude:.6f}")
                
                avg_verification = np.mean([abs(v) for v in verification])
                print(f"평균 |f(x)|: {avg_verification:.6f}")
                
                # 성능 평가
                if avg_verification < 0.01:
                    print("✅ 예측 품질: 우수")
                elif avg_verification < 0.1:
                    print("👍 예측 품질: 양호")
                elif avg_verification < 1.0:
                    print("⚠️ 예측 품질: 보통")
                else:
                    print("❌ 예측 품질: 나쁨")
                
            except ValueError:
                print("숫자 형식이 올바르지 않습니다. 다시 입력해주세요.")
            except KeyboardInterrupt:
                print("\n\n테스트를 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")

def main():
    """메인 함수"""
    # 설정
    model_path = "try_1/model_1.pth"  # 모델 파일 경로
    
    # 모델 파일 확인
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("가능한 모델 파일들을 찾는 중...")
        
        found_models = []
        for ext in ['.pth', '.pt', '.pkl', '.jit']:
            for name in ['best_polynomial_model', 'polynomial_model', 'model']:
                alt_path = f"{name}{ext}"
                if os.path.exists(alt_path):
                    found_models.append(alt_path)
        
        if found_models:
            print("발견된 모델 파일들:")
            for i, model_file in enumerate(found_models):
                print(f"  {i+1}. {model_file}")
            
            try:
                choice = int(input("사용할 모델 번호를 선택하세요: ")) - 1
                if 0 <= choice < len(found_models):
                    model_path = found_models[choice]
                else:
                    print("잘못된 선택입니다.")
                    return
            except ValueError:
                model_path = found_models[0]
                print(f"첫 번째 모델을 사용합니다: {model_path}")
        else:
            print("모델 파일이 없습니다. 먼저 모델을 훈련시켜주세요.")
            return
    
    # 테스터 생성
    try:
        tester = EnhancedPolynomialTester(model_path)
    except Exception as e:
        print(f"모델 로딩 중 오류: {e}")
        return
    
    # 메뉴 시스템
    while True:
        print(f"\n{'='*60}")
        print("다항식 근 예측 모델 테스터")
        print(f"{'='*60}")
        print("1. 빠른 성능 테스트")
        print("2. 알려진 근을 가진 다항식 테스트")
        print("3. 모델 동작 분석")
        print("4. 대화형 테스트")
        print("5. 종료")
        print(f"{'='*60}")
        
        try:
            choice = input("선택하세요 (1-5): ").strip()
            
            if choice == '1':
                tester.quick_performance_test()
            elif choice == '2':
                tester.test_known_polynomials()
            elif choice == '3':
                tester.analyze_model_behavior()
            elif choice == '4':
                tester.interactive_test()
            elif choice == '5':
                print("프로그램을 종료합니다.")
                break
            else:
                print("올바른 번호를 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()