import sympy as sp
from sympy import symbols, solve
import random
import json
import pickle

class PolynomialDataGenerator:
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples
        self.data = []
    
    def generate_data(self):
        print("5차 방정식 데이터 생성 중...")
        x = symbols('x')
        
        for i in range(self.num_samples * 3):  # 더 많이 시도해서 유효한 데이터 확보
            print(f"시도 중: {i}/{self.num_samples * 3} (현재 유효 데이터: {len(self.data)}개)")
            
            # 계수를 랜덤하게 생성 (-100 ~ 100 범위)
            coeffs = [random.uniform(-100, 100) for _ in range(6)]
            # 최고차항은 0이 아니도록 보장
            coeffs[0] = random.uniform(0.1, 100) if random.random() > 0.1 else random.uniform(-100, -0.1)
            
            # 5차 방정식 생성: ax^5 + bx^4 + cx^3 + dx^2 + ex + f = 0
            poly = coeffs[0]*x**5 + coeffs[1]*x**4 + coeffs[2]*x**3 + coeffs[3]*x**2 + coeffs[4]*x + coeffs[5]
            
            try:
                # sympy로 근 구하기
                roots = solve(poly, x)
                
                # 정확히 5개의 근이 나와야 함
                if len(roots) != 5:
                    continue
                
                # 복소수 형태로 변환하여 실부와 허부 분리
                root_pairs = []
                valid = True
                
                for root in roots:
                    try:
                        # 복소수로 변환
                        complex_root = complex(root)
                        real_part = float(complex_root.real)
                        imag_part = float(complex_root.imag)
                        
                        # 너무 큰 값은 제외 (수치적 불안정성 방지)
                        if abs(real_part) > 100 or abs(imag_part) > 100:
                            valid = False
                            break
                        
                        # NaN이나 inf 체크
                        if not (abs(real_part) < float('inf') and abs(imag_part) < float('inf')):
                            valid = False
                            break
                            
                        # [허수부, 실수부] 순서로 저장
                        root_pairs.extend([imag_part, real_part])
                        
                    except Exception as e:
                        valid = False
                        break
                
                # 유효한 데이터만 저장
                if valid and len(root_pairs) == 10:
                    data_point = {
                        'coefficients': coeffs,  # [a, b, c, d, e, f]
                        'roots': root_pairs,     # [im1, re1, im2, re2, im3, re3, im4, re4, im5, re5]
                        'equation': f"{coeffs[0]:.3f}x^5 + {coeffs[1]:.3f}x^4 + {coeffs[2]:.3f}x^3 + {coeffs[3]:.3f}x^2 + {coeffs[4]:.3f}x + {coeffs[5]:.3f} = 0"
                    }
                    self.data.append(data_point)
                    
                # 목표 개수 달성하면 종료
                if len(self.data) >= self.num_samples:
                    break
                    
            except Exception as e:
                # 근을 구할 수 없는 경우 건너뛰기
                continue
        
        print(f"데이터 생성 완료! 총 {len(self.data)}개의 유효한 데이터")
        return self.data
    
    def save_data(self, filename_base="polynomial_data"):
        """데이터를 여러 형식으로 저장"""
        if not self.data:
            print("저장할 데이터가 없습니다!")
            return
        
        # JSON 형식으로 저장
        json_filename = f"{filename_base}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"JSON 파일 저장: {json_filename}")
        
        # Pickle 형식으로 저장 (Python에서 빠른 로딩용)
        pickle_filename = f"{filename_base}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Pickle 파일 저장: {pickle_filename}")
        
        # 텍스트 파일로 간단한 요약 저장
        txt_filename = f"{filename_base}_summary.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"5차 방정식 데이터셋 요약\n")
            f.write(f"={'='*40}\n")
            f.write(f"총 데이터 개수: {len(self.data)}\n")
            f.write(f"데이터 형식: 계수 6개 -> 근 10개 (허수부, 실수부 쌍)\n\n")
            
            f.write("처음 5개 예시:\n")
            for i, item in enumerate(self.data[:5]):
                f.write(f"\n예시 {i+1}:\n")
                f.write(f"방정식: {item['equation']}\n")
                f.write(f"계수: {item['coefficients']}\n")
                f.write("근들:\n")
                roots = item['roots']
                for j in range(5):
                    imag = roots[j*2]
                    real = roots[j*2 + 1]
                    if abs(imag) < 1e-6:
                        f.write(f"  근 {j+1}: {real:.6f}\n")
                    else:
                        f.write(f"  근 {j+1}: {real:.6f} + {imag:.6f}i\n")
        
        print(f"요약 파일 저장: {txt_filename}")
    
    def load_data(self, filename):
        """저장된 데이터 불러오기"""
        if filename.endswith('.json'):
            with open(filename, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. (.json 또는 .pkl 사용)")
        
        print(f"데이터 로드 완료: {len(self.data)}개")
        return self.data
    
    def get_stats(self):
        """데이터셋 통계 출력"""
        if not self.data:
            print("데이터가 없습니다!")
            return
        
        print(f"\n데이터셋 통계:")
        print(f"총 데이터 개수: {len(self.data)}")
        
        # 실근과 복소근 분석
        real_root_counts = []
        complex_root_counts = []
        
        for item in self.data:
            roots = item['roots']
            real_count = 0
            complex_count = 0
            
            for i in range(5):
                imag = roots[i*2]
                if abs(imag) < 1e-6:  # 실근
                    real_count += 1
                else:  # 복소근
                    complex_count += 1
            
            real_root_counts.append(real_count)
            complex_root_counts.append(complex_count)
        
        print(f"평균 실근 개수: {sum(real_root_counts)/len(real_root_counts):.2f}")
        print(f"평균 복소근 개수: {sum(complex_root_counts)/len(complex_root_counts):.2f}")
        
        # 실근 개수별 분포
        from collections import Counter
        real_dist = Counter(real_root_counts)
        print(f"\n실근 개수 분포:")
        for count in sorted(real_dist.keys()):
            print(f"  {count}개 실근: {real_dist[count]}개 방정식 ({real_dist[count]/len(self.data)*100:.1f}%)")

def main():
    # 데이터 생성기 생성
    generator = PolynomialDataGenerator(num_samples=10000)
    
    # 데이터 생성
    data = generator.generate_data()
    
    # 통계 출력
    generator.get_stats()
    
    # 데이터 저장
    generator.save_data("polynomial_dataset")
    
    print("\n데이터 생성 및 저장 완료!")
    print("생성된 파일들:")
    print("- polynomial_dataset.json (JSON 형식)")
    print("- polynomial_dataset.pkl (Python Pickle 형식)")
    print("- polynomial_dataset_summary.txt (요약 텍스트)")

if __name__ == "__main__":
    main()