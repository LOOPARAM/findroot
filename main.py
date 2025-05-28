import math

e = math.e

def bisection_method(func, a, b, tolerance=1e-9, max_iterations=100):
    """
    이분법을 사용하여 방정식의 근을 찾는 함수
    
    Parameters:
    func: 함수 (f(x) = 0의 근을 찾을 함수)
    a, b: 초기 구간 [a, b]
    tolerance: 허용 오차 (기본값: 1e-6)
    max_iterations: 최대 반복 횟수 (기본값: 100)
    
    Returns:
    근의 근사값, 반복 횟수, 수렴 여부
    """
    
    # 초기 조건 확인: f(a) * f(b) < 0 이어야 함
    try:
        fa = func(a)
    except:
        return None, 0, False, 2, "정의역에서 벗어난 a값을 사용"
    
    try:
        fb = func(b)
    except:
        return None, 0, False, 2, "정의역에서 벗어난 b값을 사용"
    
    if fa * fb >= 0:
        return None, 0, False, 1, "f(a)와 f(b)의 부호가 같습니다. 구간을 다시 설정해주세요."
    
    print(f"초기 구간: [{a:.6f}, {b:.6f}]")
    print(f"f({a:.6f}) = {fa:.6f}, f({b:.6f}) = {fb:.6f}")
    print("-" * 60)
    print(f"{'반복':<4} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'|b-a|':<12}")
    print("-" * 60)
    
    for i in range(max_iterations):
        # 중점 계산
        c = (a + b) / 2
        fc = func(c)
        
        # 현재 상태 출력
        print(f"{i+1:<4} {a:<12.6f} {b:<12.6f} {c:<12.6f} {fc:<12.6f} {abs(b-a):<12.6f}")
        
        # 수렴 조건 확인
        if abs(fc) < tolerance or abs(b - a) < tolerance:
            print("-" * 60)
            print(f"수렴! 근의 근사값: {c:.8f}")
            return c, i + 1, True, 0,"성공"
        
        # 새로운 구간 설정
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    # 최대 반복 횟수 도달
    print("-" * 60)
    print(f"최대 반복 횟수에 도달했습니다. 현재 근사값: {c:.8f}")
    return c, max_iterations, False, 3, "최대 반복 횟수 도달"

def Bisection_Manager(func, a, b, tolerance=1e-9, max_iterations=100):
    # 이분법으로 구하기
    print("=== 이분법 예제 1: f(x) = ln(x)/(x^3-x-1) + sin(x) ===")
    root, iterations, converged, message, worktodo = bisection_method(example4, -100, 100)
    print(f"결과: {message}")
    if converged:
        print(f"검증: f({root:.8f}) = {example1(root):.8e}")
    
    print("\n" + "="*80 + "\n")

# 예제 함수들
def example1(x):
    """f(x) = x^3 - x - 1"""
    return x**3 - x - 1

def example2(x):
    """f(x) = cos(x) - x"""
    return math.cos(x) - x

def example3(x):
    """f(x) = e^x - 2x - 1"""
    return math.exp(x) - 2*x - 1

def example4(x):
    """f(x) = ln(x)/(x^3-x-1) + sin(x)"""
    return (math.log(x,e))/(x**3-x-1) + math.sin(x)

# 사용 예제
if __name__ == "__main__":
    