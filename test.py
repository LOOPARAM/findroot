import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 변수 정의
x = sp.Symbol('x')

# 주어진 함수 정의
# f(x) = e^x / (x^3 + 5x + (-5)/x - 4) - 1 = 0
# 이를 정리하면: e^x / (x^3 + 5x - 5/x - 4) = 1
# 즉, e^x = x^3 + 5x - 5/x - 4

print("원래 함수:")
f_original = sp.exp(x) / (x**3 + 5*x + (-5)/x - 4) - 1
print("f(x) =", f_original)
print()

# 분모를 통분하여 정리
# x^3 + 5x - 5/x - 4 = (x^4 + 5x^2 - 5 - 4x) / x
f_simplified = sp.exp(x) * x / (x**4 + 5*x**2 - 4*x - 5) - 1
print("정리된 함수:")
print("f(x) =", f_simplified)
print()

# f(x) = 0이 되려면: e^x * x / (x^4 + 5x^2 - 4x - 5) = 1
# 즉, e^x * x = x^4 + 5x^2 - 4x - 5
equation = sp.Eq(sp.exp(x) * x, x**4 + 5*x**2 - 4*x - 5)
print("풀어야 할 방정식:")
print("e^x * x =", x**4 + 5*x**2 - 4*x - 5)
print()

# 1. 기호적 근 구하기
print("=" * 60)
print("1. 기호적 근 구하기 (solve)")
print("=" * 60)

try:
    symbolic_roots = sp.solve(equation, x)
    if symbolic_roots:
        print(f"기호적 근 {len(symbolic_roots)}개 찾음:")
        for i, root in enumerate(symbolic_roots, 1):
            print(f"{i}. x = {root}")
    else:
        print("기호적 근을 찾을 수 없습니다.")
except Exception as e:
    print(f"기호적 근 계산 중 오류: {e}")
    print("이 방정식은 초월방정식으로 기호적 해를 구하기 어렵습니다.")

print()

# 2. 수치해석적으로 모든 근 찾기
print("=" * 60)
print("2. 수치해석적 근 구하기 (nsolve)")
print("=" * 60)

# 더 넓은 범위에서 초기값 설정
initial_guesses = list(range(-10, 11)) + [0.1, 0.5, 1.5, 2.5, 3.5, -0.1, -0.5, -1.5, -2.5]
real_roots = []

print("실근:")
print("-" * 30)

for guess in initial_guesses:
    try:
        if guess != 0:  # x=0은 분모가 0이 되므로 제외
            root = sp.nsolve(f_simplified, guess)
            root_float = complex(root).real if hasattr(root, 'real') else float(root)
            
            # 중복 근 제거
            is_duplicate = False
            for existing_root in real_roots:
                if abs(root_float - existing_root) < 1e-8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                real_roots.append(root_float)
                
                # 함수값 확인
                f_value = complex(f_simplified.subs(x, root_float))
                print(f"x ≈ {root_float:.10f}")
                print(f"  검증: f(x) ≈ {abs(f_value):.2e}")
                print()
                
    except Exception as e:
        continue

print(f"실근 총 {len(real_roots)}개 발견")
print()

# 3. 복소근도 찾아보기
print("=" * 60)
print("3. 복소근 탐색")
print("=" * 60)

complex_roots = []
# 복소수 초기값들
complex_guesses = [
    1+1j, 1-1j, -1+1j, -1-1j, 2+1j, 2-1j, -2+1j, -2-1j,
    0.5+2j, 0.5-2j, -0.5+2j, -0.5-2j, 1+3j, 1-3j
]

print("복소근:")
print("-" * 30)

for guess in complex_guesses:
    try:
        root = sp.nsolve(f_simplified, guess)
        
        # 실근인지 복소근인지 확인
        if abs(sp.im(root)) > 1e-8:  # 허수부가 충분히 큰 경우
            root_complex = complex(root)
            
            # 중복 근 제거
            is_duplicate = False
            for existing_root in complex_roots:
                if abs(root_complex - existing_root) < 1e-8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                complex_roots.append(root_complex)
                
                # 함수값 확인
                f_value = f_simplified.subs(x, root)
                print(f"x ≈ {root_complex:.6f}")
                print(f"  검증: f(x) ≈ {abs(complex(f_value)):.2e}")
                print()
                
    except Exception as e:
        continue

print(f"복소근 총 {len(complex_roots)}개 발견")
print()

# 4. 결과 요약
print("=" * 60)
print("4. 모든 근 요약")
print("=" * 60)

all_roots = real_roots + complex_roots
real_roots.sort()

print("실근 (정렬됨):")
for i, root in enumerate(real_roots, 1):
    print(f"{i}. x ≈ {root:.10f}")

if complex_roots:
    print("\n복소근:")
    for i, root in enumerate(complex_roots, 1):
        print(f"{i}. x ≈ {root:.6f}")

print(f"\n총 근의 개수: {len(all_roots)}개 (실근 {len(real_roots)}개 + 복소근 {len(complex_roots)}개)")

# 5. 함수 그래프 그리기 (실근만)
print("\n" + "=" * 60)
print("5. 함수 그래프")
print("=" * 60)

def f_numpy(x_val):
    if x_val == 0:
        return np.inf
    try:
        return np.exp(x_val) * x_val / (x_val**4 + 5*x_val**2 - 4*x_val - 5) - 1
    except:
        return np.nan

# 그래프 범위 설정
if real_roots:
    x_min = min(real_roots) - 2
    x_max = max(real_roots) + 2
else:
    x_min, x_max = -5, 5

x_vals = np.linspace(x_min, x_max, 2000)
x_vals = x_vals[np.abs(x_vals) > 1e-10]  # x=0 근처 제외

y_vals = [f_numpy(x_val) for x_val in x_vals]

plt.figure(figsize=(14, 8))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3, label='x=0 (특이점)')

# 실근들을 그래프에 표시
for i, root in enumerate(real_roots):
    plt.plot(root, 0, 'ro', markersize=10, 
             label=f'근 {i+1}: x≈{root:.3f}' if i < 5 else '')

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('f(x) = exp(x)·x / (x⁴ + 5x² - 4x - 5) - 1', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-20, 20)
plt.show()

print("그래프에서 빨간 점들이 실근의 위치입니다.")