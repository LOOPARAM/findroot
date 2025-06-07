import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import glob
import time
from typing import List, Dict, Optional, Tuple
from statistics import mean, stdev

# 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class PolynomialRootNet(nn.Module):
    """원래 모델 클래스"""
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
            next_size = max(next_size, 64)
            
            layers.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU(),
                nn.BatchNorm1d(next_size),
                nn.Dropout(dropout * 0.8)
            ])
            current_size = next_size
        
        # 출력층
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class EnhancedModelTester:
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.model_info = {}
        
        print(f"🔬 Enhanced Model Tester - Device: {device}")
        
    def find_model_files(self, directory="."):
        """모델 파일 찾기"""
        patterns = [
            '**/*.pth',
            '**/*.pt', 
            '**/*.pkl',
            '**/best_polynomial_model.pth',
            '**/best_*.pt',
            '**/model_*.pth',
            '**/model_*.pt',
            '**/polynomial_*.pth',
            '**/polynomial_*.pt'
        ]
        found_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            found_files.extend(files)
        
        # 중복 제거
        found_files = list(set(found_files))
        
        print(f"📁 Found {len(found_files)} model files:")
        for i, file in enumerate(found_files):
            size_mb = os.path.getsize(file) / (1024*1024)
            print(f"  {i+1}. {file} ({size_mb:.1f} MB)")
        
        return found_files
    
    def safe_load_model(self, model_path: str):
        """안전한 모델 로드 (다양한 케이스 처리)"""
        model_name = os.path.basename(model_path).split('.')[0]
        
        print(f"\n🔧 Loading: {model_path}")
        
        try:
            # 파일 형식에 따라 로드
            if model_path.endswith(('.pth', '.pt')):
                data = torch.load(model_path, map_location=self.device)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f"❌ Unsupported file format: {model_path}")
                return False
            
            # 데이터 구조 분석
            if isinstance(data, dict):
                print(f"   Data keys: {list(data.keys())}")
            
            # 모델 상태 추출 시도
            model_state = None
            arch_info = {'hidden_size': 512, 'num_layers': 4}  # 기본값
            
            if isinstance(data, dict):
                # 일반적인 체크포인트 형식
                if 'model_state_dict' in data:
                    model_state = data['model_state_dict']
                elif 'state_dict' in data:
                    model_state = data['state_dict']
                elif 'model' in data:
                    model_state = data['model']
                else:
                    # 데이터 자체가 state_dict일 수 있음
                    if any('weight' in str(key) for key in data.keys()):
                        model_state = data
                
                # 아키텍처 정보 추출
                if 'model_architecture' in data:
                    arch_info.update(data['model_architecture'])
                elif 'architecture' in data:
                    arch_info.update(data['architecture'])
                
            else:
                print(f"❌ Unexpected data type: {type(data)}")
                return False
            
            if model_state is None:
                print(f"❌ Could not find model state in {model_path}")
                return False
            
            # 모델 생성 및 로드
            hidden_size = arch_info.get('hidden_size', 512)
            num_layers = arch_info.get('num_layers', 4)
            
            model = PolynomialRootNet(
                hidden_size=hidden_size,
                num_layers=num_layers
            ).to(self.device)
            
            # 상태 로드
            model.load_state_dict(model_state)
            model.eval()
            
            # 저장
            self.models[model_name] = model
            self.model_info[model_name] = {
                'path': model_path,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'parameters': sum(p.numel() for p in model.parameters()),
                'file_size_mb': os.path.getsize(model_path) / (1024*1024)
            }
            
            print(f"   ✅ Success: {hidden_size}h-{num_layers}l ({self.model_info[model_name]['parameters']:,} params)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    def load_all_models(self, directory="."):
        """모든 모델 로드"""
        model_files = self.find_model_files(directory)
        
        if not model_files:
            print("❌ No model files found!")
            return False
        
        success_count = 0
        for model_path in model_files:
            if self.safe_load_model(model_path):
                success_count += 1
        
        print(f"\n📊 Loaded {success_count}/{len(model_files)} models successfully")
        return success_count > 0
    
    def list_loaded_models(self):
        """로드된 모델 목록"""
        if not self.models:
            print("❌ No models loaded!")
            return
        
        print(f"\n{'='*80}")
        print(f"📋 Loaded Models ({len(self.models)})")
        print(f"{'='*80}")
        
        for i, (name, info) in enumerate(self.model_info.items()):
            print(f"{i+1}. {name}")
            print(f"   📐 Architecture: {info.get('hidden_size', 'Unknown')}h-{info.get('num_layers', 'Unknown')}l")
            print(f"   🔢 Parameters: {info.get('parameters', 'Unknown'):,}")
            print(f"   💾 File Size: {info.get('file_size_mb', 0):.1f} MB")
            print()
    
    def create_test_polynomial(self):
        """테스트용 다항식 생성"""
        # 5차 다항식 계수 생성
        coeffs = np.random.uniform(-2, 2, 6)
        coeffs[0] = np.random.uniform(0.5, 2) * np.random.choice([-1, 1])  # 최고차항
        return coeffs
    
    def solve_polynomial_true(self, coeffs):
        """실제 근 계산"""
        roots = np.roots(coeffs)
        
        # [허수부, 실수부] 형태로 변환
        formatted_roots = []
        for root in roots:
            formatted_roots.extend([root.imag, root.real])
        
        return np.array(formatted_roots)
    
    def single_model_test(self, model_name, coeffs, true_roots):
        """단일 모델에 대한 단일 테스트"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        input_tensor = torch.FloatTensor(coeffs).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            pred_roots = model(input_tensor).cpu().numpy()[0]
        inference_time = time.time() - start_time
        
        # MSE 계산
        mse = np.mean((pred_roots - true_roots) ** 2)
        
        return {
            'mse': mse,
            'inference_time': inference_time,
            'predictions': pred_roots
        }
    
    def multi_run_test(self, num_runs=5, num_polynomials=20):
        """여러 번 실행하여 평균 계산"""
        if not self.models:
            print("❌ No models loaded!")
            return None
        
        print(f"\n{'='*80}")
        print(f"🎯 Multi-Run Test")
        print(f"{'='*80}")
        print(f"📊 Configuration:")
        print(f"   • Runs per model: {num_runs}")
        print(f"   • Polynomials per run: {num_polynomials}")
        print(f"   • Total tests per model: {num_runs * num_polynomials}")
        print(f"   • Device: {self.device}")
        print()
        
        all_results = {}
        
        for model_name in self.models.keys():
            print(f"🔧 Testing: {model_name}")
            
            run_results = []
            
            for run in range(num_runs):
                print(f"   Run {run+1}/{num_runs}...", end=" ")
                
                # 각 런마다 새로운 다항식들로 테스트
                mse_list = []
                time_list = []
                
                for _ in range(num_polynomials):
                    coeffs = self.create_test_polynomial()
                    true_roots = self.solve_polynomial_true(coeffs)
                    result = self.single_model_test(model_name, coeffs, true_roots)
                    
                    if result:
                        mse_list.append(result['mse'])
                        time_list.append(result['inference_time'])
                
                # 이번 런의 평균
                run_avg_mse = np.mean(mse_list)
                run_avg_time = np.mean(time_list)
                
                run_results.append({
                    'avg_mse': run_avg_mse,
                    'avg_time': run_avg_time,
                    'mse_list': mse_list,
                    'time_list': time_list
                })
                
                print(f"MSE: {run_avg_mse:.6f}")
            
            # 전체 런들의 통계 계산
            avg_mses = [r['avg_mse'] for r in run_results]
            avg_times = [r['avg_time'] for r in run_results]
            
            # 모든 개별 MSE들을 모아서 전체 통계도 계산
            all_mses = []
            all_times = []
            for r in run_results:
                all_mses.extend(r['mse_list'])
                all_times.extend(r['time_list'])
            
            model_stats = {
                'run_avg_mse_mean': np.mean(avg_mses),
                'run_avg_mse_std': np.std(avg_mses),
                'run_avg_time_mean': np.mean(avg_times),
                'run_avg_time_std': np.std(avg_times),
                'overall_mse_mean': np.mean(all_mses),
                'overall_mse_std': np.std(all_mses),
                'overall_time_mean': np.mean(all_times),
                'overall_time_std': np.std(all_times),
                'min_mse': np.min(all_mses),
                'max_mse': np.max(all_mses),
                'num_runs': num_runs,
                'num_polynomials': num_polynomials,
                'total_tests': len(all_mses)
            }
            
            all_results[model_name] = model_stats
            print(f"   ✅ Complete: avg MSE = {model_stats['overall_mse_mean']:.6f} ± {model_stats['overall_mse_std']:.6f}")
            print()
        
        return all_results
    
    def print_detailed_results(self, results):
        """상세 결과 출력"""
        if not results:
            return
        
        print(f"{'='*120}")
        print(f"📊 DETAILED MULTI-RUN RESULTS")
        print(f"{'='*120}")
        
        # 헤더
        header = (f"{'Model':<25} {'Hidden':<8} {'Params':<10} {'Size(MB)':<9} "
                 f"{'MSE(avg±std)':<18} {'Time(avg±std)':<18} {'Min/Max MSE':<16} {'CV%':<8}")
        print(header)
        print("-" * 120)
        
        # 각 모델 결과
        for model_name, stats in results.items():
            info = self.model_info[model_name]
            
            model_display = model_name[:24]
            hidden_size = info.get('hidden_size', 'Unknown')
            params = f"{info.get('parameters', 0):,}"[:9]
            size_mb = f"{info.get('file_size_mb', 0):.1f}"
            
            # MSE 통계
            mse_mean = stats['overall_mse_mean']
            mse_std = stats['overall_mse_std']
            mse_str = f"{mse_mean:.4f}±{mse_std:.4f}"
            
            # 시간 통계
            time_mean = stats['overall_time_mean']
            time_std = stats['overall_time_std']
            time_str = f"{time_mean:.4f}±{time_std:.4f}"
            
            # Min/Max
            min_max_str = f"{stats['min_mse']:.4f}/{stats['max_mse']:.4f}"
            
            # CV (Coefficient of Variation)
            cv = (mse_std / mse_mean * 100) if mse_mean > 0 else 0
            cv_str = f"{cv:.1f}%"
            
            row = (f"{model_display:<25} {hidden_size:<8} {params:<10} {size_mb:<9} "
                  f"{mse_str:<18} {time_str:<18} {min_max_str:<16} {cv_str:<8}")
            print(row)
        
        print("-" * 120)
        first_result = next(iter(results.values()))
        print(f"📈 Statistics based on {first_result['num_runs']} runs × {first_result['num_polynomials']} polynomials = {first_result['total_tests']} total tests per model")
        print("   • MSE: Lower is better | Time: Inference time in seconds | CV: Coefficient of Variation (lower = more stable)")
        print()
        
        # 랭킹
        print("🏆 RANKINGS:")
        
        # MSE 기준 랭킹
        sorted_by_mse = sorted(results.items(), key=lambda x: x[1]['overall_mse_mean'])
        print("   Best Accuracy (lowest MSE):")
        for i, (name, stats) in enumerate(sorted_by_mse[:3]):
            print(f"     {i+1}. {name}: {stats['overall_mse_mean']:.6f}")
        
        # 안정성 기준 랭킹 (CV가 낮은 순)
        sorted_by_stability = sorted(results.items(), 
                                   key=lambda x: (x[1]['overall_mse_std'] / x[1]['overall_mse_mean']) if x[1]['overall_mse_mean'] > 0 else float('inf'))
        print("   Most Stable (lowest CV):")
        for i, (name, stats) in enumerate(sorted_by_stability[:3]):
            cv = (stats['overall_mse_std'] / stats['overall_mse_mean'] * 100) if stats['overall_mse_mean'] > 0 else 0
            print(f"     {i+1}. {name}: {cv:.1f}% CV")
        
        # 속도 기준 랭킹
        sorted_by_speed = sorted(results.items(), key=lambda x: x[1]['overall_time_mean'])
        print("   Fastest Inference:")
        for i, (name, stats) in enumerate(sorted_by_speed[:3]):
            print(f"     {i+1}. {name}: {stats['overall_time_mean']:.4f}s")
        print()
    
    def plot_enhanced_comparison(self, results):
        """향상된 시각화"""
        if not results:
            print("❌ No results to plot!")
            return
        
        models = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. MSE 비교 (에러바 포함)
        ax1 = axes[0, 0]
        mse_means = [results[m]['overall_mse_mean'] for m in models]
        mse_stds = [results[m]['overall_mse_std'] for m in models]
        
        bars1 = ax1.bar(models, mse_means, alpha=0.7, color='skyblue', capsize=5)
        ax1.errorbar(models, mse_means, yerr=mse_stds, fmt='none', color='red', capsize=5, linewidth=2)
        ax1.set_title('Average MSE with Standard Deviation', fontweight='bold')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, mse, std in zip(bars1, mse_means, mse_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(mse_stds)*0.05,
                    f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 안정성 비교 (CV)
        ax2 = axes[0, 1]
        cvs = [(results[m]['overall_mse_std'] / results[m]['overall_mse_mean'] * 100) 
               if results[m]['overall_mse_mean'] > 0 else 0 for m in models]
        
        colors = ['green' if cv < 5 else 'orange' if cv < 15 else 'red' for cv in cvs]
        bars2 = ax2.bar(models, cvs, alpha=0.7, color=colors)
        ax2.set_title('Model Stability (Coefficient of Variation)', fontweight='bold')
        ax2.set_ylabel('CV (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Stable (<5%)')
        ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Moderate (<15%)')
        ax2.legend()
        
        # 값 표시
        for bar, cv in zip(bars2, cvs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cvs)*0.02,
                    f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. 성능 vs 속도 산점도
        ax3 = axes[1, 0]
        time_means = [results[m]['overall_time_mean'] for m in models]
        scatter = ax3.scatter(time_means, mse_means, 
                             s=[results[m]['total_tests']/10 for m in models],
                             alpha=0.7, c=cvs, cmap='RdYlGn_r')
        
        ax3.set_xlabel('Average Inference Time (s)')
        ax3.set_ylabel('Average MSE')
        ax3.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 모델 이름 표시
        for i, model in enumerate(models):
            ax3.annotate(model[:10], (time_means[i], mse_means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Stability (CV %)')
        
        # 4. 종합 점수 랭킹
        ax4 = axes[1, 1]
        
        # 정규화된 점수 계산 (낮을수록 좋음)
        norm_mse = np.array(mse_means) / max(mse_means)
        norm_cv = np.array(cvs) / max(cvs) if max(cvs) > 0 else np.zeros_like(cvs)
        norm_time = np.array(time_means) / max(time_means)
        
        # 종합 점수 (가중 평균: MSE 60%, CV 20%, Time 20%)
        composite_scores = 0.6 * norm_mse + 0.2 * norm_cv + 0.2 * norm_time
        
        # 점수 순으로 정렬
        sorted_indices = np.argsort(composite_scores)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [composite_scores[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
        bars4 = ax4.barh(range(len(sorted_models)), sorted_scores, color=colors, alpha=0.8)
        ax4.set_yticks(range(len(sorted_models)))
        ax4.set_yticklabels(sorted_models)
        ax4.set_xlabel('Composite Score (lower is better)')
        ax4.set_title('Overall Ranking', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 순위 표시
        for i, (bar, score) in enumerate(zip(bars4, sorted_scores)):
            ax4.text(bar.get_width() + max(sorted_scores)*0.01, bar.get_y() + bar.get_height()/2,
                    f'#{i+1} ({score:.3f})', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def select_models_for_test(self):
        """테스트할 모델 선택"""
        if not self.models:
            return []
        
        print(f"\n📋 Available Models:")
        model_list = list(self.models.keys())
        for i, model_name in enumerate(model_list):
            info = self.model_info[model_name]
            print(f"  {i+1}. {model_name} ({info.get('hidden_size', '?')}h-{info.get('num_layers', '?')}l)")
        
        print(f"\n🎯 Select models to test:")
        print(f"  • Enter numbers separated by commas (e.g., 1,3,5)")
        print(f"  • Enter 'all' for all models")
        print(f"  • Enter 'best3' for top 3 (if previous results available)")
        
        choice = input("Selection: ").strip().lower()
        
        if choice == 'all':
            return model_list
        elif choice == 'best3':
            # 간단히 처음 3개 반환 (실제로는 이전 결과 기반으로 할 수 있음)
            return model_list[:3]
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [model_list[i] for i in indices if 0 <= i < len(model_list)]
                if selected:
                    print(f"✅ Selected: {', '.join(selected)}")
                return selected
            except (ValueError, IndexError):
                print("❌ Invalid selection! Using all models.")
                return model_list

    def manual_polynomial_test(self):
        """수동 다항식 입력 테스트"""
        print(f"\n{'='*80}")
        print(f"✏️  Manual Polynomial Input")
        print(f"{'='*80}")
        print(f"📝 Enter coefficients for 5th degree polynomial:")
        print(f"   Format: a₅x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀ = 0")
        print(f"   Input: a5 a4 a3 a2 a1 a0 (space-separated)")
        print(f"   Example: 1 -6 11 -6 0 0  (roots: 0, 0, 1, 2, 3)")
        print(f"   Example: 1 0 -5 0 4 0    (x⁵ - 5x³ + 4x = 0)")
        
        while True:
            coeffs_input = input(f"\n🔢 Enter coefficients: ").strip()
            
            if coeffs_input.lower() in ['exit', 'quit', 'back']:
                return
            
            try:
                coeffs = [float(x) for x in coeffs_input.split()]
                if len(coeffs) != 6:
                    print(f"❌ Please enter exactly 6 coefficients! (got {len(coeffs)})")
                    continue
                
                if coeffs[0] == 0:
                    print(f"⚠️  Warning: Leading coefficient is 0. This is not a 5th degree polynomial.")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                break
                
            except ValueError:
                print("❌ Invalid input! Please enter numbers only.")
        
        # 방정식 표시
        self.display_polynomial_equation(coeffs)
        
        # 모델 선택
        selected_models = self.select_models_for_test()
        if not selected_models:
            print("❌ No models selected!")
            return
        
        # 실제 근 계산
        try:
            true_roots = self.solve_polynomial_true(coeffs)
            self.display_true_roots(true_roots)
        except Exception as e:
            print(f"⚠️  Warning: Could not compute true roots: {e}")
            print("Proceeding with model predictions only...")
            true_roots = None
        
        # 모델 예측
        print(f"\n{'='*80}")
        print(f"🤖 Model Predictions")
        print(f"{'='*80}")
        
        results = {}
        for model_name in selected_models:
            result = self.single_model_test(model_name, coeffs, true_roots)
            if result:
                results[model_name] = result
                
                print(f"\n🔧 {model_name}:")
                if true_roots is not None:
                    print(f"   📊 MSE: {result['mse']:.6f}")
                    print(f"   ⏱️  Time: {result['inference_time']:.4f}s")
                
                self.display_predicted_roots(result['predictions'])
        
        # 비교 결과
        if len(results) > 1 and true_roots is not None:
            print(f"\n{'='*60}")
            print(f"📈 Comparison Results")
            print(f"{'='*60}")
            
            sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
            for i, (model_name, result) in enumerate(sorted_results):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                print(f"{rank_emoji} {model_name}: MSE = {result['mse']:.6f}")
    
    def display_polynomial_equation(self, coeffs):
        """다항식 방정식 예쁘게 표시"""
        print(f"\n📐 Polynomial Equation:")
        
        terms = []
        for i, c in enumerate(coeffs):
            power = 5 - i
            if abs(c) < 1e-12:
                continue
            
            # 계수 부호 처리
            if len(terms) == 0:
                sign = "-" if c < 0 else ""
                coeff = abs(c)
            else:
                sign = " - " if c < 0 else " + "
                coeff = abs(c)
            
            # 계수가 1이고 차수가 0이 아닌 경우
            if abs(coeff - 1.0) < 1e-12 and power > 0:
                coeff_str = ""
            else:
                coeff_str = f"{coeff:g}"
            
            # 차수 처리
            if power == 0:
                term = f"{sign}{coeff:g}"
            elif power == 1:
                term = f"{sign}{coeff_str}x"
            else:
                term = f"{sign}{coeff_str}x^{power}"
            
            terms.append(term)
        
        equation = "".join(terms) + " = 0"
        print(f"   {equation}")
    
    def display_true_roots(self, true_roots):
        """실제 근 예쁘게 표시"""
        print(f"\n🎯 True Roots:")
        for i in range(5):
            real_part = true_roots[i*2 + 1] 
            imag_part = true_roots[i*2]
            
            if abs(imag_part) < 1e-10:
                print(f"   Root {i+1}: {real_part:.6f}")
            else:
                sign = "+" if imag_part >= 0 else ""
                print(f"   Root {i+1}: {real_part:.6f}{sign}{imag_part:.6f}i")
    
    def display_predicted_roots(self, pred_roots):
        """예측 근 예쁘게 표시"""
        print(f"   🔮 Predicted roots:")
        for i in range(5):
            real_part = pred_roots[i*2 + 1]
            imag_part = pred_roots[i*2]
            
            if abs(imag_part) < 1e-10:
                print(f"      Root {i+1}: {real_part:.6f}")
            else:
                sign = "+" if imag_part >= 0 else ""
                print(f"      Root {i+1}: {real_part:.6f}{sign}{imag_part:.6f}i")

    def interactive_test(self):
        """인터랙티브 테스트"""
        if not self.models:
            print("❌ No models loaded!")
            return
        
        print(f"\n{'='*80}")
        print(f"🎮 Interactive Test Mode")
        print(f"{'='*80}")
        print("Commands:")
        print("  1. Manual polynomial input & model selection")
        print("  2. Random polynomial test")
        print("  3. Quick multi-run test (5 runs × 10 polynomials)")
        print("  4. Comprehensive multi-run test (custom settings)")
        print("  5. Model comparison plot")
        print("  6. List loaded models")
        print("  7. Exit")
        
        last_results = None
        
        while True:
            try:
                choice = input(f"\n🎯 Enter choice (1-7): ").strip()
                
                if choice == '1':
                    self.manual_polynomial_test()
                
                elif choice == '2':
                    coeffs = self.create_test_polynomial()
                    true_roots = self.solve_polynomial_true(coeffs)
                    
                    print(f"\n🎲 Testing random polynomial:")
                    self.display_polynomial_equation(coeffs)
                    self.display_true_roots(true_roots)
                    
                    # 모델 선택
                    selected_models = self.select_models_for_test()
                    if selected_models:
                        print(f"\n📊 Results:")
                        for model_name in selected_models:
                            result = self.single_model_test(model_name, coeffs, true_roots)
                            if result:
                                print(f"   🔧 {model_name}: MSE = {result['mse']:.6f}, Time = {result['inference_time']:.4f}s")
                
                elif choice == '3':
                    print(f"\n🚀 Running quick multi-run test...")
                    last_results = self.multi_run_test(num_runs=5, num_polynomials=10)
                    if last_results:
                        self.print_detailed_results(last_results)
                
                elif choice == '4':
                    try:
                        num_runs = int(input("Number of runs per model (default: 5): ") or "5")
                        num_polynomials = int(input("Number of polynomials per run (default: 20): ") or "20")
                        
                        print(f"\n🔬 Running comprehensive test...")
                        last_results = self.multi_run_test(num_runs=num_runs, num_polynomials=num_polynomials)
                        if last_results:
                            self.print_detailed_results(last_results)
                            
                    except ValueError:
                        print("❌ Invalid input! Using default values.")
                        last_results = self.multi_run_test(num_runs=5, num_polynomials=20)
                        if last_results:
                            self.print_detailed_results(last_results)
                
                elif choice == '5':
                    if last_results:
                        print(f"\n📊 Generating comparison plots...")
                        self.plot_enhanced_comparison(last_results)
                    else:
                        print("❌ No test results available! Run a multi-run test first.")
                
                elif choice == '6':
                    self.list_loaded_models()
                
                elif choice == '7':
                    print("👋 Exiting...")
                    break
                
                else:
                    print("❌ Invalid choice! Please enter 1-7.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """메인 함수"""
    print("="*80)
    print("🔬 Enhanced Model Tester with Multi-Run Analysis")
    print("="*80)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # 테스터 생성
    tester = EnhancedModelTester(device)
    
    # 모델 로드 시도
    success = tester.load_all_models()
    
    if not success:
        print("\n❌ No models could be loaded. Please check:")
        print("1. Model files exist in the current directory")
        print("2. Model files are in correct format (.pth, .pt)")
        print("3. Model files contain valid PyTorch state_dict")
        return
    
    # 로드된 모델 목록 출력
    tester.list_loaded_models()
    
    # 데모 실행 여부 묻기
    demo_choice = input(f"\n🎯 Run quick demo? (y/n): ").strip().lower()
    if demo_choice == 'y':
        print(f"\n🚀 Running demo: Testing each model 3 times with 10 different polynomials each time...")
        demo_results = tester.multi_run_test(num_runs=3, num_polynomials=10)
        
        if demo_results:
            tester.print_detailed_results(demo_results)
            
            plot_choice = input(f"\n📊 Show comparison plots? (y/n): ").strip().lower()
            if plot_choice == 'y':
                tester.plot_enhanced_comparison(demo_results)
    
    # 인터랙티브 모드
    interactive_choice = input(f"\n🎮 Start interactive mode? (y/n): ").strip().lower()
    if interactive_choice == 'y':
        tester.interactive_test()
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()