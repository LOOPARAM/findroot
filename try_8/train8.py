import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd
from datetime import datetime

# 기존 클래스들을 그대로 사용 (PolynomialDataset, PolynomialRootNet 등)
class PolynomialDataset(Dataset):
    def __init__(self, data_file, train=True, train_ratio=0.8, normalize=True):
        """
        저장된 데이터 파일로부터 Dataset 생성
        
        Args:
            data_file: 데이터 파일 경로 (.json 또는 .pkl)
            train: True면 훈련용, False면 테스트용
            train_ratio: 훈련/테스트 분할 비율
            normalize: 데이터 정규화 여부
        """
        self.data = self.load_data(data_file)
        self.normalize = normalize
        self.coeffs, self.roots = self.prepare_data()
        
        # 정규화 파라미터 저장
        if normalize:
            self.coeff_mean = np.mean(self.coeffs, axis=0)
            self.coeff_std = np.std(self.coeffs, axis=0) + 1e-8  # 0으로 나누기 방지
            self.root_mean = np.mean(self.roots, axis=0)
            self.root_std = np.std(self.roots, axis=0) + 1e-8
            
            # 정규화 적용
            self.coeffs = (self.coeffs - self.coeff_mean) / self.coeff_std
            self.roots = (self.roots - self.root_mean) / self.root_std
        
        # 훈련/테스트 분할
        train_coeffs, test_coeffs, train_roots, test_roots = train_test_split(
            self.coeffs, self.roots, train_size=train_ratio, random_state=42
        )
        
        if train:
            self.coeffs = train_coeffs
            self.roots = train_roots
        else:
            self.coeffs = test_coeffs
            self.roots = test_roots
        
        print(f"{'훈련' if train else '테스트'} 데이터셋: {len(self.coeffs)}개")
        if normalize:
            print(f"데이터 정규화 적용됨")
    
    def load_data(self, filename):
        """데이터 파일 로드"""
        if filename.endswith('.json'):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. (.json 또는 .pkl 사용)")
    
    def prepare_data(self):
        """데이터를 학습용 형태로 변환"""
        coeffs = []
        roots = []
        
        for item in self.data:
            coeffs.append(item['coefficients'])
            roots.append(item['roots'])
        
        return np.array(coeffs, dtype=np.float32), np.array(roots, dtype=np.float32)
    
    def denormalize_roots(self, normalized_roots):
        """근을 원래 스케일로 되돌리기"""
        if self.normalize and hasattr(self, 'root_mean'):
            return normalized_roots * self.root_std + self.root_mean
        return normalized_roots
    
    def __len__(self):
        return len(self.coeffs)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.coeffs[idx]), torch.FloatTensor(self.roots[idx])

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

class HiddenSizeComparator:
    def __init__(self, data_file, device='cpu'):
        self.data_file = data_file
        self.device = device
        self.results = []
        
        print(f"디바이스: {device}")
        print(f"데이터 파일: {data_file}")
        
    def create_datasets(self, batch_size=32):
        """데이터셋 생성"""
        train_dataset = PolynomialDataset(self.data_file, train=True, normalize=True)
        test_dataset = PolynomialDataset(self.data_file, train=False, normalize=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, test_dataset
    
    def train_single_model(self, hidden_size, num_layers=4, epochs=50, lr=0.001, batch_size=32, dropout=0.3):
        """단일 모델 훈련"""
        print(f"\n{'='*50}")
        print(f"Hidden Size: {hidden_size}, Layers: {num_layers}")
        print(f"{'='*50}")
        
        # 데이터 로더 생성
        train_loader, test_loader, test_dataset = self.create_datasets(batch_size)
        
        # 모델 생성
        model = PolynomialRootNet(
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        ).to(self.device)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"총 파라미터 수: {total_params:,}")
        print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
        
        # 훈련 설정
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 훈련 기록
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        best_model_state = None
        
        # 훈련 시작 시간
        start_time = time.time()
        
        # 훈련 루프
        for epoch in range(epochs):
            # 훈련
            model.train()
            train_loss = 0
            num_batches = 0
            
            for coeffs, roots in train_loader:
                coeffs, roots = coeffs.to(self.device), roots.to(self.device)
                
                optimizer.zero_grad()
                pred_roots = model(coeffs)
                loss = criterion(pred_roots, roots)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            train_losses.append(train_loss)
            
            # 테스트
            model.eval()
            test_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for coeffs, roots in test_loader:
                    coeffs, roots = coeffs.to(self.device), roots.to(self.device)
                    pred_roots = model(coeffs)
                    loss = criterion(pred_roots, roots)
                    test_loss += loss.item()
                    num_batches += 1
            
            test_loss /= num_batches
            test_losses.append(test_loss)
            
            # 스케줄러 업데이트
            scheduler.step(test_loss)
            
            # 최고 모델 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
            
            # 진행 상황 출력
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Test={test_loss:.6f}, Best={best_test_loss:.6f}")
        
        # 훈련 시간 계산
        training_time = time.time() - start_time
        
        # 최고 모델 로드
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 결과 저장
        result = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'best_test_loss': best_test_loss,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'training_time': training_time,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout': dropout
        }
        
        self.results.append(result)
        
        print(f"훈련 완료 - 최고 테스트 손실: {best_test_loss:.6f}")
        print(f"훈련 시간: {training_time:.1f}초")
        
        return model, result
    
    def compare_hidden_sizes(self, hidden_sizes, num_layers=4, epochs=50, lr=0.001, batch_size=32, dropout=0.3):
        """여러 hidden_size로 비교 실험"""
        print(f"Hidden Size 비교 실험 시작")
        print(f"테스트할 Hidden Sizes: {hidden_sizes}")
        print(f"Epochs: {epochs}, Learning Rate: {lr}")
        
        for hidden_size in hidden_sizes:
            try:
                model, result = self.train_single_model(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    dropout=dropout
                )
                
                # 메모리 정리
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Hidden Size {hidden_size} 훈련 중 오류 발생: {e}")
                continue
        
        print(f"\n모든 실험 완료!")
        return self.results
    
    def plot_comparison(self, save_path=None):
        """비교 결과 시각화"""
        if not self.results:
            print("비교할 결과가 없습니다!")
            return
        
        # 데이터프레임 생성
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hidden Size 비교 실험 결과', fontsize=16, fontweight='bold')
        
        # 1. 테스트 손실 비교
        axes[0, 0].bar(range(len(df)), df['best_test_loss'], alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Hidden Size')
        axes[0, 0].set_ylabel('Best Test Loss')
        axes[0, 0].set_title('최고 테스트 손실 비교')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['hidden_size'], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 파라미터 수 vs 성능
        axes[0, 1].scatter(df['total_params'], df['best_test_loss'], s=100, alpha=0.7, color='coral')
        for i, row in df.iterrows():
            axes[0, 1].annotate(f"{row['hidden_size']}", 
                              (row['total_params'], row['best_test_loss']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('총 파라미터 수')
        axes[0, 1].set_ylabel('Best Test Loss')
        axes[0, 1].set_title('파라미터 수 vs 성능')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 훈련 시간 비교
        axes[0, 2].bar(range(len(df)), df['training_time'], alpha=0.7, color='lightgreen')
        axes[0, 2].set_xlabel('Hidden Size')
        axes[0, 2].set_ylabel('Training Time (초)')
        axes[0, 2].set_title('훈련 시간 비교')
        axes[0, 2].set_xticks(range(len(df)))
        axes[0, 2].set_xticklabels(df['hidden_size'], rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 훈련 곡선 (모든 모델)
        for i, result in enumerate(self.results):
            axes[1, 0].plot(result['test_losses'], 
                          label=f"H={result['hidden_size']}", alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].set_title('테스트 손실 곡선')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 5. 효율성 비교 (성능 / 파라미터 수)
        efficiency = df['best_test_loss'] / (df['total_params'] / 1000)  # 1000개당 손실
        axes[1, 1].bar(range(len(df)), efficiency, alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('Hidden Size')
        axes[1, 1].set_ylabel('Loss per 1K Parameters')
        axes[1, 1].set_title('효율성 (손실/파라미터)')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df['hidden_size'], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 수렴 속도 비교 (10 에포크에서의 테스트 손실)
        convergence_loss = []
        for result in self.results:
            if len(result['test_losses']) >= 10:
                convergence_loss.append(result['test_losses'][9])  # 10번째 에포크
            else:
                convergence_loss.append(result['test_losses'][-1])
        
        axes[1, 2].bar(range(len(df)), convergence_loss, alpha=0.7, color='plum')
        axes[1, 2].set_xlabel('Hidden Size')
        axes[1, 2].set_ylabel('Test Loss at Epoch 10')
        axes[1, 2].set_title('초기 수렴 속도')
        axes[1, 2].set_xticks(range(len(df)))
        axes[1, 2].set_xticklabels(df['hidden_size'], rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")
        
        plt.show()
    
    def save_results(self, filepath):
        """결과를 파일로 저장"""
        # CSV로 요약 저장
        df = pd.DataFrame(self.results)
        csv_path = filepath + '_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"요약 결과 저장: {csv_path}")
        
        # 전체 결과를 JSON으로 저장
        json_path = filepath + '_detailed.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"상세 결과 저장: {json_path}")
        
        return csv_path, json_path
    
    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            print("결과가 없습니다!")
            return
        
        print(f"\n{'='*80}")
        print(f"Hidden Size 비교 실험 요약")
        print(f"{'='*80}")
        
        # 정렬 (테스트 손실 기준)
        sorted_results = sorted(self.results, key=lambda x: x['best_test_loss'])
        
        print(f"{'순위':<4} {'Hidden':<8} {'파라미터':<12} {'테스트손실':<12} {'훈련시간':<10} {'효율성':<10}")
        print(f"{'-'*70}")
        
        for i, result in enumerate(sorted_results):
            efficiency = result['best_test_loss'] / (result['total_params'] / 1000)
            print(f"{i+1:<4} {result['hidden_size']:<8} {result['total_params']:<12,} "
                  f"{result['best_test_loss']:<12.6f} {result['training_time']:<10.1f} {efficiency:<10.4f}")
        
        # 최고 성능
        best = sorted_results[0]
        print(f"\n🏆 최고 성능:")
        print(f"   Hidden Size: {best['hidden_size']}")
        print(f"   테스트 손실: {best['best_test_loss']:.6f}")
        print(f"   파라미터 수: {best['total_params']:,}")
        print(f"   훈련 시간: {best['training_time']:.1f}초")
        
        # 가장 효율적인 모델
        most_efficient = min(sorted_results, key=lambda x: x['best_test_loss'] / (x['total_params'] / 1000))
        print(f"\n⚡ 가장 효율적인 모델:")
        print(f"   Hidden Size: {most_efficient['hidden_size']}")
        print(f"   효율성: {most_efficient['best_test_loss'] / (most_efficient['total_params'] / 1000):.4f}")
        print(f"   테스트 손실: {most_efficient['best_test_loss']:.6f}")

def main():
    # 설정
    data_file = "polynomial_dataset_2.json"  # 데이터 파일 경로
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 파일 확인
    if not os.path.exists(data_file):
        print(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        print("먼저 데이터 생성기를 실행해주세요!")
        return
    
    # 비교할 hidden_size 목록
    hidden_sizes = [128*2, 128*3, 128*4, 128*5, 128*6, 128*7, 128*8, 128*9, 128*10, 128*11, 128*12, 128*13]  # 원하는 크기로 조정 가능
    
    # 실험 설정
    experiment_config = {
        'num_layers': 6,
        'epochs': 100,        # 에포크 수 조정
        'lr': 0.004,
        'batch_size': 32,
        'dropout': 0.3
    }
    
    print(f"Hidden Size 비교 실험 설정:")
    print(f"Hidden Sizes: {hidden_sizes}")
    print(f"실험 설정: {experiment_config}")
    
    # 비교 실험 실행
    comparator = HiddenSizeComparator(data_file, device)
    results = comparator.compare_hidden_sizes(hidden_sizes, **experiment_config)
    
    # 결과 출력
    comparator.print_summary()
    
    # 결과 시각화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"hidden_size_comparison_{timestamp}.png"
    comparator.plot_comparison(save_path=plot_path)
    
    # 결과 저장
    result_path = f"hidden_size_experiment_{timestamp}"
    comparator.save_results(result_path)
    
    print(f"\n실험 완료! 결과가 저장되었습니다.")

# 개별 실험 함수 (특정 설정만 테스트하고 싶을 때)
def quick_test(hidden_sizes=[256, 512, 1024], epochs=30):
    """빠른 테스트용 함수"""
    data_file = "polynomial_dataset_2.json"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(data_file):
        print(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        return
    
    comparator = HiddenSizeComparator(data_file, device)
    results = comparator.compare_hidden_sizes(
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        lr=0.002
    )
    
    comparator.print_summary()
    comparator.plot_comparison()
    
    return results

if __name__ == "__main__":
    main()
    
    # 빠른 테스트를 원하면 아래 주석 해제
    # quick_test(hidden_sizes=[128, 512, 1024], epochs=20)