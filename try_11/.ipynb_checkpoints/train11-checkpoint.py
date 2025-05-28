import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# scipy 제거
import os

class PolynomialDataset(Dataset):
    def __init__(self, data_file, train=True, train_ratio=0.8, normalize=True):
        """
        저장된 데이터 파일로부터 Dataset 생성
        """
        self.data = self.load_data(data_file)
        self.normalize = normalize
        self.coeffs, self.roots = self.prepare_data()
        
        # 정규화 파라미터 저장
        if normalize:
            # 계수 정규화 (monic polynomial로 변환 후)
            self.coeff_mean = np.mean(self.coeffs, axis=0)
            self.coeff_std = np.std(self.coeffs, axis=0) + 1e-8
            
            # 근 정규화 (절댓값 기준)
            root_magnitudes = np.abs(self.roots.reshape(-1, 5, 2))
            self.root_scale = np.percentile(root_magnitudes, 95) + 1e-8
            
            # 정규화 적용
            self.coeffs = (self.coeffs - self.coeff_mean) / self.coeff_std
            self.roots = self.roots / self.root_scale
        
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
        """데이터를 학습용 형태로 변환 - 근을 크기 순으로 정렬"""
        coeffs = []
        roots = []
        
        for item in self.data:
            coeff = np.array(item['coefficients'])
            root = np.array(item['roots']).reshape(5, 2)  # [실수부, 허수부]
            
            # monic polynomial로 변환 (최고차항 계수로 나누기)
            if abs(coeff[0]) > 1e-8:
                coeff = coeff / coeff[0]
            
            # 근을 절댓값 기준으로 정렬 (순서 일관성 확보)
            magnitudes = np.sqrt(root[:, 0]**2 + root[:, 1]**2)
            sorted_indices = np.argsort(magnitudes)
            root = root[sorted_indices]
            
            coeffs.append(coeff[1:])  # 최고차항 제외 (monic이므로)
            roots.append(root.flatten())  # [실수부1, 허수부1, 실수부2, 허수부2, ...]
        
        return np.array(coeffs, dtype=np.float32), np.array(roots, dtype=np.float32)
    
    def denormalize_roots(self, normalized_roots):
        """근을 원래 스케일로 되돌리기"""
        if self.normalize and hasattr(self, 'root_scale'):
            return normalized_roots * self.root_scale
        return normalized_roots
    
    def __len__(self):
        return len(self.coeffs)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.coeffs[idx]), torch.FloatTensor(self.roots[idx])


class ResidualBlock(nn.Module):
    """잔차 연결이 있는 블록"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = out + residual  # 잔차 연결
        out = self.activation(out)
        out = self.dropout(out)
        return out


class PolynomialRootNet(nn.Module):
    def __init__(self, input_size=5, output_size=10, hidden_size=512, num_blocks=4, dropout=0.2):
        super().__init__()
        
        # 입력층
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout)
        )
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        
        # 특징 추출층
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout // 2)
        )
        
        # 출력층 (각 근별로)
        self.output_layer = nn.Linear(hidden_size // 2, output_size)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        # 잔차 블록들 통과
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.feature_layers(x)
        x = self.output_layer(x)
        
        return x


def magnitude_sorted_loss(pred_roots, true_roots):
    """
    근을 크기 순으로 정렬한 후 MSE 손실 계산
    """
    batch_size = pred_roots.shape[0]
    pred_roots = pred_roots.view(batch_size, 5, 2)  # [batch, 5근, 2좌표]
    true_roots = true_roots.view(batch_size, 5, 2)
    
    total_loss = 0
    
    for i in range(batch_size):
        pred_i = pred_roots[i]  # [5, 2]
        true_i = true_roots[i]  # [5, 2]
        
        # 크기 계산 및 정렬
        pred_magnitudes = torch.sqrt(pred_i[:, 0]**2 + pred_i[:, 1]**2)
        true_magnitudes = torch.sqrt(true_i[:, 0]**2 + true_i[:, 1]**2)
        
        pred_sorted_indices = torch.argsort(pred_magnitudes)
        true_sorted_indices = torch.argsort(true_magnitudes)
        
        pred_sorted = pred_i[pred_sorted_indices]
        true_sorted = true_i[true_sorted_indices]
        
        # MSE 손실 계산
        loss = torch.sum((pred_sorted - true_sorted)**2)
        total_loss += loss
    
    return total_loss / batch_size


def polynomial_verification_loss(pred_roots, coeffs, weight=0.1):
    """
    예측된 근이 실제로 방정식을 만족하는지 확인하는 손실
    """
    batch_size = pred_roots.shape[0]
    pred_roots = pred_roots.view(batch_size, 5, 2)  # [batch, 5근, 실수부+허수부]
    
    total_loss = 0
    
    for i in range(batch_size):
        coeff = coeffs[i]  # [5] (monic이므로 최고차항 계수는 1)
        roots = pred_roots[i]  # [5, 2]
        
        poly_loss = 0
        for j in range(5):
            # 복소수 근
            real, imag = roots[j, 0], roots[j, 1]
            z_real, z_imag = real, imag
            
            # z^k 계산 (k=1,2,3,4,5)
            powers_real = [1, z_real, 0, 0, 0, 0]  # z^0, z^1의 실수부
            powers_imag = [0, z_imag, 0, 0, 0, 0]  # z^0, z^1의 허수부
            
            # z^2, z^3, z^4, z^5 계산
            for k in range(2, 6):
                # z^k = z^(k-1) * z
                new_real = powers_real[k-1] * z_real - powers_imag[k-1] * z_imag
                new_imag = powers_real[k-1] * z_imag + powers_imag[k-1] * z_real
                powers_real[k] = new_real
                powers_imag[k] = new_imag
            
            # 다항식 계산: a4*z^4 + a3*z^3 + a2*z^2 + a1*z + a0 + z^5
            poly_real = (coeff[0] * powers_real[4] + coeff[1] * powers_real[3] + 
                        coeff[2] * powers_real[2] + coeff[3] * powers_real[1] + 
                        coeff[4] + powers_real[5])
            poly_imag = (coeff[0] * powers_imag[4] + coeff[1] * powers_imag[3] + 
                        coeff[2] * powers_imag[2] + coeff[3] * powers_imag[1] + 
                        powers_imag[5])
            
            # 다항식 값이 0에 가까워야 함
            poly_loss += poly_real**2 + poly_imag**2
        
        total_loss += poly_loss
    
    return weight * total_loss / batch_size


class PolynomialTrainer:
    def __init__(self, model, train_loader, test_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 훈련 기록
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, optimizer, use_verification=True, verification_weight=0.01):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for coeffs, roots in self.train_loader:
            coeffs, roots = coeffs.to(self.device), roots.to(self.device)
            
            optimizer.zero_grad()
            
            # 순전파
            pred_roots = self.model(coeffs)
            
            # 크기 순 정렬 손실 계산
            sorted_loss_val = magnitude_sorted_loss(pred_roots, roots)
            
            total_loss_batch = sorted_loss_val
            
            if use_verification:
                # 방정식 검증 손실 추가
                verification_loss_val = polynomial_verification_loss(
                    pred_roots, coeffs, verification_weight
                )
                total_loss_batch = total_loss_batch + verification_loss_val
            
            # 역전파
            total_loss_batch.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for coeffs, roots in self.test_loader:
                coeffs, roots = coeffs.to(self.device), roots.to(self.device)
                
                pred_roots = self.model(coeffs)
                loss = magnitude_sorted_loss(pred_roots, roots)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs=100, lr=0.001, use_scheduler=True, use_verification=True, 
              verification_weight=0.01, save_best=True):
        """모델 훈련"""
        print(f"훈련 시작 - Epochs: {epochs}, LR: {lr}, Device: {self.device}")
        print(f"검증 손실 사용: {use_verification}, 가중치: {verification_weight}")
        
        # 옵티마이저
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 학습률 스케줄러
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=15
            )
        
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Test Loss':<12} {'Best Test':<12}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # 훈련
            train_loss = self.train_epoch(optimizer, use_verification, verification_weight)
            self.train_losses.append(train_loss)
            
            # 테스트
            test_loss = self.test_epoch()
            self.test_losses.append(test_loss)
            
            # 스케줄러 업데이트
            if use_scheduler:
                scheduler.step(test_loss)
            
            # 최고 모델 저장
            if save_best and test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # 진행 상황 출력
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"{epoch:<6} {train_loss:<12.6f} {test_loss:<12.6f} {self.best_test_loss:<12.6f}")
        
        # 최고 모델 로드
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n최고 성능 모델 로드 완료 (Test Loss: {self.best_test_loss:.6f})")
    
    def plot_training_curve(self):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(12, 5))
        
        # 손실 곡선
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(self.test_losses, label='Test Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 최근 50 에포크 확대
        plt.subplot(1, 2, 2)
        start_idx = max(0, len(self.train_losses) - 50)
        plt.plot(range(start_idx, len(self.train_losses)), 
                self.train_losses[start_idx:], label='Train Loss', alpha=0.8)
        plt.plot(range(start_idx, len(self.test_losses)), 
                self.test_losses[start_idx:], label='Test Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Last 50 Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()


def test_model(model, test_loader, test_dataset, device, num_examples=3):
    """모델 테스트 및 결과 출력"""
    model.eval()
    
    print("\n=== 모델 테스트 결과 ===")
    
    with torch.no_grad():
        for i, (coeffs, true_roots) in enumerate(test_loader):
            if i >= num_examples:
                break
            
            coeffs, true_roots = coeffs.to(device), true_roots.to(device)
            pred_roots = model(coeffs)
            
            # 첫 번째 샘플만 출력
            if coeffs.shape[0] > 0:
                # 정규화 해제
                if hasattr(test_dataset, 'denormalize_roots'):
                    true_vals = test_dataset.denormalize_roots(true_roots[0].cpu().numpy())
                    pred_vals = test_dataset.denormalize_roots(pred_roots[0].cpu().numpy())
                else:
                    true_vals = true_roots[0].cpu().numpy()
                    pred_vals = pred_roots[0].cpu().numpy()
                
                # 계수 복원
                if hasattr(test_dataset, 'coeff_mean'):
                    coeff_vals = coeffs[0].cpu().numpy() * test_dataset.coeff_std[1:] + test_dataset.coeff_mean[1:]
                    coeff_vals = np.concatenate(([1.0], coeff_vals))  # 최고차항 추가
                else:
                    coeff_vals = np.concatenate(([1.0], coeffs[0].cpu().numpy()))
                
                print(f"\n테스트 {i+1}:")
                print(f"방정식: {coeff_vals[0]:.3f}x^5 + {coeff_vals[1]:.3f}x^4 + {coeff_vals[2]:.3f}x^3 + {coeff_vals[3]:.3f}x^2 + {coeff_vals[4]:.3f}x + {coeff_vals[5]:.3f} = 0")
                
                print("실제 근들:")
                true_roots_reshaped = true_vals.reshape(5, 2)
                for j in range(5):
                    real, imag = true_roots_reshaped[j]
                    if abs(imag) < 1e-6:
                        print(f"  근 {j+1}: {real:.6f}")
                    else:
                        print(f"  근 {j+1}: {real:.6f} + {imag:.6f}i")
                
                print("예측 근들:")
                pred_roots_reshaped = pred_vals.reshape(5, 2)
                for j in range(5):
                    real, imag = pred_roots_reshaped[j]
                    if abs(imag) < 1e-6:
                        print(f"  근 {j+1}: {real:.6f}")
                    else:
                        print(f"  근 {j+1}: {real:.6f} + {imag:.6f}i")
                
                # 크기 순 정렬 손실 계산
                sorted_error = magnitude_sorted_loss(pred_roots[0:1], true_roots[0:1])
                print(f"크기 순 정렬 손실: {sorted_error:.6f}")


def main():
    # 설정
    data_file = "polynomial_dataset_2.json"
    batch_size = 32
    epochs = 100
    learning_rate = 0.002
    
    # GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 파일 확인
    if not os.path.exists(data_file):
        print(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        return
    
    # 데이터셋 생성
    train_dataset = PolynomialDataset(data_file, train=True, normalize=True)
    test_dataset = PolynomialDataset(data_file, train=False, normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 개선된 모델 생성
    model = PolynomialRootNet(
        input_size=5,      # monic polynomial이므로 5개 계수
        output_size=10,    # 5개 근 × 2개 좌표
        hidden_size=1024,
        num_blocks=6,
        dropout=0.15
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 트레이너 생성
    trainer = PolynomialTrainer(model, train_loader, test_loader, device)
    
    # 훈련 실행 (개선된 손실함수 사용)
    trainer.train(
        epochs=epochs, 
        lr=learning_rate, 
        use_verification=True,
        verification_weight=0.05,  # 검증 손실 가중치
        use_scheduler=True
    )
    
    # 결과 시각화
    trainer.plot_training_curve()
    
    # 모델 테스트
    test_model(trainer.model, test_loader, test_dataset, device)
    
    print("\n훈련 완료!")

if __name__ == "__main__":
    main()