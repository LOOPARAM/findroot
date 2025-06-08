#222222
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 가능하면 GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

class RootDataset(Dataset):
    """근 예측용 데이터셋"""
    def __init__(self, X, y_num, y_values):
        self.X = torch.FloatTensor(X)
        self.y_num = torch.FloatTensor(y_num)
        self.y_values = torch.FloatTensor(y_values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_num[idx], self.y_values[idx]

class RootPredictorNet(nn.Module):
    """근 예측 신경망"""
    def __init__(self, input_dim=3, hidden_dims=[128, 256, 512, 256, 128]):
        super(RootPredictorNet, self).__init__()
        
        # 공통 백본 네트워크
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2 if hidden_dim <= 256 else 0.3)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 근의 개수 예측 헤드
        self.num_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),  # 0, 1, 2, 3개 근
            nn.Softmax(dim=1)
        )
        
        # 근의 값 예측 헤드
        self.values_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # 최대 3개 근의 값
        )
    
    def forward(self, x):
        # 공통 특성 추출
        features = self.backbone(x)
        
        # 두 개의 헤드로 예측
        num_roots = self.num_head(features)
        root_values = self.values_head(features)
        
        return num_roots, root_values

class RootPredictor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.model = None
        self.scaler = StandardScaler()
        self.device = device
        
    def load_and_prepare_data(self):
        """CSV 파일에서 데이터 로드 및 전처리"""
        print("데이터 로딩 및 전처리 중...")
        
        # CSV 파일 읽기
        df = pd.read_csv(self.csv_file_path)
        print(f"원본 데이터 크기: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        # 입력 변수 (a, b, c)
        X = df[['a', 'b', 'c']].values
        
        # 근의 개수
        num_roots = df['num_roots'].values
        
        # 근의 값들 (root_1, root_2, root_3, root_4가 있다면)
        root_columns = [col for col in df.columns if col.startswith('root_')]
        print(f"근 컬럼들: {root_columns}")
        
        # 실제 근의 값들 추출
        y_roots = []
        for i, row in df.iterrows():
            roots = []
            num_roots_i = int(row['num_roots'])
            
            for j in range(num_roots_i):
                if j < len(root_columns):
                    root_value = row[root_columns[j]]
                    if pd.notna(root_value):  # NaN이 아닌 경우만
                        roots.append(root_value)
            
            y_roots.append(roots)
        
        # 근의 개수별 분포 확인
        root_counts = [len(roots) for roots in y_roots]
        unique, counts = np.unique(root_counts, return_counts=True)
        print("\n근의 개수별 분포:")
        for num, count in zip(unique, counts):
            print(f"  {num}개 근: {count}개 샘플 ({count/len(y_roots)*100:.1f}%)")
        
        return X, y_roots
    
    def prepare_training_data(self, X, y_roots):
        """학습용 데이터 전처리"""
        print("학습용 데이터 변환 중...")
        
        # 입력 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 근의 개수를 원핫인코딩
        y_num = []
        y_values = []
        
        for roots in y_roots:
            # 근의 개수 (최대 3개로 제한)
            num_roots = min(len(roots), 3)
            num_onehot = [0, 0, 0, 0]
            num_onehot[num_roots] = 1
            y_num.append(num_onehot)
            
            # 근의 값들 (3개로 패딩)
            values = list(roots[:3]) + [0.0] * (3 - len(roots[:3]))
            y_values.append(values)
        
        return X_scaled, np.array(y_num), np.array(y_values)
    
    def masked_mse_loss(self, predicted_values, true_values, predicted_num):
        """마스킹된 MSE 손실"""
        batch_size = predicted_values.size(0)
        total_loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            num_roots = torch.argmax(predicted_num[i]).item()
            if num_roots > 0:
                loss = nn.MSELoss()(predicted_values[i, :num_roots], true_values[i, :num_roots])
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)
    
    def train(self, epochs=100, batch_size=128, learning_rate=0.001, validation_split=0.2):
        """모델 학습"""
        print("=== PyTorch 근 예측 신경망 학습 시작 ===")
        
        # 데이터 로드
        X, y_roots = self.load_and_prepare_data()
        X_scaled, y_num, y_values = self.prepare_training_data(X, y_roots)
        
        print(f"전처리된 데이터 크기: X={X_scaled.shape}, y_num={y_num.shape}, y_values={y_values.shape}")
        
        # 학습/검증 데이터 분할
        X_train, X_val, y_num_train, y_num_val, y_values_train, y_values_val = train_test_split(
            X_scaled, y_num, y_values, test_size=validation_split, random_state=42, stratify=np.argmax(y_num, axis=1)
        )
        
        print(f"학습 데이터: {len(X_train)}개")
        print(f"검증 데이터: {len(X_val)}개")
        
        # 데이터로더 생성
        train_dataset = RootDataset(X_train, y_num_train, y_values_train)
        val_dataset = RootDataset(X_val, y_num_val, y_values_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 생성
        self.model = RootPredictorNet().to(self.device)
        print(f"\n모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 옵티마이저와 손실함수
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        ce_loss = nn.CrossEntropyLoss()
        
        # 학습 기록
        train_losses = []
        val_losses = []
        train_num_accs = []
        val_num_accs = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print("\n=== 학습 시작 ===")
        
        for epoch in range(epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0
            train_num_correct = 0
            train_total = 0
            
            for batch_x, batch_y_num, batch_y_values in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y_num = batch_y_num.to(self.device)
                batch_y_values = batch_y_values.to(self.device)
                
                optimizer.zero_grad()
                
                # 순전파
                pred_num, pred_values = self.model(batch_x)
                
                # 손실 계산
                num_loss = ce_loss(pred_num, torch.argmax(batch_y_num, dim=1))
                values_loss = self.masked_mse_loss(pred_values, batch_y_values, batch_y_num)
                total_loss = num_loss + 10.0 * values_loss
                
                # 역전파
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # gradient clipping
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(pred_num, 1)
                _, true_labels = torch.max(batch_y_num, 1)
                train_num_correct += (predicted == true_labels).sum().item()
                train_total += batch_y_num.size(0)
            
            # 검증 모드
            self.model.eval()
            val_loss = 0
            val_num_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y_num, batch_y_values in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y_num = batch_y_num.to(self.device)
                    batch_y_values = batch_y_values.to(self.device)
                    
                    pred_num, pred_values = self.model(batch_x)
                    
                    num_loss = ce_loss(pred_num, torch.argmax(batch_y_num, dim=1))
                    values_loss = self.masked_mse_loss(pred_values, batch_y_values, batch_y_num)
                    total_loss = num_loss + 10.0 * values_loss
                    
                    val_loss += total_loss.item()
                    
                    _, predicted = torch.max(pred_num, 1)
                    _, true_labels = torch.max(batch_y_num, 1)
                    val_num_correct += (predicted == true_labels).sum().item()
                    val_total += batch_y_num.size(0)
            
            # 평균 계산
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_num_correct / train_total
            val_acc = val_num_correct / val_total
            
            # 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_num_accs.append(train_acc)
            val_num_accs.append(val_acc)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 출력
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), 'best_root_model2.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_root_model2.pth'))
        
        print("\n=== 학습 완료 ===")
        
        # 학습 과정 시각화
        self.plot_training_history(train_losses, val_losses, train_num_accs, val_num_accs)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_num_accs,
            'val_accs': val_num_accs
        }
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """학습 과정 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실
        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 정확도
        axes[1].plot(train_accs, label='Train Accuracy')
        axes[1].plot(val_accs, label='Validation Accuracy')
        axes[1].set_title('Root Count Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_roots(self, a, b, c):
        """근 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        self.model.eval()
        
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
    
    def evaluate_on_test_set(self, test_ratio=0.1):
        """테스트 세트로 성능 평가"""
        print("=== 테스트 세트 평가 ===")
        
        # 데이터 다시 로드
        X, y_roots = self.load_and_prepare_data()
        X_scaled, y_num, y_values = self.prepare_training_data(X, y_roots)
        
        # 테스트 세트 분리
        _, X_test, _, y_num_test, _, y_values_test = train_test_split(
            X_scaled, y_num, y_values, test_size=test_ratio, random_state=42, stratify=np.argmax(y_num, axis=1)
        )
        
        correct_count = 0
        total_error = 0
        
        for i in range(min(20, len(X_test))):  # 처음 20개만 출력
            # 원래 스케일로 복원
            original_params = self.scaler.inverse_transform([X_test[i]])[0]
            a, b, c = original_params
            
            # 예측
            predicted_roots, probs = self.predict_roots(a, b, c)
            
            # 실제 값
            true_num_roots = np.argmax(y_num_test[i])
            true_roots = y_values_test[i][:true_num_roots].tolist() if true_num_roots > 0 else []
            
            print(f"\n테스트 {i+1}:")
            print(f"  파라미터: a={a:.3f}, b={b:.3f}, c={c:.3f}")
            print(f"  실제 근 ({true_num_roots}개): {[f'{r:.4f}' for r in true_roots]}")
            print(f"  예측 근 ({len(predicted_roots)}개): {[f'{r:.4f}' for r in predicted_roots]}")
            print(f"  예측 확률: {[f'{p:.3f}' for p in probs]}")
            
            if len(predicted_roots) == len(true_roots):
                correct_count += 1
                if len(true_roots) > 0:
                    true_sorted = sorted(true_roots)
                    pred_sorted = sorted(predicted_roots)
                    error = np.mean([abs(t - p) for t, p in zip(true_sorted, pred_sorted)])
                    total_error += error
                    print(f"  평균 오차: {error:.4f}")
                else:
                    print("  ✅ 정확 (0개 근)")
            else:
                print("  ❌ 근의 개수 불일치")
        
        test_samples = min(20, len(X_test))
        accuracy = correct_count / test_samples
        avg_error = total_error / max(correct_count, 1)
        
        print(f"\n=== 평가 결과 ===")
        print(f"근의 개수 정확도: {accuracy:.1%}")
        print(f"평균 근의 값 오차: {avg_error:.4f}")
        
        return accuracy, avg_error

# 메인 실행 코드
if __name__ == "__main__":
    # CSV 파일 경로 설정
    csv_file_path = 'neural_network_dataset.csv'  # 파일 경로를 실제 경로로 변경하세요
    
    # 예측기 생성
    predictor = RootPredictor(csv_file_path)
    
    # 학습 실행
    history = predictor.train(
        epochs=200,
        batch_size=128,
        learning_rate=0.01,
        validation_split=0.2
    )
    
    # 테스트 세트로 성능 평가
    predictor.evaluate_on_test_set(test_ratio=0.1)
    
    # 개별 예측 테스트
    print("\n=== 개별 예측 테스트 ===")
    test_cases = [
        (1.0, 2.0, 3.0),
        (-1.0, 5.0, -2.0),
        (0.5, -3.0, 1.0),
        (2.0, 0.0, 4.0),
        (-3.262918288808181,-31.773741547149605,-24.914749445625795)
    ]
    
    for a, b, c in test_cases:
        try:
            predicted_roots, probs = predictor.predict_roots(a, b, c)
            print(f"a={a}, b={b}, c={c}")
            print(f"  예측된 근: {predicted_roots}")
            print(f"  예측 확률: {[f'{p:.3f}' for p in probs]}")
        except Exception as e:
            print(f"a={a}, b={b}, c={c} → 예측 실패: {e}")
        print()
    
    print("학습된 모델이 'best_root_model.pth'에 저장되었습니다!")