import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                   MultiLabelBinarizer)
from sklearn.model_selection import train_test_split
import math
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention механизм"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Linear transformations and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + residual)


class FeedForward(nn.Module):
    """Feed Forward Network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    """Блок Transformer с Multi-Head Attention и Feed Forward"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x = self.attention(x, mask)
        x = self.feed_forward(x)
        return x


class TapTransformer(nn.Module):
    """
    TapTransformer для multi-label classification техник обхода защиты
    Адаптирован под структуру vulnerability.json датасета
    """
    
    def __init__(
        self,
        categorical_features: Dict[str, int],  # {feature_name: vocab_size}
        numerical_features: List[str],
        num_techniques: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        embedding_dim: int = 32
    ):
        super().__init__()
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.d_model = d_model
        
        # Embedding слои для категориальных признаков
        self.embeddings = nn.ModuleDict()
        for feature, vocab_size in categorical_features.items():
            self.embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
        
        # Проекция для числовых признаков
        self.numerical_projection = nn.Linear(len(numerical_features), embedding_dim)
        
        # Общая проекция в d_model размерность
        total_features = len(categorical_features) + 1  # +1 для числовых признаков
        self.feature_projection = nn.Linear(embedding_dim, d_model)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(d_model, max_len=total_features)
        
        # Transformer блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(d_model * total_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_techniques)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, categorical_inputs: Dict[str, torch.Tensor], numerical_inputs: torch.Tensor):
        batch_size = numerical_inputs.size(0)
        embeddings = []
        
        # Обработка категориальных признаков
        for feature in self.categorical_features:
            if feature in categorical_inputs:
                emb = self.embeddings[feature](categorical_inputs[feature])
                embeddings.append(emb)
        
        # Обработка числовых признаков
        numerical_emb = self.numerical_projection(numerical_inputs)
        embeddings.append(numerical_emb)
        
        # Объединение всех признаков
        x = torch.stack(embeddings, dim=1)  # [batch_size, num_features, embedding_dim]
        
        # Проекция в d_model размерность
        x = self.feature_projection(x)  # [batch_size, num_features, d_model]
        
        # Позиционное кодирование
        x = x.transpose(0, 1)  # [num_features, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, num_features, d_model]
        
        x = self.dropout(x)
        
        # Пропуск через Transformer блоки
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Глобальное объединение признаков
        x = x.view(batch_size, -1)  # Flatten
        
        # Классификация
        logits = self.classifier(x)
        
        return logits


class VulnerabilityDataset(Dataset):
    """Dataset для данных vulnerability.json"""
    
    def __init__(
        self,
        data: List[Dict],
        categorical_encoders: Dict[str, LabelEncoder],
        numerical_scaler: StandardScaler,
        technique_encoder: MultiLabelBinarizer,
        is_training: bool = True
    ):
        self.data = data
        self.categorical_encoders = categorical_encoders
        self.numerical_scaler = numerical_scaler
        self.technique_encoder = technique_encoder
        self.is_training = is_training
        
        # Категориальные признаки из датасета
        self.categorical_features = [
            'anomaly_type', 'encoding_type', 'error_message'
        ]
        
        # Числовые признаки из датасета
        self.numerical_features = [
            'status', 'depth', 'size', 'response_time_ms', 'retry_after_seconds',
            'has_location', 'has_set_cookie', 'has_www_authenticate', 
            'has_retry_after', 'has_auth_token', 'has_cookie', 'has_x_custom_auth',
            'error_text_present', 'content_fingerprint_hash'
        ]
        
        self.prepare_data()
    
    def prepare_data(self):
        """Подготовка данных из vulnerability.json"""
        processed_data = []
        
        for item in self.data:
            processed_item = {
                'anomaly_type': item['anomaly_type'],
                'path': item.get('path', 'unknown'),  # path может отсутствовать
                'reward': item.get('reward', 1.0)
            }
            
            # Проверяем, есть ли поле features (новый формат) или данные в корне (старый формат)
            if 'features' in item:
                # Новый формат - данные в features
                features = item['features']
                processed_item.update({
                    'status': features.get('status', 200),
                    'depth': features.get('depth', 1),
                    'size': features.get('size', 0),
                    'response_time_ms': features.get('response_time_ms', 0),
                    'retry_after_seconds': features.get('retry_after_seconds', 0),
                    'has_location': int(features.get('has_location', False)),
                    'has_set_cookie': int(features.get('has_set_cookie', False)),
                    'has_www_authenticate': int(features.get('has_www_authenticate', False)),
                    'has_retry_after': int(features.get('has_retry_after', False)),
                    'has_auth_token': int(features.get('has_auth_token', False)),
                    'has_cookie': int(features.get('has_cookie', False)),
                    'has_x_custom_auth': int(features.get('has_x_custom_auth', False)),
                    'error_text_present': int(features.get('error_text_present', False)),
                    'encoding_type': features.get('encoding_type', 'none'),
                    'error_message': features.get('error_message', 'none'),
                    'content_fingerprint_hash': hash(features.get('content_fingerprint', '')) % 10000
                })
            else:
                # Старый формат - данные в корне
                processed_item.update({
                    'status': item.get('status', 200),
                    'depth': item.get('depth', 1),
                    'size': item.get('size', 0),
                    'response_time_ms': item.get('response_time_ms', 0),
                    'retry_after_seconds': item.get('retry_after_seconds', 0),
                    'has_location': int(item.get('has_location', False)),
                    'has_set_cookie': int(item.get('has_set_cookie', False)),
                    'has_www_authenticate': int(item.get('has_www_authenticate', False)),
                    'has_retry_after': int(item.get('has_retry_after', False)),
                    'has_auth_token': int(item.get('has_auth_token', False)),
                    'has_cookie': int(item.get('has_cookie', False)),
                    'has_x_custom_auth': int(item.get('has_x_custom_auth', False)),
                    'error_text_present': int(item.get('error_text_present', False)),
                    'encoding_type': item.get('encoding_type', 'none'),
                    'error_message': item.get('error_message', 'none'),
                    'content_fingerprint_hash': item.get('content_fingerprint_hash', 0)
                })
            
            # Обработка applied_techniques (может быть techniques в старом формате)
            applied_techniques = item.get('applied_techniques', item.get('techniques', []))
            if isinstance(applied_techniques, str):
                if applied_techniques == 'none':
                    applied_techniques = []
                else:
                    applied_techniques = [applied_techniques]
            
            processed_item['applied_techniques'] = applied_techniques
            processed_data.append(processed_item)
        
        self.processed_data = processed_data
        
        # Создание DataFrame для удобства обработки
        df = pd.DataFrame(processed_data)
        
        # Обработка категориальных признаков
        self.categorical_data = {}
        for feature in self.categorical_features:
            if feature in df.columns:
                if self.is_training:
                    # Обучаем энкодер
                    self.categorical_data[feature] = torch.LongTensor(
                        self.categorical_encoders[feature].fit_transform(
                            df[feature].fillna('unknown').astype(str)
                        )
                    )
                else:
                    # Используем обученный энкодер
                    encoded = []
                    for val in df[feature].fillna('unknown').astype(str):
                        try:
                            encoded.append(self.categorical_encoders[feature].transform([val])[0])
                        except ValueError:
                            # Неизвестное значение
                            encoded.append(0)
                    self.categorical_data[feature] = torch.LongTensor(encoded)
        
        # Обработка числовых признаков
        numerical_data = df[self.numerical_features].fillna(0).values.astype(float)
        if self.is_training:
            numerical_data = self.numerical_scaler.fit_transform(numerical_data)
        else:
            numerical_data = self.numerical_scaler.transform(numerical_data)
        
        self.numerical_data = torch.FloatTensor(numerical_data)
        
        # Обработка целевой переменной
        techniques_list = df['applied_techniques'].tolist()
        
        if self.is_training:
            self.targets = torch.FloatTensor(
                self.technique_encoder.fit_transform(techniques_list)
            )
        else:
            self.targets = torch.FloatTensor(
                self.technique_encoder.transform(techniques_list)
            )
        
        # Веса для семплов (reward)
        self.sample_weights = torch.FloatTensor(df['reward'].values)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        categorical_inputs = {
            feature: self.categorical_data[feature][idx]
            for feature in self.categorical_features
            if feature in self.categorical_data
        }
        
        numerical_inputs = self.numerical_data[idx]
        targets = self.targets[idx]
        sample_weight = self.sample_weights[idx]
        
        return categorical_inputs, numerical_inputs, targets, sample_weight


class SmartBruteTrainer:
    """Тренер для модели умного брутфорса"""
    
    def __init__(
        self,
        model: TapTransformer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> float:
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            categorical_inputs, numerical_inputs, targets, sample_weights = batch
            
            # Перенос на устройство
            categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(self.device)
            targets = targets.to(self.device)
            sample_weights = sample_weights.to(self.device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            logits = self.model(categorical_inputs, numerical_inputs)
            
            # Вычисление loss с весами
            loss_per_sample = self.criterion(logits, targets).mean(dim=1)
            weighted_loss = (loss_per_sample * sample_weights).mean()
            
            # Обратный проход
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += weighted_loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Оценка модели"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                categorical_inputs, numerical_inputs, targets, sample_weights = batch
                
                # Перенос на устройство
                categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
                numerical_inputs = numerical_inputs.to(self.device)
                targets = targets.to(self.device)
                sample_weights = sample_weights.to(self.device)
                
                # Прямой проход
                logits = self.model(categorical_inputs, numerical_inputs)
                
                # Loss
                loss_per_sample = self.criterion(logits, targets).mean(dim=1)
                weighted_loss = (loss_per_sample * sample_weights).mean()
                total_loss += weighted_loss.item()
                
                # Предсказания
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                num_batches += 1
        
        # Объединение всех предсказаний
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Метрики
        avg_loss = total_loss / num_batches
        
        # Hamming Loss (для multi-label)
        predicted_labels = (all_predictions > 0.5).float()
        hamming_loss = torch.mean(torch.sum(predicted_labels != all_targets, dim=1) / all_targets.size(1))
        
        # Subset Accuracy (точное совпадение всех меток)
        subset_accuracy = torch.mean((predicted_labels == all_targets).all(dim=1).float())
        
        # F1 Score для multi-label
        tp = torch.sum(predicted_labels * all_targets, dim=0)
        fp = torch.sum(predicted_labels * (1 - all_targets), dim=0)
        fn = torch.sum((1 - predicted_labels) * all_targets, dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': avg_loss,
            'hamming_loss': hamming_loss.item(),
            'subset_accuracy': subset_accuracy.item(),
            'macro_f1': f1.mean().item(),
            'micro_f1': (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)).item()
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10
    ):
        """Полный цикл обучения"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Начало обучения...")
        print(f"Устройство: {self.device}")
        print(f"Параметров в модели: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Обучение
            train_loss = self.train_epoch(train_dataloader, optimizer)
            
            # Валидация
            val_metrics = self.evaluate(val_dataloader)
            val_loss = val_metrics['loss']
            
            # Обновление learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
            print(f"  Val Subset Accuracy: {val_metrics['subset_accuracy']:.4f}")
            print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохранение лучшей модели
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("  ✓ Новая лучшая модель сохранена")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping после {epoch+1} эпох")
                break
            
            print("-" * 50)
        
        # Загрузка лучшей модели
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Обучение завершено. Загружена лучшая модель.")


def load_vulnerability_data(file_path: str) -> List[Dict]:
    """Загрузка данных из vulnerability.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main(data_file='vulnerability.json'):
    """Основная функция для обучения модели"""
    print(f"=== TapTransformer для Smart Brute Force ({data_file}) ===\n")
    
    # Загрузка данных
    print(f"1. Загрузка данных из {data_file}...")
    try:
        data = load_vulnerability_data(data_file)
        print(f"Загружено {len(data)} записей")
    except FileNotFoundError:
        print(f"Файл {data_file} не найден!")
        return
    
    # Разделение на train/test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Инициализация энкодеров
    categorical_encoders = {
        'anomaly_type': LabelEncoder(),
        'encoding_type': LabelEncoder(),
        'error_message': LabelEncoder()
    }
    numerical_scaler = StandardScaler()
    technique_encoder = MultiLabelBinarizer()
    
    print("\n2. Создание датасетов...")
    # Создание датасетов
    train_dataset = VulnerabilityDataset(
        train_data, categorical_encoders, numerical_scaler, technique_encoder, is_training=True
    )
    test_dataset = VulnerabilityDataset(
        test_data, categorical_encoders, numerical_scaler, technique_encoder, is_training=False
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Получение размеров словарей
    categorical_features = {}
    for feature, encoder in categorical_encoders.items():
        categorical_features[feature] = len(encoder.classes_)
    
    numerical_features = train_dataset.numerical_features
    num_techniques = len(technique_encoder.classes_)
    
    print(f"Категориальные признаки: {categorical_features}")
    print(f"Числовые признаки: {len(numerical_features)}")
    print(f"Количество техник: {num_techniques}")
    print(f"Техники: {list(technique_encoder.classes_)}")
    
    print("\n3. Создание модели...")
    # Создание модели
    model = TapTransformer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        num_techniques=num_techniques,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        embedding_dim=32
    )
    
    print(f"Модель создана с {sum(p.numel() for p in model.parameters()):,} параметрами")
    
    print("\n4. Обучение модели...")
    # Обучение
    trainer = SmartBruteTrainer(model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=8
    )
    
    print("\n5. Финальная оценка...")
    # Финальная оценка
    final_metrics = trainer.evaluate(test_loader)
    print("Финальные метрики на тестовой выборке:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n6. Пример предсказания...")
    # Пример предсказания
    model.eval()
    with torch.no_grad():
        # Берем первый батч из тестовых данных
        batch = next(iter(test_loader))
        categorical_inputs, numerical_inputs, targets, _ = batch
        
        categorical_inputs = {k: v.to(trainer.device) for k, v in categorical_inputs.items()}
        numerical_inputs = numerical_inputs.to(trainer.device)
        
        logits = model(categorical_inputs, numerical_inputs)
        predictions = torch.sigmoid(logits)
        
        # Показываем первый пример
        pred_techniques = technique_encoder.inverse_transform((predictions[0] > 0.5).cpu().numpy().reshape(1, -1))[0]
        true_techniques = technique_encoder.inverse_transform(targets[0].numpy().reshape(1, -1))[0]
        
        print(f"Предсказанные техники: {list(pred_techniques)}")
        print(f"Истинные техники: {list(true_techniques)}")
        
        # Показываем топ-5 вероятностей
        probs_with_names = list(zip(technique_encoder.classes_, predictions[0].cpu().numpy()))
        probs_with_names.sort(key=lambda x: x[1], reverse=True)
        print("Топ-5 техник по вероятности:")
        for technique, prob in probs_with_names[:5]:
            print(f"  {technique}: {prob:.3f}")
    
    # Сохранение энкодеров
    print("\n7. Сохранение энкодеров...")
    encoders = {
        'categorical': categorical_encoders,
        'numerical': numerical_scaler,
        'technique': technique_encoder
    }
    
    import pickle
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Энкодеры сохранены в encoders.pkl")
    
    print("\n=== Обучение завершено ===")


if __name__ == "__main__":
    main() 