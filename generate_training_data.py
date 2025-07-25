#!/usr/bin/env python3
"""
Генератор обучающих данных для TapTransformer
Преобразует результаты сканирования в формат для обучения модели
"""

import json
import random
from typing import List, Dict, Any
import argparse


class TrainingDataGenerator:
    """Генератор обучающих данных"""
    
    def __init__(self):
        # Техники для разных типов аномалий
        self.technique_mapping = {
            'REDIRECT_ANOMALY': ['TB_REDIRECT_FOLLOW', 'TB_HEADER_BYPASS'],
            'AUTH_ANOMALY': ['TB_AUTH_ATTEMPT', 'TB_HEADER_BYPASS', 'TB_NGINX_BYPASS'],
            'SERVER_ERROR': ['TB_HEADER_BYPASS', 'TB_NGINX_BYPASS', 'TB_PARAMETER_INJECTION', 'TB_STATUS_MANIPULATION'],
            'STATUS_ANOMALY': ['TB_PATH_VARIATION', 'TB_PARAM_FUZZ', 'TB_SMART_PATH_MUTATION'],
            'COOKIE_ANOMALY': ['TB_COOKIE_TWEAK', 'TB_HEADER_BYPASS'],
            'TIME_ANOMALY': ['TB_RATE_CONTROL', 'TB_PARAMETER_INJECTION'],
            'SIZE_ANOMALY': ['TB_PARAMETER_INJECTION', 'TB_CONTEXT_INFERENCE'],
            'CONTENT_ANOMALY': ['TB_PARAMETER_INJECTION', 'TB_SMART_PATH_MUTATION']
        }
        
        # Дополнительные техники на основе статус-кодов
        self.status_techniques = {
            200: ['TB_DIRECT_ACCESS', 'TB_CONTEXT_INFERENCE'],
            301: ['TB_REDIRECT_FOLLOW'],
            302: ['TB_REDIRECT_FOLLOW', 'TB_HEADER_BYPASS'],
            401: ['TB_AUTH_ATTEMPT', 'TB_HEADER_BYPASS', 'TB_NGINX_BYPASS'],
            403: ['TB_HEADER_BYPASS', 'TB_NGINX_BYPASS', 'TB_AUTH_ATTEMPT'],
            404: ['TB_PATH_VARIATION', 'TB_PARAM_FUZZ', 'TB_SMART_PATH_MUTATION'],
            405: ['TB_DIRECT_ACCESS', 'TB_STATUS_MANIPULATION'],
            500: ['TB_PARAMETER_INJECTION', 'TB_HEADER_BYPASS'],
            502: ['TB_NGINX_BYPASS', 'TB_HEADER_BYPASS'],
            503: ['TB_RATE_CONTROL', 'TB_HEADER_BYPASS', 'TB_NGINX_BYPASS']
        }
    
    def load_scan_results(self, filename: str) -> List[Dict]:
        """Загрузка результатов сканирования"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def determine_reward(self, result: Dict) -> float:
        """Определение награды для результата"""
        status = result['status_code']
        anomaly_type = result['anomaly_type']
        
        # Высокая награда за интересные находки
        if status in [200, 301, 302]:
            return 1.0
        elif status in [401, 403]:
            return 0.8
        elif status == 404:
            return 0.4
        elif status in [500, 502, 503]:
            return 0.6
        else:
            return 0.3
    
    def generate_techniques(self, result: Dict) -> List[str]:
        """Генерация техник для результата"""
        techniques = []
        anomaly_type = result['anomaly_type']
        status = result['status_code']
        
        # Техники на основе типа аномалии
        if anomaly_type in self.technique_mapping:
            techniques.extend(self.technique_mapping[anomaly_type])
        
        # Дополнительные техники на основе статус-кода
        if status in self.status_techniques:
            techniques.extend(self.status_techniques[status])
        
        # Убираем дубликаты и добавляем случайность
        techniques = list(set(techniques))
        
        # Иногда добавляем контекстный анализ
        if random.random() < 0.3:
            techniques.append('TB_CONTEXT_INFERENCE')
        
        return techniques
    
    def convert_to_training_format(self, scan_results: List[Dict]) -> List[Dict]:
        """Преобразование результатов сканирования в формат обучения"""
        training_data = []
        
        for result in scan_results:
            # Извлекаем признаки
            features = result.get('features', {})
            
            # Создаем обучающий пример
            training_example = {
                'anomaly_type': result['anomaly_type'],
                'path': result['path'],
                'features': {
                    'status': result['status_code'],
                    'depth': result['path'].count('/'),
                    'size': result['content_length'],
                    'response_time_ms': int(result['response_time'] * 1000),
                    'retry_after_seconds': 0,
                    'has_location': int('location' in result.get('headers', {})),
                    'has_set_cookie': int('set-cookie' in result.get('headers', {})),
                    'has_www_authenticate': int('www-authenticate' in result.get('headers', {})),
                    'has_retry_after': int('retry-after' in result.get('headers', {})),
                    'has_auth_token': int('authorization' in result.get('headers', {})),
                    'has_cookie': int('cookie' in result.get('headers', {})),
                    'has_x_custom_auth': int(any(h.startswith('x-') for h in result.get('headers', {}))),
                    'error_text_present': int('error' in result.get('content', '').lower()),
                    'encoding_type': result.get('headers', {}).get('content-encoding', 'none'),
                    'error_message': self.extract_error_message(result),
                    'content_fingerprint': result.get('content', '')[:100]  # Первые 100 символов
                },
                'applied_techniques': self.generate_techniques(result),
                'reward': self.determine_reward(result)
            }
            
            training_data.append(training_example)
        
        return training_data
    
    def extract_error_message(self, result: Dict) -> str:
        """Извлечение сообщения об ошибке"""
        content = result.get('content', '').lower()
        
        if 'forbidden' in content:
            return 'Forbidden'
        elif 'unauthorized' in content:
            return 'Unauthorized'
        elif 'not found' in content:
            return 'Not Found'
        elif 'internal server error' in content:
            return 'Internal Server Error'
        elif 'bad gateway' in content:
            return 'Bad Gateway'
        elif 'service unavailable' in content:
            return 'Service Unavailable'
        else:
            return 'none'
    
    def augment_data(self, training_data: List[Dict]) -> List[Dict]:
        """Аугментация данных для увеличения разнообразия"""
        augmented_data = training_data.copy()
        
        # Добавляем синтетические примеры
        synthetic_examples = [
            {
                'anomaly_type': 'AUTH_ANOMALY',
                'path': '/admin/secret',
                'features': {
                    'status': 401,
                    'depth': 2,
                    'size': 1200,
                    'response_time_ms': 150,
                    'retry_after_seconds': 0,
                    'has_location': 0,
                    'has_set_cookie': 0,
                    'has_www_authenticate': 1,
                    'has_retry_after': 0,
                    'has_auth_token': 0,
                    'has_cookie': 0,
                    'has_x_custom_auth': 0,
                    'error_text_present': 1,
                    'encoding_type': 'none',
                    'error_message': 'Unauthorized',
                    'content_fingerprint': 'authentication required'
                },
                'applied_techniques': ['TB_AUTH_ATTEMPT', 'TB_HEADER_BYPASS', 'TB_NGINX_BYPASS'],
                'reward': 0.9
            },
            {
                'anomaly_type': 'SERVER_ERROR',
                'path': '/api/internal',
                'features': {
                    'status': 500,
                    'depth': 2,
                    'size': 2000,
                    'response_time_ms': 300,
                    'retry_after_seconds': 0,
                    'has_location': 0,
                    'has_set_cookie': 0,
                    'has_www_authenticate': 0,
                    'has_retry_after': 0,
                    'has_auth_token': 0,
                    'has_cookie': 0,
                    'has_x_custom_auth': 0,
                    'error_text_present': 1,
                    'encoding_type': 'none',
                    'error_message': 'Internal Server Error',
                    'content_fingerprint': 'internal server error occurred'
                },
                'applied_techniques': ['TB_PARAMETER_INJECTION', 'TB_HEADER_BYPASS', 'TB_STATUS_MANIPULATION'],
                'reward': 0.7
            },
            {
                'anomaly_type': 'REDIRECT_ANOMALY',
                'path': '/admin',
                'features': {
                    'status': 302,
                    'depth': 1,
                    'size': 500,
                    'response_time_ms': 100,
                    'retry_after_seconds': 0,
                    'has_location': 1,
                    'has_set_cookie': 0,
                    'has_www_authenticate': 0,
                    'has_retry_after': 0,
                    'has_auth_token': 0,
                    'has_cookie': 0,
                    'has_x_custom_auth': 0,
                    'error_text_present': 0,
                    'encoding_type': 'none',
                    'error_message': 'none',
                    'content_fingerprint': 'redirecting to login'
                },
                'applied_techniques': ['TB_REDIRECT_FOLLOW', 'TB_HEADER_BYPASS'],
                'reward': 1.0
            }
        ]
        
        augmented_data.extend(synthetic_examples)
        return augmented_data
    
    def save_training_data(self, training_data: List[Dict], filename: str):
        """Сохранение обучающих данных"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Обучающие данные сохранены в {filename}")
        print(f"Всего примеров: {len(training_data)}")
    
    def print_statistics(self, training_data: List[Dict]):
        """Вывод статистики по данным"""
        print("\n=== СТАТИСТИКА ОБУЧАЮЩИХ ДАННЫХ ===")
        
        # Статистика по типам аномалий
        anomaly_counts = {}
        for item in training_data:
            anomaly_type = item['anomaly_type']
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
        
        print("Типы аномалий:")
        for anomaly, count in sorted(anomaly_counts.items()):
            print(f"  {anomaly}: {count}")
        
        # Статистика по техникам
        technique_counts = {}
        for item in training_data:
            for technique in item['applied_techniques']:
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        print("\nТоп-10 техник:")
        for technique, count in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {technique}: {count}")
        
        # Статистика по наградам
        rewards = [item['reward'] for item in training_data]
        print(f"\nНаграды: мин={min(rewards):.2f}, макс={max(rewards):.2f}, среднее={sum(rewards)/len(rewards):.2f}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Генератор обучающих данных для TapTransformer')
    parser.add_argument('--input', required=True, help='Файл с результатами сканирования')
    parser.add_argument('--output', default='new_vulnerability_data.json', help='Выходной файл с обучающими данными')
    parser.add_argument('--augment', action='store_true', help='Добавить синтетические примеры')
    
    args = parser.parse_args()
    
    generator = TrainingDataGenerator()
    
    print(f"Загружаем результаты сканирования из {args.input}...")
    scan_results = generator.load_scan_results(args.input)
    
    print(f"Преобразуем {len(scan_results)} результатов в обучающие данные...")
    training_data = generator.convert_to_training_format(scan_results)
    
    if args.augment:
        print("Добавляем синтетические примеры...")
        training_data = generator.augment_data(training_data)
    
    generator.save_training_data(training_data, args.output)
    generator.print_statistics(training_data)


if __name__ == '__main__':
    main() 