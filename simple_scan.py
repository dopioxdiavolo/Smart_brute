#!/usr/bin/env python3
"""
Простой скрипт для первичного сканирования без модели
"""

import requests
import argparse
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


def load_wordlist(file_path: str) -> List[str]:
    """Загрузка словаря путей"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # Базовый словарь если файл не найден
        return [
            'admin', 'login', 'user', 'api', 'config', 'dashboard',
            'profile', 'settings', 'logout', 'register', 'auth',
            'users', 'accounts', 'account', 'signin', 'signup',
            'index', 'home', 'main', 'root', 'private', 'public',
            'secret', 'hidden', 'backup', 'test', 'debug', 'dev',
            'configuration', 'management', 'panel', 'control'
        ]


def scan_path(base_url: str, path: str, timeout: int = 10) -> Dict[str, Any]:
    """Сканирование одного пути"""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=False)
        
        return {
            'url': url,
            'path': path,
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'content_length': len(response.content),
            'headers': dict(response.headers),
            'anomaly_type': classify_anomaly(response),
            'features': extract_features(response, path),
            'predicted_techniques': generate_techniques(response, path),
            'applied_techniques': [],
            'success': False
        }
    except Exception as e:
        return {
            'url': url,
            'path': path,
            'status_code': 0,
            'response_time': 0,
            'content_length': 0,
            'headers': {},
            'anomaly_type': 'ERROR',
            'features': {'error': str(e)},
            'predicted_techniques': [],
            'applied_techniques': [],
            'success': False
        }


def classify_anomaly(response: requests.Response) -> str:
    """Классификация аномалии"""
    if response.status_code in [301, 302, 303, 307, 308]:
        return 'REDIRECT_ANOMALY'
    elif response.status_code == 429:
        return 'RATE_LIMIT_ANOMALY'
    elif response.status_code in [401, 403]:
        return 'AUTH_ANOMALY'
    elif response.status_code in [500, 502, 503, 504]:
        return 'SERVER_ERROR'
    elif response.status_code == 200:
        return 'SUCCESS'
    else:
        return 'STATUS_ANOMALY'


def extract_features(response: requests.Response, path: str) -> Dict[str, Any]:
    """Извлечение признаков"""
    return {
        'status': response.status_code,
        'size': len(response.content),
        'response_time_ms': response.elapsed.total_seconds() * 1000,
        'depth': path.count('/'),
        'path_length': len(path),
        'has_extension': '.' in path,
        'retry_after_seconds': 0,
        'content_type': response.headers.get('content-type', ''),
        'server': response.headers.get('server', ''),
        'encoding_type': response.encoding or 'utf-8',
        'error_message': 'none'
    }


def generate_techniques(response: requests.Response, path: str) -> List[str]:
    """Генерация техник на основе эвристики"""
    techniques = []
    
    if response.status_code in [401, 403]:
        techniques.extend(['TB_AUTH_ATTEMPT', 'TB_HEADER_BYPASS'])
    elif response.status_code in [301, 302]:
        techniques.append('TB_REDIRECT_FOLLOW')
    elif response.status_code in [500, 502, 503]:
        techniques.extend(['TB_NGINX_BYPASS', 'TB_PARAMETER_INJECTION'])
    elif response.status_code == 200:
        techniques.extend(['TB_PATH_VARIATION', 'TB_PARAM_FUZZ'])
    
    return techniques


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Простое сканирование для сбора данных')
    parser.add_argument('url', help='URL для сканирования')
    parser.add_argument('--max-paths', type=int, default=50, help='Максимальное количество путей')
    parser.add_argument('--delay', type=float, default=0.1, help='Задержка между запросами')
    parser.add_argument('--threads', type=int, default=10, help='Количество потоков')
    parser.add_argument('--timeout', type=int, default=10, help='Таймаут запроса')
    parser.add_argument('--output', default='scan_results.json', help='Файл для сохранения')
    parser.add_argument('--wordlist', default='wordlists.txt', help='Файл словаря')
    
    args = parser.parse_args()
    
    print(f"🔍 Простое сканирование: {args.url}")
    print(f"📊 Максимум путей: {args.max_paths}")
    print(f"⏱️  Задержка: {args.delay}с")
    print(f"🧵 Потоков: {args.threads}")
    
    # Загрузка словаря
    wordlist = load_wordlist(args.wordlist)
    paths_to_scan = wordlist[:args.max_paths]
    
    print(f"📝 Загружено {len(paths_to_scan)} путей для сканирования")
    
    results = []
    
    # Сканирование
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_path = {
            executor.submit(scan_path, args.url, path, args.timeout): path
            for path in paths_to_scan
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                
                # Вывод результата
                status = result['status_code']
                if status == 200:
                    print(f"  🟢 /{path} [{status}]")
                elif status in [301, 302]:
                    print(f"  🔄 /{path} [{status}]")
                elif status in [401, 403]:
                    print(f"  🔐 /{path} [{status}]")
                elif status in [500, 502, 503]:
                    print(f"  🔴 /{path} [{status}]")
                else:
                    print(f"  ⚪ /{path} [{status}]")
                
                # Задержка
                if args.delay > 0:
                    time.sleep(args.delay)
                    
            except Exception as e:
                print(f"  ❌ /{path} - Ошибка: {e}")
    
    # Сохранение результатов
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Результаты сохранены в {args.output}")
    print(f"📊 Всего проверено: {len(results)} путей")
    
    # Статистика
    status_counts = {}
    for result in results:
        status = result['status_code']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("📈 Статистика по статус-кодам:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} путей")


if __name__ == '__main__':
    main() 