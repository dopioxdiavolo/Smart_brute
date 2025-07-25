#!/usr/bin/env python3
"""
Пример использования Smart Bruteforce Script с TapTransformer
"""

from smart_bruteforce import SmartBruteforcer
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)

def check_model_availability():
    """Проверка доступности TapTransformer модели"""
    model_file = "best_model.pth"
    encoders_file = "encoders.pkl"
    
    has_files = os.path.exists(model_file) and os.path.exists(encoders_file)
    
    try:
        from tap_transformer_model import TapTransformer
        has_import = True
    except ImportError:
        has_import = False
    
    return has_files, has_import

def main():
    """Пример использования умного брутфорсера"""
    
    # Проверка доступности модели
    has_files, has_import = check_model_availability()
    
    print("🔍 Проверка доступности TapTransformer:")
    print(f"  Файлы модели: {'✅' if has_files else '❌'}")
    print(f"  Импорт модуля: {'✅' if has_import else '❌'}")
    print()
    
    # Создание экземпляра брутфорсера
    if has_files and has_import:
        print("🤖 Запуск с TapTransformer моделью")
        bruteforcer = SmartBruteforcer(
            wordlist_path='wordlists.txt',
            model_path='best_model.pth',
            encoders_path='encoders.pkl',
            threads=5,
            delay=0.5,
            timeout=5
        )
    else:
        print("🎯 Запуск в демо-режиме")
        if not has_import:
            print("  Для использования TapTransformer установите: pip install torch scikit-learn")
        if not has_files:
            print("  Для создания модели запустите: python3 tap_transformer_model.py")
        print()
        
        bruteforcer = SmartBruteforcer(
            wordlist_path='wordlists.txt',
            threads=5,
            delay=0.5,
            timeout=5
        )
    
    # Целевой URL (замените на свой тестовый сервер)
    target_url = "http://httpbin.org"  # Безопасный тестовый сервер
    
    print(f"Начинаем сканирование {target_url}")
    print("-" * 60)
    
    # Запуск сканирования с ограничением на 50 путей
    results = bruteforcer.run_scan(target_url, max_paths=50)
    
    # Сохранение результатов
    bruteforcer.save_results(results, 'demo_results.json')
    
    # Вывод сводки
    bruteforcer.print_summary(results)
    
    # Детальный анализ интересных результатов
    print("\nДЕТАЛЬНЫЙ АНАЛИЗ:")
    print("-" * 60)
    
    for result in results:
        if result.predicted_techniques or result.status_code in [200, 301, 302, 401, 403]:
            print(f"\nПуть: {result.path}")
            print(f"  Статус: {result.status_code}")
            print(f"  Тип аномалии: {result.anomaly_type}")
            print(f"  Размер ответа: {result.content_length} байт")
            print(f"  Время ответа: {result.response_time:.2f}s")
            
            if result.predicted_techniques:
                print(f"  Рекомендованные техники: {result.predicted_techniques}")
                print("  Вероятности:")
                for tech, prob in result.technique_probabilities.items():
                    print(f"    {tech}: {prob:.2f}")
            
            if result.applied_techniques:
                print(f"  Применённые техники: {result.applied_techniques}")
                if result.success:
                    print("  ✓ УСПЕШНЫЙ ОБХОД!")
    
    print(f"\nРезультаты сохранены в demo_results.json")

if __name__ == '__main__':
    main() 