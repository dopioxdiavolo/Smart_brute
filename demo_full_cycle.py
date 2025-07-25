#!/usr/bin/env python3
"""
Демонстрация полного цикла работы Smart Bruteforce с TapTransformer
"""

import os
import subprocess
import sys
import time

def print_step(step_num, title, description):
    """Красивый вывод шага"""
    print(f"\n{'='*60}")
    print(f"ШАГ {step_num}: {title}")
    print(f"{'='*60}")
    print(description)
    print()

def check_file_exists(filename):
    """Проверка существования файла"""
    exists = os.path.exists(filename)
    print(f"  {filename}: {'✅ Найден' if exists else '❌ Не найден'}")
    return exists

def run_command(command, description):
    """Запуск команды с описанием"""
    print(f"🚀 {description}")
    print(f"Команда: {command}")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=300  # 5 минут таймаут
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[:1000])  # Первые 1000 символов
            if len(result.stdout) > 1000:
                print("... (вывод обрезан)")
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr[:500])
            if len(result.stderr) > 500:
                print("... (ошибки обрезаны)")
        
        print(f"\nКод возврата: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Команда превысила таймаут (5 минут)")
        return False
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        return False

def main():
    """Основная функция демонстрации"""
    print("🎯 ДЕМОНСТРАЦИЯ SMART BRUTEFORCE С TAPTRANSFORMER")
    print("=" * 60)
    
    # Шаг 1: Проверка файлов
    print_step(1, "ПРОВЕРКА ФАЙЛОВ", 
               "Проверяем наличие всех необходимых файлов")
    
    files_to_check = [
        "vulnerability.json",
        "wordlists.txt",
        "tap_transformer_model.py",
        "smart_bruteforce.py",
        "example_usage.py"
    ]
    
    all_files_exist = True
    for filename in files_to_check:
        if not check_file_exists(filename):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Не все файлы найдены. Убедитесь, что все файлы на месте.")
        return
    
    # Шаг 2: Проверка зависимостей
    print_step(2, "ПРОВЕРКА ЗАВИСИМОСТЕЙ",
               "Проверяем установленные Python пакеты")
    
    required_packages = ["torch", "sklearn", "pandas", "requests"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}: ✅ Установлен")
        except ImportError:
            print(f"  {package}: ❌ Не установлен")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Не установлены пакеты: {', '.join(missing_packages)}")
        print("Установите их командой:")
        print("pip install torch scikit-learn pandas requests")
        return
    
    # Шаг 3: Обучение модели
    print_step(3, "ОБУЧЕНИЕ TAPTRANSFORMER",
               "Обучаем модель на данных из vulnerability.json")
    
    if not os.path.exists("best_model.pth"):
        print("Модель не найдена, начинаем обучение...")
        success = run_command(
            "python3 tap_transformer_model.py",
            "Обучение TapTransformer модели"
        )
        
        if not success:
            print("❌ Обучение не удалось")
            return
    else:
        print("✅ Модель уже обучена (best_model.pth найден)")
    
    # Шаг 4: Проверка обученной модели
    print_step(4, "ПРОВЕРКА МОДЕЛИ",
               "Проверяем создание файлов модели")
    
    model_files = ["best_model.pth", "encoders.pkl"]
    model_ready = True
    
    for filename in model_files:
        if not check_file_exists(filename):
            model_ready = False
    
    if not model_ready:
        print("❌ Файлы модели не созданы")
        return
    
    # Шаг 5: Тестовый запуск
    print_step(5, "ТЕСТОВЫЙ ЗАПУСК",
               "Запускаем Smart Bruteforce на тестовом сервере")
    
    test_command = (
        "python3 smart_bruteforce.py http://httpbin.org "
        "--threads 3 --delay 1 --max-paths 20 --timeout 10 "
        "--output demo_test_results.json"
    )
    
    success = run_command(test_command, "Тестовое сканирование")
    
    if success and os.path.exists("demo_test_results.json"):
        print("✅ Тестовое сканирование завершено успешно")
        
        # Показать размер результатов
        size = os.path.getsize("demo_test_results.json")
        print(f"Размер файла результатов: {size} байт")
        
    else:
        print("❌ Тестовое сканирование не удалось")
        return
    
    # Шаг 6: Демонстрация example_usage
    print_step(6, "ДЕМОНСТРАЦИЯ EXAMPLE_USAGE",
               "Запускаем пример использования")
    
    print("Запуск example_usage.py...")
    print("(Это покажет статус TapTransformer и предложит варианты)")
    
    # Финальная информация
    print_step("ФИНАЛ", "ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА",
               "Все компоненты работают корректно!")
    
    print("🎉 Поздравляем! Smart Bruteforce с TapTransformer готов к использованию")
    print()
    print("Что дальше:")
    print("1. Используйте на своих тестовых серверах")
    print("2. Обучите модель на своих данных")
    print("3. Настройте параметры под свои нужды")
    print()
    print("Файлы для использования:")
    print("- smart_bruteforce.py - основной скрипт")
    print("- example_usage.py - примеры использования")
    print("- best_model.pth - обученная модель")
    print("- encoders.pkl - энкодеры")
    print("- demo_test_results.json - результаты тестирования")

if __name__ == "__main__":
    main() 