#!/usr/bin/env python3
"""
Полный цикл сканирования: первичное сканирование -> обучение модели -> улучшенное сканирование
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime


def run_command(cmd, description=""):
    """Запуск команды с выводом"""
    print(f"\n🔄 {description}")
    print(f"Команда: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Ошибка выполнения: {description}")
        return False
    else:
        print(f"✅ Успешно: {description}")
        return True


def clean_old_results():
    """Очистка старых результатов"""
    files_to_remove = [
        'scan_results.json',
        'improved_scan_results.json', 
        'new_scan_results.json',
        'demo_results.json',
        'demo_test_results.json',
        'best_model.pth',
        'encoders.pkl',
        'best_model_backup.pth',
        'encoders_backup.pkl',
        'new_vulnerability_data.json',
        'combined_vulnerability_data.json'
    ]
    
    print("🧹 ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Удален: {file}")
        else:
            print(f"⚪ Не найден: {file}")


def main():
    """Основная функция полного цикла"""
    parser = argparse.ArgumentParser(description='Полный цикл сканирования с переобучением')
    parser.add_argument('url', help='URL для сканирования')
    parser.add_argument('--initial-paths', type=int, default=50, help='Количество путей для первичного сканирования')
    parser.add_argument('--final-paths', type=int, default=100, help='Количество путей для финального сканирования')
    parser.add_argument('--delay', type=float, default=0.3, help='Задержка между запросами')
    parser.add_argument('--threads', type=int, default=10, help='Количество потоков')
    parser.add_argument('--skip-clean', action='store_true', help='Пропустить очистку старых результатов')
    
    args = parser.parse_args()
    
    print("🚀 ПОЛНЫЙ ЦИКЛ SMART BRUTEFORCE СКАНИРОВАНИЯ")
    print("=" * 80)
    print(f"🎯 Цель: {args.url}")
    print(f"📊 Первичное сканирование: {args.initial_paths} путей")
    print(f"🎯 Финальное сканирование: {args.final_paths} путей")
    print(f"⏱️  Задержка: {args.delay}с")
    print(f"🧵 Потоков: {args.threads}")
    print(f"🕐 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Шаг 1: Очистка (если не пропущена)
    if not args.skip_clean:
        clean_old_results()
    
    # Шаг 2: Первичное сканирование для сбора данных
    print(f"\n📡 ШАГ 1: ПЕРВИЧНОЕ СКАНИРОВАНИЕ ({args.initial_paths} путей)")
    print("=" * 80)
    
    initial_cmd = f"python3 simple_scan.py {args.url} --max-paths {args.initial_paths} --delay {args.delay} --threads {args.threads} --output initial_scan_results.json"
    
    if not run_command(initial_cmd, "Первичное сканирование с базовыми техниками"):
        print("❌ Первичное сканирование не удалось")
        return False
    
    # Проверяем результаты первичного сканирования
    if not os.path.exists('initial_scan_results.json'):
        print("❌ Файл результатов первичного сканирования не найден")
        return False
    
    # Шаг 3: Генерация обучающих данных
    print(f"\n🧠 ШАГ 2: ГЕНЕРАЦИЯ ОБУЧАЮЩИХ ДАННЫХ")
    print("=" * 80)
    
    training_cmd = f"python3 generate_training_data.py --input initial_scan_results.json --output fresh_training_data.json --augment"
    
    if not run_command(training_cmd, "Генерация обучающих данных"):
        print("❌ Генерация обучающих данных не удалась")
        return False
    
    # Объединяем с базовыми данными
    print("🔗 Объединение с базовыми данными...")
    
    try:
        # Загружаем базовые данные
        with open('vulnerability.json', 'r') as f:
            base_data = json.load(f)
        
        # Загружаем новые данные
        with open('fresh_training_data.json', 'r') as f:
            new_data = json.load(f)
        
        # Объединяем
        combined_data = base_data + new_data
        
        # Сохраняем
        with open('combined_training_data.json', 'w') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Объединено: {len(base_data)} + {len(new_data)} = {len(combined_data)} записей")
        
    except Exception as e:
        print(f"❌ Ошибка объединения данных: {e}")
        return False
    
    # Шаг 4: Обучение модели
    print(f"\n🤖 ШАГ 3: ОБУЧЕНИЕ TAPRANSFORMER МОДЕЛИ")
    print("=" * 80)
    
    # Изменяем tap_transformer_model.py чтобы использовать новые данные
    train_cmd = f"python3 -c \"import tap_transformer_model; tap_transformer_model.main('combined_training_data.json')\""
    
    if not run_command(train_cmd, "Обучение TapTransformer модели"):
        print("❌ Обучение модели не удалось")
        return False
    
    # Проверяем что модель создана
    if not os.path.exists('best_model.pth') or not os.path.exists('encoders.pkl'):
        print("❌ Модель или энкодеры не созданы")
        return False
    
    print("✅ Модель успешно обучена!")
    
    # Шаг 5: Финальное сканирование с обученной моделью
    print(f"\n🎯 ШАГ 4: ФИНАЛЬНОЕ СКАНИРОВАНИЕ С ОБУЧЕННОЙ МОДЕЛЬЮ ({args.final_paths} путей)")
    print("=" * 80)
    
    final_cmd = f"python3 smart_bruteforce.py {args.url} --max-paths {args.final_paths} --delay {args.delay} --threads {args.threads} --output final_scan_results.json"
    
    if not run_command(final_cmd, "Финальное сканирование с обученной моделью"):
        print("❌ Финальное сканирование не удалось")
        return False
    
    # Шаг 6: Генерация отчета
    print(f"\n📊 ШАГ 5: ГЕНЕРАЦИЯ ОТЧЕТА")
    print("=" * 80)
    
    if os.path.exists('final_scan_results.json'):
        report_cmd = f"python3 vulnerability_report.py --input final_scan_results.json --format both"
        run_command(report_cmd, "Генерация детального отчета")
    
    # Финальная сводка
    print(f"\n🎉 ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН!")
    print("=" * 80)
    print(f"🕐 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Результаты сохранены в:")
    print(f"   • final_scan_results.json - JSON результаты")
    print(f"   • vulnerability_report.html - HTML отчет")  
    print(f"   • vulnerability_report.txt - Текстовый отчет")
    print(f"   • best_model.pth - Обученная модель")
    print(f"   • encoders.pkl - Энкодеры")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("🎯 Все этапы выполнены успешно!")
            sys.exit(0)
        else:
            print("❌ Выполнение прервано из-за ошибок")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Выполнение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1) 