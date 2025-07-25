#!/usr/bin/env python3
"""
Переобучение TapTransformer модели на новых данных
"""

import argparse
import json
import torch
import pickle
from tap_transformer_model import main as train_main
from tap_transformer_model import load_vulnerability_data
import shutil
import os


def backup_old_model():
    """Создание резервной копии старой модели"""
    if os.path.exists('best_model.pth'):
        shutil.copy('best_model.pth', 'best_model_backup.pth')
        print("✓ Резервная копия модели создана: best_model_backup.pth")
    
    if os.path.exists('encoders.pkl'):
        shutil.copy('encoders.pkl', 'encoders_backup.pkl')
        print("✓ Резервная копия энкодеров создана: encoders_backup.pkl")


def retrain_model(data_file: str):
    """Переобучение модели на новых данных"""
    print(f"=== ПЕРЕОБУЧЕНИЕ МОДЕЛИ НА ДАННЫХ: {data_file} ===\n")
    
    # Создаем резервную копию
    backup_old_model()
    
    # Временно заменяем файл данных
    original_file = 'vulnerability.json'
    backup_original = False
    
    if os.path.exists(original_file):
        shutil.copy(original_file, 'vulnerability_original_backup.json')
        backup_original = True
        print("✓ Резервная копия оригинальных данных создана")
    
    # Копируем новые данные
    shutil.copy(data_file, original_file)
    print(f"✓ Используем данные из {data_file}")
    
    try:
        # Запускаем обучение
        print("\n🚀 Начинаем переобучение модели...")
        train_main()
        print("\n✅ Переобучение завершено успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при переобучении: {e}")
        
        # Восстанавливаем старую модель
        if os.path.exists('best_model_backup.pth'):
            shutil.copy('best_model_backup.pth', 'best_model.pth')
            print("🔄 Восстановлена старая модель")
        
        if os.path.exists('encoders_backup.pkl'):
            shutil.copy('encoders_backup.pkl', 'encoders.pkl')
            print("🔄 Восстановлены старые энкодеры")
    
    finally:
        # Восстанавливаем оригинальный файл данных
        if backup_original:
            shutil.copy('vulnerability_original_backup.json', original_file)
            print("🔄 Восстановлен оригинальный файл данных")


def compare_models():
    """Сравнение старой и новой модели"""
    print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
    
    if not os.path.exists('best_model_backup.pth'):
        print("❌ Нет резервной копии для сравнения")
        return
    
    # Загружаем тестовые данные
    try:
        with open('combined_vulnerability_data.json', 'r') as f:
            test_data = json.load(f)[:20]  # Первые 20 примеров
        
        print(f"Тестируем на {len(test_data)} примерах...")
        
        # Здесь можно добавить код для сравнения производительности
        # Пока просто выводим информацию о размерах моделей
        
        old_size = os.path.getsize('best_model_backup.pth')
        new_size = os.path.getsize('best_model.pth')
        
        print(f"Размер старой модели: {old_size:,} байт")
        print(f"Размер новой модели: {new_size:,} байт")
        print(f"Изменение размера: {((new_size - old_size) / old_size * 100):+.1f}%")
        
    except Exception as e:
        print(f"❌ Ошибка при сравнении: {e}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Переобучение TapTransformer модели')
    parser.add_argument('--data', required=True, help='Файл с новыми обучающими данными')
    parser.add_argument('--compare', action='store_true', help='Сравнить старую и новую модель')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Файл данных не найден: {args.data}")
        return
    
    # Переобучение
    retrain_model(args.data)
    
    # Сравнение моделей
    if args.compare:
        compare_models()
    
    print("\n🎉 Процесс переобучения завершен!")
    print("📁 Файлы:")
    print("  - best_model.pth (новая модель)")
    print("  - encoders.pkl (новые энкодеры)")
    print("  - best_model_backup.pth (резервная копия)")
    print("  - encoders_backup.pkl (резервная копия)")


if __name__ == '__main__':
    main() 