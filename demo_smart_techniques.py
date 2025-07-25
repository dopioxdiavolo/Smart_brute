#!/usr/bin/env python3
"""
Демонстрация умных техник обхода в Smart Bruteforce
Показывает возможности новых продвинутых техник
"""

import json
import time
from smart_bruteforce import SmartBruteforcer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_smart_techniques():
    """Демонстрация умных техник"""
    
    print("🧠 ДЕМОНСТРАЦИЯ УМНЫХ ТЕХНИК ОБХОДА")
    print("=" * 60)
    
    # Создаем брутфорсер
    bruteforcer = SmartBruteforcer(
        threads=5,
        delay=0.2,
        timeout=10
    )
    
    # Примеры для демонстрации
    demo_cases = [
        {
            "name": "Умная мутация путей",
            "description": "Генерация множественных форм и контекстных вариаций",
            "examples": [
                "user -> users, admin, profile, account",
                "admin -> administrator, management, panel",
                "api -> api/v1, api/v2, rest, graphql"
            ]
        },
        {
            "name": "Nginx bypass техники",
            "description": "Обход через статические файлы и path traversal",
            "examples": [
                "static/../admin",
                "assets/../config",
                "public/../secret",
                "%2fstatic%2f..%2fadmin"
            ]
        },
        {
            "name": "Параметрическая инъекция",
            "description": "Умное добавление параметров на основе контекста",
            "examples": [
                "?debug=1&admin=1",
                "?id=1&user_id=0",
                "?file=../config.php",
                "?role=admin&access=1"
            ]
        },
        {
            "name": "Манипуляция статус-кодов",
            "description": "Изменение HTTP методов и заголовков",
            "examples": [
                "POST вместо GET",
                "X-HTTP-Method-Override: PUT",
                "Content-Type: application/json",
                "Accept: application/xml"
            ]
        },
        {
            "name": "Контекстный вывод",
            "description": "Анализ найденных путей для поиска новых",
            "examples": [
                "/api/v1/users -> /api/v2/users",
                "/admin/123 -> /admin/0, /admin/1",
                "/config -> /config/backup, /config/old"
            ]
        }
    ]
    
    # Демонстрация каждой техники
    for i, case in enumerate(demo_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 40)
        print(f"Описание: {case['description']}")
        print("Примеры:")
        for example in case['examples']:
            print(f"  • {example}")
    
    print("\n" + "=" * 60)
    print("🚀 ДЕМОНСТРАЦИЯ НА РЕАЛЬНОМ ПРИМЕРЕ")
    print("=" * 60)
    
    # Демонстрация умных мутаций
    demo_paths = ["user", "admin", "api", "config", "file"]
    
    for path in demo_paths:
        print(f"\n🔍 Анализ пути: '{path}'")
        
        # Генерация множественных форм
        plural_forms = bruteforcer.generate_plural_forms(path)
        if plural_forms:
            print(f"  Множественные формы: {plural_forms}")
        
        # Генерация контекстных вариаций
        context_variations = bruteforcer.generate_context_variations(path)
        if context_variations:
            print(f"  Контекстные вариации: {context_variations[:5]}")
        
        # Генерация параметрических вариаций
        param_variations = bruteforcer.generate_parameter_variations(path)
        if param_variations:
            print(f"  Параметрические вариации: {param_variations[:3]}")
        
        # Генерация умных параметров
        smart_params = bruteforcer.generate_smart_parameters(path)
        if smart_params:
            print(f"  Умные параметры: {smart_params[:3]}")
    
    print("\n" + "=" * 60)
    print("🔮 КОНТЕКСТНЫЙ АНАЛИЗ")
    print("=" * 60)
    
    # Демонстрация контекстного анализа
    context_examples = [
        "api/v1/users",
        "admin/dashboard/123",
        "config/database",
        "files/upload/temp"
    ]
    
    for path in context_examples:
        print(f"\n📊 Анализ: '{path}'")
        inferred_paths = bruteforcer.infer_paths_from_context(path)
        if inferred_paths:
            print(f"  Выведенные пути: {inferred_paths[:5]}")
    
    print("\n" + "=" * 60)
    print("🎯 NGINX BYPASS ТЕХНИКИ")
    print("=" * 60)
    
    # Демонстрация Nginx bypass
    target_paths = ["admin", "config", "secret", "private"]
    
    for path in target_paths:
        print(f"\n🚀 Nginx bypass для: '{path}'")
        
        # Генерация bypass паттернов
        bypass_patterns = [
            f"static/../{path}",
            f"assets/../{path}",
            f"public/../{path}",
            f"/static/../{path}",
            f"//static/../{path}",
            f"./static/../{path}",
            f"%2fstatic%2f..%2f{path}",
            f"static/..;/{path}"
        ]
        
        print(f"  Bypass паттерны:")
        for pattern in bypass_patterns[:4]:
            print(f"    • {pattern}")
    
    print("\n" + "=" * 60)
    print("✨ ЗАКЛЮЧЕНИЕ")
    print("=" * 60)
    
    print("""
Новые умные техники значительно расширяют возможности брутфорсера:

1. 🧠 УМНАЯ МУТАЦИЯ ПУТЕЙ
   - Автоматическое создание множественных форм
   - Контекстные вариации на основе семантики
   - Параметрические расширения

2. 🚀 NGINX BYPASS
   - Обход через статические директории
   - Path traversal атаки
   - URL encoding вариации

3. 💉 ПАРАМЕТРИЧЕСКАЯ ИНЪЕКЦИЯ
   - Контекстно-зависимые параметры
   - Умное добавление ID параметров
   - Отладочные и административные флаги

4. 🔄 СТАТУС МАНИПУЛЯЦИЯ
   - Альтернативные HTTP методы
   - Заголовки для обхода ограничений
   - Content-Type манипуляции

5. 🔮 КОНТЕКСТНЫЙ АНАЛИЗ
   - Вывод новых путей из найденных
   - Анализ версий API
   - Поиск родительских/дочерних директорий

Эти техники работают совместно, создавая интеллектуальную систему
поиска уязвимостей, которая адаптируется к структуре приложения.
    """)


def demo_real_world_scenario():
    """Демонстрация реального сценария использования"""
    
    print("\n" + "=" * 60)
    print("🌐 РЕАЛЬНЫЙ СЦЕНАРИЙ ИСПОЛЬЗОВАНИЯ")
    print("=" * 60)
    
    # Имитация реального сценария
    discovered_paths = [
        "/api/v1/users",
        "/admin/dashboard", 
        "/config/settings",
        "/files/upload"
    ]
    
    bruteforcer = SmartBruteforcer()
    
    print("📍 Найденные пути:")
    for path in discovered_paths:
        print(f"  • {path}")
    
    print("\n🔍 Умный анализ и генерация новых путей:")
    
    all_generated = set()
    
    for path in discovered_paths:
        print(f"\n   Анализ: {path}")
        
        # Убираем начальный слэш для анализа
        clean_path = path.lstrip('/')
        
        # Генерируем все виды мутаций
        mutations = []
        mutations.extend(bruteforcer.generate_plural_forms(clean_path))
        mutations.extend(bruteforcer.generate_context_variations(clean_path))
        mutations.extend(bruteforcer.generate_parameter_variations(clean_path))
        mutations.extend(bruteforcer.infer_paths_from_context(clean_path))
        
        # Уникальные мутации
        unique_mutations = list(set(mutations))[:5]
        
        for mutation in unique_mutations:
            all_generated.add(mutation)
            print(f"     → {mutation}")
    
    print(f"\n📊 Итого сгенерировано {len(all_generated)} новых путей для тестирования")
    
    # Демонстрация Nginx bypass
    print(f"\n🚀 Nginx bypass техники:")
    nginx_bypasses = []
    for path in discovered_paths:
        clean_path = path.lstrip('/')
        bypasses = [
            f"static/../{clean_path}",
            f"assets/../{clean_path}",
            f"%2fstatic%2f..%2f{clean_path.replace('/', '%2f')}"
        ]
        nginx_bypasses.extend(bypasses)
    
    for bypass in nginx_bypasses[:8]:
        print(f"     → {bypass}")
    
    print(f"\n💡 Общий результат: из {len(discovered_paths)} найденных путей")
    print(f"   система сгенерировала {len(all_generated) + len(nginx_bypasses)} новых вариантов!")


if __name__ == "__main__":
    demo_smart_techniques()
    demo_real_world_scenario() 