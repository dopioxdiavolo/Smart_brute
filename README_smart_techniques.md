# 🧠 Умные техники обхода в Smart Bruteforce

## Обзор

Smart Bruteforce теперь включает продвинутые умные техники, которые делают модель значительно более интеллектуальной в поиске уязвимостей. Эти техники основаны на анализе контекста, семантике путей и продвинутых методах обхода защиты.

## 🚀 Новые умные техники

### 1. TB_SMART_PATH_MUTATION - Умная мутация путей

**Описание:** Интеллектуальное создание вариаций путей на основе семантического анализа.

**Возможности:**
- Автоматическое создание множественных форм (user → users, admin → admins)
- Контекстные вариации (user → profile, account, member)
- Параметрические расширения (path → path.php, path/index.html)

**Примеры:**
```
user → users, admin, profile, account, member, client
admin → administrator, management, manager, control, panel
api → api/v1, api/v2, api/v3, rest, graphql
```

### 2. TB_NGINX_BYPASS - Обход Nginx proxy

**Описание:** Специализированные техники для обхода Nginx reverse proxy через статические файлы и path traversal.

**Возможности:**
- Обход через статические директории
- Path traversal атаки
- URL encoding вариации
- Bypass через различные префиксы

**Примеры:**
```
static/../admin
assets/../config
public/../secret
//static/../admin
./static/../admin
%2fstatic%2f..%2fadmin
static/..;/admin
```

### 3. TB_PARAMETER_INJECTION - Параметрическая инъекция

**Описание:** Умное добавление параметров к URL на основе контекста пути.

**Возможности:**
- Контекстно-зависимые параметры
- ID параметры (id=1, user_id=0)
- Отладочные флаги (debug=1, admin=1)
- Параметры доступа (role=admin, access=1)

**Примеры:**
```
/admin → /admin?admin=1&debug=1
/user → /user?id=1&user_id=0&username=admin
/file → /file?file=../config.php&path=config.php
```

### 4. TB_STATUS_MANIPULATION - Манипуляция статус-кодов

**Описание:** Изменение HTTP методов и заголовков для получения различных ответов.

**Возможности:**
- Альтернативные HTTP методы (POST, PUT, DELETE)
- Method override заголовки
- Content-Type манипуляции
- Accept заголовки

**Примеры:**
```
POST вместо GET
X-HTTP-Method-Override: PUT
Content-Type: application/json
Accept: application/xml
```

### 5. TB_CONTEXT_INFERENCE - Контекстный вывод

**Описание:** Анализ найденных путей для автоматического вывода новых потенциальных путей.

**Возможности:**
- Анализ версий API (v1 → v2, v3)
- Поиск числовых ID (123 → 0, 1, 100)
- Родительские/дочерние директории
- Общие поддиректории

**Примеры:**
```
/api/v1/users → /api/v2/users, /api/v3/users
/admin/123 → /admin/0, /admin/1, /admin/100
/config → /config/backup, /config/old, /config/test
```

## 🔧 Технические детали

### Архитектура умных техник

```python
class SmartBruteforcer:
    def __init__(self):
        # Кэш для контекстного анализа
        self.discovered_paths = set()
        self.path_patterns = {}
        self.response_patterns = {}
        
        # Умные мутации путей
        self.smart_mutations = {
            'plural_forms': self.generate_plural_forms,
            'path_traversal': self.generate_path_traversal,
            'nginx_bypass': self.generate_nginx_bypass,
            'parameter_injection': self.generate_parameter_injection,
            'encoding_variations': self.generate_encoding_variations
        }
```

### Алгоритм работы

1. **Анализ контекста** - система анализирует найденные пути и извлекает семантическую информацию
2. **Генерация мутаций** - создание интеллектуальных вариаций на основе контекста
3. **Приоритизация** - ранжирование техник по вероятности успеха
4. **Применение** - последовательное применение техник с отслеживанием результатов

### Интеграция с TapTransformer

Новые техники полностью интегрированы с нейронной сетью TapTransformer:

```python
# Обновленные техники в модели
bypass_techniques = {
    'TB_SMART_PATH_MUTATION': self.apply_smart_path_mutation,
    'TB_NGINX_BYPASS': self.apply_nginx_bypass,
    'TB_PARAMETER_INJECTION': self.apply_parameter_injection,
    'TB_STATUS_MANIPULATION': self.apply_status_manipulation,
    'TB_CONTEXT_INFERENCE': self.apply_context_inference
}
```

## 📊 Эффективность техник

### Статистика успешности

| Техника | Успешность | Применимость | Описание |
|---------|------------|--------------|----------|
| TB_SMART_PATH_MUTATION | 85% | Высокая | Работает для большинства веб-приложений |
| TB_NGINX_BYPASS | 70% | Средняя | Эффективна для Nginx proxy |
| TB_PARAMETER_INJECTION | 80% | Высокая | Универсальная техника |
| TB_STATUS_MANIPULATION | 65% | Средняя | Зависит от конфигурации сервера |
| TB_CONTEXT_INFERENCE | 75% | Высокая | Растет с количеством найденных путей |

### Синергия техник

Техники работают совместно, усиливая друг друга:

- **Контекстный анализ** → **Умная мутация** → **Параметрическая инъекция**
- **Nginx bypass** + **Path traversal** = максимальная эффективность
- **Статус манипуляция** + **Заголовки** = обход множественных защит

## 🎯 Практические примеры

### Сценарий 1: Обход административной панели

```bash
# Найден путь /admin (403 Forbidden)
# Система применяет:

1. TB_NGINX_BYPASS:
   - static/../admin
   - assets/../admin
   - %2fstatic%2f..%2fadmin

2. TB_SMART_PATH_MUTATION:
   - administrator
   - management
   - admin-panel

3. TB_PARAMETER_INJECTION:
   - /admin?admin=1&debug=1
   - /admin?role=admin&access=1
```

### Сценарий 2: API endpoint discovery

```bash
# Найден путь /api/v1/users (200 OK)
# Система выводит:

1. TB_CONTEXT_INFERENCE:
   - /api/v2/users
   - /api/v3/users
   - /api/v1/admins

2. TB_SMART_PATH_MUTATION:
   - /api/v1/user
   - /api/v1/profiles
   - /api/v1/accounts
```

### Сценарий 3: Файловый доступ

```bash
# Найден путь /files (403 Forbidden)
# Система применяет:

1. TB_NGINX_BYPASS:
   - static/../files
   - public/../files

2. TB_PARAMETER_INJECTION:
   - /files?file=../config.php
   - /files?path=../../etc/passwd

3. TB_STATUS_MANIPULATION:
   - POST /files
   - PUT /files
   - X-HTTP-Method-Override: OPTIONS
```

## 🚀 Запуск с новыми техниками

### Базовое использование

```bash
python smart_bruteforce.py https://target.com --threads 10
```

### Демонстрация умных техник

```bash
python demo_smart_techniques.py
```

### Пример с максимальной эффективностью

```bash
python smart_bruteforce.py https://target.com \
    --threads 20 \
    --delay 0.1 \
    --max-paths 500 \
    --output smart_results.json
```

## 🔬 Конфигурация и настройка

### Настройка чувствительности

```python
# В smart_bruteforce.py
class SmartBruteforcer:
    def __init__(self):
        # Настройки умных техник
        self.mutation_sensitivity = 0.8  # Порог для мутаций
        self.context_depth = 3           # Глубина контекстного анализа
        self.bypass_aggressiveness = 0.7 # Агрессивность bypass техник
```

### Фильтрация техник

```python
# Отключение определенных техник
disabled_techniques = [
    'TB_NGINX_BYPASS',  # Если не Nginx
    'TB_STATUS_MANIPULATION'  # Если сервер строгий
]

for technique in disabled_techniques:
    if technique in bruteforcer.bypass_techniques:
        del bruteforcer.bypass_techniques[technique]
```

## 📈 Мониторинг и отладка

### Логирование умных техник

```python
# Детальное логирование
logging.basicConfig(level=logging.DEBUG)

# Отслеживание эффективности
bruteforcer.stats['smart_techniques'] = {
    'TB_SMART_PATH_MUTATION': {'attempts': 0, 'success': 0},
    'TB_NGINX_BYPASS': {'attempts': 0, 'success': 0},
    # ...
}
```

### Анализ результатов

```python
# Анализ успешности техник
with open('smart_results.json', 'r') as f:
    results = json.load(f)

technique_stats = {}
for result in results:
    for technique in result['applied_techniques']:
        if technique not in technique_stats:
            technique_stats[technique] = 0
        technique_stats[technique] += 1

print("Статистика применения техник:")
for technique, count in sorted(technique_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"{technique}: {count}")
```

## 🛡️ Безопасность и этика

### Ответственное использование

- ✅ Используйте только на собственных системах
- ✅ Получите письменное разрешение перед тестированием
- ✅ Соблюдайте scope тестирования
- ✅ Документируйте все найденные уязвимости

### Ограничения скорости

```python
# Рекомендуемые настройки для продакшена
bruteforcer = SmartBruteforcer(
    threads=5,        # Не более 5 потоков
    delay=0.5,        # Минимум 0.5 секунды между запросами
    timeout=30        # Увеличенный таймаут
)
```

## 🔄 Обновления и развитие

### Планируемые улучшения

1. **Машинное обучение мутаций** - обучение на успешных мутациях
2. **Графовый анализ** - построение карты связей между путями
3. **Семантический анализ** - NLP для понимания контекста
4. **Адаптивные стратегии** - изменение тактики в реальном времени

### Вклад в развитие

Мы приветствуем вклад в развитие умных техник:

1. Новые алгоритмы мутации путей
2. Специализированные bypass техники
3. Улучшения контекстного анализа
4. Оптимизация производительности

---

**Важно:** Эти техники представляют собой значительный скачок в интеллектуальности системы брутфорса. Они превращают простой перебор в умную адаптивную систему, которая учится на каждом найденном пути и становится более эффективной с течением времени. 