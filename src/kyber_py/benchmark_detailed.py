#!/usr/bin/env python3

import time
import inspect
import functools
import json
from collections import defaultdict
from kyber_py.kyber.default_parameters import Kyber512, Kyber768, Kyber1024

class BenchmarkTracker:
    def __init__(self, name):
        self.name = name
        self.function_calls = defaultdict(int)
        self.operation_counts = defaultdict(int)
        self.execution_times = defaultdict(float)
        
    def reset(self):
        self.function_calls.clear()
        self.operation_counts.clear()
        self.execution_times.clear()
        
    def print_results(self):
        print(f"\nДетальные результаты для {self.name}:")
        print("=" * 80)
        
        print("\n{:<50} {:<10} {:<15}".format("Функция", "Вызовы", "Время (мс)"))
        print("-" * 80)
        
        # Сортируем по количеству вызовов (по убыванию)
        for func_name, count in sorted(self.function_calls.items(), key=lambda x: x[1], reverse=True):
            time_ms = self.execution_times[func_name] * 1000
            print("{:<50} {:<10} {:<15.3f}".format(func_name, count, time_ms))
        
        print("\n{:<50} {:<15}".format("Операция", "Количество"))
        print("-" * 80)
        
        # Группируем операции по функциям
        ops_by_func = {}
        for op_name, count in self.operation_counts.items():
            if ":" in op_name:
                func_name, op = op_name.split(":", 1)
                if func_name not in ops_by_func:
                    ops_by_func[func_name] = []
                ops_by_func[func_name].append((op, count))
        
        # Выводим операции по функциям
        for func_name, ops in sorted(ops_by_func.items()):
            print(f"\nФункция: {func_name}")
            for op, count in sorted(ops):
                print(f"  {op}: {count}")

# Декоратор для отслеживания вызовов функций, операций и времени выполнения
def track_function(tracker):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Увеличиваем счетчик вызовов для этой функции
            name = func.__qualname__
            tracker.function_calls[name] += 1
            
            # Получаем исходный код функции для подсчета операций
            try:
                source = inspect.getsource(func)
                
                # Простой подсчет операций
                operators = ['+', '-', '*', '/', '%', '@', '==', '!=', '<', '>', '<=', '>=']
                for op in operators:
                    tracker.operation_counts[f"{name}:{op}"] += source.count(op)
            except (OSError, IOError):
                # Если не удается получить исходный код
                pass
            
            # Выполняем оригинальную функцию
            result = func(*args, **kwargs)
            
            # Фиксируем время выполнения
            elapsed = time.time() - start_time
            tracker.execution_times[name] += elapsed
            
            return result
        
        return wrapper
    
    return decorator

# Функция для инструментирования класса
def instrument_class(cls, tracker):
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if not name.startswith('__'):
            setattr(cls, name, track_function(tracker)(getattr(cls, name)))
    return cls

# Функция для проведения бенчмарка одной версии Kyber
def benchmark_kyber_version(kyber_instance, version_name):
    print(f"\n\nБенчмарк для {version_name}")
    print("=" * 80)
    
    # Создаем трекеры для каждой операции
    keygen_tracker = BenchmarkTracker(f"{version_name} - Генерация ключей")
    encaps_tracker = BenchmarkTracker(f"{version_name} - Инкапсуляция")
    decaps_tracker = BenchmarkTracker(f"{version_name} - Декапсуляция")
    
    # Проводим тестирование генерации ключей
    print(f"\nТестирование генерации ключей для {version_name}...")
    instrument_class(kyber_instance.__class__, keygen_tracker)
    start_time = time.time()
    pk, sk = kyber_instance.keygen()
    keygen_time = time.time() - start_time
    print(f"Время генерации ключей: {keygen_time:.6f} секунд")
    keygen_tracker.print_results()
    
    # Сохраняем исходный класс для следующего шага
    kyber_instance.__class__ = type(kyber_instance.__class__.__name__, 
                                  kyber_instance.__class__.__bases__, 
                                  {k: v for k, v in kyber_instance.__class__.__dict__.items()})
    
    # Проводим тестирование инкапсуляции
    print(f"\nТестирование инкапсуляции для {version_name}...")
    instrument_class(kyber_instance.__class__, encaps_tracker)
    start_time = time.time()
    key, ciphertext = kyber_instance.encaps(pk)
    encaps_time = time.time() - start_time
    print(f"Время инкапсуляции: {encaps_time:.6f} секунд")
    encaps_tracker.print_results()
    
    # Сохраняем исходный класс для следующего шага
    kyber_instance.__class__ = type(kyber_instance.__class__.__name__, 
                                  kyber_instance.__class__.__bases__, 
                                  {k: v for k, v in kyber_instance.__class__.__dict__.items()})
    
    # Проводим тестирование декапсуляции
    print(f"\nТестирование декапсуляции для {version_name}...")
    instrument_class(kyber_instance.__class__, decaps_tracker)
    start_time = time.time()
    key2 = kyber_instance.decaps(sk, ciphertext)
    decaps_time = time.time() - start_time
    print(f"Время декапсуляции: {decaps_time:.6f} секунд")
    decaps_tracker.print_results()
    
    # Проверка корректности
    assert key == key2, "Ошибка: ключи не совпадают"
    
    # Сохраняем результаты в JSON
    results = {
        "version": version_name,
        "total_time": keygen_time + encaps_time + decaps_time,
        "keygen": {
            "time": keygen_time,
            "function_calls": dict(keygen_tracker.function_calls),
            "operation_counts": dict(keygen_tracker.operation_counts),
            "execution_times": dict(keygen_tracker.execution_times)
        },
        "encaps": {
            "time": encaps_time,
            "function_calls": dict(encaps_tracker.function_calls),
            "operation_counts": dict(encaps_tracker.operation_counts),
            "execution_times": dict(encaps_tracker.execution_times)
        },
        "decaps": {
            "time": decaps_time,
            "function_calls": dict(decaps_tracker.function_calls),
            "operation_counts": dict(decaps_tracker.operation_counts),
            "execution_times": dict(decaps_tracker.execution_times)
        }
    }
    
    # Записываем результаты в файл
    with open(f"{version_name.lower()}_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nРезультаты для {version_name} сохранены в файл {version_name.lower()}_benchmark.json")
    
    return results

if __name__ == "__main__":
    print("Детальный бенчмарк для алгоритма Kyber")
    print("=" * 80)
    
    # Запуск бенчмарков для всех версий Kyber
    results_512 = benchmark_kyber_version(Kyber512, "Kyber512")
    results_768 = benchmark_kyber_version(Kyber768, "Kyber768")
    results_1024 = benchmark_kyber_version(Kyber1024, "Kyber1024")
    
    # Сводка результатов
    print("\n\nСводная таблица времени выполнения (секунды):")
    print("=" * 80)
    print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "Версия", "Keygen", "Encaps", "Decaps", "Всего"))
    print("-" * 80)
    
    for results, name in [(results_512, "Kyber512"), 
                         (results_768, "Kyber768"), 
                         (results_1024, "Kyber1024")]:
        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            name, 
            results["keygen"]["time"], 
            results["encaps"]["time"], 
            results["decaps"]["time"],
            results["total_time"]
        )) 