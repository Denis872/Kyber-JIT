#!/usr/bin/env python3

import time
import inspect
import functools
from collections import defaultdict
from kyber_py.kyber.default_parameters import Kyber512, Kyber768, Kyber1024

# Словарь для подсчета вызовов функций
function_calls = defaultdict(int)
# Словарь для подсчета операций
operation_counts = defaultdict(int)

# Декоратор для подсчета вызовов функций и операций
def count_calls_and_ops(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Увеличиваем счетчик вызовов для этой функции
        name = func.__qualname__
        function_calls[name] += 1
        
        # Получаем исходный код функции для подсчета операций
        source = inspect.getsource(func)
        
        # Простой подсчет операций (можно реализовать более сложный анализ)
        operators = ['+', '-', '*', '/', '%', '@', '==', '!=', '<', '>', '<=', '>=']
        for op in operators:
            operation_counts[f"{name}:{op}"] += source.count(op)
        
        # Выполняем оригинальную функцию
        result = func(*args, **kwargs)
        return result
    
    return wrapper

# Применяем декоратор ко всем методам класса Kyber
def instrument_class(cls):
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if not name.startswith('__'):
            setattr(cls, name, count_calls_and_ops(getattr(cls, name)))
    return cls

# Инструментируем классы Kyber
for kyber_class in [Kyber512, Kyber768, Kyber1024]:
    instrument_class(kyber_class.__class__)

# Сбрасываем счетчики перед новым запуском
def reset_counters():
    function_calls.clear()
    operation_counts.clear()

# Функция для отображения результатов
def display_results(kyber_name):
    print(f"Результаты для {kyber_name}:")
    print("-" * 60)
    print("{:<40} {:<10}".format("Функция", "Вызовы"))
    print("-" * 60)
    
    for func_name, count in sorted(function_calls.items()):
        print("{:<40} {:<10}".format(func_name, count))
    
    print("\n", "-" * 60)
    print("{:<40} {:<10}".format("Операция", "Количество"))
    print("-" * 60)
    
    for op_name, count in sorted(operation_counts.items()):
        print("{:<40} {:<10}".format(op_name, count))
    print("\n")

# Запускаем тестирование для Kyber512
def benchmark_kyber512():
    reset_counters()
    print("\nТестирование Kyber512...\n")
    start_time = time.time()
    
    # Генерация ключей
    pk, sk = Kyber512.keygen()
    
    # Инкапсуляция
    key, ciphertext = Kyber512.encaps(pk)
    
    # Декапсуляция
    key2 = Kyber512.decaps(sk, ciphertext)
    
    # Проверка корректности
    assert key == key2, "Ошибка: ключи не совпадают"
    
    elapsed = time.time() - start_time
    print(f"Время выполнения: {elapsed:.6f} секунд")
    display_results("Kyber512")

# Запускаем тестирование для Kyber768
def benchmark_kyber768():
    reset_counters()
    print("\nТестирование Kyber768...\n")
    start_time = time.time()
    
    # Генерация ключей
    pk, sk = Kyber768.keygen()
    
    # Инкапсуляция
    key, ciphertext = Kyber768.encaps(pk)
    
    # Декапсуляция
    key2 = Kyber768.decaps(sk, ciphertext)
    
    # Проверка корректности
    assert key == key2, "Ошибка: ключи не совпадают"
    
    elapsed = time.time() - start_time
    print(f"Время выполнения: {elapsed:.6f} секунд")
    display_results("Kyber768")

# Запускаем тестирование для Kyber1024
def benchmark_kyber1024():
    reset_counters()
    print("\nТестирование Kyber1024...\n")
    start_time = time.time()
    
    # Генерация ключей
    pk, sk = Kyber1024.keygen()
    
    # Инкапсуляция
    key, ciphertext = Kyber1024.encaps(pk)
    
    # Декапсуляция
    key2 = Kyber1024.decaps(sk, ciphertext)
    
    # Проверка корректности
    assert key == key2, "Ошибка: ключи не совпадают"
    
    elapsed = time.time() - start_time
    print(f"Время выполнения: {elapsed:.6f} секунд")
    display_results("Kyber1024")

if __name__ == "__main__":
    print("Бенчмарк для алгоритма Kyber")
    print("=" * 60)
    
    # Запуск всех бенчмарков
    benchmark_kyber512()
    benchmark_kyber768()
    benchmark_kyber1024() 