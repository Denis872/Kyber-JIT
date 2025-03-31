#!/usr/bin/env python3

import time
import inspect
import functools
import json
import csv
from collections import defaultdict
from kyber_py.kyber.default_parameters import Kyber512, Kyber768, Kyber1024
from kyber_py.polynomials.polynomials import PolynomialRing

class PolynomialBenchmark:
    def __init__(self, kyber_version):
        self.kyber_version = kyber_version
        self.kyber_instance = {
            "Kyber512": Kyber512,
            "Kyber768": Kyber768,
            "Kyber1024": Kyber1024
        }[kyber_version]
        
        # Получаем экземпляр полиномиального кольца
        self.ring = self.kyber_instance.M.ring
        
        # Счетчики
        self.func_calls = defaultdict(int)
        self.operation_counts = defaultdict(int)
        self.execution_times = defaultdict(float)
    
    def reset_counters(self):
        self.func_calls.clear()
        self.operation_counts.clear()
        self.execution_times.clear()
    
    def instrument_ring(self):
        """Инструментирует все методы полиномиального кольца"""
        self.original_methods = {}
        
        for name, method in inspect.getmembers(self.ring.__class__, inspect.isfunction):
            if not name.startswith('__'):
                # Сохраняем оригинальный метод
                self.original_methods[name] = getattr(self.ring.__class__, name)
                
                # Заменяем на инструментированный
                setattr(self.ring.__class__, name, self._create_wrapper(name, method))
    
    def restore_original_methods(self):
        """Восстанавливает оригинальные методы"""
        for name, method in self.original_methods.items():
            setattr(self.ring.__class__, name, method)
    
    def _create_wrapper(self, name, method):
        """Создает обертку для метода с отслеживанием вызовов и времени"""
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            qualified_name = f"{self.ring.__class__.__name__}.{name}"
            self.func_calls[qualified_name] += 1
            
            # Подсчет операций в исходном коде
            try:
                source = inspect.getsource(method)
                operators = ['+', '-', '*', '/', '%', '@', '==', '!=', '<', '>', '<=', '>=']
                
                for op in operators:
                    op_count = source.count(op)
                    if op_count > 0:
                        self.operation_counts[f"{qualified_name}:{op}"] += op_count
            except:
                pass
                
            # Вызов оригинального метода
            result = method(*args, **kwargs)
            
            # Замер времени
            elapsed = time.time() - start_time
            self.execution_times[qualified_name] += elapsed
            
            return result
            
        return wrapper
    
    def benchmark_polynomial_operations(self, iterations=100):
        """Запускает бенчмарк основных полиномиальных операций"""
        print(f"\nБенчмарк полиномиальных операций для {self.kyber_version}")
        print("=" * 80)
        
        self.reset_counters()
        self.instrument_ring()
        
        # Создаем тестовые полиномы
        print(f"Генерация тестовых полиномов...")
        
        # Случайные полиномы
        polys = []
        for i in range(5):
            coeffs = [self.kyber_instance.random_bytes(1)[0] % self.ring.q for _ in range(self.ring.n)]
            polys.append(self.ring(coeffs))
        
        # Запуск бенчмарка отдельных операций
        operations = [
            ("NTT", lambda: polys[0].to_ntt()),
            ("Inverse NTT", lambda: polys[0].to_ntt().from_ntt()),
            ("Addition", lambda: polys[0] + polys[1]),
            ("Subtraction", lambda: polys[0] - polys[1]),
            ("Multiplication", lambda: polys[0] * polys[1]),
            ("Multiplication (NTT domain)", lambda: polys[0].to_ntt() * polys[1].to_ntt()),
            ("Compression/Decompression", lambda: polys[0].compress(4).decompress(4)),
            ("Centered binomial distribution", lambda: self.ring.cbd(self.kyber_instance.random_bytes(128), 2))
        ]
        
        # Таблица результатов для отдельных операций
        op_results = []
        
        for op_name, op_func in operations:
            # Сбрасываем счетчики перед каждой операцией
            self.reset_counters()
            
            # Разогрев
            op_func()
            
            print(f"Тестирование операции: {op_name}...")
            
            # Основной тест
            start_time = time.time()
            for _ in range(iterations):
                op_func()
            total_time = time.time() - start_time
            
            # Сохраняем результаты
            avg_time = total_time / iterations
            op_results.append({
                "operation": op_name,
                "average_time": avg_time,
                "total_time": total_time,
                "iterations": iterations,
                "function_calls": dict(self.func_calls),
                "operation_counts": dict(self.operation_counts)
            })
            
            print(f"  Среднее время: {avg_time*1000:.3f} мс ({total_time:.3f} с всего)")
        
        # Тест полного шифрования/дешифрования
        self.reset_counters()
        print("\nТестирование полного цикла шифрования Kyber...")
        
        start_time = time.time()
        pk, sk = self.kyber_instance.keygen()
        key, ciphertext = self.kyber_instance.encaps(pk)
        key2 = self.kyber_instance.decaps(sk, ciphertext)
        assert key == key2, "Ошибка: ключи не совпадают"
        total_time = time.time() - start_time
        
        print(f"Время полного цикла: {total_time:.3f} с")
        
        # Восстанавливаем оригинальные методы
        self.restore_original_methods()
        
        # Анализ результатов
        self._analyze_and_save_results(op_results, total_time)
        
        return op_results
    
    def _analyze_and_save_results(self, op_results, total_time):
        """Анализирует и сохраняет результаты бенчмарка"""
        # Сохраняем общие результаты в JSON
        results = {
            "version": self.kyber_version,
            "total_time": total_time,
            "operations": op_results
        }
        
        json_filename = f"{self.kyber_version.lower()}_polynomial_benchmark.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        # Создаем CSV с детальными результатами по времени
        csv_filename = f"{self.kyber_version.lower()}_polynomial_timing.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Операция", "Среднее время (мс)", "Всего вызовов функций"])
            
            for op in op_results:
                total_calls = sum(op["function_calls"].values())
                writer.writerow([
                    op["operation"], 
                    op["average_time"] * 1000, 
                    total_calls
                ])
        
        print(f"\nРезультаты сохранены в {json_filename} и {csv_filename}")
        
        # Вывод TOP-10 наиболее затратных функций
        print("\nTOP-10 самых затратных функций (по количеству вызовов):")
        all_calls = defaultdict(int)
        for op in op_results:
            for func, count in op["function_calls"].items():
                all_calls[func] += count
        
        for func, count in sorted(all_calls.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{func}: {count} вызовов")

def run_all_benchmarks():
    print("Бенчмарк полиномиальных операций для всех версий Kyber")
    print("=" * 80)
    
    versions = ["Kyber512", "Kyber768", "Kyber1024"]
    results = {}
    
    for version in versions:
        benchmark = PolynomialBenchmark(version)
        results[version] = benchmark.benchmark_polynomial_operations()
    
    # Сравнительная таблица времени выполнения операций
    print("\n\nСравнительная таблица времени выполнения (мс):")
    print("=" * 80)
    
    # Получаем все уникальные операции
    operations = set()
    for version_results in results.values():
        for op in version_results:
            operations.add(op["operation"])
    
    # Заголовок таблицы
    header = ["Операция"] + versions
    print(" | ".join(header))
    print("-" * (sum(len(h) for h in header) + len(header) * 3))
    
    # Данные таблицы
    for op_name in sorted(operations):
        row = [op_name]
        for version in versions:
            version_results = results[version]
            op_time = next((op["average_time"] * 1000 for op in version_results 
                           if op["operation"] == op_name), "N/A")
            row.append(f"{op_time:.3f}" if isinstance(op_time, float) else op_time)
        print(" | ".join(row))

if __name__ == "__main__":
    run_all_benchmarks() 