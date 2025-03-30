from ..utilities.utils import bit_count
from .polynomials_generic import PolynomialRing, Polynomial
import numba as nb
import numpy as np

@nb.njit
def _ntt_forward_transform(coeffs, zetas, q=3329):
    """
    Оптимизированная реализация прямого NTT преобразования
    """
    k, l = 1, 128
    coeffs_copy = coeffs.copy()
    
    while l >= 2:
        start = 0
        while start < 256:
            zeta = zetas[k]
            k = k + 1
            for j in range(start, start + l):
                t = (zeta * coeffs_copy[j + l]) % q
                coeffs_copy[j + l] = (coeffs_copy[j] - t) % q
                coeffs_copy[j] = (coeffs_copy[j] + t) % q
            start = start + 2 * l
        l = l >> 1

    # Финальное модульное сокращение
    for j in range(256):
        coeffs_copy[j] = coeffs_copy[j] % q
    
    return coeffs_copy

@nb.njit
def _ntt_inverse_transform(coeffs, zetas, ntt_f, q=3329):
    """
    Оптимизированная реализация обратного NTT преобразования
    """
    l, l_upper = 2, 128
    k = l_upper - 1
    coeffs_copy = coeffs.copy()
    
    while l <= 128:
        start = 0
        while start < 256:
            zeta = zetas[k]
            k = k - 1
            for j in range(start, start + l):
                t = coeffs_copy[j]
                coeffs_copy[j] = (t + coeffs_copy[j + l]) % q
                coeffs_copy[j + l] = (coeffs_copy[j + l] - t) % q
                coeffs_copy[j + l] = (zeta * coeffs_copy[j + l]) % q
            start = start + 2 * l
        l = l << 1

    for j in range(256):
        coeffs_copy[j] = (coeffs_copy[j] * ntt_f) % q
    
    return coeffs_copy

@nb.njit
def _ntt_base_mul(a0, a1, b0, b1, zeta, q=3329):
    """
    Оптимизированное базовое умножение для NTT
    """
    r0 = (a0 * b0 + zeta * a1 * b1) % q
    r1 = (a1 * b0 + a0 * b1) % q
    return r0, r1

@nb.njit
def _ntt_coefficient_mul(f_coeffs, g_coeffs, zetas, q=3329):
    """
    Оптимизированное умножение коэффициентов полиномов в NTT форме
    """
    new_coeffs = np.zeros(256, dtype=np.int32)
    for i in range(64):
        # Базовое умножение для первой пары
        a0, a1 = f_coeffs[4*i+0], f_coeffs[4*i+1]
        b0, b1 = g_coeffs[4*i+0], g_coeffs[4*i+1]
        zeta = zetas[64 + i]
        r0 = (a0 * b0 + zeta * a1 * b1) % q
        r1 = (a1 * b0 + a0 * b1) % q
        
        # Базовое умножение для второй пары
        a0, a1 = f_coeffs[4*i+2], f_coeffs[4*i+3]
        b0, b1 = g_coeffs[4*i+2], g_coeffs[4*i+3]
        zeta_neg = q - zetas[64 + i]  # -zeta mod q
        r2 = (a0 * b0 + zeta_neg * a1 * b1) % q
        r3 = (a1 * b0 + a0 * b1) % q
        
        # Сохраняем результаты
        new_coeffs[4*i+0] = r0
        new_coeffs[4*i+1] = r1
        new_coeffs[4*i+2] = r2
        new_coeffs[4*i+3] = r3
    
    return new_coeffs

@nb.njit
def _cbd_optimized(input_bytes_array, eta, q=3329):
    """
    Оптимизированная реализация Centered Binomial Distribution.
    Работает напрямую с байтами как numpy массивом.
    """
    coefficients = np.zeros(256, dtype=np.int32)
    mask = (1 << eta) - 1
    mask2 = (1 << 2 * eta) - 1
    
    # Расчет количества бит на одну выборку
    bits_per_sample = 2 * eta
    
    # Текущее значение, накопленные биты и позиция
    current_val = 0
    bits_accumulated = 0
    coef_index = 0
    
    # Проходим по всем байтам
    for i in range(len(input_bytes_array)):
        byte_val = input_bytes_array[i]
        
        # Добавляем 8 бит к текущим битам
        current_val |= (int(byte_val) << bits_accumulated)
        bits_accumulated += 8
        
        # Пока у нас достаточно бит для извлечения выборки
        while bits_accumulated >= bits_per_sample and coef_index < 256:
            # Извлекаем выборку
            x = current_val & mask2
            
            # Подсчет установленных битов
            a_bits = x & mask
            b_bits = (x >> eta) & mask
            
            a = 0
            b = 0
            
            # Подсчет установленных битов для a
            while a_bits > 0:
                a += a_bits & 1
                a_bits >>= 1
                
            # Подсчет установленных битов для b
            while b_bits > 0:
                b += b_bits & 1
                b_bits >>= 1
            
            # Сохраняем результат
            coefficients[coef_index] = (a - b) % q
            coef_index += 1
            
            # Сдвигаем биты для следующей выборки
            current_val >>= bits_per_sample
            bits_accumulated -= bits_per_sample
    
    return coefficients

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _compress_element(x, d, q=3329):
    """
    Векторизованная функция сжатия коэффициента
    Compute round((2^d / q) * x) % 2^d
    """
    t = 1 << d
    y = (t * x + 1664) // q  # 1664 = 3329 // 2
    return y % t

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _decompress_element(x, d, q=3329):
    """
    Векторизованная функция распаковки коэффициента
    Compute round((q / 2^d) * x)
    """
    t = 1 << (d - 1)
    y = (q * x + t) >> d
    return y

@nb.njit
def _encode_polynomial(coeffs, d):
    """
    Оптимизированное кодирование полинома в байты.
    Используем более безопасный подход для работы с большими числами.
    """
    # Создаем буфер для результата как numpy массив вместо bytearray
    result_bytes = np.zeros(32 * d, dtype=np.uint8)
    
    # Обрабатываем каждый коэффициент напрямую, не используя битовые сдвиги для больших чисел
    # вместо этого заполняем байты напрямую
    acc = 0  # аккумулятор для битов
    bit_pos = 0  # текущая позиция бита
    byte_pos = 0  # текущая позиция байта
    
    # Обрабатываем коэффициенты в правильном порядке
    for i in range(256):
        coef = coeffs[i]
        # Добавляем d бит от текущего коэффициента в аккумулятор
        acc |= (coef << bit_pos)
        bit_pos += d
        
        # Пока у нас есть хотя бы 8 бит, записываем их в результат
        while bit_pos >= 8:
            if byte_pos < len(result_bytes):
                result_bytes[byte_pos] = acc & 0xFF
                byte_pos += 1
                acc >>= 8
                bit_pos -= 8
            else:
                # Защита от переполнения буфера
                break
    
    # Записываем оставшиеся биты, если они есть
    if bit_pos > 0 and byte_pos < len(result_bytes):
        result_bytes[byte_pos] = acc & 0xFF
    
    return result_bytes

@nb.njit
def _decode_polynomial(input_bytes_array, d, n=256, q=3329):
    """
    Оптимизированное декодирование полинома из байтов.
    Работает напрямую с байтами как numpy массивом.
    """
    if d == 12:
        m = q
    else:
        m = 1 << d
        
    # Подготовим массив для коэффициентов
    coeffs = np.zeros(n, dtype=np.int32)
    
    mask = (1 << d) - 1
    bit_pos = 0  # текущая битовая позиция
    coef_idx = 0  # индекс текущего коэффициента
    current_val = 0  # текущее накопленное значение
    bits_in_val = 0  # сколько бит уже накоплено
        
    # Проходим по всем байтам входных данных
    for i in range(len(input_bytes_array)):
        byte_val = input_bytes_array[i]
        
        # Добавляем 8 бит к текущему значению
        current_val |= (int(byte_val) << bits_in_val)
        bits_in_val += 8
        
        # Пока у нас достаточно бит для извлечения d-битного значения
        while bits_in_val >= d and coef_idx < n:
            # Извлекаем d бит
            coef = current_val & mask
            coeffs[coef_idx] = coef % m
            coef_idx += 1
            
            # Сдвигаем для следующего коэффициента
            current_val >>= d
            bits_in_val -= d
    
    return coeffs

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _ntt_scalar_mul(coef, scalar, q=3329):
    """
    Векторизованное умножение NTT-коэффициента на скаляр по модулю q
    """
    return (coef * scalar) % q

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _ntt_add_mod_q(x, y, q=3329):
    """
    Векторизованное сложение двух NTT-коэффициентов по модулю q
    """
    return (x + y) % q

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _ntt_sub_mod_q(x, y, q=3329):
    """
    Векторизованное вычитание двух NTT-коэффициентов по модулю q
    """
    return (x - y) % q

class PolynomialRingKyber(PolynomialRing):
    """
    Initialise the polynomial ring:

        R = GF(3329) / (X^256 + 1)
    """

    def __init__(self):
        self.q = 3329
        self.n = 256
        self.element = PolynomialKyber
        self.element_ntt = PolynomialKyberNTT

        root_of_unity = 17
        self.ntt_zetas = [
            pow(root_of_unity, self._br(i, 7), 3329) for i in range(128)
        ]
        self.ntt_f = pow(128, -1, 3329)

    @staticmethod
    def _br(i, k):
        """
        bit reversal of an unsigned k-bit integer
        """
        bin_i = bin(i & (2**k - 1))[2:].zfill(k)
        return int(bin_i[::-1], 2)

    def ntt_sample(self, input_bytes):
        """
        Algorithm 1 (Parse)
        https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf

        Algorithm 6 (Sample NTT)

        Parse: B^* -> R
        """
        i, j = 0, 0
        coefficients = [0 for _ in range(self.n)]
        while j < self.n:
            d1 = input_bytes[i] + 256 * (input_bytes[i + 1] % 16)
            d2 = (input_bytes[i + 1] // 16) + 16 * input_bytes[i + 2]

            if d1 < 3329:
                coefficients[j] = d1
                j = j + 1

            if d2 < 3329 and j < self.n:
                coefficients[j] = d2
                j = j + 1

            i = i + 3
        return self(coefficients, is_ntt=True)

    def cbd(self, input_bytes, eta, is_ntt=False):
        """
        Algorithm 2 (Centered Binomial Distribution)
        https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf

        Algorithm 6 (Sample Poly CBD)

        Expects a byte array of length (eta * deg / 4)
        For Kyber, this is 64 eta.
        """
        assert 64 * eta == len(input_bytes)
        
        # Преобразуем байты в numpy массив перед вызовом JIT-функции
        input_bytes_array = np.array(list(input_bytes), dtype=np.uint8)
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _cbd_optimized(input_bytes_array, eta)
        
        # Преобразуем результат обратно в список Python
        return self(result_coeffs.tolist(), is_ntt=is_ntt)

    def decode(self, input_bytes, d, is_ntt=False):
        """
        Decode (Algorithm 3)

        decode: B^32l -> R_q
        """
        # Ensure the value d is set correctly
        if 256 * d != len(input_bytes) * 8:
            raise ValueError(
                f"input bytes must be a multiple of (polynomial degree) / 8, {256*d = }, {len(input_bytes)*8 = }"
            )

        # Преобразуем байты в numpy массив перед вызовом JIT-функции
        input_bytes_array = np.array(list(input_bytes), dtype=np.uint8)
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _decode_polynomial(input_bytes_array, d)
        
        # Преобразуем результат обратно в список Python и создаем полином
        return self(result_coeffs.tolist(), is_ntt=is_ntt)

    def __call__(self, coefficients, is_ntt=False):
        if not is_ntt:
            element = self.element
        else:
            element = self.element_ntt

        if isinstance(coefficients, int):
            return element(self, [coefficients])
        if not isinstance(coefficients, list):
            raise TypeError(
                f"Polynomials should be constructed from a list of integers, of length at most n = {256}"
            )
        return element(self, coefficients)


class PolynomialKyber(Polynomial):
    def __init__(self, parent, coefficients):
        self.parent = parent
        self.coeffs = self._parse_coefficients(coefficients)

    def encode(self, d):
        """
        Encode (Inverse of Algorithm 3)
        """
        # Преобразуем список коэффициентов в numpy массив
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        
        # Вызываем оптимизированную JIT-функцию
        result_np_array = _encode_polynomial(coeffs_array, d)
        
        # Преобразуем результат в bytes
        return bytes(result_np_array)

    def compress(self, d):
        """
        Compress the polynomial by compressing each coefficient

        NOTE: This is lossy compression
        """
        # Преобразуем список коэффициентов в numpy массив
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        
        # Вызываем векторизованную JIT-функцию с явной передачей всех параметров
        result_coeffs = _compress_element(coeffs_array, d, 3329)
        
        # Преобразуем результат обратно в список Python
        self.coeffs = result_coeffs.tolist()
        return self

    def decompress(self, d):
        """
        Decompress the polynomial by decompressing each coefficient

        NOTE: This as compression is lossy, we have
        x' = decompress(compress(x)), which x' != x, but is
        close in magnitude.
        """
        # Преобразуем список коэффициентов в numpy массив
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        
        # Вызываем векторизованную JIT-функцию с явной передачей всех параметров
        result_coeffs = _decompress_element(coeffs_array, d, 3329)
        
        # Преобразуем результат обратно в список Python
        self.coeffs = result_coeffs.tolist()
        return self

    def to_ntt(self):
        """
        Convert a polynomial to number-theoretic transform (NTT) form.
        The input is in standard order, the output is in bit-reversed order.
        """
        # Преобразуем список коэффициентов в numpy массив для Numba
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        zetas_array = np.array(self.parent.ntt_zetas, dtype=np.int32)
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _ntt_forward_transform(coeffs_array, zetas_array)
        
        # Преобразуем результат обратно в список Python
        return self.parent(result_coeffs.tolist(), is_ntt=True)

    def from_ntt(self):
        """
        Not supported, raises a ``TypeError``
        """
        raise TypeError(f"Polynomial not in the NTT domain: {type(self) = }")


class PolynomialKyberNTT(PolynomialKyber):
    def __init__(self, parent, coefficients):
        self.parent = parent
        self.coeffs = self._parse_coefficients(coefficients)

    def to_ntt(self):
        """
        Not supported, raises a ``TypeError``
        """
        raise TypeError(
            f"Polynomial is already in the NTT domain: {type(self) = }"
        )

    def from_ntt(self):
        """
        Convert a polynomial from number-theoretic transform (NTT) form in place
        The input is in bit-reversed order, the output is in standard order.
        """
        # Преобразуем список коэффициентов в numpy массив для Numba
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        zetas_array = np.array(self.parent.ntt_zetas, dtype=np.int32)
        ntt_f = self.parent.ntt_f
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _ntt_inverse_transform(coeffs_array, zetas_array, ntt_f)
        
        # Преобразуем результат обратно в список Python
        return self.parent(result_coeffs.tolist(), is_ntt=False)

    @staticmethod
    def _ntt_base_multiplication(a0, a1, b0, b1, zeta):
        """
        Base case for ntt multiplication
        """
        return _ntt_base_mul(a0, a1, b0, b1, zeta)

    def _ntt_coefficient_multiplication(self, f_coeffs, g_coeffs):
        """
        Given the coefficients of two polynomials compute the coefficients of
        their product
        """
        # Преобразуем списки коэффициентов в numpy массивы
        f_array = np.array(f_coeffs, dtype=np.int32)
        g_array = np.array(g_coeffs, dtype=np.int32)
        zetas_array = np.array(self.parent.ntt_zetas, dtype=np.int32)
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _ntt_coefficient_mul(f_array, g_array, zetas_array)
        
        # Преобразуем результат обратно в список Python
        return result_coeffs.tolist()

    def _ntt_multiplication(self, other):
        """
        Number Theoretic Transform multiplication.
        """
        new_coeffs = self._ntt_coefficient_multiplication(
            self.coeffs, other.coeffs
        )
        return new_coeffs

    def _add_(self, other):
        if isinstance(other, type(self)):
            # Преобразуем списки коэффициентов в numpy массивы
            x_array = np.array(self.coeffs, dtype=np.int32)
            y_array = np.array(other.coeffs, dtype=np.int32)
            
            # Вызываем векторизованную JIT-функцию с явной передачей всех параметров
            result_coeffs = _ntt_add_mod_q(x_array, y_array, 3329)
            
            # Преобразуем результат обратно в список Python
            return result_coeffs.tolist()
        elif isinstance(other, int):
            # Для сложения с целым числом копируем коэффициенты и изменяем нулевой
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = (new_coeffs[0] + other) % 3329
            return new_coeffs
        else:
            raise NotImplementedError(
                f"Polynomials can only be added to each other, {type(other) = }, {type(self) = }"
            )
    
    def _sub_(self, other):
        if isinstance(other, type(self)):
            # Преобразуем списки коэффициентов в numpy массивы
            x_array = np.array(self.coeffs, dtype=np.int32)
            y_array = np.array(other.coeffs, dtype=np.int32)
            
            # Вызываем векторизованную JIT-функцию с явной передачей всех параметров
            result_coeffs = _ntt_sub_mod_q(x_array, y_array, 3329)
            
            # Преобразуем результат обратно в список Python
            return result_coeffs.tolist()
        elif isinstance(other, int):
            # Для вычитания целого числа копируем коэффициенты и изменяем нулевой
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = (new_coeffs[0] - other) % 3329
            return new_coeffs
        else:
            raise NotImplementedError(
                f"Polynomials can only be subtracted from each other, {type(other) = }, {type(self) = }"
            )

    def __add__(self, other):
        new_coeffs = self._add_(other)
        return self.parent(new_coeffs, is_ntt=True)

    def __sub__(self, other):
        new_coeffs = self._sub_(other)
        return self.parent(new_coeffs, is_ntt=True)

    def __mul__(self, other):
        if isinstance(other, type(self)):
            new_coeffs = self._ntt_multiplication(other)
        elif isinstance(other, int):
            # Преобразуем список коэффициентов в numpy массив
            coeffs_array = np.array(self.coeffs, dtype=np.int32)
            
            # Вызываем векторизованную JIT-функцию
            result_coeffs = _ntt_scalar_mul(coeffs_array, other)
            
            # Преобразуем результат обратно в список Python
            new_coeffs = result_coeffs.tolist()
        else:
            raise NotImplementedError(
                f"Polynomials can only be multiplied by each other, or scaled by integers, {type(other) = }, {type(self) = }"
            )
        return self.parent(new_coeffs, is_ntt=True)
