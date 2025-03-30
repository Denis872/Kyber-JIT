import random
import numba as nb
import numpy as np

@nb.njit
def _optimized_schoolbook_multiplication(a, b, n, q):
    """
    Оптимизированная реализация умножения полиномов методом школы.
    a, b - массивы коэффициентов полиномов
    n - степень полинома
    q - модуль поля
    """
    new_coeffs = np.zeros(n, dtype=np.int32)
    
    # Первая часть умножения: положительные индексы
    for i in range(n):
        for j in range(0, n - i):
            new_coeffs[i + j] = (new_coeffs[i + j] + a[i] * b[j]) % q
    
    # Вторая часть умножения: учет отрицательных индексов из-за модуля X^n + 1
    for j in range(1, n):
        for i in range(n - j, n):
            new_coeffs[i + j - n] = (new_coeffs[i + j - n] - a[i] * b[j]) % q
    
    return new_coeffs

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _scalar_multiply(coef, scalar, q):
    """
    Векторизованное умножение коэффициента на скаляр по модулю q
    """
    return (coef * scalar) % q

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _add_mod_q_vec(x, y, q):
    """
    Векторизованное сложение двух чисел по модулю q
    """
    return (x + y) % q

@nb.vectorize([nb.int32(nb.int32, nb.int32, nb.int32)])
def _sub_mod_q_vec(x, y, q):
    """
    Векторизованное вычитание двух чисел по модулю q
    """
    return (x - y) % q

@nb.vectorize([nb.int32(nb.int32, nb.int32)])
def _reduce_coefficient(c, q):
    """
    Векторизованное сокращение коэффициента по модулю q
    """
    return c % q

@nb.vectorize([nb.int32(nb.int32, nb.int32)])
def _negate_coefficient(c, q):
    """
    Векторизованное отрицание коэффициента по модулю q
    """
    return (-c) % q

class PolynomialRing:
    """
    Initialise the polynomial ring:

        R = GF(q) / (X^n + 1)
    """

    def __init__(self, q, n):
        self.q = q
        self.n = n
        self.element = Polynomial

    def gen(self):
        """
        Return the generator `x` of the polynomial ring
        """
        return self([0, 1])

    def random_element(self):
        """
        Compute a random element of the polynomial ring with coefficients in the
        canonical range: ``[0, q-1]``
        """
        coefficients = [random.randint(0, self.q - 1) for _ in range(self.n)]
        return self(coefficients)

    def __call__(self, coefficients):
        if isinstance(coefficients, int):
            return self.element(self, [coefficients])
        if not isinstance(coefficients, list):
            raise TypeError(
                f"Polynomials should be constructed from a list of integers, of length at most d = {self.n}"
            )
        return self.element(self, coefficients)

    def __repr__(self):
        return f"Univariate Polynomial Ring in x over Finite Field of size {self.q} with modulus x^{self.n} + 1"


class Polynomial:
    def __init__(self, parent, coefficients):
        self.parent = parent
        self.coeffs = self._parse_coefficients(coefficients)

    def is_zero(self):
        """
        Return if polynomial is zero: f = 0
        """
        return all(c == 0 for c in self.coeffs)

    def is_constant(self):
        """
        Return if polynomial is constant: f = c
        """
        return all(c == 0 for c in self.coeffs[1:])

    def _parse_coefficients(self, coefficients):
        """
        Helper function which right pads with zeros
        to allow polynomial construction as
        f = R([1,1,1])
        """
        l = len(coefficients)
        if l > self.parent.n:
            raise ValueError(
                f"Coefficients describe polynomial of degree greater than maximum degree {self.parent.n}"
            )
        elif l < self.parent.n:
            coefficients = coefficients + [0 for _ in range(self.parent.n - l)]
        return coefficients

    def reduce_coefficients(self):
        """
        Reduce all coefficients modulo q
        """
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        result_coeffs = _reduce_coefficient(coeffs_array, self.parent.q)
        self.coeffs = result_coeffs.tolist()
        return self

    def _add_mod_q(self, x, y):
        """
        add two coefficients modulo q
        """
        return (x + y) % self.parent.q

    def _sub_mod_q(self, x, y):
        """
        sub two coefficients modulo q
        """
        return (x - y) % self.parent.q

    def _schoolbook_multiplication(self, other):
        """
        Naive implementation of polynomial multiplication
        suitible for all R_q = F_1[X]/(X^n + 1)
        """
        n = self.parent.n
        q = self.parent.q
        a = np.array(self.coeffs, dtype=np.int32)
        b = np.array(other.coeffs, dtype=np.int32)
        
        # Вызываем оптимизированную JIT-функцию
        result_coeffs = _optimized_schoolbook_multiplication(a, b, n, q)
        
        # Преобразуем результат обратно в список Python
        return result_coeffs.tolist()

    def __neg__(self):
        """
        Returns -f, by negating all coefficients
        """
        coeffs_array = np.array(self.coeffs, dtype=np.int32)
        result_coeffs = _negate_coefficient(coeffs_array, self.parent.q)
        return self.parent(result_coeffs.tolist())

    def _add_(self, other):
        q = self.parent.q
        if isinstance(other, type(self)):
            # Преобразуем списки коэффициентов в numpy массивы
            x_array = np.array(self.coeffs, dtype=np.int32)
            y_array = np.array(other.coeffs, dtype=np.int32)
            
            # Вызываем векторизованную JIT-функцию
            result_coeffs = _add_mod_q_vec(x_array, y_array, q)
            return result_coeffs.tolist()
        elif isinstance(other, int):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = self._add_mod_q(new_coeffs[0], other)
            return new_coeffs
        else:
            raise NotImplementedError(
                "Polynomials can only be added to each other"
            )

    def __add__(self, other):
        new_coeffs = self._add_(other)
        return self.parent(new_coeffs)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self + other
        return self

    def _sub_(self, other):
        q = self.parent.q
        if isinstance(other, type(self)):
            # Преобразуем списки коэффициентов в numpy массивы
            x_array = np.array(self.coeffs, dtype=np.int32)
            y_array = np.array(other.coeffs, dtype=np.int32)
            
            # Вызываем векторизованную JIT-функцию
            result_coeffs = _sub_mod_q_vec(x_array, y_array, q)
            return result_coeffs.tolist()
        elif isinstance(other, int):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = self._sub_mod_q(new_coeffs[0], other)
            return new_coeffs
        else:
            raise NotImplementedError(
                "Polynomials can only be subtracted from each other"
            )

    def __sub__(self, other):
        new_coeffs = self._sub_(other)
        return self.parent(new_coeffs)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        self = self - other
        return self

    def __mul__(self, other):
        if isinstance(other, type(self)):
            new_coeffs = self._schoolbook_multiplication(other)
        elif isinstance(other, int):
            coeffs_array = np.array(self.coeffs, dtype=np.int32)
            result_coeffs = _scalar_multiply(coeffs_array, other, self.parent.q)
            new_coeffs = result_coeffs.tolist()
        else:
            raise NotImplementedError(
                "Polynomials can only be multiplied by each other, or scaled by integers"
            )
        return self.parent(new_coeffs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self = self * other
        return self

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError(
                "Exponentiation of a polynomial must be done using an integer."
            )

        # Deal with negative scalar multiplication
        if n < 0:
            raise ValueError(
                "Negative powers are not supported for elements of a Polynomial Ring"
            )
        f = self
        g = self.parent(1)
        while n > 0:
            if n % 2 == 1:
                g = g * f
            f = f * f
            n = n // 2
        return g

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.coeffs == other.coeffs
        elif isinstance(other, int):
            if (
                self.is_constant()
                and (other % self.parent.q) == self.coeffs[0]
            ):
                return True
        return False

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __repr__(self):
        if self.is_zero():
            return "0"

        info = []
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    info.append(f"{c}")
                elif i == 1:
                    if c == 1:
                        info.append("x")
                    else:
                        info.append(f"{c}*x")
                else:
                    if c == 1:
                        info.append(f"x^{i}")
                    else:
                        info.append(f"{c}*x^{i}")
        return " + ".join(info)

    def __str__(self):
        return self.__repr__()
