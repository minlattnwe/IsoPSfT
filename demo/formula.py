"""
formula.py

formulas to generate surface
"""
from abc import ABC, abstractclassmethod

import numpy


class Formula(ABC):

    @abstractclassmethod
    def depth(self, x, y):
        ...

    @abstractclassmethod
    def param_position(self, x, y, z) -> numpy.ndarray:
        ...


def get_formula(**kwarg) -> Formula:
    pow = kwarg['formula_x_power']
    if pow == 2:
        return QuadraticFormula(**kwarg)
    if pow == 3:
        return CubedFormula(**kwarg)
    raise NotImplementedError()


class QuadraticFormula(Formula):

    def __init__(self, **kwarg):
        self.coef = kwarg['formula_x_coef']
        self.C = kwarg['formula_constant']

    def __str__(self):
        coef = self.coef
        if coef == 1:
            coef = ''
        sign = '+'
        if self.C < 0:
            sign = '-'
        return f'{coef}x^2 {sign} {abs(self.C)}'

    def depth(self, x, y):
        return numpy.power(x, 2) *self.coef +self.C

    def param_position(self, x, y, z):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        nx = 2 *self.coef *x
        x = ( nx *numpy.sqrt(nx **2+1) + numpy.arcsinh(nx) ) /4 /self.coef
        return numpy.hstack([x, y])


class CubedFormula(Formula):

    def __init__(self, **kwarg):
        self.coef = kwarg['formula_x_coef']
        self.C = kwarg['formula_constant']

    def __str__(self):
        coef = self.coef
        if coef == 1:
            coef = ''
        sign = '+'
        if self.C < 0:
            sign = '-'
        return f'{coef}x^2 {sign} {abs(self.C)}'

    def depth(self, x, y):
        return numpy.power(x, 3) *self.coef +self.C

    def param_position(self, x, y, z):
        """
        see: https://www.wolframalpha.com/input/?i=integral+sqrt(+1%2B(3nx%5E2)%5E2+)
        """
        from mpmath import ellipf
        x = x.reshape(-1).astype(numpy.complex)

        # coefs shortcut
        n = self.coef
        ni = numpy.sqrt(n *1j)
        sq3 = numpy.sqrt(3)

        sq9n2x4 = numpy.sqrt(9 *(n**2) *(x**4) +1)

        # calculation
        num1 = 27 *(n**3) *(x**5)
        num2 = 2 *sq3 *sq9n2x4

        num3 = 1j *numpy.arcsinh( sq3 *ni *x )
        for i, n in enumerate(num3):
            num3[i] = ellipf(n, -1)

        num4 = 3 *n *x

        den = 9 *n *sq9n2x4

        x = (num1 -num2 *num3 +num4) / den
        x = numpy.real(x)
        return numpy.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
