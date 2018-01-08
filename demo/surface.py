"""
surface.py

customized surface descriptor object to render
"""
from abc import ABC, abstractclassmethod

import numpy

from .formula import get_formula


class Surface(ABC):

    def __init__(self, xrang, xsamp, yrang, ysamp):
        self.xrange = numpy.array(xrang)
        self.yrange = numpy.array(yrang)
        self.setNumSamp(xsamp, ysamp)

    def setNumSamp(self, xsamp, ysamp):
        self.shape = ysamp, xsamp
        xx, yy = numpy.meshgrid(
            numpy.linspace(*self.xrange, xsamp),
            numpy.linspace(*self.yrange, ysamp)
        )

        # estimate z
        zz = self.depth(xx, yy)

        self._points = numpy.stack([xx, yy, zz], axis=2)
        self.zrange = numpy.array([
                numpy.min(zz), numpy.max(zz)
            ])
        self.center = numpy.average(numpy.vstack([
                self.xrange, self.yrange, self.zrange
            ]), axis=1)

        # estimate normal
        self._normal = numpy.empty(self._points.shape)
        for i, row in enumerate(self._points):
            for j, point in enumerate(row):
                self._normal[i, j] = self.normal(*point)

        self._points -= self.center[None, None, :]

    @abstractclassmethod
    def depth(self, xx, yy) -> numpy.ndarray:
        ...

    @abstractclassmethod
    def normal(self, x, y, z) -> numpy.ndarray:
        ...

    def __getitem__(self, index) -> (numpy.ndarray, numpy.ndarray):
        point = self._points[index]
        normal = self._normal[index]
        return point, normal


class TruthSurface(Surface):

    def __init__(self, formula, xrang, xsamp, yrang, ysamp):
        self.formula = formula
        super().__init__(xrang, xsamp, yrang, ysamp)

    def depth(self, xx, yy):
        return self.formula.depth(xx, yy)

    def normal(self, x, y, z):
        vecX = numpy.array([1, 0, self.formula.depth(x+1, y) -z])
        vecY = numpy.array([0, 1, self.formula.depth(x, y+1) -z])
        return numpy.cross(vecY, vecX)


class EstimatedSurface(Surface):

    def __init__(self, solution, xrang, yrang):
        self.solution = solution
        super().__init__(xrang, 2, yrang, 2)

    def depth(self, xx, yy):
        x = xx.reshape(-1, 1)
        y = yy.reshape(-1, 1)
        return self.solution(numpy.hstack([x, y]))[:,2].reshape(xx.shape)

    def normal(self, x, y, z):
        vecX = numpy.array([1, 0, self.depth(x+1, y) -z])
        vecY = numpy.array([0, 1, self.depth(x, y+1) -z])
        return numpy.cross(vecY, vecX)
