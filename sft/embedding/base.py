from abc import ABC, abstractmethod

import numpy


class Embdding(ABC):

    @abstractmethod
    def __call__(self, pt) -> numpy.ndarray:
        ...
