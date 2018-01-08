from .base import Embdding

import numpy


class IdentityEmbdding(Embdding):
    """
    The embedding maps R^2 parameterization space point directly to R^3 tempalte space
    """

    def __call__(self, points) -> numpy.ndarray:
        if isinstance(points, numpy.ndarray) and len(points.shape) == 2:
            # already 2d array
            assert points.shape[1] == 2
            return numpy.hstack([
                points,
                numpy.ones((points.shape[0], 1))
            ])

        else:
            # otherwise
            assert len(points) == 2
            return numpy.array((*points, 1), dtype=float).reshape(-1)
