"""
isop.py

implement IsoPSfT algorithm
"""
import numpy

from .embedding import Embdding, TPS


def _1xNpoint(point):
    """
    *assistant func* ensure point datatype and shape
    """
    return numpy.array(point, dtype=float).reshape(-1)


def _Nx1point(point):
    """
    *assistant func* ensure point datatype and shape
    """
    return numpy.array(point, dtype=float).reshape(-1, 1)


def differentiate(func, pt):
    """
    *assistant func* get gradient on specific function at specific point
    """
    original_point = _1xNpoint(func(pt))

    gradient = numpy.empty((original_point.size, pt.size))
    for i, _ in enumerate(pt):
        pd = pt.copy()
        pd[i] += 1

        new_point = _1xNpoint(func(pd))
        diff = new_point - original_point

        gradient[:, i] = diff

    return gradient


class IsoPSfT(Embdding):
    """
    The isometric perspective shape-from-template embedding
    """

    def __init__(
        self,
        warp                :Embdding,
        template_embedding  :Embdding,
        control_points      :numpy.ndarray
    ):
        """
        ## Arguments
            + `warp` - the warp function from parameterization space to image
                     space
            + `template_embedding` - the embedding from parameterization space to
                                   template space
            + `control_points` - n by 2 matrix of points set on parameterization
                               space
        """
        if not isinstance(warp, Embdding):
            raise TypeError('`warp` is not a Embedding instance.')
        if not isinstance(template_embedding, Embdding):
            raise TypeError('`template_embedding` is not a Embedding instance.')

        control_points = numpy.array(control_points, dtype=float)
        assert control_points.shape[1] == 2, \
            f'Dimension of `control_points` (d={control_points.shape[1]})' \
             ' does not match requirements (d=2).'

        # calculate
        source_points = []
        target_points = []
        for point in control_points:
            # image space
            q = _Nx1point(warp(point))
            qt = _Nx1point((*q, 1))
            qt_norm = numpy.linalg.norm(qt)

            deriv_q = differentiate(warp, point)

            # template space
            P = template_embedding(point)
            deriv_P = differentiate(template_embedding, point)

            # matrix K
            K = numpy.dot(deriv_P.T, deriv_P)

            # matrix G
            G_pos = numpy.dot(deriv_q.T, deriv_q)
            G_post = numpy.dot(numpy.dot(numpy.dot(deriv_q.T, q), q.T), deriv_q)
            G = G_pos - G_post / (qt_norm **2)

            # lambda
            if numpy.abs(numpy.linalg.det(G)) < 1E-6: # singular
                continue

            try:
                KG = numpy.dot(K, numpy.linalg.inv(G))
                lambda2 = numpy.min(numpy.linalg.eig(KG)[0])
            except numpy.linalg.LinAlgError:
                continue

            source_points.append(P)
            target_points.append(qt.reshape(-1) *lambda2)

        self.tps = TPS(source_points, target_points)
        self.template_embedding = template_embedding

    def __call__(self, points) -> numpy.ndarray:
        points = numpy.array(points, dtype=float)
        return self.tps(self.template_embedding(points))
