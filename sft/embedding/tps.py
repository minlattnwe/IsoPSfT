"""
These code originally was matlab script in `daeyun/TPS-Deformation`_ and
is rewrote into python code by tzing.

.. _daeyun/TPS-Deformation: https://github.com/daeyun/TPS-Deformation/

-------

Copyright (c) 2014, Daeyun Shin.
Copyright (c) 2017, tzing.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the project nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from .base import Embdding

import numpy
from scipy.spatial.distance import cdist


def _2darray(data) -> numpy.ndarray:
    """
    *assistant func* ensure data type and
    """
    data = numpy.array(data, dtype=float)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return data


def pairwise_radial_basis(A, B):
    """
    Compute the TPS radial basis function phi(r) between every row-pair of A
    and B where r is the Euclidean distance.

    ## Usage
        ```python
        P = pairwise_radial_basis(A, B)
        ```

    ## Arguments
        + `A` - n by d vector containing n d-dimensional points.
        + `B` - m by d vector containing m d-dimensional points.

    ## Returns
        ```math
        P - P(i, j) = phi(norm(A(i,:)-B(j,:)))
                where phi(r) = r^2*log(r) for r >= 1
                               r*log(r^r) for r <  1
        ```

    ## References
        1. https://en.wikipedia.org/wiki/Polyharmonic_spline
        2. https://en.wikipedia.org/wiki/Radial_basis_function
    """
    # r_mat(i, j) is the Euclidean distance between A(i, :) and B(j, :).
    r_mat = cdist(A, B)

    pwise_cond_ind1 = r_mat >= 1
    pwise_cond_ind2 = r_mat < 1
    r_mat_p1 = r_mat[pwise_cond_ind1]
    r_mat_p2 = r_mat[pwise_cond_ind2]

    # P correcponds to the matrix A from [1].
    P = numpy.empty(r_mat.shape)
    P[pwise_cond_ind1] = (r_mat_p1 **2) *numpy.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 *numpy.log(numpy.power(r_mat_p2, r_mat_p2))

    return P


class TPS(Embdding):
    """
    The thin plate spline warpping
    """

    def __init__(
        self,
        source_points   :numpy.ndarray,
        target_points   :numpy.ndarray,
        alpha           :float = 0
    ):
        """
        Given a set of control points and their displacements, compute the
        coefficients of the TPS interpolant f(S) deforming surface S.

        ## Arguments
            + `source_points` - p by d vector of control points.
            + `target_points` - p by d vector of corresponding control points
                                in the mapping function f(S).
            + `alpha`         - regularization parameter. See page 4 of [3].

        ## References
            1. http://en.wikipedia.org/wiki/Polyharmonic_spline
            2. http://en.wikipedia.org/wiki/Thin_plate_spline
            3. http://cseweb.ucsd.edu/~sjb/pami_tps.pdf
        """
        # ensure data type and shape
        source_points = _2darray(source_points)
        target_points = _2darray(target_points)
        assert source_points.shape == target_points.shape, \
            f'Size of `source_points` {source_points.shape} does not equal' \
            f' to `target_points` {target_points.shape}.'

        p, d = source_points.shape

        # This correcponds to the matrix A from [1]
        A = pairwise_radial_basis(source_points, source_points)

        # Relax the exact interpolation requirement by means of regularization. [3]
        A = A + alpha *numpy.identity(p)

        # This correcponds to V from [1]
        V = numpy.hstack([
            numpy.ones((p, 1)),
            source_points
        ])

        # Target points
        M = numpy.vstack([
            numpy.hstack([A, V]),
            numpy.hstack([V.T, numpy.zeros((d+1, d+1))])
        ])
        Y = numpy.vstack([target_points, numpy.zeros((d+1, d))])

        # solve for M*X = Y.
        # At least d+1 control points should not be in a subspace; e.g. for d=2, at
        # least 3 points are not on a straight line. Otherwise M will be singular.
        X = numpy.linalg.solve(M, Y)

        self.source_points = source_points
        self.d = d
        self.coef = X

    def __call__(self, points) -> numpy.ndarray:
        """
        Given a set of control points and mapping coefficients, compute a
        deformed surface f(S) using a thin plate spline radial basis function
        phi(r) as shown in [1].

        ## Arguments
            + `points` - n by 3 matrix of X, Y, Z components of the surface

        ## Returns
            n by 3 vectors of X, Y, Z compoments of the deformed surface.

        ## References
            1. http://en.wikipedia.org/wiki/Polyharmonic_spline
        """
        points = _2darray(points)
        assert points.shape[1] == self.d, \
            f'Dimension of input array (d={points.shape[1]})' \
            f' not match with this instance (d={self.d}).'

        n = points.shape[0]

        A = pairwise_radial_basis(points, self.source_points)
        V = numpy.hstack([
            numpy.ones((n, 1)),
            points
        ])

        f_surface = numpy.dot(
            numpy.hstack([A, V]),
            self.coef
        )

        return f_surface
