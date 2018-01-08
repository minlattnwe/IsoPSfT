"""
compute.py

perform shape-from-template
"""
import numpy

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend import Legend

from .surface import EstimatedSurface
from .formula import get_formula

from sft import TPS, IdentityEmbdding, IsoPSfT


CONFIG_CONTROL_POINTS = {
    'color': 'r',
    'marker': 'x'
}

CONFIG_OTHER_POINTS = {
    'color': 'b',
    'marker': '.'
}

CONFIG_ESTIMATED = {
    'color': 'g',
    'marker': '+'
}


def solve(**kwarg):
    fig = kwarg['fig']
    fig.clear()

    def error_text(pos, msg):
        ax = fig.add_subplot(2, 2, pos)
        ax.text(.5, .5, msg, color='red', verticalalignment='center', horizontalalignment='center')

    # get all points
    xx, yy = numpy.meshgrid(
        numpy.linspace(*kwarg['xrang'], kwarg['xsamp']),
        numpy.linspace(*kwarg['yrang'], kwarg['ysamp'])
    )
    zz = kwarg['formula'].depth(xx, yy)

    apoints_target = numpy.stack([xx, yy, zz], axis=2).reshape(-1, 3)

    # sampling
    sidx = numpy.random.permutation(numpy.arange(xx.size))[:kwarg['nsamp']]

    mask = numpy.zeros((xx.size), dtype=bool)
    mask[sidx] = True

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title('ground truth point set')
    ax.scatter3D(*apoints_target[mask].T, **CONFIG_CONTROL_POINTS)
    ax.scatter3D(*apoints_target[~mask].T, **CONFIG_OTHER_POINTS)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # project to image space
    f = kwarg['focal']

    apoints_image = numpy.dot(
        numpy.array([
            [f, 0, 0],
            [0, f, 0],
            [0, 0, 1]]
        ),
        apoints_target.T
    )

    apoints_image = apoints_image[:2] / apoints_image[2, None]
    apoints_image = apoints_image.T

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('point set on image')
    ax.grid()
    ax.scatter(*apoints_image[mask].T, **CONFIG_CONTROL_POINTS)
    ax.scatter(*apoints_image[~mask].T, **CONFIG_OTHER_POINTS)

    # project to parameterization space & solve
    apoints_param = kwarg['formula'].param_position(*apoints_target.T)

    xrang_param = numpy.min(apoints_param[:,0]), numpy.max(apoints_param[:,0])
    yrang_param = numpy.min(apoints_param[:,1]), numpy.max(apoints_param[:,1])
    cpoints_param = apoints_param[mask]

    # solve tps warp
    try:
        warp = TPS(cpoints_param, apoints_image[mask])
    except numpy.linalg.LinAlgError:
        error_text(4, 'Can\'t solve thin plate spline')
        return

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('estimated warp')
    ax.scatter(*warp(apoints_param).T, **CONFIG_ESTIMATED)

    # solve sft
    try:
        solution = IsoPSfT(warp, IdentityEmbdding(), cpoints_param)
    except numpy.linalg.LinAlgError:
        error_text(2, 'Can\'t solve shape from template')
        return

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title('estimated surface')
    ax.scatter3D(*solution(apoints_param).T, **CONFIG_ESTIMATED)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return EstimatedSurface(solution, xrang_param, yrang_param)
