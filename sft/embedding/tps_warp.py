from .base import Embdding
from .tps import TPS

from itertools import groupby

import numpy


class TpsWarp(Embdding):
    """
    The projection form 2d parameterization plane to 2d image
    """

    def __init__(
        self,
        keypoints_template  :numpy.ndarray,
        keypoints_image     :numpy.ndarray,
        matches             :numpy.ndarray,
        cluster_labels      :numpy.ndarray
    ):
        matches = sorted(matches, key=lambda m: cluster_labels[m.queryIdx])

        source_pt = []
        target_pt = []
        for _, cluster in groupby(matches, lambda m: cluster_labels[m.queryIdx]):
            cluster = list(cluster)

            if len(cluster) == 1:
                match = cluster[0]
            else:
                match = min(cluster, key=lambda m: m.distance)

            kp_from = keypoints_template[match.queryIdx]
            kp_to = keypoints_image[match.trainIdx]

            source_pt.append(kp_from.pt)
            target_pt.append(kp_to.pt)

        # solve thin plane spline
        self._tps = TPS(source_pt, target_pt)

    def __call__(self, pt) -> numpy.ndarray:
        return self._tps(pt)
