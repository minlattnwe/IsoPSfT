"""
gor.py

the geometrical outlier removal model
"""
import numpy
from sklearn.mixture import BayesianGaussianMixture

from .keypoints import get_matched_pair


THRESHOLD_BRMM = 1E5


def brmm_gor(
        keypoints_template :numpy.ndarray,
        keypoints_deformed :numpy.ndarray,
        matches            :numpy.ndarray
    ) -> numpy.ndarray:
    """
    the bayesian robust mixture model(BRMM) geometrical outlier remover
    """
    fmat = numpy.empty((len(matches), 5))
    for m, f in zip(matches, fmat):
        kp_t = keypoints_template[m.queryIdx]
        kp_d = keypoints_deformed[m.trainIdx]

        f[0] = kp_t.pt[0] - kp_d.pt[0]
        f[1] = kp_t.pt[1] - kp_d.pt[1]
        f[2] = kp_t.size - kp_d.size

        difAng = (kp_t.angle - kp_d.angle) /180 *numpy.pi
        f[3] = numpy.sin(difAng)
        f[4] = numpy.cos(difAng)

    brmm = BayesianGaussianMixture(len(fmat))
    brmm.fit(fmat)

    inliers_matches = []
    for model, match in zip(brmm.covariances_, matches):
        if numpy.linalg.det(model) < THRESHOLD_BRMM:
            inliers_matches.append(match)

    return numpy.array(inliers_matches)
