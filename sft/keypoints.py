"""
keypoints.py

keypoint extraction & selection
"""
import random

import cv2
import numpy


# constants
#
FLANN_INDEX_LSH = 6     # required by FLANN
FLANN_INDEX_KDTREE = 0

THRESHOLD_MATCH = .75
THRESHOLD_RANSAC = 5

MIN_MATCH_COUNT = 10


# global vars
#
detector = cv2.xfeatures2d.SURF_create()

flann = cv2.FlannBasedMatcher({
        'algorithm': FLANN_INDEX_KDTREE,
        'trees': 5
    })


# funcs
#
def get_keypoint_and_desc(img :numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    """
    get keypoints and descriptor

    this is a shortcut func then users can know nothing about the feature extractor api
    """
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return numpy.array(keypoints), descriptors


def select_best_keypoints(
        image :numpy.ndarray,
        keypoints :numpy.ndarray =None,
        descriptors :numpy.ndarray =None,
        num_attempt :int =10,
        thres_score :int =1
    ) -> (numpy.ndarray, numpy.ndarray):
    """
    try to find out the best keypoint of a image then it can save time on matching
    """
    # get keypoint and descriptors
    if keypoints is None:
        keypoints = detector.detect(image)
    if descriptors is None:
        descriptors = detector.compute(image, keypoints)

    # scoring keypoints by RANSAC
    scores = numpy.zeros(len(keypoints))
    for _ in range(num_attempt):
        scores += score_keypoints(image, keypoints, descriptors)

    # returning the index with high-enogh score
    return scores > thres_score


def gen_random_rotation(h, w):
    """
    generate random 3d rotation
    """
    rtheta, rphi, rgamma = numpy.random.normal(scale=1/3, size=3) *numpy.pi

    d = numpy.sqrt(h**2 + w**2)
    f = d / (2 * numpy.sin(rgamma) if numpy.sin(rgamma) != 0 else 1)

    # Projection 2D -> 3D matrix
    A1 = numpy.matrix([[1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = numpy.matrix([[1, 0, 0, 0],
                        [0, numpy.cos(rtheta), -numpy.sin(rtheta), 0],
                        [0, numpy.sin(rtheta), numpy.cos(rtheta), 0],
                        [0, 0, 0, 1]])

    RY = numpy.matrix([[numpy.cos(rphi), 0, -numpy.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [numpy.sin(rphi), 0, numpy.cos(rphi), 0],
                        [0, 0, 0, 1]])

    RZ = numpy.matrix([[numpy.cos(rgamma), -numpy.sin(rgamma), 0, 0],
                        [numpy.sin(rgamma), numpy.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation matrix
    T = numpy.matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, f],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = numpy.matrix([[f, 0, 0, 0],
                        [0, f, 0, 0],
                        [0, 0, 1, 0]])

    # first form of transformation matrix
    mat = A2 *T *RX *RY *RZ *A1

    # estimate bounding box size
    bbox = mat * numpy.matrix([
            [0, w, w, 0],
            [0, 0, h, h],
            [1, 1, 1, 1]
    ])

    bbox = numpy.asarray(bbox[:2, :]) / numpy.tile(bbox[2,:], (2,1))

    lt = numpy.min(bbox, axis=1)
    br = numpy.max(bbox, axis=1)
    size = numpy \
            .asarray(br - lt) \
            .reshape(-1) \
            .round() \
            .astype(int)

    # translate
    mat = numpy.matrix([[1, 0, -lt[0, 0]],
                        [0, 1, -lt[1, 0]],
                        [0, 0, 1]]) *mat

    return mat, tuple(size)


def get_matched_pair(descriptor1, descriptor2, is_robust=True) -> numpy.ndarray:
    """
    Return well-matched pairs
    """
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    if not is_robust:
        return numpy.array(matches)[:,0].reshape(-1)

    # find good matches
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue

        m, n = match
        if m.distance < THRESHOLD_MATCH *n.distance:
            good_matches.append(m)

    return numpy.array(good_matches)


def score_keypoints(
        base_image      :numpy.ndarray,
        base_keypoint   :numpy.ndarray,
        base_descriptor :numpy.ndarray,
    ):
    """
    Scoring the keypoints by random rotate the image
    """
    # random rotate image and gen keypoint and descriptor
    mat, sz = gen_random_rotation(*base_image.shape[:2])
    img = cv2.warpPerspective(base_image, mat, sz)

    sample_keypoints, sample_descriptors = get_keypoint_and_desc(img)

    # find matches
    matches = get_matched_pair(base_descriptor, sample_descriptors)

    # skip if no good result
    if len(matches) < MIN_MATCH_COUNT:
        return numpy.zeros(len(base_keypoint))

    kp_base = base_keypoint[[ s.queryIdx for s in matches ]]
    kp_img = sample_keypoints[[ s.trainIdx for s in matches ]]

    pt_base = numpy.array([ k.pt for k in kp_base ]).reshape(-1,1,2)
    pt_img = numpy.array([ k.pt for k in kp_img ]).reshape(-1,1,2)

    _, used_mask = cv2.findHomography(pt_base, pt_img, cv2.RANSAC, THRESHOLD_RANSAC)
    used_mask = used_mask.reshape(-1).astype(bool)
    used_idx = [ s.queryIdx for s in matches[used_mask] ]

    score = numpy.zeros(len(base_keypoint))
    score[used_idx] = 1
    return score
