import numpy as np
import math
from enum import Enum

class OtsuType(Enum):
    classic = 0
    kurita_otsu_abdelmalek = 1
    kittler_lllingworth = 2



def _compute_otsu_criteria(im, th, otsu_type):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    sigma0 = np.sqrt(var0)
    sigma1 = np.sqrt(var1)

    if otsu_type == OtsuType.classic:
        return weight0 * var0 + weight1 * var1
    elif otsu_type == OtsuType.kurita_otsu_abdelmalek:
        return weight0 * np.log(weight0 + 1e-6) - np.log(sigma0 + 1e-6) + weight1 * np.log(weight1 + 1e-6) - np.log(sigma1 + 1e-6)
    elif otsu_type == OtsuType.kittler_lllingworth:
        return weight0 * np.log(sigma0 / (weight0  + 1e-6) + 1e-6) + weight1 * np.log(sigma1 / (weight1 + 1e-6) + 1e-6)
    else:
        raise RuntimeError('Invalid otsu_type.')


def otsu(img: np.ndarray, criteria_type: OtsuType = OtsuType.improved) -> int:
    threshold_range = range(np.max(img) + 1)
    criterias = [_compute_otsu_criteria(img, th, criteria_type) for th in threshold_range]
    return threshold_range[np.argmin(criterias)]
        