import itertools
import math
from collections.abc import Iterable
import numpy as np

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}


def _integral_image(image, *, dtype=None):
    if dtype is None and image.real.dtype.kind == 'f':
        # default to at least double precision cumsum for accuracy
        dtype = np.promote_types(image.dtype, np.float64)

    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S


def _supported_float_type(input_dtype, allow_complex=False):
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def _get_view(padded, kernel_shape, idx, val):
    sl_shift = tuple([slice(c, s - (w_ - 1 - c))
                      for c, w_, s in zip(idx, kernel_shape, padded.shape)])
    v = padded[sl_shift]
    if val == 1:
        return v
    return val * v


def _correlate_sparse(image, kernel_shape, kernel_indices, kernel_values):
    idx, val = kernel_indices[0], kernel_values[0]
    # implementation assumes this corner is first in kernel_indices_in_values
    if tuple(idx) != (0,) * image.ndim:
        raise RuntimeError("Unexpected initial index in kernel_indices")
    # make a copy to avoid modifying the input image
    out = _get_view(image, kernel_shape, idx, val).copy()
    for idx, val in zip(kernel_indices[1:], kernel_values[1:]):
        out += _get_view(image, kernel_shape, idx, val)
    return out


def _validate_window_size(axis_sizes):
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = (f'Window size for `threshold_sauvola` or '
                   f'`threshold_niblack` must not be even on any dimension. '
                   f'Got {axis_sizes}')
            raise ValueError(msg)



def _mean_std(image, w):
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)

    float_dtype = _supported_float_type(image.dtype)
    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype(float_dtype, copy=False), pad_width,
                    mode='reflect')

    integral = _integral_image(padded, dtype=np.float64)
    padded *= padded
    integral_sq = _integral_image(padded, dtype=np.float64)


    # Create lists of non-zero kernel indices and values
    kernel_indices = list(itertools.product(*tuple([(0, _w) for _w in w])))
    kernel_values = [(-1) ** (image.ndim % 2 != np.sum(indices) % 2)
                     for indices in kernel_indices]

    total_window_size = math.prod(w)
    kernel_shape = tuple(_w + 1 for _w in w)
    m = _correlate_sparse(integral, kernel_shape, kernel_indices,
                          kernel_values)
    m = m.astype(float_dtype, copy=False)
    m /= total_window_size
    g2 = _correlate_sparse(integral_sq, kernel_shape, kernel_indices,
                           kernel_values)
    g2 = g2.astype(float_dtype, copy=False)
    g2 /= total_window_size

    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    print(total_window_size)
    return m, s


def threshold_niblack(image, window_size=15, k=0.2):
    m, s = _mean_std(image, window_size)
    return m - k * s

def threshold_niblack_const(image, window_size=15, k=0.2, c=0.0):
    m, s = _mean_std(image, window_size)
    return m - k * s - c