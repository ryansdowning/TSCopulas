from typing import Any, Optional

import numpy as np


def apply_lag_transform(data, lag: int, axis: int = 0, drop: bool = False, fill: Optional[Any] = np.nan):
    """N-dimensional transform function which returns an N+1-dimensional matrix such that the last dimension has axis
    length <lag + 1> and the i'th index of the last dimension is the input matrix lagged by i

    .. math::
        Let data be X \in \mathbb{R}^n, f(X) = Y such that Y \in \mathbb{R}^{n+1} and Y[...,i] = shift(X, i),  # noqa
        i \in \{0, ..., lag\}  # noqa

    Args:
        data: [description]
        lag: [description]
        axis: [description]
        drop: [description]
        fill: [description]

    Returns:

    """
    if axis < 0:
        axis = data.ndim + axis
    if axis != 0:
        axis = axis - 1

    length = data.shape[axis]
    if lag > length:
        raise ValueError("...")

    out_shape = data.shape + (lag+1,)
    out = np.full(out_shape, fill)

    slices = [slice(None, None, None) for _ in range(out.ndim)]
    for i in range(lag + 1):
        slices[axis] = slice(i, length)
        slices[-1] = i
        lag_slices = slices[:-1]
        lag_slices[axis] = slice(0, length-i)
        out[tuple(slices)] = data[tuple(lag_slices)]

    if drop:
        slices[axis] = slice(lag, length, None)
        slices[-1] = slice(None, None, None)
        return out[tuple(slices)]
    return out
