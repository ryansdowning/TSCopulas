import numpy as np

from tscopulas.preprocessing import lag_transform as lt


def test_1d_apply_lag_transform():
    arr = np.arange(10)
    out = lt.apply_lag_transform(arr, 3)

    assert out.shape == (10, 4)
    np.testing.assert_array_equal(out[...,0], arr)
    np.testing.assert_array_equal(out[...,1:,1], arr[:-1])
    np.testing.assert_array_equal(out[...,2:,2], arr[:-2])
    np.testing.assert_array_equal(out[...,3:,3], arr[:-3])
