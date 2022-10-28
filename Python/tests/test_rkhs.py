import nlsa.rkhs as rkhs
import numpy as np

def test_energy():
    w = np.array([1, 2, 3])
    engy = rkhs.energy(w)
    c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    e = engy(c)
    assert np.all(e == (np.array([1, 4, 9]) * 6))
