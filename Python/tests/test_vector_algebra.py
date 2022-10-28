import numpy as np
from nlsa import vector_algebra as vec
from nlsa.abstract_algebra import compose_by, conjugate_by
from nptyping import Int, NDArray, Shape

V = NDArray[Shape["3"], Int]
M = NDArray[Shape["3, 3"], Int]


def test_add():
    u: V = np.array([1, 2, 3])
    v: V = np.array([-1, -2, -3])
    w: V = vec.add(u, v)
    assert np.all(w == 0)


def test_compose():
    u: V = np.array([1, 2, 3])
    v: V = np.array([1, 0, 1])
    w = vec.compose(u, v)
    print(w)
    assert np.all(w == np.array([1, 0, 3]))


def test_compose_by():
    m: M = np.identity(3, dtype=int)
    v: V = np.array([1, 1, 1])
    u = compose_by(vec, v)
    w = u(m)
    assert np.all(w == m)


def conjugate_by():
    u: V = np.array([2, 2, 2])
    v: V = np.array([3, 3, 3])
    m = np.identity(3, dtype=int)
    t = conjugate_by(vec, u, vec, v)
    w = t(m)
    assert np.all(w == 6 * m)




