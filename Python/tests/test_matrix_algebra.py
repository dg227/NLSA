import numpy as np
from nlsa import matrix_algebra as mat
from nlsa.abstract_algebra import compose_by
from nptyping import Int, NDArray, Shape

V = NDArray[Shape["3"], Int]
M = NDArray[Shape["3, 3"], Int]


def test_add():
    u: V = np.array([1, 2, 3])
    v: V = np.array([-1, -2, -3])
    w: V = mat.add(u, v)
    assert np.all(w == 0)


def test_compose():
    m: M = np.identity(3, dtype=int)
    v: V = np.array([1, 1, 1])
    w = mat.compose(m, v)
    assert np.all(w == 1)


def test_compose_by():
    m: M = np.identity(3, dtype=int)
    v: V = np.array([1, 1, 1])
    u = compose_by(mat, v)
    w = u(m)
    assert np.all(w == 1)


def test_pure_state():
    m: M = np.diag([1, 2, 3])
    v: V = np.array([0, 1, 0])
    omega = mat.pure_state(v)
    print(v)
    print(m)
    y = omega(m)
    print(y)
    assert y == 2
