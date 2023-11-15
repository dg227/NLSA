import numpy as np
from nlsa.abstract_algebra2 import ImplementsPower,\
        ImplementsUnitalStarAlgebraLRModule
from nlsa.numpy.vector_algebra import VectorAlgebra, counting_measure
from numpy.typing import NDArray
from typing import Literal


K = np.int32
V = NDArray[K]
N = Literal[2]


def test_vector_algebra():
    vec: VectorAlgebra[N, K] = VectorAlgebra(dim=2, dtype=np.int32)
    v1 = np.array([1, 2], dtype=np.int32)
    v2 = np.array([3, -4], dtype=np.int32)
    k = np.array(np.int32(5))
    assert isinstance(vec, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(vec, ImplementsPower)
    assert np.all(vec.add(v1, v2) == np.array([4, -2]))
    assert np.all(vec.sub(v1, v2) == np.array([-2, 6]))
    assert np.all(vec.mul(v1, v2) == np.array([3, -8]))
    assert np.all(vec.smul(k, v1) == np.array([5, 10]))
    assert np.all(vec.power(v1, k) == np.array([1, 32]))
    assert np.all(vec.lmul(v1, v2) == np.array([3, -8]))
    assert np.all(vec.rmul(v1, v2) == np.array([3, -8]))
    assert np.all(vec.rdiv(v2, v1) == np.array([3, -2]))
    assert np.all(vec.ldiv(v1, v2) == np.array([3, -2]))
    assert np.all(vec.unit() == np.array([1, 1]))
    assert np.all(vec.star(v1) == np.array([1, 2]))
    assert np.all(counting_measure(np.array([v1, v2])) == np.array([3, -1]))
