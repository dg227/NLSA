from nlsa.abstract_algebra2 import ImplementsVectorSpace
from nlsa.function_algebra2 import Lift, apply, compose, identity
from typing import Callable, Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')
L = Callable[[V], V]


def make_unit() -> Callable[[], L[V]]:
    """Make identity operator (unit of operator algebra)."""
    return identity


class OperatorAlgebra(Generic[V, K]):
    """Implement operator algebra structure on a vector space."""
    def __init__(self, hilb: ImplementsVectorSpace[V, K]):
        lft: Lift[V] = Lift()
        self.scl = hilb.scl
        self.hilb = hilb
        self.zero: Callable[[], L[V]] = lft.constant(hilb.zero)
        self.add: Callable[[L[V], L[V]], L[V]] = lft.binary(hilb.add)
        self.sub: Callable[[L[V], L[V]], L[V]] = lft.binary(hilb.sub)
        self.neg: Callable[[L[V]], L[V]] = lft.unary(hilb.neg)
        self.smul: Callable[[K, L[V]], L[V]] = lft.left(hilb.smul)
        self.mul: Callable[[L[V], L[V]], L[V]] = compose
        self.unit: Callable[[], L[V]] = make_unit
        self.app: Callable[[L[V], V], V] = apply
