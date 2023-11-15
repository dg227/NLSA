import numpy as np
from nlsa.abstract_algebra2 import FromScalarField,\
        ImplementsUnitalStarAlgebraLRModule
from nlsa.function_algebra2 import FunctionLRModule, BivariateFunctionSpace
from nlsa.scalar_algebra2 import IntField
from numpy.typing import NDArray
from typing import Callable, Literal


FZ = Callable[[float], int]
VZ = NDArray[np.int32]
N = Literal[2]


def test_functionlrmodule_int():
    fun: FunctionLRModule[float, int, int, int, int]\
        = FunctionLRModule(codomain=FromScalarField(IntField))
    assert isinstance(fun, ImplementsUnitalStarAlgebraLRModule)

    def f(x: float) -> int:
        return int(round(x)) + 1

    def g(x: float) -> int:
        return int(round(x)) - 1

    h_add = fun.add(f, g)
    assert h_add(1) == 2
    h_sub = fun.sub(f, g)
    assert h_sub(1) == 2
    h_smul = fun.smul(-1, f)
    assert h_smul(1) == -2
    h_mul = fun.mul(f, g)
    assert h_mul(1) == 0
    h_power = fun.power(f, 2)
    assert h_power(1) == 4
    h_star = fun.star(f)
    assert h_star(1) == 2
    h_unit = fun.unit()
    assert h_unit(2) == 1
    h_inv = fun.inv(f)
    assert h_inv(1) == 0
    h_div = fun.div(f, g)
    assert h_div(2) == 3
    h_lmul = fun.lmul(1, f)
    assert h_lmul(1) == 2
    h_ldiv = fun.ldiv(2, f)
    assert h_ldiv(1) == 1
    h_rmul = fun.rmul(f, 1)
    assert h_rmul(1) == 2
    h_rdiv = fun.rdiv(f, 2)
    assert h_rdiv(1) == 1


def test_bivariatefunctionalgebra():
    fun: BivariateFunctionSpace[float, float, int, int]\
        = BivariateFunctionSpace(codomain=FromScalarField(IntField))
    assert isinstance(fun, ImplementsUnitalStarAlgebraLRModule)

    def f(x1: float, x2: float) -> int:
        return int(round(x1 + 2 * x2))

    def g(x: float) -> int:
        return int(round(3 * x))

    h_lmul = fun.lmul(g, f)
    assert h_lmul(1, 2) == 15
    h_rmul = fun.rmul(f, g)
    assert h_rmul(1, 2) == 30
    h_ldiv = fun.ldiv(g, f)
    assert h_ldiv(1, 2) == 1  #(1+4)/3
    h_rdiv = fun.rdiv(f, g)
    assert h_rdiv(1, 2) == 0
