import numpy as np
from functools import partial
from more_itertools import take
from nlsa import abstract_algebra as alg
from nlsa import function_algebra as fun
from nlsa import scalars as scl
from nlsa.abstract_algebra import compose_by, id_map, iterate
from nptyping import Int, NDArray, Shape
from typing import Callable, TypeVar

V = NDArray[Shape["2"], Int]
M = NDArray[Shape["2, 2"], Int]
F = Callable[[int], int]
FV = Callable[[int], V]
FM = Callable[[int], M]


def test_add():
    f: F = lambda i: i + 1
    g: F = lambda i: i - 1
    h: F = fun.add(f, g)
    assert h(0) == 0


def test_higher_order_add():
    def a(f: F) -> int:
        return f(-1)

    def b(f: F) -> int:
        return f(1)

    def e(x: int) -> int:
        return x

    c = fun.add(a, b)
    assert c(e) == 0


def test_vector_valued_add():
    f: FV = lambda i: np.array([i, i + 1, i + 2])
    g: FV = lambda i: np.array([-i, -i - 1, -i - 2])
    h = fun.add(f, g)
    assert np.all(h(0) == np.array([0, 0, 0]))


def test_function_valued_add():
    def f(x: int) -> Callable[[int], int]:
        def fx(y: int) -> int:
            z = x + y
            return z
        return fx

    def g(x: int) -> Callable[[int], int]:
        def gx(y: int) -> int:
            z = x - y
            return z
        return gx

    h: F = fun.lift_from(fun).add(f, g)
    a = h(0)
    assert a(0) == 0


def test_sub():
    f: F = lambda i: i + 1
    g: F = lambda i: i + 1
    h: F = fun.sub(f, g)
    assert h(0) == 0


def test_higher_order_sub():
    def a(f: F) -> int:
        return f(-1)
    def b(f: F) -> int:
        return f(1)
    def e(x: int) -> int:
        return x ** 2

    c = fun.sub(a, b)
    assert c(e) == 0


def test_vector_valued_sub():
    f: FV = lambda i: np.array([i, i + 1, i + 2])
    g: FV = lambda i: np.array([i, i - 1, i - 2])
    h = fun.sub(f, g)
    assert np.all(h(0) == np.array([0, 2, 4]))


def test_function_valued_sub():
    def f(x: int) -> Callable[[int], int]:
        def fx(y: int) -> int:
            z = x + y
            return z
        return fx

    def g(x: int) -> Callable[[int], int]:
        def gx(y: int) -> int:
            z = x - y
            return z
        return gx

    h: F = fun.lift_from(fun).sub(f, g)
    a = h(1)
    assert a(1) == 2


def test_fmap():
    def f(x: int) -> int:
        return 2 * x
    g = fun.fmap(f)
    h = g(id_map)
    assert h(0) == 0
    assert h(1) == 2


def test_algmul():
    def f(x: int) -> M:
        y = np.diag([x, 0, x])
        return y
    def g(x: int) -> M:
        y = np.diag([0, x, 0])
        return y

    h: FM = fun.algmul(f, g)
    assert np.all(h(0) == 0)


def test_higher_order_algmul():
    def a(f: FM) -> M:
        return f(-1)
    def b(f: FM) -> M:
        return f(1)
    def e(x: int) -> M:
        return np.diag([x, x])

    c = fun.algmul(a, b)
    assert np.all(c(e) == -1 * np.eye(2))  


def test_integer_valued_algmul():
    def f(x: int) -> int:
        y = 2 * x
        return y
    def g(x: int) -> int:
        y = 3 * x
        return y

    h = fun.lift_from(scl).algmul(f, g)
    assert h(1) == 6


def test_compose():
    f: F = lambda i: i + 1
    g: F = lambda i: i - 1
    h: F = fun.compose(f, g)
    assert h(0) == 0


def test_compose2():
    f = lambda i: 2 * i
    g = lambda i, j: i + j
    h = fun.compose2(f, g)
    assert h(2, 1) == 6
    
def test_evaluate_at():
    f = lambda i: 2 * i
    x = 1
    assert fun.evaluate_at(x, f) == 2


def test_iterate():
    f = lambda i: 2 * i
    f_iter = iterate(fun, f)
    x = 1
    evalx = partial(fun.evaluate_at, x)
    fx = map(evalx, f_iter) 
    assert np.all(take(3, fx) == np.array([2, 4, 8]))
    f_iter = iterate(fun, f, initial=id_map)
    fx = map(evalx, f_iter) 
    assert np.all(take(4, fx) == np.array([1, 2, 4, 8]))


def test_compose_by():
    f: F = lambda i: i + 1
    g: F = lambda i: i - 1
    u = compose_by(fun, g)
    h = u(f)
    assert h(0) == 0
