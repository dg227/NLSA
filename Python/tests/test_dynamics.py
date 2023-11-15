import nlsa.dynamics as dn
import numpy as np
from functools import partial
from more_itertools import take
from multiprocess import Pool
from nlsa.abstract_algebra import id_map
from scipy import sparse


def test_cocycle_orbit():
    def phi(x: int) -> int:
        return (-1) ** x
    def psi(x: int, y:int) -> int:
        return x * y
    x = 0
    y = 1
    x_orb = dn.orbit(x, phi)
    y_orb = dn.cocycle_orbit(y, x_orb, psi)
    xn = take(5, x_orb)
    yn = take(5, y_orb)
    print(xn)
    print(yn)


def test_cocycle_orbit2():
    def psi(x: int, y:int) -> int:
        return x * y
    x_orb = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y = 1
    y_orb = dn.cocycle_orbit(y, x_orb, psi)
    xn = take(5, x_orb)
    yn = take(5, y_orb)
    print(xn)
    print(yn)

def test_stepanoff_generator():
    k = 1
    alpha = 3
    v = np.imag(dn.stepanoff_generator(alpha, k).todense())
    assert np.all(v == v.transpose())


def test_flow():
    dt = 1.0
    x = np.array([0.0])
    n_iter = 3
    phi = dn.flow(id_map, dt)
    x_orb = dn.orbit(x, phi)
    xn = take(n_iter + 1, x_orb)
    assert np.all(np.abs(xn - np.arange(0, n_iter + 1) <= 1E-6))


def test_flow2():
    dt = 1.0
    x = np.array([0.0, 0.0, 0.0])
    n_iter = 3
    phi = dn.flow(id_map, dt)
    x_orb = dn.orbit(x, phi)
    xn = take(n_iter + 1, x_orb)
    assert np.all(np.shape(xn) == np.array([4, 3]))


def test_flows():
    dt = 1.0
    x = np.array([0.0])
    n_iter = 3
    ts =  dt * np.arange(n_iter)
    phi = dn.flows(id_map, ts)
    xs = phi(x)
    assert np.all(np.abs(xs - np.arange(n_iter) <= 1E-6))


dt = 1.0
f = dn.flow(id_map, dt)
def phi0(x):
   return f(x)


def test_flow_ensemble():
    x = np.array([[0.0], [1.0], [2.0]])
    n_iter = 3
    p = Pool(2)
    y = p.map(phi0, x)
    assert np.all(np.shape(y) == np.shape(x))

    phi = partial(p.map, phi0)
    x_orb = dn.orbit(x, phi)
    xn = take(n_iter + 1, x_orb)
    assert np.all(np.shape(xn) == np.array([n_iter + 1, 3, 1]))


def test_stepanoff_flow():
    dt = 0.1
    alpha = np.sqrt(30)
    x = np.array([0.0, 0.0])
    n_iter = 3
    phi = dn.flow(dn.stepanoff_vec(alpha), dt)
    x_orb = dn.orbit(x, phi)
    xn = take(n_iter + 1, x_orb)
    assert np.all(np.shape(xn) == np.array([4, 2]))



# def test_stepanoff_flow_ensemble():
#     dt = 0.1
#     alpha = np.sqrt(30)
#     x = np.array([[0.0, 0.0], [1.0, 1.0]])
#     n_iter = 3
#     step = dn.flow(dn.stepanoff_vec(alpha), dt)
#     p = Pool(2)
#     y = p.map(step, x)
#     assert np.all(np.shape(y) == np.shape(x))

#     phi = partial(p.map, step)
#     x_orb = dn.orbit(x, phi)
#     xn = take(n_iter + 1, x_orb)
#     assert np.all(np.shape(xn) == np.array([n_iter + 1, 2, 2]))
