import nlsa.dynamics as dn
import numpy as np
from functools import partial
from more_itertools import take
# from multiprocessing import Pool
from multiprocess import Pool
from nlsa.abstract_algebra import id_map
# from pathos.multiprocessing import ProcessPool as Pool
from scipy import sparse

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

    
