import numpy as np
from nlsa import fourier_s1 as f1
from nlsa import fourier_t2 as f2
from nlsa import function_algebra as fun

def test_dual_s1():
    ks = f1.dual_group(2)
    assert np.all(ks == np.array([-2, -1, 0, 1, 2]))


def test_rkha_weights_s1():
    p = 0.5
    tau = 0.1
    k = 2
    w = f1.rkha_weights(p=p, tau=tau)
    wk = w(f1.dual_group(k))
    assert np.all(wk >= 1.0)
    assert np.all(np.shape(wk) == np.array([2 * k + 1]))


def test_fourier_basis_s1():
    k = 2
    phi = f1.fourier_basis(k)
    x = np.array([0, np.pi])
    y = phi(x)
    # assert np.all(y == np.array([[1, 1, 1, 1, 1], [1, -1, 1, -1, 1]]))
    assert np.all(np.shape(y) == np.array([2, 2 * k + 1]))
    

def test_rkha_basis_s1():
    p = 0.5
    tau = 0.1
    k = 2
    psi = f1.rkha_basis(p, tau, k)
    x = np.array([0, np.pi])
    y = psi(x)
    # assert np.all(y == np.array([[1, 1, 1, 1, 1], [1, -1, 1, -1, 1]]))
    assert np.all(np.shape(y) == np.array([2, 2 * k + 1]))


def test_synthesis_fourier_s1():
    kappa = 1
    mu = np.pi
    k = 2
    n = 5
    x = np.linspace(0, 2 * np.pi, n)
    f_hat = f1.von_mises_fourier(kappa, mu)
    c = f_hat(f1.dual_group(k))
    phi = f1.fourier_basis(k)
    f = fun.synthesis(phi, c)
    y = f(x)
    assert np.all(np.shape(y) == np.array([n]))


def test_mult_op_s1():
    k_max = 1
    v = np.array([-2, -1, 0, 1, 2])
    mult_op = f1.mult_op_fourier(k_max)
    m = mult_op(v)
    assert np.all(np.diag(m, k=-2) == v[4])
    assert np.all(np.diag(m, k=-1) == v[3])
    assert np.all(np.diag(m, k=0) == v[2])
    assert np.all(np.diag(m, k=1) == v[1])
    assert np.all(np.diag(m, k=2) == v[0])


def test_von_mises_feature_map():
    k_max = 3
    epsilon = 0.1
    x = np.array([0, 1, 2, 3])
    xi = f1.von_mises_feature_map(epsilon, k_max)
    y = xi(x)
    assert np.all(y.shape == np.array([4, 7]))


def test_dual_t2():
    k_max = (3, 5)
    k = f2.dual_group(k_max)
    n_k = f2.dual_size(k_max)
    assert np.all(np.shape(k) == np.array([2, n_k]))


def test_rkha_weights_t2():
    p = 0.5
    tau = 0.1
    k = (2, 3)
    n_k = f2.dual_size(k)
    w = f2.rkha_weights(p=p, tau=tau)
    wk = w(f2.dual_group(k))
    assert np.all(wk >= 1.0)
    assert np.all(np.shape(wk) == np.array([n_k]))


def test_fourier_basis_t2():
    k = (1, 2)
    n_x = 7
    n_k = f2.dual_size(k)
    phi = f2.fourier_basis(k)
    x1 = np.linspace(0, 2 * np.pi, n_x)
    x2 = (x1 + np.pi) % (2 * np.pi) 
    x = np.vstack((x1, x2)).T
    y = phi(x)
    assert np.all(np.shape(y) == np.array([n_x, n_k]))


def test_synthesis_fourier_t2():
    kappa = (1, 2)
    mu = (np.pi, np.pi)
    k = (1, 2)
    n_x = 7
    n_k = f2.dual_size(k)
    x1 = np.linspace(0, 2 * np.pi, n_x)
    x2 = (x1 + np.pi) % (2 * np.pi) 
    x = np.vstack((x1, x2)).T
    f_hat = f2.von_mises_fourier(kappa, mu)
    c = f_hat(f2.dual_group(k))
    phi = f2.fourier_basis(k)
    f = fun.synthesis(phi, c)
    y = f(x)
    assert np.all(np.shape(y) == np.array([n_x]))


def test_mult_op_t2():
    k = (1, 2)
    v1 = np.arange(-2 * k[0], 2 * k[0] + 1)
    v2 = np.arange(-2 * k[1], 2 * k[1] + 1)
    v = np.kron(v2, v1)
    mult_op = f2.mult_op_fourier(k)
    m = mult_op(v)
    n = f2.dual_size(k)
    assert np.all(np.shape(m) == np.array([n, n]))
