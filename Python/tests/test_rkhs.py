import nlsa.function_algebra as fun
import nlsa.rkhs as rkhs
import nlsa.matrix_algebra as mat
import nlsa.vector_algebra as vec
import nlsa.vector_algebra_pairwise as vep
import numpy as np
from functools import partial
from more_itertools import take
from nlsa.abstract_algebra import make_l2_innerp, riesz_dual, synthesis_operator
from nlsa.dynamics import circle_rotation, circle_embedding_r2, orbit
from nlsa.function_algebra import compose2


def test_energy():
    w = np.array([1, 2, 3])
    engy = rkhs.energy(w)
    c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    e = engy(c)
    assert np.all(e == (np.array([1, 4, 9]) * 6))


def test_gaussian():
    epsilon = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    x = np.array([[1, 0, 0]]) 
    y = k(x, x)
    assert np.all(y == np.array([1]))


def test_eval_sampling_measure():
    def f(x): 
        y = np.ones(x.shape)
        return y
    xs = np.array([[0, 1, 2]]) 
    fxs = f(xs)
    assert np.all(fxs == np.array([[1, 1, 1]]))
    fbar = rkhs.eval_sampling_measure(xs, f)
    assert fbar == 1


def test_kernel_operator():
    epsilon = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    xs = np.array([[0, 0, 0,], [1, 1, 1]])
    mu = riesz_dual(vec.innerp, vec.uniform(2))
    incl = partial(fun.evaluate_at, xs)
    k_op = rkhs.kernel_operator(vec, incl, mu, k)
    v = np.array([0, 0])
    v = v[:, np.newaxis]
    f = k_op(v)
    y = np.array([[0, 0, 0]])
    assert np.all(f(y) == np.array([[0]]))


def test_rnormalize():
    epsilon = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    xs = np.array([[0, 0, 0,], [1, 1, 1]])
    mu = riesz_dual(vec.innerp, vec.uniform(2))
    incl = partial(fun.evaluate_at, xs)
    k_op = rkhs.kernel_operator(vec, incl, mu, k)
    v = np.array([1, 1])
    v = v[:, np.newaxis]
    r = k_op(v)
    kr = rkhs.rnormalize(vec, k, r)
    kr_mat = kr(xs, xs)


def test_lnormalize():
    epsilon = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    xs = np.array([[0, 0, 0,], [1, 1, 1]])
    mu = riesz_dual(vec.innerp, np.ones(2))
    incl = partial(fun.evaluate_at, xs)
    k_op = rkhs.kernel_operator(vec, incl, mu, k)
    v = np.array([1, 1])
    v = v[:, np.newaxis]
    l = k_op(v)
    kl = rkhs.lnormalize(vec, k, l)
    kl_mat = kl(xs, xs)
    print(np.sum(kl_mat, axis=1))
    # assert(np.all(np.sum(kl_mat, axis=1) == np.array([1, 1])))


def test_dm():
    # Dataset generation
    a = 1 / np.sqrt(30)
    x0 = 0.0
    n = 100
    thetas = np.array(take(n, orbit(x0, circle_rotation(2 * np.pi * a))))
    xs = circle_embedding_r2(thetas) 

    # Diffusion maps kernel matrix
    epsilon = 0.15
    alpha = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    unit = np.ones(n)
    mu = riesz_dual(vec.innerp, unit)
    incl = partial(fun.evaluate_at, xs)
    p = rkhs.dm_normalize(vec, alpha, unit, incl, mu, k)
    p_mat = p(xs, xs)
    p_sum = np.sum(p_mat, axis=1)
    p_mat2 = p(xs[0], xs)
    np.testing.assert_allclose(p_sum, 1)
    
    # Eigenvalue problem
    n_eig = 5
    lamb, phi = mat.eig_sorted(p_mat, n_eig)
    lapl = lamb - 1
    lapl = lapl / lapl[1]
    np.testing.assert_allclose(lapl, np.array([0, 1, 1, 4, 4]), rtol=0.1,
                               atol=1)

    # Integral operator
    # We use inclv in order to support evaluation of multiple functions (phi's)
    # on multiple test points (xs).
    idx = 1
    n_test = 2
    inclv = partial(fun.evaluate_at_vectorized, xs)
    p_op = rkhs.kernel_operator(vec, inclv, mu, p)
    f = p_op(phi.T) 
    y = f(xs[0:n_test])
    print(y.shape)



    #Nystrom operator
    i_test = 1
    coeffs = np.identity(n_eig)
    nyst = rkhs.nystrom_basis(vec, p_op, lamb, phi.T)
    synth = synthesis_operator(fun, nyst)
    # varphi = synth(np.atleast_2d(coeffs[:, idx]).T)
    varphi = synth(np.atleast_2d(coeffs))
    phi_test = varphi(xs[i_test])
    # np.testing.assert_allclose(phi_test, phi[i_test, :])

    # print(np.atleast_2d(coeffs[:, 0:2]).shape)
    # print(phi_test.shape)
    
def test_dm_sym():
    # Dataset generation
    a = 1 / np.sqrt(30)
    x0 = 0.0
    n = 100
    thetas = np.array(take(n, orbit(x0, circle_rotation(2 * np.pi * a))))
    xs = circle_embedding_r2(thetas) 

    # Symmetric diffusion maps kernel matrix
    epsilon = 0.15
    alpha = 1
    shape_func = partial(rkhs.gaussian, epsilon)
    k = compose2(shape_func, rkhs.dist2)
    unit = np.ones(n)
    mu = riesz_dual(vec.innerp, unit)
    incl = partial(fun.evaluate_at, xs)
    s = rkhs.dmsym_normalize(vec, alpha, unit, incl, mu, k)
    s_mat = s(xs, xs)
    np.testing.assert_array_equal(s_mat.shape, [n, n])

    # Eigenvalue problem
    n_eig = 5
    with_eig = rkhs.dm_eigen(s_mat, n_eig, solver='eig')
    with_eigs = rkhs.dm_eigen(s_mat, n_eig, solver='eigs')
    np.testing.assert_allclose(with_eig[0], with_eigs[0])
    np.testing.assert_allclose(with_eig[2], with_eigs[2])
    lamb = with_eig[0]
    phi = with_eig[1]
    w = with_eig[2]

    # Non-symmetric kernel matrix and Nystrom operator
    i_test = [1, 2]
    p = rkhs.dm_normalize(vec, alpha, unit, incl, mu, k)
    incl = partial(fun.evaluate_at, xs)
    p_op = rkhs.kernel_operator(vep, incl, mu, p)
    nyst = rkhs.nystrom_basis(vec, p_op, lamb, phi.T)
    synth = synthesis_operator(fun, nyst)
    coeffs = np.identity(n_eig)
    varphi = synth(np.atleast_2d(coeffs))
    phi_test = varphi(xs[i_test])

    # Reconstruct coordinates of observation map
    w_inner = make_l2_innerp(vep, riesz_dual(vec.innerp, w))
    c = w_inner(phi.T, xs.T)
    f = synth(c)
    x_rec = f(xs)
    np.testing.assert_allclose(x_rec, xs, rtol=20, atol=0.12)

