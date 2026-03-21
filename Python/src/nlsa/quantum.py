"""Provide generic quantum mechanical operations."""

import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from functools import partial

type F[X, Y] = Callable[[X], Y]


def make_vector_state[A, V, K](
    impl: alg.ImplementsOperatorAlgebra[A, V, K], v: V, /
) -> Callable[[A], K]:
    """Make vector state of operator algebra."""

    def phi(a: A) -> K:
        return impl.codom.innerp(v, impl.app(a, v))

    return phi


def gelfand[A, K](a: A) -> Callable[[Callable[[A], K]], K]:
    """Compute Gelfand transform of algebra element."""

    def g(phi: Callable[[A], K]) -> K:
        return phi(a)

    return g


def make_qeval[A, V, K, X](
    impl: alg.ImplementsOperatorAlgebra[A, V, K],
    feature_map: Callable[[X], V],
    normalize: bool = False,
) -> Callable[[X], F[A, K]]:
    """Make quantum pointwise evaluation functional from feature map."""
    if normalize:
        feat = fun.compose(partial(alg.normalize, impl.dom), feature_map)
    else:
        feat = feature_map

    def eval_at(x: X) -> Callable[[A], K]:
        phi = make_vector_state(impl, feat(x))

        def evalx(a: A) -> K:
            return phi(a)

        return evalx

    return eval_at
