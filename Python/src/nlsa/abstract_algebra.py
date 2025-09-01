"""Provide protocols and generic functions for abstract algebraic structures.

"""
from collections.abc import Callable, Iterable
from functools import partial, reduce
from typing import Optional, Protocol, runtime_checkable

type F[X, Y] = Callable[[X], Y]


@runtime_checkable
class ImplementsZero[V](Protocol):
    """Implement additive zero."""
    zero: Callable[[], V]


@runtime_checkable
class ImplementsAdd[V](Protocol):
    """Implement addition."""
    add: Callable[[V, V], V]


@runtime_checkable
class ImplementsSub[V](Protocol):
    """Implement subtraction."""
    sub: Callable[[V, V], V]


@runtime_checkable
class ImplementsNeg[V](Protocol):
    """Implement additive negation."""
    neg: Callable[[V], V]


@runtime_checkable
class ImplementsSmul[K, V](Protocol):
    """Implement scalar multiplication."""
    smul: Callable[[K, V], V]


@runtime_checkable
class ImplementsSdiv[K, V](Protocol):
    """Implement scalar division."""
    sdiv: Callable[[K, V], V]


@runtime_checkable
class ImplementsMul[A](Protocol):
    """Implement algebraic multiplication."""
    mul: Callable[[A, A], A]


@runtime_checkable
class ImplementsUnit[A](Protocol):
    """Implement algebraic unit."""
    unit: Callable[[], A]


@runtime_checkable
class ImplementsDiv[A](Protocol):
    """Implement algebraic division."""
    div: Callable[[A, A], A]


@runtime_checkable
class ImplementsInv[A](Protocol):
    """Implement algebraic inversion."""
    inv: Callable[[A], A]


@runtime_checkable
class ImplementsMod[A](Protocol):
    """Implement algebraic modulus."""
    mod: Callable[[A], A]


@runtime_checkable
class ImplementsLmul[L, V](Protocol):
    """Implement left module multiplication."""
    lmul: Callable[[L, V], V]


@runtime_checkable
class ImplementsRmul[V, R](Protocol):
    """Implement right module multiplication."""
    rmul: Callable[[V, R], V]


@runtime_checkable
class ImplementsPower[A, K](Protocol):
    """Implement algebraic power (exponentiation)."""
    power: Callable[[A, K], A]


@runtime_checkable
class ImplementsSqrt[A](Protocol):
    """Implement square root."""
    sqrt: Callable[[A], A]


@runtime_checkable
class ImplementsExp[A, B](Protocol):
    """Implement exponentiation."""
    exp: Callable[[A], B]


@runtime_checkable
class ImplementsAdj[A](Protocol):
    """Implement algebraic adjunction."""
    adj: Callable[[A], A]


@runtime_checkable
class ImplementsLdiv[L, V](Protocol):
    """Implement left module division."""
    ldiv: Callable[[L, V], V]


@runtime_checkable
class ImplementsRdiv[V, R](Protocol):
    """Implement right module division."""
    rdiv: Callable[[V, R], V]


@runtime_checkable
class ImplementsNorm[V, R](Protocol):
    """Implement norm."""
    norm: Callable[[V], R]


@runtime_checkable
class ImplementsInnerp[V, K](Protocol):
    """Implement inner product."""
    innerp: Callable[[V, V], K]


@runtime_checkable
class ImplementsIncl[X, Y, V](Protocol):
    """Implement function inclusion."""
    incl: Callable[[F[X, Y]], V]


@runtime_checkable
class ImplementsIntegrate[V, K](Protocol):
    """Implement integration."""
    integrate: Callable[[V], K]


@runtime_checkable
class ImplementsCompose[F, G, H](Protocol):
    """Implement composition."""
    compose: Callable[[F, G], H]


@runtime_checkable
class ImplementsScalarField[K](ImplementsZero[K], ImplementsAdd[K],
                               ImplementsSub[K], ImplementsNeg[K],
                               ImplementsMul[K], ImplementsUnit[K],
                               ImplementsDiv[K], ImplementsInv[K],
                               ImplementsAdj[K], ImplementsSqrt[K],
                               ImplementsPower[K, K], ImplementsMod[K],
                               Protocol):
    """Implement scalar field operations."""
    pass


@runtime_checkable
class ImplementsVectorSpace[T, K](ImplementsZero[T], ImplementsAdd[T],
                                  ImplementsSub[T], ImplementsNeg[T],
                                  ImplementsSmul[K, T], ImplementsSdiv[K, T],
                                  Protocol):
    """Implement vector space operations."""
    scl: ImplementsScalarField[K]


@runtime_checkable
class ImplementsNormedSpace[T, K](ImplementsVectorSpace[T, K],
                                  ImplementsNorm[T, K], Protocol):
    """Implement normed space operations."""
    pass


@runtime_checkable
class ImplementsInnerProductSpace[V, K](ImplementsNormedSpace[V, K],
                                        ImplementsInnerp[V, K], Protocol):
    """Implement inner product space operations."""
    pass


@runtime_checkable
class ImplementsAlgebra[A, K](ImplementsVectorSpace[A, K], ImplementsMul[A],
                              ImplementsDiv[A], ImplementsInv[A],
                              ImplementsUnit[A], ImplementsAdj[A],
                              ImplementsPower[A, K], ImplementsSqrt[A],
                              ImplementsMod[A], Protocol):
    """Implement algebra operations."""
    pass


@runtime_checkable
class ImplementsNormedAlgebra[A, K](ImplementsAlgebra[A, K],
                                    ImplementsNorm[A, K], Protocol):
    """Implement normed algebra operations."""
    pass


@runtime_checkable
class ImplementsInnerProductAlgebra[V, K](ImplementsNormedAlgebra[V, K],
                                          ImplementsInnerp[V, K], Protocol):
    """Implement inner product space operations."""
    pass


# TODO: Check if we need dom and codom.
@runtime_checkable
class ImplementsOperatorSpace[T, V, W, K](ImplementsVectorSpace[T, K],
                                          Protocol):
    """
    Implement vector space operations for linear maps on inner product spaces.
    """
    dom: ImplementsInnerProductSpace[V, K]
    codom: ImplementsInnerProductSpace[W, K]
    app: Callable[[T, V], W]


@runtime_checkable
class ImplementsOperatorAlgebra[A, V, K](ImplementsOperatorSpace[A, V, V, K],
                                         ImplementsAlgebra[A, K],
                                         Protocol):
    """Implement algebra operations on a Hilbert space."""


@runtime_checkable
class ImplementsLModule[M, K, L](ImplementsVectorSpace[M, K],
                                 ImplementsLmul[L, M], ImplementsLdiv[L, M],
                                 Protocol):
    """Implement left module operations."""
    pass


@runtime_checkable
class ImplementsRModule[M, K, R](ImplementsVectorSpace[M, K],
                                 ImplementsRmul[M, R], ImplementsRdiv[M, R],
                                 Protocol):
    """Implement right module operations."""
    pass


@runtime_checkable
class ImplementsBimodule[M, K, L, R](ImplementsLModule[M, K, L],
                                     ImplementsRModule[M, K, R],
                                     Protocol):
    """Implement bimodulde operations."""
    pass


@runtime_checkable
class ImplementsInnerProductLModule[M, K, L](ImplementsInnerProductSpace[M, K],
                                             ImplementsLmul[L, M],
                                             ImplementsLdiv[L, M],
                                             Protocol):
    """Implement inner-product left module operations."""
    pass


@runtime_checkable
class ImplementsInnerProductRModule[M, K, R](ImplementsInnerProductSpace[M, K],
                                             ImplementsRmul[M, R],
                                             ImplementsRdiv[M, R],
                                             Protocol):
    """Implement inner-product right module operations."""
    pass


@runtime_checkable
class ImplementsInnerProductBimodule[M, K, L, R](
        ImplementsInnerProductLModule[M, K, L],
        ImplementsInnerProductRModule[M, K, R], Protocol):
    """Implement inner product bimodulde operations."""
    pass


@runtime_checkable
class ImplementsMeasurableFnAlgebra[X, Y, V, K](ImplementsAlgebra[V, K],
                                                ImplementsIncl[X, Y, V],
                                                Protocol):
    """Implement operations on equivalence classes of functions."""
    pass


@runtime_checkable
class ImplementsMeasureFnAlgebra[X, Y, V, K](
        ImplementsMeasurableFnAlgebra[X, Y, V, K], ImplementsIntegrate[V, Y],
        Protocol):
    """Implement operations on measure function space."""
    pass


@runtime_checkable
class ImplementsL2FnAlgebra[X, Y, V, K](ImplementsMeasureFnAlgebra[X, Y, V, K],
                                        ImplementsInnerp[V, K],
                                        Protocol):
    """Implement operations on L2 function space with algebra structure."""
    pass


@runtime_checkable
class ImplementsAnalysisOperators[V, Ks](Protocol):
    """Implement analysis operators associated with frame."""
    anal: Callable[[V], Ks]
    dual_anal: Callable[[V], Ks]


@runtime_checkable
class ImplementsSynthesisOperators[V, Ks](Protocol):
    """Implement synthesis operators associated with fame."""
    synth: Callable[[Ks], V]
    dual_synth: Callable[[Ks], V]


@runtime_checkable
class ImplementsFrame[V, Ks, I](ImplementsAnalysisOperators[V, Ks],
                                ImplementsSynthesisOperators[V, Ks],
                                Protocol):
    """Implement analysis and synthesis operators associated with frame."""
    vec: Callable[[I], V]
    dual_vec: Callable[[I], V]


@runtime_checkable
class ImplementsDimensionedFrame[V, Ks, I](ImplementsFrame[V, Ks, I],
                                           Protocol):
    """Implement operators of frame with known dimension."""
    dim: int


@runtime_checkable
class ImplementsFnAnalysisOperators[X, Y, Ks](Protocol):
    """Implement function analysis operators."""
    fn_anal: Callable[[F[X, Y]], Ks]
    dual_fn_anal: Callable[[F[X, Y]], Ks]


@runtime_checkable
class ImplementsFnSynthesisOperators[X, Y, Ks](Protocol):
    """Implement function synthesis operators."""
    fn_synth: Callable[[Ks], F[X, Y]]
    dual_fn_synth: Callable[[Ks], F[X, Y]]


@runtime_checkable
class ImplementsL2FnFrame[X, Y, V, Ks, I](
        ImplementsFrame[V, Ks, I], ImplementsFnSynthesisOperators[X, Y, Ks],
        ImplementsFnAnalysisOperators[X, Y, Ks], Protocol):
    """Implement frame of L2 function space."""
    fn: Callable[[I], F[X, Y]]
    dual_fn: Callable[[I], F[X, Y]]


@runtime_checkable
class ImplementsDimensionedL2FnFrame[X, Y, V, Ks, I](
        ImplementsL2FnFrame[X, Y, V, Ks, I], Protocol):
    """Implement operators of L2 function frame with known dimension."""
    dim: int


def compose_by[F, G, H](impl: ImplementsCompose[F, G, H], g: G, /) \
        -> Callable[[F], H]:
    """Make composition map."""
    def u(f: F) -> H:
        h = impl.compose(f, g)
        return h
    return u


def precompose_by[F, G, H](impl: ImplementsCompose[F, G, H], f: F, /) \
        -> Callable[[G], H]:
    """Make pre-composition map."""
    return partial(impl.compose, f)


def conjugate_by[F, G, H, U, V](impl1: ImplementsCompose[U, H, F], u: U,
                                impl2: ImplementsCompose[G, V, H], v: V, /) \
        -> Callable[[G], F]:
    """Conjugation map."""
    def c(g: G) -> F:
        h = impl2.compose(g, v)
        f = impl1.compose(u, h)
        return f
    return c


def multiply_by[A](impl: ImplementsMul[A], a: A, /) -> Callable[[A], A]:
    """Make multiplication operator."""
    def m(b: A) -> A:
        c = impl.mul(a, b)
        return c
    return m


def divide_by[A](impl: ImplementsDiv[A], a: A, /) -> Callable[[A], A]:
    """Make division operator."""
    def m(b: A) -> A:
        c = impl.div(b, a)
        return c
    return m


def smultiply_by[K, V](impl: ImplementsSmul[K, V], a: K, /) \
        -> Callable[[V], V]:
    """Make scalar multiplication operator."""
    def m(b: V) -> V:
        c = impl.smul(a, b)
        return c
    return m


def ldivide_by[L, M](impl: ImplementsLdiv[L, M], a: L, /) -> Callable[[M], M]:
    """Make left division operator."""
    def m(b: M) -> M:
        c = impl.ldiv(a, b)
        return c
    return m


def exponentiate_by[A, K](impl: ImplementsPower[A, K], a: K, /) \
        -> Callable[[A], A]:
    """Make exponentiation map."""
    def exp_a(b: A) -> A:
        c = impl.power(b, a)
        return c
    return exp_a


def identity[X](x: X, /) -> X:
    """Identity map."""
    return x


def lapp[T, V, W, K](impl: ImplementsOperatorSpace[T, V, W, K], a: T, v: V,
                     /) -> W:
    """Apply linear map to vector."""
    return impl.app(a, v)


def make_linear_operator[T, V, W, K](impl: ImplementsOperatorSpace[T, V, W, K],
                                     a: T, /) -> Callable[[V], W]:
    """Make linear map."""
    def op(v: V) -> W:
        return impl.app(a, v)
    return op


def make_form[T, V, W, K](impl: ImplementsOperatorSpace[T, V, W, K], a: T)\
        -> Callable[[W, V], K]:
    """Make bilinear or sesquilinear form."""
    def b(w: W, v: V) -> K:
        return impl.codom.innerp(w, impl.app(a, v))
    return b


def normalize[V, K](impl: ImplementsNormedSpace[V, K], v: V) -> V:
    """Normalize vector in normed space."""
    return impl.sdiv(impl.norm(v), v)


def make_vector_state[A, V, K](impl: ImplementsOperatorAlgebra[A, V, K],
                               v: V, /) -> Callable[[A], K]:
    """Make vector state of operator algebra."""
    def phi(a: A) -> K:
        return impl.codom.innerp(v, impl.app(a, v))
    return phi


def gelfand[A, K](impl: ImplementsAlgebra[A, K], a: A)\
        -> Callable[[Callable[[A], K]], K]:
    """Compute Gelfand transform of algebra element."""
    def g(phi: Callable[[A], K]) -> K:
        return phi(a)
    return g


def make_qeval[A, V, K, X](impl: ImplementsOperatorAlgebra[A, V, K],
                           feat: Callable[[X], V])\
        -> Callable[[X], F[A, K]]:
    """Make quantum pointwise evaluation functional from feature map."""
    def eval_at(x: X) -> Callable[[A], K]:
        phi = make_vector_state(impl, feat(x))

        def evalx(a: A) -> K:
            return phi(a)

        return evalx
    return eval_at


def sum[V, K](impl: ImplementsVectorSpace[V, K], vs: Iterable[V],
              initializer: Optional[V] = None) -> V:
    """Sum a collection of elements of a vector space."""
    if initializer is None:
        initializer = impl.zero()
    return reduce(impl.add, vs, initializer)


def product[A, K](impl: ImplementsAlgebra[A, K], vs: Iterable[A],
                  initializer: Optional[A] = None) -> A:
    """Multiply a collection of algebra elements."""
    if initializer is None:
        initializer = impl.unit()
    return reduce(impl.mul, vs, initializer)


def linear_combination[V, K](impl: ImplementsVectorSpace[V, K],
                             cs: Iterable[K], vs: Iterable[V]) -> V:
    """Form linear combination of elements of a vector space."""
    cvs = map(impl.smul, cs, vs)
    return sum(impl, cvs)
