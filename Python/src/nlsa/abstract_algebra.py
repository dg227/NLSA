"""Provide protocols and functions for abstract algebraic structures."""

from collections.abc import Callable, Iterable
from functools import partial, reduce
from typing import (
    Optional,
    Protocol,
    SupportsComplex,
    SupportsFloat,
    final,
    runtime_checkable,
)

type F[X, Y] = Callable[[X], Y]


@runtime_checkable
class ImplementsZero[V](Protocol):
    """Implement additive zero."""

    @property
    def zero(self) -> Callable[[], V]:
        """Additive zero."""
        ...


@runtime_checkable
class ImplementsAdd[V](Protocol):
    """Implement addition."""

    @property
    def add(self) -> Callable[[V, V], V]:
        """Addition."""
        ...


@runtime_checkable
class ImplementsSub[V](Protocol):
    """Implement subtraction."""

    @property
    def sub(self) -> Callable[[V, V], V]:
        """Subtraction."""
        ...


@runtime_checkable
class ImplementsNeg[V](Protocol):
    """Implement additive negation."""

    @property
    def neg(self) -> Callable[[V], V]:
        """Additive negation."""
        ...


@runtime_checkable
class ImplementsSmul[K, V](Protocol):
    """Implement scalar multiplication."""

    @property
    def smul(self) -> Callable[[K, V], V]:
        """Scalar multiplication."""
        ...


@runtime_checkable
class ImplementsSdiv[K, V](Protocol):
    """Implement scalar division."""

    @property
    def sdiv(self) -> Callable[[K, V], V]:
        """Scalar division."""
        ...


@runtime_checkable
class ImplementsMul[A](Protocol):
    """Implement algebraic multiplication."""

    @property
    def mul(self) -> Callable[[A, A], A]:
        """Algebraic multiplication."""
        ...


@runtime_checkable
class ImplementsUnit[A](Protocol):
    """Implement algebraic unit."""

    @property
    def unit(self) -> Callable[[], A]:
        """Algebraic unit."""
        ...


@runtime_checkable
class ImplementsDiv[A](Protocol):
    """Implement algebraic division."""

    @property
    def div(self) -> Callable[[A, A], A]:
        """Algebraic division."""
        ...


@runtime_checkable
class ImplementsInv[A](Protocol):
    """Implement algebraic inversion."""

    @property
    def inv(self) -> Callable[[A], A]:
        """Algebraic inversion."""
        ...


@runtime_checkable
class ImplementsLog[A](Protocol):
    """Implement natural logarithm."""

    @property
    def log(self) -> Callable[[A], A]:
        """Natural logarithm."""
        ...


@runtime_checkable
class ImplementsLog10[A](Protocol):
    """Implement base 10 logarithm."""

    @property
    def log10(self) -> Callable[[A], A]:
        """Base 10 logarithm."""
        ...


@runtime_checkable
class ImplementsAbs[A](Protocol):
    """Implement algebraic modulus (absolute value)."""

    @property
    def abs(self) -> Callable[[A], A]:
        """Algebraic modulus."""
        ...


@runtime_checkable
class ImplementsLmul[L, V](Protocol):
    """Implement left module multiplication."""

    @property
    def lmul(self) -> Callable[[L, V], V]:
        """Left module multiplication."""
        ...


@runtime_checkable
class ImplementsRmul[V, R](Protocol):
    """Implement right module multiplication."""

    @property
    def rmul(self) -> Callable[[V, R], V]:
        """Right module multiplication."""
        ...


@runtime_checkable
class ImplementsPower[A, K](Protocol):
    """Implement monoidal power."""

    @property
    def power(self) -> Callable[[A, K], A]:
        """Monoidal power."""
        ...


@runtime_checkable
class ImplementsMPower[A](Protocol):
    """Implement monoidal power."""

    @property
    def mpower(self) -> Callable[[A, int], A]:
        """Monoidal power."""
        ...


@runtime_checkable
class ImplementsSqrt[A](Protocol):
    """Implement square root."""

    @property
    def sqrt(self) -> Callable[[A], A]:
        """Square root."""
        ...


@runtime_checkable
class ImplementsExp[A, B](Protocol):
    """Implement exponentiation."""

    @property
    def exp(self) -> Callable[[A], B]:
        """Exponentiation."""
        ...


@runtime_checkable
class ImplementsExp10[A](Protocol):
    """Implement exponentiation with base 10."""

    @property
    def exp10(self) -> Callable[[A], A]:
        """Base 10 exponentiation."""
        ...


@runtime_checkable
class ImplementsAdj[A](Protocol):
    """Implement algebraic adjunction."""

    @property
    def adj(self) -> Callable[[A], A]:
        """Algebraic adjunction."""
        ...


@runtime_checkable
class ImplementsLdiv[L, V](Protocol):
    """Implement left module division."""

    @property
    def ldiv(self) -> Callable[[L, V], V]:
        """Left module division."""
        ...


@runtime_checkable
class ImplementsRdiv[V, R](Protocol):
    """Implement right module division."""

    @property
    def rdiv(self) -> Callable[[V, R], V]:
        """Right module division."""
        ...


@runtime_checkable
class ImplementsNorm[V, R](Protocol):
    """Implement norm."""

    @property
    def norm(self) -> Callable[[V], R]:
        """Norm."""
        ...


@runtime_checkable
class ImplementsInnerp[V, K](Protocol):
    """Implement inner product."""

    @property
    def innerp(self) -> Callable[[V, V], K]:
        """Inner product."""
        ...


@runtime_checkable
class ImplementsIncl[X, Y, V](Protocol):
    """Implement function inclusion."""

    @property
    def incl(self) -> Callable[[F[X, Y]], V]:
        """Function inclusion."""
        ...


@runtime_checkable
class ImplementsIntegrate[V, K](Protocol):
    """Implement integration."""

    @property
    def integrate(self) -> Callable[[V], K]:
        """Integration."""
        ...


@runtime_checkable
class ImplementsCompose[F, G, H](Protocol):
    """Implement composition."""

    @property
    def compose(self) -> Callable[[F, G], H]:
        """Composition."""
        ...


@runtime_checkable
class ImplementsScalarField[K](
    ImplementsZero[K],
    ImplementsAdd[K],
    ImplementsSub[K],
    ImplementsNeg[K],
    ImplementsMul[K],
    ImplementsUnit[K],
    ImplementsDiv[K],
    ImplementsInv[K],
    ImplementsMPower[K],
    Protocol,
):
    """Implement scalar field."""

    pass


@runtime_checkable
class ImplementsRealScalarField[K](
    ImplementsScalarField[K],
    ImplementsSqrt[K],
    ImplementsPower[K, K],
    ImplementsAbs[K],
    ImplementsExp[K, K],
    ImplementsExp10[K],
    ImplementsLog[K],
    ImplementsLog10[K],
    Protocol,
):
    """Implement real scalar field."""

    @property
    def from_pyscalar(self) -> Callable[[SupportsFloat], K]:
        """Convert Python float to scalar."""
        ...


@runtime_checkable
class ImplementsComplexScalarField[K](
    ImplementsRealScalarField[K],
    ImplementsAdj[K],
    Protocol,
):
    """Implement complex scalar field."""

    @property
    def from_pyscalar(self) -> Callable[[SupportsFloat | SupportsComplex], K]:
        """Convert Python float or complex to scalar."""
        ...


@runtime_checkable
class ImplementsVectorSpace[T, K](
    ImplementsZero[T],
    ImplementsAdd[T],
    ImplementsSub[T],
    ImplementsNeg[T],
    ImplementsSmul[K, T],
    ImplementsSdiv[K, T],
    Protocol,
):
    """Implement vector space."""

    @property
    def scl(self) -> ImplementsScalarField[K]:
        """Scalar field associated with vector space."""
        ...


@runtime_checkable
class ImplementsRealVectorSpace[T, K](
    ImplementsVectorSpace[T, K],
    Protocol,
):
    """Implement real vector space."""

    @property
    def scl(self) -> ImplementsRealScalarField[K]:
        """Scalar field associated with real vector space."""
        ...


@runtime_checkable
class ImplementsComplexVectorSpace[T, K](
    ImplementsRealVectorSpace[T, K],
    Protocol,
):
    """Implement complex vector space."""

    @property
    def scl(self) -> ImplementsComplexScalarField[K]:
        """Scalar field associated with complex vector space."""
        ...


@runtime_checkable
class ImplementsNormedSpace[T, K](
    ImplementsRealVectorSpace[T, K], ImplementsNorm[T, K], Protocol
):
    """Implement normed space operations."""

    pass


@runtime_checkable
class ImplementsInnerProductSpace[V, K](
    ImplementsNormedSpace[V, K], ImplementsInnerp[V, K], Protocol
):
    """Implement inner product space."""

    pass


@runtime_checkable
class ImplementsAlgebra[A, K](
    ImplementsVectorSpace[A, K],
    ImplementsMul[A],
    ImplementsUnit[A],
    ImplementsMPower[A],
    Protocol,
):
    """Implement algebra."""

    pass


@runtime_checkable
class ImplementsRealAlgebra[A, K](
    ImplementsRealVectorSpace[A, K],
    ImplementsAlgebra[A, K],
    Protocol,
):
    """Implement algebra over the reals."""

    pass


@runtime_checkable
class ImplementsStarAlgebra[A, K](
    ImplementsComplexVectorSpace[A, K],
    ImplementsAlgebra[A, K],
    ImplementsAdj[A],
    Protocol,
):
    """Implement algebra over the reals."""

    pass


@runtime_checkable
class ImplementsNormedAlgebra[A, K](
    ImplementsRealAlgebra[A, K], ImplementsNorm[A, K], Protocol
):
    """Implement normed algebra."""

    pass


@runtime_checkable
class ImplementsNormedStarAlgebra[A, K](
    ImplementsStarAlgebra[A, K], ImplementsNorm[A, K], Protocol
):
    """Implement normed star algebra."""

    pass


@runtime_checkable
class ImplementsInnerProductAlgebra[V, K](
    ImplementsNormedAlgebra[V, K], ImplementsInnerp[V, K], Protocol
):
    """Implement algebra with inner product."""

    pass


@runtime_checkable
class ImplementsInnerProductStarAlgebra[V, K](
    ImplementsNormedStarAlgebra[V, K], ImplementsInnerp[V, K], Protocol
):
    """Implement star algebra with inner product."""

    pass


@runtime_checkable
class ImplementsAlgebraWithDivision[A, K](
    ImplementsAlgebra[A, K], ImplementsDiv[A], ImplementsInv[A], Protocol
):
    """Implement algebra with division and inverse."""

    pass


@runtime_checkable
class ImplementsRealAlgebraWithDivision[A, K](
    ImplementsRealAlgebra[A, K],
    ImplementsAlgebraWithDivision[A, K],
    Protocol,
):
    """Implement algebra over the reals with division and inverse."""

    pass


@runtime_checkable
class ImplementsStarAlgebraWithDivision[A, K](
    ImplementsStarAlgebra[A, K],
    ImplementsAlgebraWithDivision[A, K],
    Protocol,
):
    """Implement star algebra with division and inverse."""

    pass


@runtime_checkable
class ImplementsCalculus[A, K](
    ImplementsPower[A, K],
    ImplementsSqrt[A],
    ImplementsAbs[A],
    Protocol,
):
    """Implement basic functional calculus operations."""

    pass


@runtime_checkable
class ImplementsAlgebraWithCalculus[A, K](
    ImplementsRealAlgebraWithDivision[A, K],
    ImplementsCalculus[A, K],
    Protocol,
):
    """Implement real algebra operations with functional calculus."""

    pass


@runtime_checkable
class ImplementsStarAlgebraWithCalculus[A, K](
    ImplementsStarAlgebraWithDivision[A, K],
    ImplementsCalculus[A, K],
    ImplementsAdj[A],
    Protocol,
):
    """Implement star algebra operations with functional calculus."""

    pass


@runtime_checkable
class ImplementsNormedAlgebraWithCalculus[A, K](
    ImplementsAlgebraWithCalculus[A, K], ImplementsNorm[A, K], Protocol
):
    """Implement normed algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsInnerProductAlgebraWithCalculus[V, K](
    ImplementsNormedAlgebraWithCalculus[V, K],
    ImplementsInnerp[V, K],
    Protocol,
):
    """Implement real algebra with inner product and functional calculus."""

    pass


@runtime_checkable
class ImplementsNormedStarAlgebraWithCalculus[A, K](
    ImplementsStarAlgebraWithCalculus[A, K], ImplementsNorm[A, K], Protocol
):
    """Implement normed star algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsInnerProductStarAlgebraWithCalculus[V, K](
    ImplementsNormedStarAlgebraWithCalculus[V, K],
    ImplementsInnerp[V, K],
    Protocol,
):
    """Implement star algebra with inner product and functional calculus."""

    pass


# TODO: Check if we need dom and codom.
@runtime_checkable
class ImplementsOperatorSpace[T, V, W, K](
    ImplementsRealVectorSpace[T, K], Protocol
):
    """Implement vector space of linear maps between Hilbert spaces."""

    @property
    def dom(self) -> ImplementsInnerProductSpace[V, K]:
        """Domain of operators in operator space."""
        ...

    @property
    def codom(self) -> ImplementsInnerProductSpace[W, K]:
        """Codomain of operators in operator space."""
        ...

    @property
    def app(self) -> Callable[[T, V], W]:
        """Application of operators in operator space."""
        ...


@runtime_checkable
class ImplementsOperatorSystem[T, V, W, K](
    ImplementsComplexVectorSpace[T, K],
    ImplementsOperatorSpace[T, V, W, K],
    ImplementsAdj[T],
    Protocol,
):
    """Implement operator system between Hilbert spaces."""

    pass


@runtime_checkable
class ImplementsNormedOperatorSpace[T, V, W, K](
    ImplementsOperatorSpace[T, V, W, K],
    ImplementsNorm[T, K],
    Protocol,
):
    """Implement operator space with norm."""

    pass


@runtime_checkable
class ImplementsNormedOperatorSystem[T, V, W, K](
    ImplementsOperatorSystem[T, V, W, K],
    ImplementsNorm[T, K],
    Protocol,
):
    """Implement operator system with norm."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorSpace[T, V, W, K](
    ImplementsOperatorSpace[T, V, W, K],
    ImplementsInnerp[T, K],
    Protocol,
):
    """Implement operator space with inner product."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorSystem[T, V, W, K](
    ImplementsOperatorSystem[T, V, W, K],
    ImplementsInnerp[T, K],
    Protocol,
):
    """Implement operator system with inner product."""

    pass


@runtime_checkable
class ImplementsOperatorAlgebra[A, V, K](
    ImplementsOperatorSpace[A, V, V, K],
    ImplementsAlgebra[A, K],
    Protocol,
):
    """Implement operator algebra on Hilbert space."""

    pass


@runtime_checkable
class ImplementsOperatorStarAlgebra[A, V, K](
    ImplementsOperatorSystem[A, V, V, K],
    ImplementsStarAlgebra[A, K],
    Protocol,
):
    """Implement operator algebra on Hilbert space."""

    pass


@runtime_checkable
class ImplementsOperatorAlgebraWithCalculus[A, V, K](
    ImplementsOperatorAlgebra[A, V, K],
    ImplementsCalculus[A, K],
    Protocol,
):
    """Implement operator algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsOperatorStarAlgebraWithCalculus[A, V, K](
    ImplementsOperatorStarAlgebra[A, V, K],
    ImplementsCalculus[A, K],
    Protocol,
):
    """Implement operator star algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsNormedOperatorAlgebra[A, V, K](
    ImplementsOperatorAlgebra[A, V, K],
    ImplementsNorm[A, K],
    Protocol,
):
    """Implement normed operator algebra."""

    pass


@runtime_checkable
class ImplementsNormedOperatorAlgebraWithCalculus[A, V, K](
    ImplementsOperatorAlgebraWithCalculus[A, V, K],
    ImplementsNorm[A, K],
    Protocol,
):
    """Implement normed operator algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsNormedOperatorStarAlgebra[A, V, K](
    ImplementsOperatorStarAlgebra[A, V, K],
    ImplementsNorm[A, K],
    Protocol,
):
    """Implement normed operator star algebra."""

    pass


@runtime_checkable
class ImplementsNormedOperatorStarAlgebraWithCalculus[A, V, K](
    ImplementsOperatorStarAlgebraWithCalculus[A, V, K],
    ImplementsNorm[A, K],
    Protocol,
):
    """Implement normed operator star algebra with functional calculus."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorAlgebra[A, V, K](
    ImplementsNormedOperatorAlgebra[A, V, K],
    ImplementsInnerp[A, K],
    Protocol,
):
    """Implement operator algebra with inner product."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorStarAlgebra[A, V, K](
    ImplementsNormedOperatorStarAlgebra[A, V, K],
    ImplementsInnerp[A, K],
    Protocol,
):
    """Implement operator star algebra with inner product."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorAlgebraWithCalculus[A, V, K](
    ImplementsInnerProductOperatorAlgebra[A, V, K],
    ImplementsCalculus[A, K],
    Protocol,
):
    """Implement operator algebra with inner product, functional calculus."""

    pass


@runtime_checkable
class ImplementsInnerProductOperatorStarAlgebraWithCalculus[A, V, K](
    ImplementsInnerProductOperatorStarAlgebra[A, V, K],
    ImplementsCalculus[A, K],
    Protocol,
):
    """Implement operator star algebra with inner product, func. calculus."""

    pass


@runtime_checkable
class ImplementsLModule[M, K, L](
    ImplementsVectorSpace[M, K],
    ImplementsLmul[L, M],
    Protocol,
):
    """Implement left module over a vector space."""

    pass


@runtime_checkable
class ImplementsRModule[M, K, R](
    ImplementsVectorSpace[M, K],
    ImplementsRmul[M, R],
    Protocol,
):
    """Implement right module over a vector space."""

    pass


@runtime_checkable
class ImplementsBimodule[M, K, L, R](
    ImplementsLModule[M, K, L], ImplementsRModule[M, K, R], Protocol
):
    """Implement bimodule over a vector space."""

    pass


@runtime_checkable
class ImplementsLDivModule[M, K, L](
    ImplementsLModule[M, K, L],
    ImplementsLdiv[L, M],
    Protocol,
):
    """Implement left division module over a vector space."""

    pass


@runtime_checkable
class ImplementsRDivModule[M, K, R](
    ImplementsRModule[M, K, R],
    ImplementsRdiv[M, R],
    Protocol,
):
    """Implement right division module over a vector space."""

    pass


@runtime_checkable
class ImplementsDivBimodule[M, K, L, R](
    ImplementsLDivModule[M, K, L], ImplementsRDivModule[M, K, R], Protocol
):
    """Implement division bimodule over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductLModule[M, K, L](
    ImplementsInnerProductSpace[M, K],
    ImplementsLmul[L, M],
    ImplementsLdiv[L, M],
    Protocol,
):
    """Implement inner-product left module over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductRModule[M, K, R](
    ImplementsInnerProductSpace[M, K],
    ImplementsRmul[M, R],
    ImplementsRdiv[M, R],
    Protocol,
):
    """Implement inner-product right module over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductBimodule[M, K, L, R](
    ImplementsInnerProductLModule[M, K, L],
    ImplementsInnerProductRModule[M, K, R],
    Protocol,
):
    """Implement inner product bimodulde over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductLDivModule[M, K, L](
    ImplementsLDivModule[M, K, L],
    ImplementsInnerProductSpace[M, K],
    Protocol,
):
    """Implement inner-product left division module over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductRDivModule[M, K, R](
    ImplementsRDivModule[M, K, R],
    ImplementsInnerProductSpace[M, K],
    Protocol,
):
    """Implement inner-product right division module over a vector space."""

    pass


@runtime_checkable
class ImplementsInnerProductDivBimodule[M, K, L, R](
    ImplementsInnerProductLDivModule[M, K, L],
    ImplementsInnerProductRDivModule[M, K, R],
    Protocol,
):
    """Implement inner product division bimodulde over a vector space."""

    pass


@runtime_checkable
class ImplementsMeasurableFnAlgebra[X, Y, V, K](
    ImplementsAlgebraWithCalculus[V, K], ImplementsIncl[X, Y, V], Protocol
):
    """Implement operations on equivalence classes of functions."""

    pass


@runtime_checkable
class ImplementsMeasurableFnStarAlgebra[X, Y, V, K](
    ImplementsStarAlgebraWithCalculus[V, K], ImplementsIncl[X, Y, V], Protocol
):
    """Implement operations on equivalence classes of functions."""

    pass


@runtime_checkable
class ImplementsMeasureFnAlgebra[X, Y, V, K](
    ImplementsMeasurableFnAlgebra[X, Y, V, K],
    ImplementsIntegrate[V, Y],
    Protocol,
):
    """Implement operations on measure function space."""

    pass


@runtime_checkable
class ImplementsMeasureFnStarAlgebra[X, Y, V, K](
    ImplementsMeasurableFnStarAlgebra[X, Y, V, K],
    ImplementsIntegrate[V, Y],
    Protocol,
):
    """Implement operations on measure function space."""

    pass


@runtime_checkable
class ImplementsL2FnAlgebra[X, Y, V, K](
    ImplementsMeasureFnStarAlgebra[X, Y, V, K],
    ImplementsInnerp[V, K],
    Protocol,
):
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
class ImplementsFrame[V, Ks, I](
    ImplementsAnalysisOperators[V, Ks],
    ImplementsSynthesisOperators[V, Ks],
    Protocol,
):
    """Implement analysis and synthesis operators associated with frame."""

    vec: Callable[[I], V]
    dual_vec: Callable[[I], V]


@runtime_checkable
class ImplementsDimensionedFrame[V, Ks, I](
    ImplementsFrame[V, Ks, I], Protocol
):
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
    ImplementsFrame[V, Ks, I],
    ImplementsFnSynthesisOperators[X, Y, Ks],
    ImplementsFnAnalysisOperators[X, Y, Ks],
    Protocol,
):
    """Implement frame of L2 function space."""

    fn: Callable[[I], F[X, Y]]
    dual_fn: Callable[[I], F[X, Y]]


@runtime_checkable
class ImplementsDimensionedL2FnFrame[X, Y, V, Ks, I](
    ImplementsL2FnFrame[X, Y, V, Ks, I], Protocol
):
    """Implement operators of L2 function frame with known dimension."""

    dim: int


@final
class AsBimodule[A, K](ImplementsBimodule[A, K, A, A]):
    """Implement algebra as bimodule over itself."""

    def __init__(self, alg: ImplementsAlgebra[A, K]):
        """Initialize bimodule implementation from algebra field."""
        self._alg = alg

    @property
    def scl(self) -> ImplementsScalarField[K]:
        """Return scl property of AsAlgebra object."""
        return self._alg.scl

    @property
    def zero(self) -> Callable[[], A]:
        """Return zero property of AsBimodule object."""
        return self._alg.zero

    @property
    def add(self) -> Callable[[A, A], A]:
        """Return add property of AsBimodule object."""
        return self._alg.add

    @property
    def sub(self) -> Callable[[A, A], A]:
        """Return sub property of AsBimodule object."""
        return self._alg.sub

    @property
    def neg(self) -> Callable[[A], A]:
        """Return neg property of AsBimodule object."""
        return self._alg.neg

    @property
    def smul(self) -> Callable[[K, A], A]:
        """Return smul property of AsBimodule object."""
        return self._alg.smul

    @property
    def sdiv(self) -> Callable[[K, A], A]:
        """Return sdiv property of AsAlgebra object."""
        return self._alg.sdiv

    @property
    def lmul(self) -> Callable[[A, A], A]:
        """Return lmul property of AsBimodule object."""
        return self._alg.mul

    @property
    def rmul(self) -> Callable[[A, A], A]:
        """Return rmul property of AsBimodule object."""
        return self._alg.mul


def compose_by[F, G, H](
    impl: ImplementsCompose[F, G, H], g: G, /
) -> Callable[[F], H]:
    """Make composition map."""

    def u(f: F) -> H:
        h = impl.compose(f, g)
        return h

    return u


def precompose_by[F, G, H](
    impl: ImplementsCompose[F, G, H], f: F, /
) -> Callable[[G], H]:
    """Make pre-composition map."""
    return partial(impl.compose, f)


def conjugate_by[F, G, H, U, V](
    impl1: ImplementsCompose[U, H, F],
    u: U,
    impl2: ImplementsCompose[G, V, H],
    v: V,
    /,
) -> Callable[[G], F]:
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


def smultiply_by[K, V](
    impl: ImplementsSmul[K, V], a: K, /
) -> Callable[[V], V]:
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


def exponentiate_by[A, K](
    impl: ImplementsPower[A, K], a: K, /
) -> Callable[[A], A]:
    """Make exponentiation map."""

    def exp_a(b: A) -> A:
        c = impl.power(b, a)
        return c

    return exp_a


def mexponentiate_by[A](
    impl: ImplementsMPower[A], p: int, /
) -> Callable[[A], A]:
    """Make exponentiation map."""

    def mexp_p(a: A) -> A:
        c = impl.mpower(a, p)
        return c

    return mexp_p


def identity[X](x: X, /) -> X:
    """Identity map."""
    return x


def lapp[T, V, W, K](
    impl: ImplementsOperatorSpace[T, V, W, K], a: T, v: V, /
) -> W:
    """Apply linear map to vector."""
    return impl.app(a, v)


def make_linear_operator[T, V, W, K](
    impl: ImplementsOperatorSpace[T, V, W, K], a: T, /
) -> Callable[[V], W]:
    """Make linear map."""

    def op(v: V) -> W:
        return impl.app(a, v)

    return op


def make_form[T, V, W, K](
    impl: ImplementsOperatorSpace[T, V, W, K], a: T
) -> Callable[[W, V], K]:
    """Make bilinear or sesquilinear form."""

    def b(w: W, v: V) -> K:
        return impl.codom.innerp(w, impl.app(a, v))

    return b


def normalize[V, K](impl: ImplementsNormedSpace[V, K], v: V) -> V:
    """Normalize vector in normed space."""
    return impl.sdiv(impl.norm(v), v)


def sum[V, K](
    impl: ImplementsComplexVectorSpace[V, K],
    vs: Iterable[V],
    initializer: Optional[V] = None,
) -> V:
    """Sum a collection of elements of a vector space."""
    if initializer is None:
        initializer = impl.zero()
    return reduce(impl.add, vs, initializer)


def product[A, K](
    impl: ImplementsStarAlgebraWithCalculus[A, K],
    vs: Iterable[A],
    initializer: Optional[A] = None,
) -> A:
    """Multiply a collection of algebra elements."""
    if initializer is None:
        initializer = impl.unit()
    return reduce(impl.mul, vs, initializer)


def linear_combination[V, K](
    impl: ImplementsComplexVectorSpace[V, K], cs: Iterable[K], vs: Iterable[V]
) -> V:
    """Form linear combination of elements of a vector space."""
    cvs = map(impl.smul, cs, vs)
    return sum(impl, cvs)
