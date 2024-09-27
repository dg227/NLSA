"""Provides functions and protocols for abstract algebraic structures.

"""
from functools import partial, reduce
from nlsa.utils import swap_args
from typing import Callable, Generic, Iterable, Optional, Protocol, \
        TypeGuard, TypeVar, TypeVarTuple, runtime_checkable

F = TypeVar('F')
G = TypeVar('G')
H = TypeVar('H')
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
S = TypeVar('S')
K = TypeVar('K')
R = TypeVar('R')
L = TypeVar('L')
V_cov = TypeVar('V_cov', covariant=True)
T_cov = TypeVar('T_cov', covariant=True)
T_con = TypeVar('T_con', contravariant=True)
K_con = TypeVar('K_con', contravariant=True)
L_con = TypeVar('L_con', contravariant=True)
R_con = TypeVar('R_con', contravariant=True)
S_con = TypeVar('S_con', contravariant=True)
S_cov = TypeVar('S_cov', covariant=True)
U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')
X = TypeVar('X')


@runtime_checkable
class ImplementsZero(Protocol[T]):
    """Implement additive zero."""
    zero: Callable[[], T]


@runtime_checkable
class ImplementsAdd(Protocol[T]):
    """Implement addition."""
    add: Callable[[T, T], T]
    # def add(self, a: T, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsSub(Protocol[T]):
    """Implement subtraction."""
    sub: Callable[[T, T], T]
    # def sub(self, a: T, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsNeg(Protocol[T]):
    """Implement additive negation."""
    neg: Callable[[T], T]
    # def neg(self, a: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsSmul(Protocol[K, T]):
    """Implement scalar multiplication."""
    smul: Callable[[K, T], T]
    # def smul(self, a: K_con, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsMul(Protocol[T]):
    """Implement algebraic multiplication."""
    mul: Callable[[T, T], T]
    # def mul(self, a: T, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsUnit(Protocol[T]):
    """Implement algebraic unit."""
    unit: Callable[[], T]
    # def unit(self) -> T_cov:
    #     ...


@runtime_checkable
class ImplementsDiv(Protocol[T]):
    """Implement algebraic division."""
    div: Callable[[T, T], T]
    # def div(self, a: T, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsInv(Protocol[T]):
    """Implement algebraic inversion."""
    inv: Callable[[T], T]
    # def inv(self, a: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsLmul(Protocol[K, T]):
    """Implement left module multiplication."""
    lmul: Callable[[K, T], T]
    # def lmul(self, a: L_con, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsRmul(Protocol[T, K]):
    """Implement right module multiplication."""
    rmul: Callable[[T, K], T]
    # def rmul(self, b: T, a: R_con, /) -> T:
    #     ...


@runtime_checkable
class ImplementsPower(Protocol[T, S]):
    """Implement algebraic power (exponentiation)."""
    power: Callable[[T, S], T]


@runtime_checkable
class ImplementsSqrt(Protocol[T]):
    """Implement square root."""
    sqrt: Callable[[T], T]
    # def sqrt(self, a: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsExp(Protocol[T]):
    """Implement exponentiation."""
    exp: Callable[[T], T]
    # def exp(self, a: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsStar(Protocol[T]):
    """Implement algebraic adjunction."""
    star: Callable[[T], T]
    # def star(self, a: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsLdiv(Protocol[L, T]):
    """Implement left module division."""
    ldiv: Callable[[L, T], T]
    # def ldiv(self, a: K_con, b: T, /) -> T:
    #     ...


@runtime_checkable
class ImplementsRdiv(Protocol[T, R]):
    """Implement right module division."""
    rdiv: Callable[[T, R], T]
    # def rdiv(self, a: T, b: K_con, /) -> T:
    #     ...


@runtime_checkable
class ImplementsNorm(Protocol[T, K]):
    """Implement norm."""
    norm: Callable[[T], K]


@runtime_checkable
class ImplementsInnerp(Protocol[T, K]):
    """Implement inner product."""
    innerp: Callable[[T, T], K]
    # def innerp(self, a: T_con, b: T_con, /) -> S_cov:
    #     ...


@runtime_checkable
class ImplementsScalarField(ImplementsZero[K], ImplementsAdd[K],
                            ImplementsSub[K], ImplementsNeg[K],
                            ImplementsMul[K], ImplementsUnit[K],
                            ImplementsDiv[K], ImplementsInv[K],
                            ImplementsStar[K], Protocol[K]):
    """Implement scalar field operations."""
    pass


@runtime_checkable
class ImplementsLScalarField(ImplementsScalarField[K], ImplementsLmul[K, K],
                             ImplementsLdiv[K, K], Protocol[K]):
    """Implement scalar field operations with specialized left
    multiplication/division operations (e.g., for broadcasting).
    """
    pass


@runtime_checkable
class ImplementsRootScalarField(ImplementsScalarField[K], ImplementsSqrt[K],
                                Protocol[K]):
    """Implement scalar field operations with square root."""
    pass


@runtime_checkable
class ImplementsPowerScalarField(ImplementsScalarField[K],
                                 ImplementsPower[K, K], Protocol[K]):
    """Implement scalar field operations with exponentiation."""
    pass


@runtime_checkable
class ImplementsRootLScalarField(ImplementsRootScalarField[K],
                                 ImplementsLmul[K, K], ImplementsLdiv[K, K],
                                 Protocol[K]):
    """Implement scalar field operations with specialized left
    multiplication/division (e.g., for broadcasting), square root, and
    exponentiation operations.
    """
    pass


@runtime_checkable
class ImplementsVectorSpace(ImplementsZero[T], ImplementsAdd[T],
                            ImplementsSub[T], ImplementsNeg[T],
                            ImplementsSmul[K, T], Protocol[T, K]):
    """Implement vector space operations."""
    scl: ImplementsScalarField[K]

    def sdiv(self, k: K, v: T, /) -> T:
        return self.smul(self.scl.inv(k), v)


@runtime_checkable
class ImplementsBanachSpace(ImplementsVectorSpace[T, K],
                            ImplementsNorm[T, K],
                            Protocol[T, K]):
    """Implement Banach space operations."""
    pass


@runtime_checkable
class ImplementsHilbertSpace(ImplementsBanachSpace[T, K],
                             ImplementsInnerp[T, K],
                             Protocol[T, K]):
    """Implement Hilbert space operations."""
    pass


@runtime_checkable
class ImplementsAlgebra(ImplementsVectorSpace[T, K], ImplementsMul[T],
                        Protocol[T, K]):
    """Implement algebra operations."""
    pass


@runtime_checkable
class ImplementsStarAlgebra(ImplementsAlgebra[T, K], ImplementsStar[T],
                            Protocol[T, K]):
    """Implement star algebra operations."""
    pass


@runtime_checkable
class ImplementsUnitalAlgebra(ImplementsAlgebra[T, K], ImplementsDiv[T],
                              ImplementsInv[T], ImplementsUnit[T],
                              Protocol[T, K]):
    """Implement unital algebra operations."""
    pass


@runtime_checkable
class ImplementsUnitalStarAlgebra(ImplementsUnitalAlgebra[T, K],
                                  ImplementsStar[T], Protocol[T, K]):
    """Implement unital star algebra operations."""
    pass


@runtime_checkable
class ImplementsBanachAlgebra(ImplementsAlgebra[T, K], ImplementsNorm[T, K],
                              Protocol[T, K]):
    """Implement Banach algebra operations."""
    pass


@runtime_checkable
class ImplementsOperatorSpace(ImplementsVectorSpace[T, K], ImplementsStar[T],
                              Protocol[T, V, W, K]):
    """Implement vector space operations for linear maps between Hilbert spaces.
    """
    hilb1: ImplementsHilbertSpace[V, K]
    hilb2: ImplementsHilbertSpace[W, K]
    app: Callable[[T, V], W]


@runtime_checkable
class ImplementsOperatorAlgebra(ImplementsUnitalStarAlgebra[T, K],
                                Protocol[T, V, K]):
    """Implement algebra operations on a Hilbert space."""
    hilb: ImplementsHilbertSpace[V, K]
    app: Callable[[T, V], V]


@runtime_checkable
class ImplementsPowerAlgebra(ImplementsAlgebra[T, K], ImplementsPower[T, K],
                             Protocol[T, K]):
    """Implement algebra operations with exponentiation."""
    pass


@runtime_checkable
class ImplementsRootAlgebra(ImplementsAlgebra[T, K], ImplementsSqrt[T],
                            Protocol[T, K]):
    """Implement algebra operations with square root."""
    pass
    # scl: ImplementsRootScalarField[K]


@runtime_checkable
class ImplementsLModule(ImplementsVectorSpace[T, K],
                        ImplementsLmul[L, T],
                        ImplementsLdiv[L, T],
                        Protocol[T, K, L]):
    """Implement left module operations."""
    pass


@runtime_checkable
class ImplementsAlgebraLModule(ImplementsLModule[T, K, L],
                               ImplementsAlgebra[T, K],
                               Protocol[T, K, L]):
    """Implement algebra and left module operations."""
    pass


@runtime_checkable
class ImplementsUnitalAlgebraLModule(ImplementsLModule[T, K, L],
                                     ImplementsUnitalAlgebra[T, K],
                                     Protocol[T, K, L]):
    """Implement unital algebra and left module operations."""
    pass


@runtime_checkable
class ImplementsStarAlgebraLModule(ImplementsLModule[T, K, L],
                                   ImplementsStarAlgebra[T, K],
                                   Protocol[T, K, L]):
    """Implement star algebra and left module operations."""
    pass


@runtime_checkable
class ImplementsRootAlgebraLModule(ImplementsLModule[T, K, L],
                                   ImplementsRootAlgebra[T, K],
                                   Protocol[T, K, L]):
    """Implement root algebra and left module operations."""
    pass


@runtime_checkable
class ImplementsRModule(ImplementsVectorSpace[T, K],
                        ImplementsRmul[T, R],
                        ImplementsRdiv[T, R],
                        Protocol[T, K, R]):
    """Implement right module operations."""
    pass


@runtime_checkable
class ImplementsAlgebraRModule(ImplementsRModule[T, K, R],
                               ImplementsAlgebra[T, K],
                               Protocol[T, K, R]):
    """Implement algebra and right module operations."""
    pass


@runtime_checkable
class ImplementsUnitalAlgebraRModule(ImplementsRModule[T, K, R],
                                     ImplementsUnitalAlgebra[T, K],
                                     Protocol[T, K, R]):
    """Implement unital algebra and right module operations."""
    pass


@runtime_checkable
class ImplementsStarAlgebraRModule(ImplementsRModule[T, K, R],
                                   ImplementsStarAlgebra[T, K],
                                   Protocol[T, K, R]):
    """Implement star algebra and right module operations."""
    pass


@runtime_checkable
class ImplementsRootAlgebraRModule(ImplementsRModule[T, K, R],
                                   ImplementsRootAlgebra[T, K],
                                   Protocol[T, K, R]):
    """Implement root algebra and right module operations."""
    pass


@runtime_checkable
class ImplementsLRModule(ImplementsLModule[T, K, L],
                         ImplementsRModule[T, K, R],
                         Protocol[T, K, L, R]):
    """Implement bimodulde operations."""
    pass


@runtime_checkable
class ImplementsAlgebraLRModule(ImplementsLRModule[T, K, L, R],
                                ImplementsAlgebra[T, K],
                                Protocol[T, K, L, R]):
    """Implement algebra and bimodule module operations."""
    pass


@runtime_checkable
class ImplementsUnitalAlgebraLRModule(ImplementsLRModule[T, K, L, R],
                                      ImplementsUnitalAlgebra[T, K],
                                      Protocol[T, K, L, R]):
    """Implement unital algebra and left-right module operations."""
    ...


@runtime_checkable
class ImplementsStarAlgebraLRModule(ImplementsLRModule[T, K, L, R],
                                    ImplementsStarAlgebra[T, K],
                                    Protocol[T, K, L, R]):
    """Implement star algebra and left-right module operations."""
    ...


@runtime_checkable
class ImplementsUnitalStarAlgebraLRModule(
                                    ImplementsLRModule[T, K, L, R],
                                    ImplementsUnitalStarAlgebra[T, K],
                                    Protocol[T, K, L, R]):
    """Implement unital star algebra and left-right module operations."""
    ...


@runtime_checkable
class ImplementsRootAlgebraLRModule(ImplementsLRModule[T, K, L, R],
                                    ImplementsRootAlgebra[T, K],
                                    Protocol[T, K, L, R]):
    """Implement root algebra and left-right module operations."""
    ...


@runtime_checkable
class ImplementsIncl(Protocol[S_cov, K_con, T_cov]):
    """Implement function inclusion."""
    def incl(self, f: Callable[[S_cov], K_con], /) -> T_cov:
        ...


@runtime_checkable
class ImplementsIntegrate(Protocol[T_con, S_cov]):
    """Implement integration."""
    def integrate(self, a: T_con, /) -> S_cov:
        ...


@runtime_checkable
class ImplementsMeasurableFnAlgebra(ImplementsRootAlgebra[T, K],
                                    ImplementsIncl[S_cov, K, T],
                                    Protocol[S_cov, T, K]):
    """Implement operations on space of equivalence classes of functions."""
    pass


@runtime_checkable
class ImplementsMeasureFnAlgebra(ImplementsMeasurableFnAlgebra[S_cov, T, K],
                                 ImplementsIntegrate[T, K],
                                 Protocol[S_cov, T, K]):
    """Implement operations on measure function space."""
    pass


@runtime_checkable
class ImplementsMeasurableUnitalFnAlgebra(ImplementsRootAlgebra[T, K],
                                          ImplementsUnitalAlgebra[T, K],
                                          ImplementsIncl[S_cov, K, T],
                                          Protocol[S_cov, T, K]):
    """Implement operations on space of equivalence classes of functions with
    unit.
    """
    pass


@runtime_checkable
class ImplementsMeasureUnitalFnAlgebra(ImplementsMeasurableUnitalFnAlgebra[S_cov,
                                                                           T, K],
                                       ImplementsIntegrate[T, K],
                                       Protocol[S_cov, T, K]):
    """Implement operations on measure function spaces with unit."""
    pass


@runtime_checkable
class ImplementsCompose(Protocol[G, F, H]):
    """Implement composition."""
    compose: Callable[[F, G], H]


ScalarField = ImplementsScalarField[K]\
        | ImplementsLScalarField[K]\
        | ImplementsRootScalarField[K]\
        | ImplementsRootLScalarField[K]


def implements_scalar_field(scl: ScalarField[K])\
        -> TypeGuard[ImplementsScalarField[K]]:
    return True


def implements_lscalar_field(scl: ScalarField[K])\
        -> TypeGuard[ImplementsLScalarField[K]]:
    return isinstance(scl, ImplementsLScalarField)


def implements_root_scalar_field(scl: ScalarField[K])\
        -> TypeGuard[ImplementsRootScalarField[K]]:
    return isinstance(scl, ImplementsRootScalarField)


def implements_power_scalar_field(scl: ScalarField[K])\
        -> TypeGuard[ImplementsPowerScalarField[K]]:
    return isinstance(scl, ImplementsPowerScalarField)


class FromScalarField(ImplementsUnitalStarAlgebraLRModule[K, K, K, K],
                      ImplementsAlgebraLRModule[K, K, K, K],
                      Generic[K]):
    """Lift scalar field to bimodule over itself."""
    def __init__(self, scl: ScalarField[K]):

        self.zero = scl.zero
        self.add = scl.add
        self.sub = scl.sub
        self.neg = scl.neg
        self.mul = scl.mul
        self.unit = scl.unit
        self.div = scl.div
        self.inv = scl.inv
        self.star = scl.star
        self.scl = scl
        self.smul = scl.mul
        self.rmul = scl.mul
        self.rdiv = scl.div
        # self.lmul = make_lmul(scl)
        # self.ldiv = make_ldiv(scl)

        if implements_lscalar_field(scl):
            self.ldiv = scl.ldiv
            self.lmul = scl.lmul
        elif implements_scalar_field(scl):
            self.ldiv = swap_args(scl.div)
            self.lmul = swap_args(scl.mul)

        if implements_power_scalar_field(scl):
            self.power = scl.power

        if implements_root_scalar_field(scl):
            self.sqrt = scl.sqrt

    # def ldiv(self, a: K, b: K, /) -> K:
    #     return self._ldiv(a, b)

    # def lmul(self, a: K, b: K, /) -> K:
    #     return self._lmul(a, b)


def compose_by(impl: ImplementsCompose[G, F, H], g: G) -> Callable[[F], H]:
    """Make composition map."""
    def u(f: F) -> H:
        h = impl.compose(f, g)
        return h
    return u


def precompose_by(impl: ImplementsCompose[G, F, H], f: F) -> Callable[[G], H]:
    """Make pre-composition map"""
    return partial(impl.compose, f)


def conjugate_by(impl1: ImplementsCompose[H, U, F], u: U,
                 impl2: ImplementsCompose[V, G, H], v: V) -> Callable[[G], F]:
    """Conjugation map."""
    def c(g: G) -> F:
        h = impl2.compose(g, v)
        f = impl1.compose(u, h)
        return f
    return c


def multiply_by(impl: ImplementsMul[T], a: T) -> Callable[[T], T]:
    """Make multiplication operator."""
    def m(b: T) -> T:
        c = impl.mul(a, b)
        return c
    return m


def divide_by(impl: ImplementsDiv[T], a: T) -> Callable[[T], T]:
    """Make division operator."""
    def m(b: T) -> T:
        c = impl.div(b, a)
        return c
    return m


def smultiply_by(impl: ImplementsSmul[S, T], a: S) -> Callable[[T], T]:
    """Make scalar multiplication operator."""
    def m(b: T) -> T:
        c = impl.smul(a, b)
        return c
    return m


def ldivide_by(impl: ImplementsLdiv[K, T], a: K) -> Callable[[T], T]:
    """Make left division operator."""
    def m(b: T) -> T:
        c = impl.ldiv(a, b)
        return c
    return m


def exponentiate_by(impl: ImplementsPower[T, S], a: S)\
        -> Callable[[T], T]:
    """Make exponentiation map."""
    def exp_a(b: T) -> T:
        c = impl.power(b, a)
        return c
    return exp_a


def identity(x: T) -> T:
    """Identity map."""
    return x


def lapp(impl: ImplementsOperatorSpace[T, V, W, K], a: T, v: V, /) -> W:
    """Apply linear map to vector."""
    return impl.app(a, v)

# TODO: Consider merging make_linear_map and make_linear_operator into a single
# overloaded function.


def make_linear_map(impl: ImplementsOperatorSpace[T, V, W, K], a: T, /)\
        -> Callable[[V], W]:
    """Make linear map."""
    def op(v: V) -> W:
        return impl.app(a, v)
    return op


def make_linear_operator(impl: ImplementsOperatorAlgebra[T, V, K], a: T, /)\
        -> Callable[[V], V]:
    """Make linear operator."""
    def op(v: V) -> V:
        return impl.app(a, v)
    return op


def make_bilinear_form(impl: ImplementsOperatorAlgebra[T, V, K], a: T)\
        -> Callable[[V, V], K]:
    """Make bilinear form."""
    def b(u: V, v: V) -> K:
        return impl.hilb.innerp(u, impl.app(a, v))
    return b


def normalize(impl: ImplementsBanachSpace[V, K], v: V) -> V:
    """Normalize vector in Banach space."""
    return impl.sdiv(impl.norm(v), v)


def make_vector_state(impl: ImplementsOperatorAlgebra[T, V, K], v: V, /)\
        -> Callable[[T], K]:
    """Make vector state of operator algebra."""
    def phi(a: T) -> K:
        return impl.hilb.innerp(v, impl.app(a, v))
    return phi


def gelfand(impl: ImplementsAlgebra[T, K], a: T, /)\
        -> Callable[[Callable[[T], K]], K]:
    """Compute Gelfand transform of algebra element."""
    def a_hat(phi: Callable[[T], K]) -> K:
        return phi(a)
    return a_hat


def make_qeval(impl: ImplementsOperatorAlgebra[T, V, K],
               feat: Callable[[X], V], /)\
        -> Callable[[X], Callable[[T], K]]:
    """Make quantum pointwise evaluation functional from feature map."""
    def eval_at(x: X) -> Callable[[T], K]:
        phi = make_vector_state(impl, feat(x))

        def evalx(a: T) -> K:
            return phi(a)

        return evalx
    return eval_at


def sum(impl: ImplementsVectorSpace[V, K], vs: Iterable[V],
        initializer: Optional[V] = None) -> V:
    """Sum a collection of elements of a vector space."""
    if initializer is None:
        initializer = impl.zero()
    return reduce(impl.add, vs, initializer)


def product(impl: ImplementsUnitalAlgebra[V, K], vs: Iterable[V],
            initializer: Optional[V] = None) -> V:
    """Multiply a collection of elements of a unital algebra."""
    if initializer is None:
        initializer = impl.unit()
    return reduce(impl.mul, vs, initializer)


def linear_combination(impl: ImplementsVectorSpace[V, K], cs: Iterable[K],
                       vs: Iterable[V]) -> V:
    """Form linear combination of elements of a vector space."""
    cvs = map(impl.smul, cs, vs)
    return sum(impl, cvs)
