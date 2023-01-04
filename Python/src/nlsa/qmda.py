"""Provides functions for quantum mechanical data assimilation (QMDA).

"""
from dataclasses import dataclass
from nlsa.abstract_algebra import ImplementsCompose, ImplementsPureState
from typing import Callable, Protocol, TypeVar, runtime_checkable

A = TypeVar('A')
A_contra = TypeVar('A_contra', contravariant=True)
X = TypeVar('X')
Y = TypeVar('Y')
K = TypeVar('K', float, complex)


@dataclass
class Qmda:
    """Class for keeping track of QMDA parameters."""

    nTO: int = 1
    """Number of timesteps between observations."""

    nTF: int = 1
    """Number of forecast timesteps."""

    nL: int = 1
    """Number of basis functions."""

    shape_fun: str = 'bump'
    """Kernel shape function."""

    epsilon: float = 1
    """Kernel bandwidth scaling."""

    ifVB: bool = True
    """Use variable-bandwidth kernel."""

    ifSqrtm: bool = False
    """Perform analysis step (Bayesian update) using matrix square root of
    feature map."""

    def tag(self) -> str:
        s = f'nTO{self.nTO}' \
          + f'_nTF{self.nTF}' \
          + f'_nL{self.nL}' \
          + '_' + self.shape_fun \
          + f'_eps{self.epsilon:.2f}'
        if self.ifVB:
            s += '_vb'
        if self.ifSqrtm:
            s += '_sqrtm'
        return s


def make_da(forecast: Callable[[X], X], analysis: Callable[[Y, X], X])\
        -> Callable[[Y, X], X]:
    """Build dataa assimilation (DA) state update from forecast and analysis
    functions.

    :forecast: Forecast map.
    :analysis: Analysis map.
    :returns: Combined forecast-analysis map.

    """
    def da(y: Y, x: X) -> X:
        x_prior = forecast(x)
        x_posterior = analysis(y, x_prior)
        return x_posterior
    return da


def make_pure_state_prediction(t: ImplementsPureState[Y, A], a: A) \
        -> Callable[[Y], K]:
    """Build prediction function for observable from pure state.

    :t: Object implementing the ImplementsPureState protocol.
    :a: Quantum mechanical observable.
    :returns: Prediction function.

    """
    def f(y: Y) -> K:
        return t.pure_state(y)(a)
    return f


@runtime_checkable
class ImplementsComposeAndPureState(ImplementsCompose[Y, A_contra, Y],
                                    ImplementsPureState[Y, A_contra],
                                    Protocol[Y, A_contra]):
    pass


def pure_state_prediction(t: ImplementsComposeAndPureState[Y, A],
                          a: A, u: A, xi: Y) -> K:
    """Quantum mechanical Koopman prediction from pure state.
    :t: Object implementing the ImplementsPureState and ImplementsMatmul
    protocols.
    :a: Quantum mechanical observable.
    :u: Transfer operator.
    :xi: Initial state vector
    :returns: Prediction of time-evolved observable.


    """
    return t.pure_state(t.compose(u, xi))(a)
