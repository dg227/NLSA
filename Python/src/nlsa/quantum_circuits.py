"""Implements quantum circuits for simulation of Koopman evolution of
observables.

"""
import numpy as np
import scipy.linalg as la
from nptyping import Complex, Double, Int, NDArray, Shape
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.extensions import UnitaryGate
from typing import Literal, Tuple, TypeVar

# We use Literal instead of Shape because for some reason mypy is giving
# errors.
X = NDArray[Literal['N'], Double]
Ints = NDArray[Literal['N'], Int]
V = TypeVar('V', NDArray[Shape['N'], Double], NDArray[Shape['N'], Complex])
M = TypeVar('M',
            NDArray[Shape['N, N'], Double],
            NDArray[Shape['N, N'], Complex])


def koopman_circuit(q: int, v: M, u: M, xi: V) -> QuantumCircuit:
    """Build Qiskit circuit that implements Koopman evolution of an observable.

    :q: Number of qubits.
    :v: Unitary matrix that contains the eigenvectors of the observable.
    :u: Koopman operator matrix.
    :xi: Initial state vector.
    :returns: Qiskit circuit of q qubits that implements the corresponding
    Koopman evolution and measurement.

    """
    extra_dims = 2 ** q - xi.size
    id_pad = np.identity(extra_dims)
    zero_pad = np.zeros(extra_dims)

    state_vec = np.concatenate((xi, zero_pad))
    v_gate = UnitaryGate(la.block_diag(v.conj().T, id_pad),
                         label='Eigenbasis rotation')
    u_gate = UnitaryGate(la.block_diag(u, id_pad), label='Koopman')

    circ = QuantumCircuit(q)
    circ.initialize(state_vec, circ.qubits)
    circ.append(u_gate, circ.qubits)
    circ.append(v_gate, circ.qubits)
    circ.measure_all()
    return circ


def ensemble_measurement(circ: QuantumCircuit, shots: int = 512, backend=None) \
        -> Tuple[Ints, X]:
    """Ensemble measurement of Qiskit circuit."""

    if backend is None:
        backend = Aer.get_backend('aer_simulator')
    job = backend.run(transpile(circ, backend), shots=shots)
    counts = job.result().get_counts(circ)
    idx: Ints = np.array([int(ket, 2) for ket in sorted(counts.keys())])
    density: X = np.array([counts[ket]
                           for ket in sorted(counts.keys())]) / shots
    return idx, density
