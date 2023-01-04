import numpy as np
from nlsa.quantum_circuits import koopman_circuit, ensemble_measurement

def test_predict_with_circuit():
    xi =  np.ones(5)
    xi = xi / np.linalg.norm(xi)
    u = np.identity(5)
    c = np.identity(5)
    circ = koopman_circuit(4, c, u, xi)
    i, x = ensemble_measurement(circ)
