import nlsa.matrix_algebra as mat
import numpy as np

from nlsa.qmda import Qmda, make_pure_state_prediction

def test_Qmda():
    q = Qmda(nTO=1, nTF=10, nL=10, shape_fun="bump", epsilon=1)
    print(q)
    print(q.tag())

def test_make_pure_state_prediction():
    a = np.identity(3)
    xi = np.array([1, 0, 0])
    f = make_pure_state_prediction(mat, a)
    assert(f(xi) == 1)

    
