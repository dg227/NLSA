import numpy as np
from more_itertools import take

def phi_basis():
    j = 0
    while True:
        if j == 0:
            phi = lambda x: x
        elif j % 2 == 1:
            k = (j + 1) / 2
            phi = lambda x: np.sqrt(2) * np.sin(k * x) 
        else: 
            k = j / 2
            phi = lambda x: np.sqrt(2) * np.cos(k * x) 

        yield phi
        j = j + 1


def triple_prod(f, g, h):
    def fgh(x):
        return f(x) * g(x) * h(x)
    return fgh


l_max = 17
phis = take(l_max, phi_basis())
c = np.empty((l_max,l_max,l_max))

for i in range(0, l_max):
    for j in range(0, l_max):
        for k in range(0, l_max):
            f = triple_prod(phis[i], phis[j], phis[k])
            c[i, j, k] = f(0)
            print(c[i, j, k])

"""
ALTERNATIVE APPROACH 

def structure_consts(phis, integrate):
    def c_fun(i, j, k): 
        f = triple_prod(phis[i], phis[j], phis[k])
        c = integrate(f) 
        return c
    return c_fun

c_fun = structure_consts(phis, quad)


for i in range(0, l_max):
    for j in range(0, l_max):
        for k in range(0, l_max):
            c[i, j, k] = c_fun(i, j, k)

"""


