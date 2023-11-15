import jax.numpy as jnp
from nlsa.jax.fourier_t2 import flip_conj

if __name__ == '__main__':
    u = jnp.array([[1, 2, 3], [2, 3, 4]])
    v = flip_conj(u)
    w = flip_conj(v)
    print(u)
    print(v)
    print(w)
