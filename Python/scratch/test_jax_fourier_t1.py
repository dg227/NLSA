import jax.numpy as jnp
from nlsa.jax.fourier_t1 import make_fourier_synthesis

if __name__ == '__main__':
    ks = jnp.arange(-1, 2)
    synth = make_fourier_synthesis(ks)
    v = jnp.ones(3)
    f = synth(v)
    x = jnp.array([0.])
    print(f(x))

