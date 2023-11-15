import jax.numpy as jnp
from jax import Array, jit, vmap
from nlsa.jax.dynamics import flow, make_stepanoff_generator_fourier,\
        make_stepanoff_vector_field

R2 = Array
T2 = Array
X = Array

if __name__ == '__main__':
    # alpha = jnp.sqrt(30)
    alpha = 0.3

    v = make_stepanoff_vector_field(alpha)
    phi = flow(v, 1.0)
    vphi = jit(vmap(phi))
    x = jnp.array([[0., 0.], [0., 0.]])
    y = vphi(x)
    print(y)

    k1 = 1
    n1 = 2 * k1 + 1
    k2 = 2
    n2 = 2 * k2 + 1
    g = make_stepanoff_generator_fourier(alpha, (k1, k2), hermitian=True)
    a = jnp.zeros((n1, n2))
    a = a.at[k1, k2].set(1.)
    f1 = a.reshape(n1 * n2)
    f2 = g(f1)
    print(f1)
    print(f2)
