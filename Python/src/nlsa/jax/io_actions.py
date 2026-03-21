"""Provide wrappers to perform I/O actions for JAX arrays."""

import jax.numpy as jnp
from collections.abc import Callable
from functools import wraps
from jax import Array
from nlsa.io_actions import IO
from pathlib import Path
from typing import Literal, Optional


def npyit[**P](
    f: Callable[P, Array],
    io: IO,
    mode: Literal["calc", "calcsave", "read"] = "calc",
    fname: str = "Untitled",
    callback: Optional[Callable[[Array], Array]] = None,
) -> Callable[P, Array]:
    """Wrap computation to perform saving to/reading from npy file."""

    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> Array:
        match mode:
            case "calc":
                y = f(*args, **kwargs)
            case "calcsave":
                y = f(*args, **kwargs)
                pth: Path = io.cwd / ".".join((fname, "npy"))
                pth.parent.mkdir(parents=True, exist_ok=True)
                jnp.save(pth, y)
                print(f"Data saved at {pth}")
            case "read":
                pth: Path = io.cwd / ".".join((fname, "npy"))
                if callback is not None:
                    y = callback(jnp.load(pth))
                else:
                    y = jnp.load(pth)
                assert isinstance(y, Array)
                print(f"Data read from {pth}")
        return y

    return f_wrapped
