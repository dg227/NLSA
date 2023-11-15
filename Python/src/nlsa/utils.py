from typing import Callable, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".

    Copied from https://stackoverflow.com/questions/1015307
    /python-bind-an-unbound-method#comment8431145_1015405

    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def swap_args(f: Callable[[X, Y], Z], /) -> Callable[[Y, X], Z]:
    """Swap arguments of bivariate function."""
    def g(y: Y, x: X, /) -> Z:
        z: Z = f(x, y)
        return z
    return g
