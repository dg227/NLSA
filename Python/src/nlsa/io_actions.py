"""Provide wrappers to perform I/O actions."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
from collections.abc import Callable, Mapping
from functools import wraps
from dataclasses import dataclass, field
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, DTypeLike
from pandas import DataFrame
from pathlib import Path
from typeguard import check_type
from typing import Literal, Optional, Self

type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


@dataclass(frozen=True)
class IO:
    """Keep record of IO actions."""
    root: Path = Path.cwd()
    paths: list[Path] = field(default_factory=list[Path])

    def __itruediv__(self, p: str | Path) -> Self:
        """Append relative path to the path history."""
        match p:
            case str():
                pth = Path(p)
            case Path():
                pth = p
        if self.paths:
            self.paths.append(self.paths[-1] / pth)
        else:
            self.paths.append(pth)
        return self

    def __imatmul__(self, p: str | Path) -> Self:
        """Append full path to the path history."""
        match p:
            case str():
                pth = Path(p)
            case Path():
                pth = p
        self.paths.append(pth)
        return self

    @property
    def cwd(self) -> Path:
        if self.paths:
            return self.root / self.paths[-1]
        else:
            return self.root


def pickleit[T, **P](f: Callable[P, T], io: IO,
                     mode: Literal['calc', 'calcsave', 'read'] = "calc",
                     fname: str = "Untitled",
                     cls: Optional[type[T]] = None) -> Callable[P, T]:
    ...
    """Wrap computation to perform pickling/unpickling."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        match mode:
            case "calc":
                y = f(*args, **kwargs)
            case "calcsave":
                y = f(*args, **kwargs)
                pth: Path = io.cwd / ".".join((fname, "pkl"))
                pth.parent.mkdir(parents=True, exist_ok=True)
                with open(pth, 'wb') as file:
                    pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Data saved at {pth}")
            case "read":
                assert cls is not None
                pth: Path = io.cwd / ".".join((fname, "pkl"))
                with open(pth, 'rb') as file:
                    y = pickle.load(file)
                check_type(y, cls)
                print(f"Data read from {pth}")
        return y
    return f_wrapped


def h5it[T: Mapping[str, object], **P](
            f: Callable[P, T], io: IO,
            mode: Literal['calc', 'calcsave', 'read'] = "calc",
            fname: str = "Untitled", dtype: Optional[DTypeLike] = None,
            cls: Optional[type[T]] = None,
            callback: Optional[Callable[[dict[str, ArrayLike]], T]] = None) \
        -> Callable[P, T]:
    ...
    """Wrap computation to perform saving to/reading from HDF5 file."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        match mode:
            case "calc":
                y = f(*args, **kwargs)
            case "calcsave":
                y = f(*args, **kwargs)
                pth: Path = io.cwd / ".".join((fname, "h5"))
                pth.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(pth, "w") as file:
                    for key, value in y.items():
                        try:
                            file.create_dataset(key,
                                                data=np.asarray(value,
                                                                dtype=dtype))
                        except TypeError as exc:
                            raise TypeError("Dict value must be ArrayLike") \
                                    from exc
                print(f"Data saved at {pth}")
            case "read":
                assert cls is not None
                assert callback is not None
                pth: Path = io.cwd / ".".join((fname, "h5"))
                with h5py.File(pth, "r") as file:
                    y_dict: dict[str, ArrayLike] = {}
                    for key, value in file.items():
                        assert isinstance(value, h5py.Dataset)
                        y_dict[key] = np.asarray(value)
                    y = callback(y_dict)
                print(f"Data read from {pth}")
        return y
    return f_wrapped


def npyit[T: ArrayLike, **P](
            f: Callable[P, T], io: IO,
            mode: Literal['calc', 'calcsave', 'read'] = "calc",
            fname: str = "Untitled",
            cls: Optional[type[T]] = None,
            callback: Optional[Callable[[ArrayLike], T]] = None) \
        -> Callable[P, T]:
    """Wrap computation to perform saving to/reading from npy file."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        match mode:
            case "calc":
                y = f(*args, **kwargs)
            case "calcsave":
                y = f(*args, **kwargs)
                pth: Path = io.cwd / ".".join((fname, "npy"))
                pth.parent.mkdir(parents=True, exist_ok=True)
                np.save(pth, np.asarray(y))
                print(f"Data saved at {pth}")
            case "read":
                assert cls is not None
                pth: Path = io.cwd / ".".join((fname, "npy"))
                if callback is not None:
                    y = callback(np.load(pth))
                else:
                    y = np.load(pth)
                check_type(y, cls)
                print(f"Data read from {pth}")
        return y
    return f_wrapped


def csvit[**P](
            f: Callable[P, DataFrame], io: IO,
            mode: Literal['calc', 'calcsave', 'read'] = "calc",
            fname: str = "Untitled",
            index: Optional[bool | str] = None,
            callback: Optional[Callable[[DataFrame], DataFrame]] = None) \
        -> Callable[P, DataFrame]:
    """Wrap computation to perform saving to/reading from csv file."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> DataFrame:
        match mode:
            case "calc":
                df = f(*args, **kwargs)
            case "calcsave":
                if index is None:
                    _index = False
                else:
                    assert isinstance(index, bool)
                    _index = index
                df = f(*args, **kwargs)
                pth: Path = io.cwd / ".".join((fname, "csv"))
                pth.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(pth, index=_index)
                print(f"Data saved at {pth}")
            case "read":
                pth: Path = io.cwd / ".".join((fname, "npy"))
                df = pd.read_csv(pth)
                assert isinstance(pd, DataFrame)
                if callback is not None:
                    df = callback(df)
                if index is not None:
                    assert isinstance(index, str)
                    df = df.set_index(index)
                print(f"Data read from {pth}")
        return df
    return f_wrapped


def timeit[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    """Add basic timing wrapper to function."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        y = f(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{f.__name__}: Took {total_time:.4f} seconds")
        return y
    return f_wrapped


def pauseit[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    """Pause execution until key input."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        y = f(*args, **kwargs)
        input("Press any key to continue...")
        return y
    return f_wrapped


def plotit[**P](f: Callable[P, Figure],
                io: IO,
                mode: Optional[Literal['save', 'show', 'saveshow']] = "show",
                fname: str = "Untitled",
                dpi: Literal['figure'] | float = "figure",
                fmt: str = "png") -> Callable[P, Optional[Figure]]:
    """Wrap plotting function to implement saving/showing to screen."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) -> Optional[Figure]:
        if mode is not None:
            fig = f(*args, **kwargs)
            if "save" in mode:
                pth: Path = io.cwd / ".".join((fname, fmt))
                pth.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(pth, bbox_inches="tight", dpi=dpi)
                print(f"Figure saved at {pth}")
            if "show" in mode:
                plt.show(block=False)
                input("Press any key to continue...")
            return fig
    return f_wrapped


def plotem[**P](f: Callable[P, tuple[Figure, F[int, None]]],
                io: IO,
                mode: Optional[Literal['save', 'show', 'saveshow']] = "show",
                block: bool | Literal['user'] = True, fname: str = "Untitled",
                dpi: Literal['figure'] | float = "figure",
                fmt: str = "png") \
        -> Callable[P, tuple[Optional[Figure], F[int, None]]]:
    """Wrap plotting function to implement saving/showing of multiple figs."""
    @wraps(f)
    def f_wrapped(*args: P.args, **kwargs: P.kwargs) \
            -> tuple[Optional[Figure], F[int, None]]:
        if mode is not None:
            fig, g = f(*args, **kwargs)

            def g_wrapped(i: int) -> None:
                g(i)
                if mode is not None and "save" in mode:
                    fname_i = "_".join((fname, str(i)))
                    pth: Path = io.cwd / ".".join((fname_i, fmt))
                    pth.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(pth, bbox_inches="tight", dpi=dpi)
                    print(f"Figure saved at {pth}")
                if mode is not None and "show" in mode:
                    plt.show(block=False)
                    input("Press any key to continue...")
            return fig, g_wrapped
        else:
            def g_wrapped(_: int) -> None:
                pass
            return None, g_wrapped
    return f_wrapped
