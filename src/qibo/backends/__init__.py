import os
from importlib import import_module

import numpy as np

from qibo.backends.abstract import Backend
from qibo.backends.clifford import CliffordBackend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.backends.pytorch import PyTorchBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import log, raise_error

QIBO_NATIVE_BACKENDS = ("numpy", "tensorflow", "pytorch", "qulacs")


class MissingBackend(ValueError):
    """Impossible to locate backend provider package."""


class MetaBackend:
    """Meta-backend class which takes care of loading the qibo backends."""

    @staticmethod
    def load(backend: str, **kwargs) -> Backend:
        """Loads the native qibo backend.

        Args:
            backend (str): Name of the backend to load.
            kwargs (dict): Additional arguments for the qibo backend.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if backend == "numpy":
            return NumpyBackend()
        elif backend == "tensorflow":
            return TensorflowBackend()
        elif backend == "pytorch":
            return PyTorchBackend()
        elif backend == "clifford":
            engine = kwargs.pop("platform", None)
            kwargs["engine"] = engine
            return CliffordBackend(**kwargs)
        elif backend == "qulacs":
            from qibo.backends.qulacs import QulacsBackend

            return QulacsBackend()
        else:
            raise_error(
                ValueError,
                f"Backend {backend} is not available. The native qibo backends are {QIBO_NATIVE_BACKENDS}.",
            )

    def list_available(self) -> dict:
        """Lists all the available native qibo backends."""
        available_backends = {}
        for backend in QIBO_NATIVE_BACKENDS:
            try:
                MetaBackend.load(backend)
                available = True
            except:  # pragma: no cover
                available = False
            available_backends[backend] = available
        return available_backends


class _Global:
    _backend = None
    _transpiler = None

    _dtypes = {"double": "complex128", "single": "complex64"}
    _default_order = [
        {"backend": "qibojit", "platform": "cupy"},
        {"backend": "qibojit", "platform": "numba"},
        {"backend": "tensorflow"},
        {"backend": "numpy"},
        {"backend": "pytorch"},
    ]

    @classmethod
    def backend(cls):  # pragma: no cover
        if cls._backend is not None:
            return cls._backend

        backend = os.environ.get("QIBO_BACKEND")
        if backend:
            # Create backend specified by user
            platform = os.environ.get("QIBO_PLATFORM")
            cls._backend = construct_backend(backend, platform=platform)
        else:
            # Create backend according to default order
            for kwargs in cls._default_order:
                try:
                    cls._backend = construct_backend(**kwargs)
                    break
                except (ModuleNotFoundError, ImportError):
                    pass

        if cls._backend is None:
            raise_error(RuntimeError, "No backends available.")

        log.info(f"Using {cls._backend} backend on {cls._backend.device}")
        return cls._backend

    @classmethod
    def set_backend(cls, backend, **kwargs):  # pragma: no cover
        if (
            cls._backend is None
            or cls._backend.name != backend
            or cls._backend.platform != kwargs.get("platform")
        ):
            cls._backend = construct_backend(backend, **kwargs)
            log.info(f"Using {cls._backend} backend on {cls._backend.device}")
        else:
            log.info(f"Backend {backend} is already loaded.")

    @classmethod
    def get_backend(cls):
        return cls._backend

    @classmethod
    def transpiler(cls):  # pragma: no cover
        from qibo.transpiler.pipeline import Passes

        if cls._transpiler is not None:
            return cls._transpiler

        cls._transpiler = Passes(passes=[])
        return cls._transpiler

    @classmethod
    def set_transpiler(cls, transpiler):  # pragma: no cover
        cls._transpiler = transpiler
        # TODO: check if transpiler is valid on the backend

    @classmethod
    def get_transpiler(cls):  # pragma: no cover
        return cls._transpiler

    @classmethod
    def resolve_global(cls):
        if cls._backend is None:
            cls._backend = cls.backend()
        if cls._transpiler is None:
            # TODO: add default transpiler for hardware backends
            cls._transpiler = cls.transpiler()


class QiboMatrices:
    def __init__(self, dtype="complex128"):
        self.create(dtype)

    def create(self, dtype):
        self.matrices = NumpyMatrices(dtype)
        self.I = self.matrices.I(2)
        self.X = self.matrices.X
        self.Y = self.matrices.Y
        self.Z = self.matrices.Z
        self.SX = self.matrices.SX
        self.H = self.matrices.H
        self.S = self.matrices.S
        self.SDG = self.matrices.SDG
        self.CNOT = self.matrices.CNOT
        self.CY = self.matrices.CY
        self.CZ = self.matrices.CZ
        self.CSX = self.matrices.CSX
        self.CSXDG = self.matrices.CSXDG
        self.SWAP = self.matrices.SWAP
        self.iSWAP = self.matrices.iSWAP
        self.SiSWAP = self.matrices.SiSWAP
        self.SiSWAPDG = self.matrices.SiSWAPDG
        self.FSWAP = self.matrices.FSWAP
        self.ECR = self.matrices.ECR
        self.SYC = self.matrices.SYC
        self.TOFFOLI = self.matrices.TOFFOLI
        self.CCZ = self.matrices.CCZ


matrices = QiboMatrices()


def get_backend():
    return str(_Global.get_backend())


def set_backend(backend, **kwargs):
    _Global.set_backend(backend, **kwargs)


def get_transpiler():
    return str(_Global.get_transpiler())


def set_transpiler(transpiler):
    _Global.set_transpiler(transpiler)


def get_precision():
    return _Global.get_backend().precision


def set_precision(precision):
    _Global.get_backend().set_precision(precision)
    matrices.create(_Global.get_backend().dtype)


def get_device():
    return _Global.get_backend().device


def set_device(device):
    parts = device[1:].split(":")
    if device[0] != "/" or len(parts) < 2 or len(parts) > 3:
        raise_error(
            ValueError,
            "Device name should follow the pattern: /{device type}:{device number}.",
        )
    backend = _Global.get_backend()
    backend.set_device(device)
    log.info(f"Using {backend} backend on {backend.device}")


def get_threads():
    return _Global.get_backend().nthreads


def set_threads(nthreads):
    if not isinstance(nthreads, int):
        raise_error(TypeError, "Number of threads must be integer.")
    if nthreads < 1:
        raise_error(ValueError, "Number of threads must be positive.")
    _Global.get_backend().set_threads(nthreads)


def _check_backend(backend):
    if backend is None:
        return _Global.backend()

    return backend


def list_available_backends(*providers: str) -> dict:
    """Lists all the backends that are available."""
    available_backends = MetaBackend().list_available()
    for backend in providers:
        try:
            module = import_module(backend.replace("-", "_"))
            available = getattr(module, "MetaBackend")().list_available()
        except:
            available = False
        available_backends.update({backend: available})
    return available_backends


def construct_backend(backend, **kwargs) -> Backend:
    """Construct a generic native or non-native qibo backend.

    Args:
        backend (str): Name of the backend to load.
        kwargs (dict): Additional arguments for constructing the backend.
    Returns:
        qibo.backends.abstract.Backend: The loaded backend.
    """
    if backend in QIBO_NATIVE_BACKENDS + ("clifford",):
        return MetaBackend.load(backend, **kwargs)

    provider = backend.replace("-", "_")
    try:
        module = import_module(provider)
        return getattr(module, "MetaBackend").load(**kwargs)
    except ImportError as e:
        # pylint: disable=unsupported-membership-test
        if provider not in e.msg:
            raise e
        raise_error(
            MissingBackend,
            f"The '{backend}' backends' provider is not available. Check that a Python "
            f"package named '{provider}' is installed, and it is exposing valid Qibo "
            "backends.",
        )


def _check_backend_and_local_state(seed, backend):
    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    backend = _check_backend(backend)

    if seed is None or isinstance(seed, int):
        if backend.__class__.__name__ in [
            "CupyBackend",
            "CuQuantumBackend",
        ]:  # pragma: no cover
            local_state = backend.np.random.default_rng(seed)
        else:
            local_state = np.random.default_rng(seed)
    else:
        local_state = seed

    return backend, local_state
