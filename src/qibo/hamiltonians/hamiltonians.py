"""Module defining Hamiltonian classes."""

from itertools import chain
from math import prod
from typing import Optional

import numpy as np
import sympy

from qibo.backends import PyTorchBackend, _check_backend
from qibo.config import EINSUM_CHARS, log, raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.symbols import Z


class Hamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a dense or sparse matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape :math:`2^{n} \\times 2^{n}`.
            Sparse matrices based on ``scipy.sparse`` for ``numpy`` / ``qibojit`` backends
            (or on ``tf.sparse`` for the ``tensorflow`` backend) are also supported.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.
    """

    def __init__(self, nqubits, matrix, backend=None):
        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)

        if not (
            isinstance(matrix, self.backend.tensor_types)
            or self.backend.is_sparse(matrix)
        ):
            raise_error(
                TypeError,
                f"Matrix of invalid type {type(matrix)} given during Hamiltonian initialization",
            )
        matrix = self.backend.cast(matrix)

        super().__init__()
        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @property
    def matrix(self):
        """Returns the full matrix representation.

        For :math:`n` qubits, can be a dense :math:`2^{n} \\times 2^{n}` array or a sparse
        matrix, depending on how the Hamiltonian was created.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        shape = tuple(mat.shape)
        if shape != 2 * (2**self.nqubits,):
            raise_error(
                ValueError,
                f"The Hamiltonian is defined for {self.nqubits} qubits "
                + f"while the given matrix has shape {shape}.",
            )
        self._matrix = mat

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, backend=None):
        """Creates a :class:`qibo.hamiltonian.Hamiltonian` from a symbolic Hamiltonian.

        We refer to :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): full Hamiltonian written with ``sympy`` symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair ``(target, matrix)``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
                in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
                Defaults to ``None``.

        Returns:
            :class:`qibo.hamiltonians.SymbolicHamiltonian`: object that implements the
            Hamiltonian represented by the given symbolic expression.
        """
        log.warning(
            "`Hamiltonian.from_symbolic` and the use of symbol maps is "
            "deprecated. Please use `SymbolicHamiltonian` and Qibo symbols "
            "to construct Hamiltonians using symbols."
        )
        return SymbolicHamiltonian(
            symbolic_hamiltonian, symbol_map=symbol_map, backend=backend
        )

    def eigenvalues(self, k=6):
        if self._eigenvalues is None:
            self._eigenvalues = self.backend.calculate_eigenvalues(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.backend.calculate_eigenvectors(
                self.matrix, k
            )
        return self._eigenvectors

    def exp(self, a):
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=C0415
            matrix_exponentiation,
        )

        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = matrix_exponentiation(
                a, self.matrix, self._eigenvectors, self._eigenvalues, self.backend
            )
        return self._exp.get("result")

    def expectation(self, state, normalize=False):
        if isinstance(state, self.backend.tensor_types):
            state = self.backend.cast(state)
            shape = tuple(state.shape)
            if len(shape) == 1:  # state vector
                return self.backend.calculate_expectation_state(self, state, normalize)

            if len(shape) == 2:  # density matrix
                return self.backend.calculate_expectation_density_matrix(
                    self, state, normalize
                )

            raise_error(
                ValueError,
                "Cannot calculate Hamiltonian expectation value "
                + f"for state of shape {shape}",
            )

        raise_error(
            TypeError,
            "Cannot calculate Hamiltonian expectation "
            + f"value for state of type {type(state)}",
        )

    def expectation_from_samples(self, freq, qubit_map=None):
        obs = self.matrix
        if (
            self.backend.np.count_nonzero(
                obs - self.backend.np.diag(self.backend.np.diagonal(obs))
            )
            != 0
        ):
            raise_error(
                NotImplementedError,
                "Observable is not diagonal. Expectation of non diagonal observables starting from samples is currently supported for `qibo.hamiltonians.hamiltonians.SymbolicHamiltonian` only.",
            )
        keys = list(freq.keys())
        if qubit_map is None:
            qubit_map = list(range(int(np.log2(len(obs)))))
        counts = np.array(list(freq.values())) / sum(freq.values())
        expval = 0
        size = len(qubit_map)
        for j, k in enumerate(keys):
            index = 0
            for i in qubit_map:
                index += int(k[qubit_map.index(i)]) * 2 ** (size - 1 - i)
            expval += obs[index, index] * counts[j]
        return self.backend.np.real(expval)

    def eye(self, dim: Optional[int] = None):
        """Generate Identity matrix with dimension ``dim``"""
        if dim is None:
            dim = int(self.matrix.shape[0])
        return self.backend.cast(self.backend.matrices.I(dim), dtype=self.matrix.dtype)

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation:

        .. math::
            \\Xi_{k}(\\mu) = \\sqrt{\\bra{\\mu} \\, H^{2} \\, \\ket{\\mu}
                - \\bra{\\mu} \\, H \\, \\ket{\\mu}^2} \\, .

        for a given state :math:`\\ket{\\mu}`.

        Args:
            state (ndarray): quantum state to be used to compute the energy fluctuation.

        Returns:
            float: Energy fluctuation value.
        """
        state = self.backend.cast(state)
        energy = self.expectation(state)
        h = self.matrix
        h2 = Hamiltonian(nqubits=self.nqubits, matrix=h @ h, backend=self.backend)
        average_h2 = self.backend.calculate_expectation_state(h2, state, normalize=True)
        return self.backend.np.sqrt(self.backend.np.abs(average_h2 - energy**2))

    def __add__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be added.",
                )
            new_matrix = self.matrix + o.matrix
        elif isinstance(o, self.backend.numeric_types):
            new_matrix = self.matrix + o * self.eye()
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian addition to {type(o)} not implemented.",
            )
        return self.__class__(self.nqubits, new_matrix, backend=self.backend)

    def __sub__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be subtracted.",
                )
            new_matrix = self.matrix - o.matrix
        elif isinstance(o, self.backend.numeric_types):
            new_matrix = self.matrix - o * self.eye()
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(o)} not implemented.",
            )
        return self.__class__(self.nqubits, new_matrix, backend=self.backend)

    def __rsub__(self, o):
        if isinstance(o, self.__class__):  # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != o.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be added.",
                )
            new_matrix = o.matrix - self.matrix
        elif isinstance(o, self.backend.numeric_types):
            new_matrix = o * self.eye() - self.matrix
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(o)} not implemented.",
            )
        return self.__class__(self.nqubits, new_matrix, backend=self.backend)

    def __mul__(self, o):
        if isinstance(o, self.backend.tensor_types):
            o = complex(o)
        elif not isinstance(o, self.backend.numeric_types):
            raise_error(
                NotImplementedError,
                f"Hamiltonian multiplication to {type(o)} not implemented.",
            )
        new_matrix = self.matrix * o
        r = self.__class__(self.nqubits, new_matrix, backend=self.backend)
        o = self.backend.cast(o)
        if self._eigenvalues is not None:
            if self.backend.np.real(o) >= 0:  # TODO: check for side effects K.qnp
                r._eigenvalues = o * self._eigenvalues
            elif not self.backend.is_sparse(self.matrix):
                axis = (0,) if isinstance(self.backend, PyTorchBackend) else 0
                r._eigenvalues = o * self.backend.np.flip(self._eigenvalues, axis)
        if self._eigenvectors is not None:
            if self.backend.np.real(o) > 0:  # TODO: see above
                r._eigenvectors = self._eigenvectors
            elif o == 0:
                r._eigenvectors = self.eye(int(self._eigenvectors.shape[0]))
        return r

    def __matmul__(self, o):
        if isinstance(o, self.__class__):
            matrix = self.backend.calculate_hamiltonian_matrix_product(
                self.matrix, o.matrix
            )
            return self.__class__(self.nqubits, matrix, backend=self.backend)

        if isinstance(o, self.backend.tensor_types):
            return self.backend.calculate_hamiltonian_state_product(self.matrix, o)

        raise_error(
            NotImplementedError,
            f"Hamiltonian matmul to {type(o)} not implemented.",
        )


class SymbolicHamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a symbolic representation.

    Calculations using symbolic Hamiltonians are either done directly using
    the given ``sympy`` expression as it is (``form``) or by parsing the
    corresponding ``terms`` (which are :class:`qibo.core.terms.SymbolicTerm`
    objects). The latter approach is more computationally costly as it uses
    a ``sympy.expand`` call on the given form before parsing the terms.
    For this reason the ``terms`` are calculated only when needed, for example
    during Trotterization.
    The dense matrix of the symbolic Hamiltonian can be calculated directly
    from ``form`` without requiring ``terms`` calculation (see
    :meth:`qibo.core.hamiltonians.SymbolicHamiltonian.calculate_dense` for details).

    Args:
        form (sympy.Expr): Hamiltonian form as a ``sympy.Expr``. Ideally the
            Hamiltonian should be written using Qibo symbols.
            See :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
            example for more details.
        symbol_map (dict): Dictionary that maps each ``sympy.Symbol`` to a tuple
            of (target qubit, matrix representation). This feature is kept for
            compatibility with older versions where Qibo symbols were not available
            and may be deprecated in the future.
            It is not required if the Hamiltonian is constructed using Qibo symbols.
            The symbol_map can also be used to pass non-quantum operator arguments
            to the symbolic Hamiltonian, such as the parameters in the
            :meth:`qibo.hamiltonians.models.MaxCut` Hamiltonian.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.
    """

    def __init__(self, form=None, nqubits=None, symbol_map={}, backend=None):
        super().__init__()
        self._form = None
        self._terms = None
        self.constant = 0  # used only when we perform calculations using ``_terms``
        self._dense = None
        self.symbol_map = symbol_map
        # if a symbol in the given form is not a Qibo symbol it must be
        # included in the ``symbol_map``

        from qibo.symbols import Symbol  # pylint: disable=import-outside-toplevel

        self._qiboSymbol = Symbol  # also used in ``self._get_symbol_matrix``

        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)

        if form is not None:
            self.form = form
        if nqubits is not None:
            self.nqubits = nqubits

    @property
    def dense(self) -> "MatrixHamiltonian":
        """Creates the equivalent Hamiltonian matrix."""
        if self._dense is None:
            log.warning(
                "Calculating the dense form of a symbolic Hamiltonian. "
                "This operation is memory inefficient."
            )
            self.dense = self.calculate_dense()
        return self._dense

    @dense.setter
    def dense(self, hamiltonian):
        assert isinstance(hamiltonian, Hamiltonian)
        self._dense = hamiltonian
        self._eigenvalues = hamiltonian._eigenvalues
        self._eigenvectors = hamiltonian._eigenvectors
        self._exp = hamiltonian._exp

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, form):
        # Check that given form is a ``sympy`` expression
        if not isinstance(form, sympy.Expr):
            raise_error(
                TypeError,
                f"Symbolic Hamiltonian should be a ``sympy`` expression but is {type(form)}.",
            )
        # Calculate number of qubits in the system described by the given
        # Hamiltonian formula
        nqubits = 0
        for symbol in form.free_symbols:
            if isinstance(symbol, self._qiboSymbol):
                q = symbol.target_qubit
            elif isinstance(symbol, sympy.Expr):
                if symbol not in self.symbol_map:
                    raise_error(ValueError, f"Symbol {symbol} is not in symbol map.")
                q, matrix = self.symbol_map.get(symbol)
                if not isinstance(matrix, self.backend.tensor_types):
                    # ignore symbols that do not correspond to quantum operators
                    # for example parameters in the MaxCut Hamiltonian
                    q = 0
            if q > nqubits:
                nqubits = q

        self._form = form
        self.nqubits = nqubits + 1

    @property
    def terms(self):
        """List of terms of which the Hamiltonian is a sum of.

        Terms will be objects of type :class:`qibo.core.terms.HamiltonianTerm`.
        """
        if self._terms is None:
            # Calculate terms based on ``self.form``
            from qibo.hamiltonians.terms import (  # pylint: disable=import-outside-toplevel
                SymbolicTerm,
            )

            form = sympy.expand(self.form)
            terms = []
            for f, c in form.as_coefficients_dict().items():
                term = SymbolicTerm(c, f, self.symbol_map)
                if term.target_qubits:
                    terms.append(term)
                else:
                    self.constant += term.coefficient
            self._terms = terms
        return self._terms

    @terms.setter
    def terms(self, terms):
        self._terms = terms
        self.nqubits = max(q for term in self._terms for q in term.target_qubits) + 1

    @property
    def matrix(self):
        """Returns the full matrix representation.

        Consisting of :math:`2^{n} \\times 2^{n}`` elements.
        """
        return self.dense.matrix

    def eigenvalues(self, k=6):
        return self.dense.eigenvalues(k)

    def eigenvectors(self, k=6):
        return self.dense.eigenvectors(k)

    def ground_state(self):
        return self.eigenvectors()[:, 0]

    def exp(self, a):
        return self.dense.exp(a)

    def _get_symbol_matrix(self, term):
        """Calculates numerical matrix corresponding to symbolic expression.

        This is partly equivalent to sympy's ``.subs``, which does not work
        in our case as it does not allow us to substitute ``sympy.Symbol``
        with numpy arrays and there are different complication when switching
        to ``sympy.MatrixSymbol``. Here we calculate the full numerical matrix
        given the symbolic expression using recursion.
        Helper method for ``_calculate_dense_from_form``.

        Args:
            term (sympy.Expr): Symbolic expression containing local operators.

        Returns:
            ndarray: matrix corresponding to the given expression as an array
            of shape ``(2 ** self.nqubits, 2 ** self.nqubits)``.
        """
        if isinstance(term, sympy.Add):
            # symbolic op for addition
            result = sum(
                self._get_symbol_matrix(subterm) for subterm in term.as_ordered_terms()
            )

        elif isinstance(term, sympy.Mul):
            # symbolic op for multiplication
            # note that we need to use matrix multiplication even though
            # we use scalar symbols for convenience
            factors = term.as_ordered_factors()
            result = self._get_symbol_matrix(factors[0])
            for subterm in factors[1:]:
                result = result @ self._get_symbol_matrix(subterm)

        elif isinstance(term, sympy.Pow):
            # symbolic op for power
            base, exponent = term.as_base_exp()
            matrix = self._get_symbol_matrix(base)
            # multiply ``base`` matrix ``exponent`` times to itself
            result = matrix
            for _ in range(exponent - 1):
                result = result @ matrix

        elif isinstance(term, sympy.Symbol):
            # if the term is a ``Symbol`` then it corresponds to a quantum
            # operator for which we can construct the full matrix directly
            if isinstance(term, self._qiboSymbol):
                # if we have a Qibo symbol the matrix construction is
                # implemented in :meth:`qibo.core.terms.SymbolicTerm.full_matrix`.
                result = term.full_matrix(self.nqubits)
            else:
                q, matrix = self.symbol_map.get(term)
                if not isinstance(matrix, self.backend.tensor_types):
                    # symbols that do not correspond to quantum operators
                    # for example parameters in the MaxCut Hamiltonian
                    result = complex(matrix) * np.eye(2**self.nqubits)
                else:
                    # if we do not have a Qibo symbol we construct one and use
                    # :meth:`qibo.core.terms.SymbolicTerm.full_matrix`.
                    result = self._qiboSymbol(q, matrix).full_matrix(self.nqubits)

        elif term.is_number:
            # if the term is number we should return in the form of identity
            # matrix because in expressions like `1 + Z`, `1` is not correspond
            # to the float 1 but the identity operator (matrix)
            result = complex(term) * np.eye(2**self.nqubits)

        else:
            raise_error(
                TypeError,
                f"Cannot calculate matrix for symbolic term of type {type(term)}.",
            )

        return result

    def _calculate_dense_from_form(self) -> Hamiltonian:
        """Calculates equivalent Hamiltonian using symbolic form.

        Useful when the term representation is not available.
        """
        matrix = self._get_symbol_matrix(self.form)
        return Hamiltonian(self.nqubits, matrix, backend=self.backend)

    def _calculate_dense_from_terms(self) -> Hamiltonian:
        """Calculates equivalent Hamiltonian using the term representation."""
        if 2 * self.nqubits > len(EINSUM_CHARS):  # pragma: no cover
            # case not tested because it only happens in large examples
            raise_error(NotImplementedError, "Not enough einsum characters.")

        matrix = 0
        chars = EINSUM_CHARS[: 2 * self.nqubits]
        for term in self.terms:
            ntargets = len(term.target_qubits)
            tmat = np.reshape(term.matrix, 2 * ntargets * (2,))
            n = self.nqubits - ntargets
            emat = np.reshape(np.eye(2**n, dtype=tmat.dtype), 2 * n * (2,))
            gen = lambda x: (chars[i + x] for i in term.target_qubits)
            tc = "".join(chain(gen(0), gen(self.nqubits)))
            ec = "".join(c for c in chars if c not in tc)
            matrix += np.einsum(f"{tc},{ec}->{chars}", tmat, emat)
        matrix = np.reshape(matrix, 2 * (2**self.nqubits,))
        return Hamiltonian(self.nqubits, matrix, backend=self.backend) + self.constant

    def calculate_dense(self):
        if self._terms is None:
            # calculate dense matrix directly using the form to avoid the
            # costly ``sympy.expand`` call
            return self._calculate_dense_from_form()
        return self._calculate_dense_from_terms()

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def expectation_from_circuit(
        self, circuit: "Circuit", qubit_map: dict = None, nshots: int = 1000
    ) -> float:
        """
        Calculate the expectation value from a circuit.
        This even works for observables not completely diagonal in the computational
        basis, but only diagonal at a term level in a defined basis. Namely, for
        an observable of the form :math:``H = \\sum_i H_i``, where each :math:``H_i``
        consists in a `n`-qubits pauli operator :math:`P_0 \\otimes P_1 \\otimes \\cdots \\otimes P_n`,
        the expectation value is computed by rotating the input circuit in the suitable
        basis for each term :math:``H_i`` thus extracting the `term-wise` expectations
        that are then summed to build the global expectation value.
        Each term of the observable is treated separately, by measuring in the correct
        basis and re-executing the circuit.

        Args:
            circuit (Circuit): input circuit.
            qubit_map (dict): qubit map, defaults to ``{0: 0, 1: 1, ..., n: n}``
            nshots (int): number of shots, defaults to 1000.

        Returns:
            (float): the calculated expectation value.
        """
        from qibo import Circuit, gates

        if qubit_map is None:
            qubit_map = list(range(self.nqubits))

        rotated_circuits = []
        coefficients = []
        Z_observables = []
        q_maps = []
        for term in self.terms:
            # store coefficient
            coefficients.append(term.coefficient)
            # build diagonal observable
            Z_observables.append(
                SymbolicHamiltonian(
                    prod([Z(q) for q in term.target_qubits]),
                    nqubits=circuit.nqubits,
                    backend=self.backend,
                )
            )
            # prepare the measurement basis and append it to the circuit
            measurements = [
                gates.M(factor.target_qubit, basis=factor.gate.__class__)
                for factor in set(term.factors)
            ]
            circ_copy = circuit.copy(True)
            circ_copy.add(measurements)
            rotated_circuits.append(circ_copy)
            # map the original qubits into 0, 1, 2, ...
            q_maps.append(
                dict(
                    zip(
                        [qubit_map[q] for q in term.target_qubits],
                        range(len(term.target_qubits)),
                    )
                )
            )
        frequencies = [
            result.frequencies()
            for result in self.backend.execute_circuits(rotated_circuits, nshots=nshots)
        ]
        return sum(
            [
                coeff * obs.expectation_from_samples(freq, qmap)
                for coeff, freq, obs, qmap in zip(
                    coefficients, frequencies, Z_observables, q_maps
                )
            ]
        )

    def expectation_from_samples(self, freq: dict, qubit_map: dict = None) -> float:
        """
        Calculate the expectation value from the samples.
        The observable has to be diagonal in the computational basis.

        Args:
            freq (dict): input frequencies of the samples.
            qubit_map (dict): qubit map.

        Returns:
            (float): the calculated expectation value.
        """
        for term in self.terms:
            # pylint: disable=E1101
            for factor in term.factors:
                if not isinstance(factor, Z):
                    raise_error(
                        NotImplementedError, "Observable is not a Z Pauli string."
                    )

        if qubit_map is None:
            qubit_map = list(range(self.nqubits))

        keys = list(freq.keys())
        counts = self.backend.cast(list(freq.values()), self.backend.precision) / sum(
            freq.values()
        )
        expvals = []
        for term in self.terms:
            qubits = term.target_qubits
            expvals.extend(
                [
                    term.coefficient.real
                    * (-1) ** [state[qubit_map[q]] for q in qubits].count("1")
                    for state in keys
                ]
            )
        expvals = self.backend.cast(expvals, dtype=counts.dtype).reshape(
            len(self.terms), len(freq)
        )
        return self.backend.np.sum(expvals @ counts.T) + self.constant.real

    def __add__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be added.",
                )
            new_ham = self.__class__(
                symbol_map=dict(self.symbol_map), backend=self.backend
            )
            if self._form is not None and o._form is not None:
                new_ham.form = self.form + o.form
                new_ham.symbol_map.update(o.symbol_map)
            if self._terms is not None and o._terms is not None:
                new_ham.terms = self.terms + o.terms
                new_ham.constant = self.constant + o.constant
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense + o.dense

        elif isinstance(o, self.backend.numeric_types):
            new_ham = self.__class__(
                symbol_map=dict(self.symbol_map), backend=self.backend
            )
            if self._form is not None:
                new_ham.form = self.form + o
            if self._terms is not None:
                new_ham.terms = self.terms
                new_ham.constant = self.constant + o
            if self._dense is not None:
                new_ham.dense = self.dense + o

        else:
            raise_error(
                NotImplementedError,
                f"SymbolicHamiltonian addition to {type(o)} not implemented.",
            )
        return new_ham

    def __sub__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be subtracted.",
                )
            new_ham = self.__class__(
                symbol_map=dict(self.symbol_map), backend=self.backend
            )
            if self._form is not None and o._form is not None:
                new_ham.form = self.form - o.form
                new_ham.symbol_map.update(o.symbol_map)
            if self._terms is not None and o._terms is not None:
                new_ham.terms = self.terms + [-1 * x for x in o.terms]
                new_ham.constant = self.constant - o.constant
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense - o.dense

        elif isinstance(o, self.backend.numeric_types):
            new_ham = self.__class__(
                symbol_map=dict(self.symbol_map), backend=self.backend
            )
            if self._form is not None:
                new_ham.form = self.form - o
            if self._terms is not None:
                new_ham.terms = self.terms
                new_ham.constant = self.constant - o
            if self._dense is not None:
                new_ham.dense = self.dense - o

        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(o)} " "not implemented.",
            )
        return new_ham

    def __rsub__(self, o):
        if isinstance(o, self.backend.numeric_types):
            new_ham = self.__class__(
                symbol_map=dict(self.symbol_map), backend=self.backend
            )
            if self._form is not None:
                new_ham.form = o - self.form
            if self._terms is not None:
                new_ham.terms = [-1 * x for x in self.terms]
                new_ham.constant = o - self.constant
            if self._dense is not None:
                new_ham.dense = o - self.dense
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(o)} not implemented.",
            )
        return new_ham

    def __mul__(self, o):
        if not isinstance(o, (self.backend.numeric_types, self.backend.tensor_types)):
            raise_error(
                NotImplementedError,
                f"Hamiltonian multiplication to {type(o)} not implemented.",
            )
        o = complex(o)
        new_ham = self.__class__(symbol_map=dict(self.symbol_map), backend=self.backend)
        if self._form is not None:
            new_ham.form = o * self.form
        if self._terms is not None:
            new_ham.terms = [o * x for x in self.terms]
            new_ham.constant = self.constant * o
        if self._dense is not None:
            new_ham.dense = o * self._dense
        return new_ham

    def apply_gates(self, state, density_matrix=False):
        """Applies gates corresponding to the Hamiltonian terms.

        Gates are applied to the given state.

        Helper method for :meth:`qibo.hamiltonians.SymbolicHamiltonian.__matmul__`.
        """
        total = 0
        for term in self.terms:
            total += term(
                self.backend,
                self.backend.cast(state, copy=True),
                self.nqubits,
                density_matrix=density_matrix,
            )
        if self.constant:  # pragma: no cover
            total += self.constant * state
        return total

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            if self._form is None or o._form is None:
                raise_error(
                    NotImplementedError,
                    "Multiplication of symbolic Hamiltonians "
                    "without symbolic form is not implemented.",
                )
            new_form = self.form * o.form
            new_symbol_map = dict(self.symbol_map)
            new_symbol_map.update(o.symbol_map)
            new_ham = self.__class__(
                new_form, symbol_map=new_symbol_map, backend=self.backend
            )
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense @ o.dense
            return new_ham

        if isinstance(o, self.backend.tensor_types):
            rank = len(tuple(o.shape))
            if rank not in (1, 2):
                raise_error(
                    NotImplementedError,
                    f"Cannot multiply Hamiltonian with rank-{rank} tensor.",
                )
            state_qubits = int(np.log2(int(o.shape[0])))
            if state_qubits != self.nqubits:
                raise_error(
                    ValueError,
                    f"Cannot multiply Hamiltonian on {self.nqubits} qubits to "
                    + f"state of {state_qubits} qubits.",
                )
            if rank == 1:  # state vector
                return self.apply_gates(o)

            return self.apply_gates(o, density_matrix=True)

        raise_error(
            NotImplementedError,
            f"Hamiltonian matmul to {type(o)} not implemented.",
        )

    def circuit(self, dt, accelerators=None):
        """Circuit that implements a Trotter step of this Hamiltonian.

        Args:
            dt (float): Time step used for Trotterization.
            accelerators (dict, optional): Dictionary with accelerators for distributed circuits.
                Defaults to ``None``.
        """
        from qibo import Circuit  # pylint: disable=import-outside-toplevel
        from qibo.hamiltonians.terms import (  # pylint: disable=import-outside-toplevel
            TermGroup,
        )

        groups = TermGroup.from_terms(self.terms)
        circuit = Circuit(self.nqubits, accelerators=accelerators)
        circuit.add(
            group.term.expgate(dt / 2.0) for group in chain(groups, groups[::-1])
        )

        return circuit


class TrotterHamiltonian:
    """"""

    def __init__(self, *parts):
        raise_error(
            NotImplementedError,
            "`TrotterHamiltonian` is substituted by `SymbolicHamiltonian` "
            + "and is no longer supported. Please check the documentation "
            + "of `SymbolicHamiltonian` for more details.",
        )

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map):
        return cls()
