from qibo.gates.abstract import Gate, Channel
from qibo.gates.gates import X, Y, Z, M, Unitary
from qibo.config import raise_error


class KrausChannel(Channel):
    """General channel defined by arbitrary Krauss operators.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\sum _k A_k \\rho A_k^\\dagger

    where A are arbitrary Kraus operators given by the user. Note that Kraus
    operators set should be trace preserving, however this is not checked.
    Simulation of this gate requires the use of density matrices.
    For more information on channels and Kraus operators please check
    `J. Preskill's notes <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_.

    Args:
        ops (list): List of Kraus operators as pairs ``(qubits, Ak)`` where
          ``qubits`` refers the qubit ids that ``Ak`` acts on and ``Ak`` is
          the corresponding matrix as a ``np.ndarray`` or ``tf.Tensor``.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import Circuit
            from qibo import gates
            # initialize circuit with 3 qubits
            c = Circuit(3, density_matrix=True)
            # define a sqrt(0.4) * X gate
            a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
            # define a sqrt(0.6) * CNOT gate
            a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 0, 1], [0, 0, 1, 0]])
            # define the channel rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
            channel = gates.KrausChannel([((1,), a1), ((0, 2), a2)])
            # add the channel to the circuit
            c.add(channel)
    """

    def __init__(self, ops):
        super(KrausChannel, self).__init__()
        self.name = "KrausChannel"
        self.density_matrix = True
        if isinstance(ops[0], Gate):
            self.gates = tuple(ops)
            self.target_qubits = tuple(sorted(set(
                q for gate in ops for q in gate.target_qubits)))
        else:
            self.gates, self.target_qubits = self._from_matrices(ops)
        self.init_args = [self.gates]

    def _from_matrices(self, matrices):
        """Creates gates from qubits and matrices list."""
        gatelist, qubitset = [], set()
        for qubits, matrix in matrices:
            # Check that given operators have the proper shape.
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise_error(ValueError, "Invalid Krauss operator shape {} for "
                                        "acting on {} qubits."
                                        "".format(shape, len(qubits)))
            qubitset.update(qubits)
            gatelist.append(Unitary(matrix, *list(qubits)))
            gatelist[-1].density_matrix = True
        return tuple(gatelist), tuple(sorted(qubitset))


class UnitaryChannel(KrausChannel):
    """Channel that is a probabilistic sum of unitary operations.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\left (1 - \\sum _k p_k \\right )\\rho +
                                \\sum _k p_k U_k \\rho U_k^\\dagger

    where U are arbitrary unitary operators and p are floats between 0 and 1.
    Note that unlike :class:`qibo.abstractions.gates.KrausChannel` which requires
    density matrices, it is possible to simulate the unitary channel using
    state vectors and probabilistic sampling. For more information on this
    approach we refer to :ref:`Using repeated execution <repeatedexec-example>`.

    Args:
        p (list): List of floats that correspond to the probability that each
            unitary Uk is applied.
        ops (list): List of  operators as pairs ``(qubits, Uk)`` where
            ``qubits`` refers the qubit ids that ``Uk`` acts on and ``Uk`` is
            the corresponding matrix as a ``np.ndarray``/``tf.Tensor``.
            Must have the same length as the given probabilities ``p``.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, p, ops, seed=None):
        if len(p) != len(ops):
            raise_error(ValueError, "Probabilities list has length {} while "
                                    "{} gates were given."
                                    "".format(len(p), len(ops)))
        for pp in p:
            if pp < 0 or pp > 1:
                raise_error(ValueError, "Probabilities should be between 0 "
                                        "and 1 but {} was given.".format(pp))
        super(UnitaryChannel, self).__init__(ops)
        self.name = "UnitaryChannel"
        self.probs = p
        self.psum = sum(p)
        self.seed = seed
        self.density_matrix = False
        self.init_args = [p, self.gates]
        self.init_kwargs = {"seed": seed}


class PauliNoiseChannel(UnitaryChannel):
    """Noise channel that applies Pauli operators with given probabilities.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_x - p_y - p_z) \\rho + p_x X\\rho X + p_y Y\\rho Y + p_z Z\\rho Z

    which can be used to simulate phase flip and bit flip errors.
    This channel can be simulated using either density matrices or state vectors
    and sampling with repeated execution.
    See :ref:`How to perform noisy simulation? <noisy-example>` for more
    information.

    Args:
        q (int): Qubit id that the noise acts on.
        px (float): Bit flip (X) error probability.
        py (float): Y-error probability.
        pz (float): Phase flip (Z) error probability.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, px=0, py=0, pz=0, seed=None):
        probs, gates = [], []
        for p, gate in [(px, X), (py, Y), (pz, Z)]:
            if p > 0:
                probs.append(p)
                gates.append(gate(q))

        super(PauliNoiseChannel, self).__init__(probs, gates, seed=seed)
        self.name = "PauliNoiseChannel"
        assert self.target_qubits == (q,)

        self.init_args = [q]
        self.init_kwargs = {"px": px, "py": py, "pz": pz, "seed": seed}


class ResetChannel(UnitaryChannel):
    """Single-qubit reset channel.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_0 - p_1) \\rho
        + p_0 (|0\\rangle \\langle 0| \\otimes \\tilde{\\rho })
        + p_1 (|1\\rangle \langle 1| \otimes \\tilde{\\rho })

    with

    .. math::
        \\tilde{\\rho } = \\frac{\langle 0|\\rho |0\\rangle }{\mathrm{Tr}\langle 0|\\rho |0\\rangle}

    Args:
        q (int): Qubit id that the channel acts on.
        p0 (float): Probability to reset to 0.
        p1 (float): Probability to reset to 1.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, p0=0.0, p1=0.0, seed=None):
        probs = [p0, p1]
        gates = [M(q, collapse=True), X(q)]
        super(ResetChannel, self).__init__(probs, gates, seed=seed)
        self.name = "ResetChannel"
        assert self.target_qubits == (q,)

        self.init_args = [q]
        self.init_kwargs = {"p0": p0, "p1": p1, "seed": seed}


class ThermalRelaxationChannel:
    """Single-qubit thermal relaxation error channel.

    Implements the following transformation:

    If :math:`T_1 \\geq T_2`:

    .. math::
        \\mathcal{E} (\\rho ) = (1 - p_z - p_0 - p_1)\\rho + p_zZ\\rho Z
        + p_0 (|0\\rangle \\langle 0| \\otimes \\tilde{\\rho })
        + p_1 (|1\\rangle \langle 1| \otimes \\tilde{\\rho })

    with

    .. math::
        \\tilde{\\rho } = \\frac{\langle 0|\\rho |0\\rangle }{\mathrm{Tr}\langle 0|\\rho |0\\rangle}

    while if :math:`T_1 < T_2`:

    .. math::
        \\mathcal{E}(\\rho ) = \\mathrm{Tr} _\\mathcal{X}\\left [\\Lambda _{\\mathcal{X}\\mathcal{Y}}(\\rho _\\mathcal{X} ^T \\otimes \\mathbb{I}_\\mathcal{Y})\\right ]

    with

    .. math::
        \\Lambda = \\begin{pmatrix}
        1 - p_1 & 0 & 0 & e^{-t / T_2} \\\\
        0 & p_1 & 0 & 0 \\\\
        0 & 0 & p_0 & 0 \\\\
        e^{-t / T_2} & 0 & 0 & 1 - p_0
        \\end{pmatrix}

    where :math:`p_0 = (1 - e^{-t / T_1})(1 - \\eta )` :math:`p_1 = (1 - e^{-t / T_1})\\eta`
    and :math:`p_z = 1 - e^{-t / T_1} + e^{-t / T_2} - e^{t / T_1 - t / T_2}`.
    Here :math:`\\eta` is the ``excited_population``
    and :math:`t` is the ``time``, both controlled by the user.
    This gate is based on
    `Qiskit's thermal relaxation error channel <https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.thermal_relaxation_error.html#qiskit.providers.aer.noise.thermal_relaxation_error>`_.

    Args:
        q (int): Qubit id that the noise channel acts on.
        t1 (float): T1 relaxation time. Should satisfy ``t1 > 0``.
        t2 (float): T2 dephasing time.
            Should satisfy ``t1 > 0`` and ``t2 < 2 * t1``.
        time (float): the gate time for relaxation error.
        excited_population (float): the population of the excited state at
            equilibrium. Default is 0.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        self.name = "ThermalRelaxationChannel"
        self.init_args = [q, t1, t2, time]
        self.init_kwargs = {"excited_population": excited_population,
                            "seed": seed}

    def calculate_probabilities(self, t1, t2, time, excited_population):
        if excited_population < 0 or excited_population > 1:
            raise_error(ValueError, "Invalid excited state population {}."
                                    "".format(excited_population))
        if time < 0:
            raise_error(ValueError, "Invalid gate_time ({} < 0)".format(time))
        if t1 <= 0:
            raise_error(ValueError, "Invalid T_1 relaxation time parameter: "
                                    "T_1 <= 0.")
        if t2 <= 0:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 <= 0.")
        if t2 > 2 * t1:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 greater than 2 * T_1.")


class _ThermalRelaxationChannelA(UnitaryChannel):
    """Implements thermal relaxation when T1 >= T2."""

    def calculate_probabilities(self, t1, t2, time, excited_population): # pragma: no cover
        # function not tested because it is redefined in `qibo.core.cgates._ThermalRelaxationChannelA`
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        probs = self.calculate_probabilities(t1, t2, time, excited_population)
        gates = [Z(q), M(q, collapse=True), X(q)]
        super(_ThermalRelaxationChannelA, self).__init__(
            probs, gates, seed=seed)
        ThermalRelaxationChannel.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        assert self.target_qubits == (q,)


class _ThermalRelaxationChannelB(Gate):
    """Implements thermal relaxation when T1 < T2."""

    def calculate_probabilities(self, t1, t2, time, excited_population): # pragma: no cover
        # function not tested because it is redefined in `qibo.core.cgates._ThermalRelaxationChannelB`
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        probs = self.calculate_probabilities(t1, t2, time, excited_population)
        self.exp_t2, self.preset0, self.preset1 = probs # pylint: disable=E0633

        super(_ThermalRelaxationChannelB, self).__init__()
        self.target_qubits = (q,)
        ThermalRelaxationChannel.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        # this case can only be applied to density matrices
        self.density_matrix = True