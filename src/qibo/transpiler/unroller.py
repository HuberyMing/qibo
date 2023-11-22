from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler.abstract import NativeGates, Unroller
from qibo.transpiler.decompositions import cz_dec, gpi2_dec, iswap_dec, opt_dec, u3_dec
from qibo.transpiler.exceptions import DecompositionError


# TODO: Make setting single-qubit native gates more flexible
class DefaultUnroller(Unroller):
    """Translates a circuit to native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to translate
            into native gates.
        native_gates (:class:`qibo.transpiler.abstract.NativeGates`): native gates supported by the quantum hardware.

    Returns:
        (:class:`qibo.models.circuit.Circuit`): equivalent circuit with native gates.
    """

    def __init__(
        self,
        native_gates: NativeGates,
    ):
        self.native_gates = native_gates

    def __call__(self, circuit: Circuit):
        translated_circuit = circuit.__class__(circuit.nqubits)
        for gate in circuit.queue:
            translated_circuit.add(
                translate_gate(
                    gate,
                    self.native_gates,
                )
            )
        return translated_circuit

    def is_satisfied(self, circuit: Circuit):
        """Return True if a circuit is correctly decomposed into native gates, otherwise False."""
        try:
            assert_decomposition(circuit=circuit, native_gates=self.native_gates)
        except DecompositionError:
            return False
        return True


def assert_decomposition(
    circuit: Circuit,
    native_gates: NativeGates,
):
    """Checks if a circuit has been correctly decomposed into native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        native_gates (:class:`qibo.transpiler.abstract.NativeGates`):
            native gates supported by the quantum hardware.
    """
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if len(gate.qubits) == 1:
            try:
                native_type_gate = NativeGates.from_gate(gate)
                if not (native_type_gate & native_gates.single_qubit_natives()):
                    raise_error(
                        DecompositionError,
                        f"{gate.name} is not a single qubit native gate.",
                    )
            except ValueError:
                raise_error(
                    DecompositionError,
                    f"{gate.name} is not a single qubit native gate.",
                )
        elif len(gate.qubits) == 2:
            try:
                native_type_gate = NativeGates.from_gate(gate)
                if not (native_type_gate & native_gates.two_qubit_natives()):
                    raise_error(
                        DecompositionError,
                        f"{gate.name} is not a two qubits native gate.",
                    )
            except ValueError:
                raise_error(
                    DecompositionError, f"{gate.name} is not a two qubits native gate."
                )
        else:
            raise_error(
                DecompositionError, f"{gate.name} acts on more than two qubits."
            )


def translate_gate(
    gate,
    native_gates: NativeGates,
):
    """Maps gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        native_gates (:class:`qibo.transpiler.abstract.NativeGates`): native gates supported by the quantum hardware.

    Returns:
        (list): List of native gates that decompose the input gate.
    """
    if isinstance(gate, (gates.M, gates.I, gates.Align)):
        return gate
    elif len(gate.qubits) == 1:
        return _translate_single_qubit_gates(gate, native_gates.single_qubit_natives())
    else:
        decomposition_2q = _translate_two_qubit_gates(gate, native_gates)
        final_decomposition = []
        for gate in decomposition_2q:
            if len(gate.qubits) == 1:
                final_decomposition += _translate_single_qubit_gates(
                    gate, native_gates.single_qubit_natives()
                )
            else:
                final_decomposition.append(gate)
        return final_decomposition


def _translate_single_qubit_gates(gate: gates.Gate, single_qubit_natives: NativeGates):
    """Helper method for :meth:`translate_gate`.

    Maps single qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        single_qubit_natives (:class:`qibo.transpiler.abstract.NativeGates`): single qubit native gates.

    Returns:
        (list): List of native gates that decompose the input gate.
    """
    if NativeGates.U3 & single_qubit_natives:
        return u3_dec(gate)
    elif NativeGates.GPI2 & single_qubit_natives:
        return gpi2_dec(gate)
    else:
        raise DecompositionError("Use U3 or GPI2 as single qubit native gates")


def _translate_two_qubit_gates(gate: gates.Gate, native_gates: NativeGates):
    """Helper method for :meth:`translate_gate`.

    Maps two qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        native_gates (:class:`qibo.transpiler.abstract.NativeGates`): native gates supported by the quantum hardware.

    Returns:
        (list): List of native gates that decompose the input gate.
    """
    two_qubit_natives = native_gates.two_qubit_natives()
    if two_qubit_natives is NativeGates.CZ | NativeGates.iSWAP:
        # Check for a special optimized decomposition.
        if gate.__class__ in opt_dec.decompositions:
            return opt_dec(gate)
        # Check if the gate has a CZ decomposition
        if not gate.__class__ in iswap_dec.decompositions:
            return cz_dec(gate)
        # Check the decomposition with less 2 qubit gates.
        else:
            if cz_dec.count_2q(gate) < iswap_dec.count_2q(gate):
                return cz_dec(gate)
            elif cz_dec.count_2q(gate) > iswap_dec.count_2q(gate):
                return iswap_dec(gate)
            # If equal check the decomposition with less 1 qubit gates.
            # This is never used for now but may be useful for future generalization
            elif cz_dec.count_1q(gate) < iswap_dec.count_1q(gate):  # pragma: no cover
                return cz_dec(gate)
            else:  # pragma: no cover
                return iswap_dec(gate)
    elif two_qubit_natives is NativeGates.CZ:
        return cz_dec(gate)
    elif two_qubit_natives is NativeGates.iSWAP:
        if gate.__class__ in iswap_dec.decompositions:
            return iswap_dec(gate)
        else:
            # First decompose into CZ
            cz_decomposed = cz_dec(gate)
            # Then CZ are decomposed into iSWAP
            iswap_decomposed = []
            for g in cz_decomposed:
                # Need recursive function as gates.Unitary is not in iswap_dec
                for g_translated in translate_gate(g, native_gates=native_gates):
                    iswap_decomposed.append(g_translated)
            return iswap_decomposed
    else:  # pragma: no cover
        raise_error(DecompositionError, "Use only CZ and/or iSWAP as native gates")
