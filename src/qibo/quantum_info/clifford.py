from dataclasses import dataclass
from functools import reduce
from itertools import product

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary


def _string_product(operators):
    """Calculates the tensor product of a list of operators represented as strings.

    Args:
        operators (list): The list of operators.

    Returns:
        product (str): The string representing the tensor product of the operators.
    """
    # calculate global sign
    phases = np.array(["-" in op for op in operators], dtype=bool)
    # remove the - signs
    operators = "|".join(operators).replace("-", "").split("|")
    prod = []
    for op in zip(*operators):
        tmp = "".join([o for o in op if o != "I"])
        if tmp == "":
            tmp = "I"
        prod.append(tmp)
    result = "-" if len(phases.nonzero()[0]) % 2 == 1 else ""
    return f"{result}({')('.join(prod)})"


def _list_of_matrices_product(operators):
    """Calculates the product of a list of operators as np.ndarrays.

    Args:
        operators (list): The list of operators.

    Returns:
        product (np.ndarray): The tensor product of the operators.
    """
    return reduce(np.matmul, operators)  # faster
    # return np.einsum(*[d for i, op in enumerate(operators) for d in (op, (i, i + 1))])


@dataclass
class Clifford:
    """The object storing the results of a circuit execution with the ``qibo.backends.CliffordBackend``.

    Args:
        tableau (np.ndarray): The tableu of the state.
        measurements (list): A list of measurements gates ``qibo.gates.M``.
        nqubits (int): The number of qubits of the state.
        nshots (int): The number of shots used for sampling the measurements.
    """

    tableau: np.ndarray
    measurements: list = None
    nqubits: int = None
    nshots: int = 1000

    _backend: CliffordBackend = CliffordBackend()
    _measurement_gate = None
    _samples = None

    def __post_init__(self):
        # adding the scratch row if not provided
        if self.tableau.shape[0] % 2 == 0:
            self.tableau = np.vstack((self.tableau, np.zeros(self.tableau.shape[1])))
        self.nqubits = int((self.tableau.shape[1] - 1) / 2)

    @classmethod
    def from_circuit(
        cls, circuit: Circuit, initial_state: np.ndarray = None, nshots: int = 1000
    ):
        """Allows to create a ``Clifford`` object by executing the input circuit.

        Args:
            circuit (qibo.models.Circuit): The clifford circuit to run.
            initial_state (np.ndarray): The initial tableu state.
            nshots (int): The number of shots to perform.

        Returns:
            result (qibo.quantum_info.Clifford): The object storing the result of the circuit execution.
        """
        return cls._backend.execute_circuit(circuit, initial_state, nshots)

    def generators(self, return_array=False):
        """Extracts the generators of the stabilizers.

        Args:
            return_array (bool): If ``True`` returns the generators as np.ndarray matrices, otherwise their representation as strings is returned.

        Returns:
            (generators, phases) (list, list): List of the generators and their corresponding phases.
        """
        generators, phases = self._backend.tableau_to_generators(
            self.tableau, return_array
        )
        return generators[self.nqubits :], phases[self.nqubits :]

    def get_destabilizers_generators(self, return_array=False):
        """Extracts the generators of the de-stabilizers.

        Args:
            return_array (bool): If ``True`` returns the generators as np.ndarray matrices, otherwise their representation as strings is returned.

        Returns:
            (generators, phases) (list, list): List of the generators and their corresponding phases.
        """
        generators, phases = self._backend.tableau_to_generators(
            self.tableau, return_array
        )
        return generators[: self.nqubits], phases[: self.nqubits]

    def _construct_operators(self, generators, phases, is_array=False):
        """Helper function to construct all the operators from their generators.

        Args:
            generators (list): List of the generators.
            phases (list): List of the phases of the generators.
            is_array (bool): Whether the generators are np.ndarrays.

        Returns:
            operators (list): A list containining all the operators generated by the generators.
        """
        if is_array:
            operators = np.array(generators) * phases.reshape(-1, 1, 1)
        else:
            operators = generators.copy()
            for i in (phases == -1).nonzero()[0]:
                operators[i] = f"-{operators[i]}"
        identity = (
            np.eye(2**self.nqubits)
            if is_array
            else "".join(["I" for _ in range(self.nqubits)])
        )
        operators = [(g, identity) for g in operators]
        if is_array:
            return [_list_of_matrices_product(ops) for ops in product(*operators)]
        return [_string_product(ops) for ops in product(*operators)]

    def stabilizers(self, return_array=False):
        """Extracts the stabilizers of the state.

        Args:
            return_array (bool): If ``True`` returns the stabilizers as np.narrays.

        Returns:
            stabilizers (list): List of the stabilizers of the state.
        """
        generators, phases = self.get_stabilizers_generators(return_array)
        return self._construct_operators(generators, phases, return_array)

    def destabilizers(self, return_array=False):
        """Extracts the de-stabilizers of the state.

        Args:
            return_array (bool): If ``True`` returns the de-stabilizers as np.narrays.

        Returns:
            destabilizers (list): List of the de-stabilizers of the state.
        """
        generators, phases = self.get_destabilizers_generators(return_array)
        return self._construct_operators(generators, phases, return_array)

    def state(self):
        """Builds the density matrix representation of the state.

        Returns:
            state (np.ndarray): The density matrix of the state.
        """
        stabilizers = self.get_stabilizers(True)
        return np.sum(stabilizers, 0) / len(stabilizers)

    @property
    def measurement_gate(self):
        """Single measurement gate containing all measured qubits.

        Useful for sampling all measured qubits at once when simulating.
        """
        if self._measurement_gate is None:
            for gate in self.measurements:
                if self._measurement_gate is None:
                    self._measurement_gate = M(*gate.init_args, **gate.init_kwargs)
                else:
                    self._measurement_gate.add(gate)

        return self._measurement_gate

    def samples(self, binary: bool = True, registers: bool = False):
        """Returns raw measurement samples.

        Args:
            binary (bool, optional): Return samples in binary or decimal form.
            registers (bool, optional): Group samples according to registers.

        Returns:
            If ``binary`` is ``True``
                samples are returned in binary form as a tensor
                of shape ``(nshots, n_measured_qubits)``.
            If ``binary`` is ``False``
                samples are returned in decimal form as a tensor
                of shape ``(nshots,)``.
            If ``registers`` is ``True``
                samples are returned in a ``dict`` where the keys are the register
                names and the values are the samples tensors for each register.
            If ``registers`` is ``False``
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        if not self.measurements:
            raise_error(RuntimeError, "No measurement provided.")
        measured_qubits = self.measurement_gate.qubits
        if self._samples is None:
            if self.measurements[0].result.has_samples():
                samples = np.concatenate(
                    [gate.result.samples() for gate in self.measurements], axis=1
                )
            else:
                samples = self._backend.sample_shots(
                    self.tableau, measured_qubits, self.nqubits, self.nshots
                )
            if self.measurement_gate.has_bitflip_noise():
                p0, p1 = self.measurement_gate.bitflip_map
                bitflip_probabilities = [
                    [p0.get(q) for q in measured_qubits],
                    [p1.get(q) for q in measured_qubits],
                ]
                samples = self._backend.apply_bitflips(samples, bitflip_probabilities)
            # register samples to individual gate ``MeasurementResult``
            qubit_map = {
                q: i for i, q in enumerate(self.measurement_gate.target_qubits)
            }
            self._samples = np.array(samples, dtype="int32")
            for gate in self.measurements:
                rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                gate.result.register_samples(self._samples[:, rqubits], self._backend)
        if registers:
            return {
                gate.register_name: gate.result.samples(binary)
                for gate in self.measurements
            }

        if binary:
            return self._samples

        return self._backend.samples_to_decimal(self._samples, len(measured_qubits))

    def frequencies(self, binary: bool = True, registers: bool = False):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool, optional): Return frequency keys in binary or decimal form.
            registers (bool, optional): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If ``binary`` is ``True``
                the keys of the `Counter` are in binary form, as strings of
                :math:`0`s and :math`1`s.
            If ``binary`` is ``False``
                the keys of the ``Counter`` are integers.
            If ``registers`` is ``True``
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If ``registers`` is ``False``
                a single ``Counter`` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        measured_qubits = self.measurement_gate.target_qubits
        freq = self._backend.calculate_frequencies(self.samples(False))
        if registers:
            if binary:
                return {
                    gate.register_name: frequencies_to_binary(
                        self._backend.calculate_frequencies(gate.result.samples(False)),
                        len(gate.target_qubits),
                    )
                    for gate in self.measurements
                }
            return {
                gate.register_name: self._backend.calculate_frequencies(
                    gate.result.samples(False)
                )
                for gate in self.measurements
            }

        if binary:
            return frequencies_to_binary(freq, len(measured_qubits))
        else:
            return freq

    def probabilities(self, qubits=None):
        """Computes the probabilities of the selected qubits from the measured samples.

        Args:
            qubits (tuple): Qubits for which to compute the probabilities.

        Returns:
            probabilities (np.ndarray): The measured probabilities.
        """
        measured_qubits = self.measurement_gate.qubits
        if qubits is not None:
            if not set(qubits).issubset(set(measured_qubits)):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, but only qubits {measured_qubits} were measured.",
                )
            qubits = [measured_qubits.index(q) for q in qubits]
        else:
            qubits = range(len(measured_qubits))

        probs = [0 for _ in range(2 ** len(measured_qubits))]
        samples = self.samples(False)
        for s in samples:
            probs[s] += 1
        probs = self._backend.cast(probs) / len(samples)
        return self._backend.calculate_probabilities(
            np.sqrt(probs), qubits, len(measured_qubits)
        )
