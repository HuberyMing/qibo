
import numpy as np
from qibo import gates
from qibo.models import Circuit

def create_measurement_circuits(circuit, labels, circuit_name = 'GHZ', label_format="little_endian"):
    """
    Prepares measurement circuits
    - labels: string of Pauli matrices (e.g. XYZXX)
    """
    
    measurement_circuit_names = []
    measurement_circuits = []

    nqubits = circuit.nqubits

    for label in labels:

        # for aligning to the natural little_endian way of iterating through bits below
        if label_format == "big_endian":
            effective_label = label[::-1]
        else:
            effective_label = label[:]
        probe_circuit = Circuit(nqubits)

        for qubit, letter in zip(*[range(circuit.nqubits), effective_label]):
            if letter == "X":
                probe_circuit.add(gates.H(qubit))  # H

            elif letter == "Y":
                probe_circuit.add(gates.S(qubit).dagger())  # S^dg
                probe_circuit.add(gates.H(qubit))  # H

        probe_circuit.add(gates.M(*range(nqubits)))

        measurement_circuit_name = "{}-{}".format(circuit_name, label)
        measurement_circuit = circuit + probe_circuit

        measurement_circuit_names.append(measurement_circuit_name)
        measurement_circuits.append(measurement_circuit)

    return measurement_circuits, measurement_circuit_names

def execute_measurement_circuits(
    circuit,
    labels,
    circuit_name = 'GHZ',
    num_shots=100,
    label_format="little_endian",
):
    """
    Executes measurement circuits
    - labels: string of Pauli matrices (e.g. XYZXX)
    - backend: 'numpy', 'qibojit', 'pytorch', 'tensorflow' prvided by qibo
    - num_shots: number of shots measurement is taken to get empirical frequency through counts
    """

    circuit_name = 'GHZ'

    measurement_circuits, measurement_circuit_names = create_measurement_circuits(circuit, labels, circuit_name)

    data_dict_list = []
    for i, label in enumerate(labels):
        result = measurement_circuits[i](nshots=num_shots)
        count_dict = result.frequencies()

        measurement_circuit_name = "{}-{}".format(circuit_name, label)

        data_dict = {
            "measurement_circuit_name": measurement_circuit_name,
            "circuit_name": circuit_name,
            "label": label,
            "count_dict": count_dict,
            "num_shots": num_shots,
        }
        data_dict_list.append(data_dict)
    return data_dict_list


def get_state_vector(circuit):
    """
    obtain state vector
    """
    return circuit.execute().state()

def get_state_matrix(circuit):
    """
    Obtain density matrix by taking an outer product of state vector
    """
    state_vector = get_state_vector(circuit)
    state_matrix = np.outer(state_vector, state_vector.conj())
    return state_matrix


def GHZstate(nqubits):

    circuit = Circuit(nqubits)

    circuit.add(gates.H(0))
    for i in range(1, nqubits):
        circuit.add(gates.CNOT(0, i))

    circuit_name = 'GHZ'

    return circuit, circuit_name


if __name__ == '__main__':

    nqubits = 3
    circuit, circuit_name = GHZstate(nqubits)

    labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ"]
    #measurement_circuits, measurement_circuit_names = create_measurement_circuits(circuit, labels)
    data_dict_list = execute_measurement_circuits(circuit, labels, circuit_name)

    statevec  = get_state_vector(circuit)
    DenMat    = get_state_matrix(circuit)
