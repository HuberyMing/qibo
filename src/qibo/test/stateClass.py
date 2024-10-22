#
#   class states
#   

import numpy as np
from qibo import gates
from qibo.models import Circuit


class State:
    def __init__(self, nqubits):
        """
        Initializes State classß
        - nqubits: number of qubits
        - quantum register: object to hold quantum information
        - classical register: object to hold classical information
        - circuit_name: circuit name; defined in each subclass (GHZState, HadamardState, RandomState)
        """

        self.nqubits = nqubits

        self.circuit_name = None
        self.circuit = None
        self.measurement_circuit_names = []
        self.measurement_circuits = []


    def create_measurement_circuits(self, labels, label_format="little_endian"):
        """
        Prepares measurement circuits
        - labels: string of Pauli matrices (e.g. XYZXX)
        """

        #qubits = range(self.nqubits)
        nqubits = self.nqubits

        for label in labels:

            # for aligning to the natural little_endian way of iterating through bits below
            if label_format == "big_endian":
                effective_label = label[::-1]
            else:
                effective_label = label[:]
            probe_circuit = Circuit(self.nqubits)

            for qubit, letter in zip(*[range(self.nqubits), effective_label]):
                if letter == "X":
                    probe_circuit.add(gates.H(qubit))  # H

                elif letter == "Y":
                    probe_circuit.add(gates.S(qubit).dagger())  # S^dg
                    probe_circuit.add(gates.H(qubit))  # H

            probe_circuit.add(gates.M(*range(self.nqubits)))

            measurement_circuit_name = "{}-{}".format(self.circuit_name, label)
            measurement_circuit = self.circuit + probe_circuit

            self.measurement_circuit_names.append(measurement_circuit_name)
            self.measurement_circuits.append(measurement_circuit)


    def execute_measurement_circuits(
        self,
        labels,
        num_shots=100,
        label_format="little_endian",
    ):
        """
        Executes measurement circuits
        - labels: string of Pauli matrices (e.g. XYZXX)
        - backend: 'numpy', 'qibojit', 'pytorch', 'tensorflow' prvided by qibo
        - num_shots: number of shots measurement is taken to get empirical frequency through counts
        """
        if self.measurement_circuit_names == []:
            self.create_measurement_circuits(labels, label_format)

        data_dict_list = []
        for i, label in enumerate(labels):
            result = self.measurement_circuits[i](nshots=num_shots)
            count_dict = result.frequencies()

            measurement_circuit_name = "{}-{}".format(self.circuit_name, label)

            data_dict = {
                "measurement_circuit_name": measurement_circuit_name,
                "circuit_name": self.circuit_name,
                "label": label,
                "count_dict": count_dict,
                "num_shots": num_shots,
            }
            data_dict_list.append(data_dict)
        return data_dict_list

    def get_state_vector(self):
        """
        Executes circuit by connecting to Qiskit object, and obtain state vector
        """
        if self.circuit is None:
            self.create_circuit()

        return self.circuit.execute().state()
 
    def get_state_matrix(self):
        """
        Obtain density matrix by taking an outer product of state vector
        """
        state_vector = self.get_state_vector()
        state_matrix = np.outer(state_vector, state_vector.conj())
        return state_matrix


class GHZState(State):
    """
    Constructor for GHZState class
    """

    def __init__(self, nqubits):
        State.__init__(self, nqubits)
        self.circuit_name = "GHZ"
        self.create_circuit()

    def create_circuit(self):
        circuit = Circuit(self.nqubits)

        circuit.add(gates.H(0))
        for i in range(1, self.nqubits):
            circuit.add(gates.CNOT(0, i))

        self.circuit = circuit

    

if __name__ == '__main__':

    nqubits = 3
    labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ"]

    state = GHZState(nqubits)
    data_dict_list = state.execute_measurement_circuits(labels)

    print(state.get_state_vector())
    print(state.get_state_matrix())
