
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.models.encodings import ghz_state

import re


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

    #circuit_name = 'GHZ'

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

def Test_original_measure_to_coef(circuit, circuit_name, labels, num_shots):

    #measurement_circuits, measurement_circuit_names = create_measurement_circuits(circuit, labels)
    data_dict_list = execute_measurement_circuits(circuit, labels, circuit_name, num_shots=num_shots)

    measurement_list = []
    data_dict = {data_dict_list[ii]['label']: data_dict_list[ii]['count_dict'] for ii in range(len(labels))}
    for label in labels:
        count_dict = data_dict[label]

        coef = count_dict_2_PaulCoef(label, count_dict)
        measurement_list.append(coef)

    return measurement_list


def Pauli_Measure_circuit(label):
    """ prepare the measurement circuit according to Pauli label

    Args:
        label (str): the label for Pauli measurement
                (eg)  'XIZ', 'YYI', 'XZY' for the 3 qubit measurement
    Returns:
        (circuit): the measurement circuit
    """
    posI = [m.start() for m in re.finditer('I', label)]
    posX = [m.start() for m in re.finditer('X', label)]
    posY = [m.start() for m in re.finditer('Y', label)]
    posZ = [m.start() for m in re.finditer('Z', label)]

    probe_circuit = Circuit(nqubits)
    if len(posI) > 0:
        probe_circuit.add(gates.M(*posI))
    if len(posX) > 0:
        probe_circuit.add(gates.M(*posX, basis=gates.X))
    if len(posY) > 0:
        probe_circuit.add(gates.M(*posY, basis=gates.Y))
    if len(posZ) > 0:
        probe_circuit.add(gates.M(*posZ, basis=gates.Z))

    return probe_circuit

def effective_parity(key, label):
    """Calculates the effective number of '1' in the given key

    Args:
        key (str): the measurement outcome in the form of 0 or 1
                (eg)  '101' or '110' for the 3 qubit measurement
        label (str): the label for Pauli measurement
                (eg)  'XIZ', 'YYI', 'XZY' for the 3 qubit measurement
    Returns:
        int: the number of effective '1' in the key
    """
    indices = [i for i, symbol in enumerate(label) if symbol == "I"]
    digit_list = list(key)
    for i in indices:
        digit_list[i] = "0"
    effective_key = "".join(digit_list)

    return effective_key.count("1")

def count_dict_2_PaulCoef(label, count_dict):
    """to convert the shot measurement result for the given Pauli label
        into the coefficient of the label in the Pauli operator basis

    Args:
        label (str): the label for Pauli measurement
            (eg)  'XIYZ', 'XYYI', 'ZXZY' for the 4 qubit measurement
        count_dict (dict): the shot measurement result for the label
            (eg) {'0011': 9, '0100': 7, '1001': 8, '0101': 11, '1101': 3,
                  '0001': 9, '1000': 9, '0000': 12, '1110': 19, '1111': 4,
                  '0111': 1, '1100': 2, '1010': 5, '0110': 1}

    Returns:
        (float): the coefficient in the Pauli operator basis corresponding to the label
    """
    num_shots = sum(count_dict.values())

    freq = {k: (v) / (num_shots) for k, v in count_dict.items()}
    parity_freq = {k: (-1) ** effective_parity(k, label) * v for k, v in freq.items()}
    coef = sum(parity_freq.values())
    #data_Pauli_coef = {label: coef}

    return coef

def shot_measure_to_PauliCoef(circuit, labels, num_shots):
    """ given the circuit, do the shot measurements, and convert the results into 
    the coefficients corresponding to the labels in the Pauli operator basis

    Args:
        circuit (circuit): the circuit generating the state
        labels (str): all the sampled Pauli measurement labels 
        num_shots (int): number of shots

    Returns:
        (list): the coefficients corresponding to the Pauli operator labels
    """

    measurement_list = []
    for label in labels:
        probe_circuit = Pauli_Measure_circuit(label)   
        Measure_Circ = circuit + probe_circuit

        result = Measure_Circ(nshots=num_shots)
        count_dict = result.frequencies()
    
        #Measure_Circ.draw()
        #print(count_dict)

        coef = count_dict_2_PaulCoef(label, count_dict)
        measurement_list.append(coef)

    return measurement_list


if __name__ == '__main__':

    nqubits = 3
    circuit, circuit_name = GHZstate(nqubits)
    circ_ghz = ghz_state(nqubits)

    labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ"]
    num_shots = 100

    measurement_list0a = Test_original_measure_to_coef(circuit, circuit_name, labels, num_shots)
    measurement_list0b = Test_original_measure_to_coef(circ_ghz, circuit_name, labels, num_shots)

    measurement_list1a = shot_measure_to_PauliCoef(circuit, labels, num_shots)
    measurement_list1b = shot_measure_to_PauliCoef(circ_ghz, labels, num_shots)



    statevec  = get_state_vector(circuit)
    DenMat    = get_state_matrix(circuit)
