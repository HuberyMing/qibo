#
#   test GHZ circuit
#

import numpy as np
from qibo import models, gates

from qibo.models import Circuit

#from qibo import Circuit, gates


def create_measurement_circuits(circuit, labels, label_format="little_endian"):
    """
    Prepares measurement circuits
    - labels: string of Pauli matrices (e.g. XYZXX)
    """

    circuit_name = 'GHZ'
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

def sort_sample(sample) -> dict:
    """
    Sort raw measurements into count dictionary
    """
    sorted_sample = {}
    for shot in sample:
        s = "".join(map(str, shot))
        if s in sorted_sample:
            sorted_sample[s] += 1
        else:
            sorted_sample[s] = 1
    return sorted_sample


def execute_measurement_circuits(
    circuit,
    labels,
    backend="numpy",
    num_shots=100,
    # num_shots = 10000,
    label_format="little_endian",
):
    """
    Executes measurement circuits
    - labels: string of Pauli matrices (e.g. XYZXX)
    - backend: 'numpy', 'qibojit', 'pytorch', 'tensorflow' prvided by qibo
    - num_shots: number of shots measurement is taken to get empirical frequency through counts
    """

    circuit_name = 'GHZ'

    measurement_circuits, measurement_circuit_names = create_measurement_circuits(circuit, labels)

    #qibo.set_backend(backend)

    data_dict_list = []
    for i, label in enumerate(labels):
        #result = measurement_circuits[i].execute(nshots=num_shots)
        result = measurement_circuits[i](nshots=num_shots)
        result.samples(binary=True, registers=False)

        count_dict = sort_sample(result.to_dict()["samples"])
        count_freq = result.frequencies()
        print(' {}  ------'.format(label))
        print(count_dict)
        print(count_freq)


        measurement_circuit_name = "{}-{}".format(circuit_name, label)

        data_dict = {
            "measurement_circuit_name": measurement_circuit_name,
            "circuit_name": circuit_name,
            "label": label,
            "count_dict": count_dict,
            "count_freq": count_freq,
            "backend": "qibo: " + backend,
            "num_shots": num_shots,
        }
        data_dict_list.append(data_dict)
    return data_dict_list





example = 3
if example == 0:
    # create a circuit for N=3 qubits
    circuit = models.Circuit(3)


    # add some gates in the circuit
    circuit.add([gates.H(0), gates.X(1)])
    circuit.add(gates.RX(0, theta=np.pi/6))
    # execute the circuit and obtain the
    # final state as a tf.Tensor
    final_state = circuit()

    #circuit.add(gates.M(0, 1))         # RuntimeError: Cannot add gates to a circuit after it is executed. 
    #final_state.add(gates.M(0, 1))     # RuntimeError: Cannot add gates to a circuit after it is executed.

    circuit2 = Circuit(
        nqubits=10,
        accelerators={"/GPU:0": 1, "/GPU:1": 1}
        )
    #circuit2.add(gates.M(*range(2,8)))
    circuit2.add(gates.M(*range(10)))
    #circuit2(nshots=100)                #  NotImplementedError: numpy does not support distributed execution.

    # create a circuit with all parameters set to 0.
    c = Circuit(3)
    c.add(gates.RX(0, theta=0))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))

    # set new values to the circuit's parameters
    params = [0.123, 0.456, (0.789, 0.321)]
    c.set_parameters(params)

    # do the measurement
    c.add(gates.M(0,1,2))
    result = c(nshots=100)

    print(result.samples(binary=False))
    print(result.samples(binary=True))

    print(result.frequencies(binary=True))
    print(result.frequencies(binary=False))


elif example == 1:
    # Construct the circuit
    c = Circuit(2)
    # Add some gates
    c.add(gates.H(0))
    c.add(gates.H(1))
    # Define an initial state (optional - default initial state is |00>)
    initial_state = np.ones(4) / 2.0
    # Execute the circuit and obtain the final state
    result = c(initial_state) # c.execute(initial_state) also works
    print(result.state())
    # should print `tf.Tensor([1, 0, 0, 0])`
    print(result.state())
    # should print `np.array([1, 0, 0, 0])`

    # Add a measurement register on both qubits
    c.add(gates.M(0, 1))        #  RuntimeError: Cannot add gates to a circuit after it is executed.
    result.add(gates.M(0,1))    #  AttributeError: 'QuantumState' object has no attribute 'add'



elif example == 2:
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.H(2))
    c.add(gates.TOFFOLI(0, 1, 2))
    print(c.summary())
    # Prints
    '''
    Circuit depth = 5
    Total number of gates = 6
    Number of qubits = 3
    Most common gates:
    h: 3
    cx: 2
    ccx: 1
    '''

elif example == 3:
    c = Circuit(5)
    c.add(gates.X(0))
    c.add(gates.X(4))
    c.add(gates.M(0, 1, register_name="A"))
    c.add(gates.M(3, 4, register_name="B"))

    c.display()

    result = c(nshots=100)

    print(result.samples(binary=False))
    print(result.samples(binary=True))

    print(result.frequencies(binary=True))
    print(result.frequencies(binary=False))

    c.display()
    print()
    print(result)
    print(result.state())       #  same as  result2.to_dict()["state"]
 
    result2 = c.execute()
    state_vector = result2.to_dict()["state"]
    print(state_vector)

elif example == 4:              #  GHZ state

    n = 3

    circuit = Circuit(n)

    circuit.add(gates.H(0))
    for i in range(1, n):
        circuit.add(gates.CNOT(0, i))


    labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ"]
    #measurement_circuits, measurement_circuit_names = create_measurement_circuits(circuit, labels)
    data_dict_list = execute_measurement_circuits(circuit, labels)

    #circuit.display()
    #result = circuit(nshots=100)

elif example == 5:      # How to collapse state during measurements?
    # [Advanced examples - Qibo · v0.2.12](https://qibo.science/qibo/stable/code-examples/advancedexamples.html#collapse-examples)

    c = Circuit(1, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.H(0))
    result = c(nshots=1)
    print(result)
    # prints |+><+| if 0 is measured
    # or |-><-| if 1 is measured

elif example == 51:     # Conditioning gates on measurement outcomes

    c = Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output.symbols[0] / 4))
    result = c()

    print(result)
    print(result.state())

elif example == 6:      # How to use parametrized gates?
    # https://qibo.science/qibo/stable/code-examples/advancedexamples.html#using-density-matrices
    #

    c = Circuit(3)
    g0 = gates.RX(0, theta=0)
    g1 = gates.RY(1, theta=0)
    g2 = gates.fSim(0, 2, theta=0, phi=0)
    c.add([g0, g1, gates.CZ(1, 2), g2, gates.H(2)])

    # set new values to the circuit's parameters using a dictionary
    params = {g0: 0.123, g1: 0.456, g2: (0.789, 0.321)}
    c.set_parameters(params)
    # equivalently the parameter's can be update with a list as
    params = [0.123, 0.456, (0.789, 0.321)]
    c.set_parameters(params)
    # or with a flat list as
    params = [0.123, 0.456, 0.789, 0.321]
    c.set_parameters(params)


