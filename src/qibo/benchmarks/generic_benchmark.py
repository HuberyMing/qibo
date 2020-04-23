"""
Generic benchmark script that runs circuits defined in `benchmark_models.py`.

The type of the circuit is selected using the ``--type`` flag.
"""
import argparse
import os
import time
from qibo.benchmarks import utils, benchmark_models
from typing import List, Optional


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="3-10", type=str)
parser.add_argument("--nlayers", default=None, type=int)
parser.add_argument("--nshots", default=None, type=int)
parser.add_argument("--type", default="qft", type=str)
parser.add_argument("--directory", default=None, type=str)
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--compile", action="store_true")


def main(nqubits_list: List[int],
         type: str,
         nlayers: Optional[int] = None,
         nshots: Optional[int] = None,
         directory: Optional[str] = None,
         name: Optional[str] = None,
         compile: bool = False):
    """Runs benchmarks for the Quantum Fourier Transform.

    If `directory` is specified this saves an `.h5` file that contains the
    following keys:
        * nqubits: List with the number of qubits that were simulated.
        * simulation_time: List with simulation times for each number of qubits.
        * compile_time (optional): List with compile times for each number of
            qubits. This is saved only if `compile` is `True`.

    Args:
        nqubits_list: List with the number of qubits to run for.
        directory: Directory to save the log files.
            If `None` then logs are not saved.
        name: Name of the run to be used when saving logs.
            This should be specified if a directory in given. Otherwise it
            is ignored.
        compile: If `True` then the Tensorflow graph is compiled using
            `circuit.compile()`. In this case the compile time is also logged.

    Raises:
        FileExistsError if the file with the `name` specified exists in the
        given `directory`.
    """
    if directory is not None:
        if name is None:
            raise ValueError("A run name should be given in order to save "
                             "logs.")

        # Generate log file name
        log_name = [name]
        if compile:
            log_name.append("compiled")
        log_name = "{}.h5".format("_".join(log_name))
        # Generate log file path
        file_path = os.path.join(directory, log_name)
        if os.path.exists(file_path):
            raise FileExistsError("File {} already exists in {}."
                                  "".format(log_name, directory))

        print("Saving logs in {}.".format(file_path))

    # Create log dict
    logs = {"nqubits": [], "simulation_time": []}
    if compile:
        logs["compile_time"] = []

    # Set circuit type
    print("Running {} benchmarks.".format(type))
    create_circuit_func = benchmark_models.circuits[type]

    for nqubits in nqubits_list:
        if nlayers is None:
            circuit = create_circuit_func(nqubits)
        else:
            circuit = create_circuit_func(nqubits, nlayers)

        print("\nSimulating {} qubits...".format(nqubits))

        if compile:
            start_time = time.time()
            circuit.compile()
            # Try executing here so that compile time is not included
            # in the simulation time
            final_state = circuit.execute(nshots=nshots)
            logs["compile_time"].append(time.time() - start_time)

        start_time = time.time()
        final_state = circuit.execute(nshots=nshots)
        logs["simulation_time"].append(time.time() - start_time)

        logs["nqubits"].append(nqubits)

        # Write updated logs in file
        if directory is not None:
            utils.update_file(file_path, logs)

        # Print results during run
        if compile:
            print("Compile time:", logs["compile_time"][-1])
        print("Simulation time:", logs["simulation_time"][-1])


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["nqubits_list"] = utils.parse_nqubits(args.pop("nqubits"))
    main(**args)