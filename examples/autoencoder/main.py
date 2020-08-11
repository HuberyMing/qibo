#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo.models import Circuit
from qibo import hamiltonians, gates, models
from qibo import matrices
from scipy.optimize import minimize
import argparse


def multikron(matrices):
    h = 1
    for m in matrices:
        h = np.kron(h, m)
    return h


def ising(nqubits, lamb=1.0):
    """Implements the Ising model. The function uses the Identity and the
    Z-Pauli and X-Pauli matrices, and builds the final Hamiltonian:

    .. math::
        H = \sum_{i=0}^{nqubits} sz_i sz_{i+1} +
                                    \\lamb \cdot \sum_{i=0}^{nqubits} sx_i.

    Args:
        nqubits (int): number of quantum bits.
        lamb (float): coefficient for the X component (default 1.0).

    Returns:
        ``Hamiltonian`` object for the Ising model.
    """
    eye = matrices.I
    sz = matrices.Z
    sx = matrices.X
    ising = sum(multikron((sz if i in {j % nqubits, (j+1) % nqubits} else eye
                           for j in range(nqubits))) for i in range(nqubits))
    ising += lamb * sum(multikron((sx if i == j % nqubits else eye
                                   for j in range(nqubits))) for i in range(nqubits))
    return hamiltonians.Hamiltonian(nqubits, ising)


def encoder_hamiltonian(nqubits, ncompress):
    """Implements the Encoding hamiltonian. The function uses the Identity and
    Z-Pauli matrices:

    .. math::
        H = 1/2 * \sum_{i=0}^{ncompress} 1 - sz_i

    Args:
        nqubits (int): number of quantum bits.
        ncompress (int): number of compressed/discarded qubits.

    Returns:
        ``Hamiltonian`` object for the Encoding hamiltonian.
    """
    eye = matrices.I
    sz = matrices.Z
    encoder = sum(multikron((sz if i == j else eye
                             for j in range(nqubits))) for i in range(ncompress))
    encoder += ncompress * np.eye(2 ** nqubits, dtype=eye.dtype)
    encoder /= 2
    return hamiltonians.Hamiltonian(nqubits, encoder)


def main(nqubits, layers, compress, lambdas):

    def cost_function(params, count):
        """Evaluates the cost function to be minimized.

        Args:
            params (array or list): values of the parameters.

        Returns:
            Value of the cost function.
        """
        circuit = Circuit(nqubits)

        for l in range(layers):
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            for q in range(0, nqubits-1, 2):
                circuit.add(gates.CZ(q, q+1))
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            for q in range(1, nqubits-2, 2):
                circuit.add(gates.CZ(q, q+1))
            circuit.add(gates.CZ(0, nqubits-1))
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=0))

        cost = 0
        # this will change all thetas to the appropriate values
        circuit.set_parameters(params)
        for i in range(len(ising_groundstates)):
            final_state = circuit(np.copy(ising_groundstates[i]))
            cost += encoder.expectation(final_state).numpy().real

        if count[0] % 50 == 0:
            print(count[0], cost/len(ising_groundstates))
        count[0] += 1

        return cost/len(ising_groundstates)

    nparams = 2 * nqubits * layers + nqubits
    initial_params = np.random.uniform(0, 2*np.pi, nparams)
    encoder = encoder_hamiltonian(nqubits, compress)

    ising_groundstates = []
    for lamb in lambdas:
        ising_ham = ising(nqubits, lamb=lamb)
        ising_groundstates.append(ising_ham.eigenvectors()[0])

    count = [0]
    result = minimize(lambda p: cost_function(p, count), initial_params,
                      method='L-BFGS-B', options={'maxiter': 2.0e3, 'maxfun': 2.0e3})

    print('Final parameters: ', result.x)
    print('Final cost function: ', result.fun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--compress", default=2, type=int)
    parser.add_argument(
        "--lambdas", default=[0.9, 0.95, 1.0, 1.05, 1.10], type=list)
    args = parser.parse_args()
    main(**vars(args))
