import pytest
from qibo import set_backend

from qibo.models.encodings import ghz_state

from qibo.hamiltonians import SymbolicHamiltonian

from functools import reduce

from qibo.symbols import I, X, Y, Z

symbol_map = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "I": I,
}

@pytest.mark.parametrize("label", ["ZIY", "YXY", "IXZ", "YXY", "ZIZ", "IZI", "ZII", "IIZ", "IZZ", "III"])
# @pytest.mark.parametrize("label", ["ZIY", "YXY", "IXZ", "YXY"])         # ok
# @pytest.mark.parametrize("label", ["III"])                              # acutally wrong answer
# @pytest.mark.parametrize("label", ["ZIZ", "IZI", "ZII", "IIZ", "IZZ"])  # NotImplementedError: Observable is not a Z Pauli string.
def test_exp_from_samples(label):
    """
    # label = "ZIY" #"YXY" #"IXZ" #"YXY"                  # OK
    label = "IIZ" #"ZIZ" #"IZI" #"ZII" #"IIZ" #"IZZ"      
    """

    nqubits = 3
    num_shots = 200

    stateGHZ = ghz_state(nqubits)

    qubitPauli = [symbol_map[Ps](i) for i, Ps in enumerate(label)]

    symbolPauli = SymbolicHamiltonian(reduce(lambda x, y: x * y, qubitPauli))

    from qibochem.measurement import expectation, expectation_from_samples
    coef_Pauli_exact = expectation(stateGHZ, symbolPauli)                                       # OK
    coef_Pauli_shots = expectation_from_samples(stateGHZ, symbolPauli, n_shots=num_shots)       # error from here

    print(f"symbolPauli = {symbolPauli}")
    print(f'coef_Pauli_exact  =  {coef_Pauli_exact}')
    print(f'coef_Pauli_shots  =  {coef_Pauli_shots}')



