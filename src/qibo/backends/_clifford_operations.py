from functools import cache, reduce

import numpy as np
from scipy import sparse

name = "numpy"


def _get_rxz(symplectic_matrix, nqubits):
    return (
        symplectic_matrix[:, -1],
        symplectic_matrix[:, :nqubits],
        symplectic_matrix[:, nqubits:-1],
    )


def I(symplectic_matrix, q, nqubits):
    return symplectic_matrix


def H(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & z[:, q])
    symplectic_matrix[:, [q, nqubits + q]] = symplectic_matrix[:, [nqubits + q, q]]
    return symplectic_matrix


def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:, -1]
    xcq = symplectic_matrix[:, control_q]
    xtq = symplectic_matrix[:, target_q]
    ztq = symplectic_matrix[:, ind_zt]
    zcq = symplectic_matrix[:, ind_zc]
    symplectic_matrix[:, -1] = r ^ (xcq & ztq) & (xtq ^ ~zcq)
    symplectic_matrix[:, target_q] = xtq ^ xcq
    symplectic_matrix[:, ind_zc] = zcq ^ ztq
    return symplectic_matrix


def CZ(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-H"""
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:, -1]
    xcq = symplectic_matrix[:, control_q]
    xtq = symplectic_matrix[:, target_q]
    ztq = symplectic_matrix[:, ind_zt]
    zcq = symplectic_matrix[:, ind_zc]
    ztq_xor_xcq = ztq ^ xcq
    symplectic_matrix[:, -1] = (
        r ^ (xtq & ztq) ^ (xcq & xtq & (ztq ^ ~zcq)) ^ (xtq & ztq_xor_xcq)
    )
    z_control_q = xtq ^ zcq
    z_target_q = ztq_xor_xcq
    symplectic_matrix[:, ind_zc] = z_control_q
    symplectic_matrix[:, ind_zt] = z_target_q
    return symplectic_matrix


def S(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    r[:] = r ^ (x[:, q] & z[:, q])
    z[:, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def Z(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ ((x[:, q] & z[:, q]) ^ x[:, q] & (z[:, q] ^ x[:, q]))
    return symplectic_matrix


def X(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (z[:, q] & x[:, q])
    return symplectic_matrix


def Y(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
        r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    )
    return symplectic_matrix


def SX(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SDG(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, nqubits + q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SXDG(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & x[:, q])
    symplectic_matrix[:, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def RX(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return X(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        return SX(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        return SXDG(symplectic_matrix, q, nqubits)


def RZ(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return Z(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        return S(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        return SDG(symplectic_matrix, q, nqubits)


def RY_pi(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, [nqubits + q, q]] = symplectic_matrix[:, [q, nqubits + q]]
    return symplectic_matrix


def RY_3pi_2(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H-S-S-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, [nqubits + q, q]] = symplectic_matrix[:, [q, nqubits + q]]
    return symplectic_matrix


def RY(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return Y(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        """Decomposition --> H-S-S"""
        return RY_pi(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        """Decomposition --> H-S-S-H-S-S-H-S-S"""
        return RY_3pi_2(symplectic_matrix, q, nqubits)


def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> CNOT-CNOT-CNOT"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
        r
        ^ (x[:, control_q] & z[:, target_q] & (x[:, target_q] ^ ~z[:, control_q]))
        ^ (
            (x[:, target_q] ^ x[:, control_q])
            & (z[:, target_q] ^ z[:, control_q])
            & (z[:, target_q] ^ ~x[:, control_q])
        )
        ^ (
            x[:, target_q]
            & z[:, control_q]
            & (x[:, control_q] ^ x[:, target_q] ^ z[:, control_q] ^ ~z[:, target_q])
        )
    )
    symplectic_matrix[
        :, [control_q, target_q, nqubits + control_q, nqubits + target_q]
    ] = symplectic_matrix[
        :, [target_q, control_q, nqubits + target_q, nqubits + control_q]
    ]
    return symplectic_matrix


def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
        r
        ^ (x[:, target_q] & z[:, target_q])
        ^ (x[:, control_q] & z[:, control_q])
        ^ (x[:, control_q] & (z[:, control_q] ^ x[:, control_q]))
        ^ (
            (z[:, control_q] ^ x[:, control_q])
            & (z[:, target_q] ^ x[:, target_q])
            & (x[:, target_q] ^ ~x[:, control_q])
        )
        ^ (
            (x[:, target_q] ^ z[:, control_q] ^ x[:, control_q])
            & (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q])
            & (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q] ^ ~z[:, control_q])
        )
        ^ (x[:, control_q] & (x[:, target_q] ^ x[:, control_q] ^ z[:, control_q]))
    )
    z_control_q = x[:, target_q] ^ z[:, target_q] ^ x[:, control_q]
    z_target_q = x[:, target_q] ^ z[:, control_q] ^ x[:, control_q]
    symplectic_matrix[:, nqubits + control_q] = z_control_q
    symplectic_matrix[:, nqubits + target_q] = z_target_q
    symplectic_matrix[:, [control_q, target_q]] = symplectic_matrix[
        :, [target_q, control_q]
    ]
    return symplectic_matrix


def FSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, np.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, -np.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def CY(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> S-CNOT-SDG"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
        r
        ^ (x[:, target_q] & (z[:, target_q] ^ x[:, target_q]))
        ^ (
            x[:, control_q]
            & (x[:, target_q] ^ z[:, target_q])
            & (z[:, control_q] ^ ~x[:, target_q])
        )
        ^ ((x[:, target_q] ^ x[:, control_q]) & (z[:, target_q] ^ x[:, target_q]))
    )
    x_target_q = x[:, control_q] ^ x[:, target_q]
    z_control_q = z[:, control_q] ^ z[:, target_q] ^ x[:, target_q]
    z_target_q = z[:, target_q] ^ x[:, control_q]
    symplectic_matrix[:, target_q] = x_target_q
    symplectic_matrix[:, nqubits + control_q] = z_control_q
    symplectic_matrix[:, nqubits + target_q] = z_target_q
    return symplectic_matrix


def CRX(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CY(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Y(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return Y(symplectic_matrix, target_q, nqubits)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CZ(symplectic_matrix, control_q, target_q, nqubits)


def CRZ(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CNOT(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return X(symplectic_matrix, target_q, nqubits)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        return X(symplectic_matrix, target_q, nqubits)


def CRY(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        return CZ(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        return CRZ(symplectic_matrix, control_q, target_q, nqubits, theta)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return Z(symplectic_matrix, target_q, nqubits)


def ECR(symplectic_matrix, control_q, target_q, nqubits):
    symplectic_matrix = S(symplectic_matrix, control_q, nqubits)
    symplectic_matrix = SX(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def _exponent(x1, z1, x2, z2):
    """Helper function that computes the exponent to which i is raised for the product of the x and z paulis encoded in the symplectic matrix. This is used in _rowsum. The computation is performed parallely over the separated paulis x1[i], z1[i], x2[i] and z2[i].

    Args:
        x1 (np.array): Bits of the first x paulis.
        z1 (np.array): Bits of the first z paulis.
        x2 (np.array): Bits of the second x paulis.
        z2 (np.array): Bits of the second z paulis.

    Returns:
        (np.array): The calculated exponents.
    """
    return 2 * (x1 * x2 * (z2 - z1) + z1 * z2 * (x1 - x2)) - x1 * z2 + x2 * z1


def _rowsum(symplectic_matrix, h, i, dim, determined=False):
    """Helper function that updates the symplectic matrix by setting the h-th generator equal to the (i+h)-th one. This is done to keep track of the phase of the h-th row of the symplectic matrix (r[h]). The function is applied parallely over all the rows h and i passed.

    Args:
        symplectic_matrix (np.array): Input symplectic matrix.
        h (np.array): Indices of the rows encoding the generators to update.
        i (np.array): Indices of the rows encoding the generators to use.
        nqubits (int): Total number of qubits.

    Returns:
        (np.array): The updated symplectic matrix.
    """
    xi, zi = symplectic_matrix[i, :dim], symplectic_matrix[i, dim:-1]
    xh, zh = symplectic_matrix[h, :dim], symplectic_matrix[h, dim:-1]
    exponents = _exponent(xi, zi, xh, zh)
    ind = (
        2 * symplectic_matrix[h, -1]
        + 2 * symplectic_matrix[i, -1]
        + np.sum(exponents, axis=-1)
    ) % 4 == 0
    r = np.ones(h.shape[0], dtype=np.uint8)
    r[ind] = 0

    xi_xh = xi ^ xh
    zi_zh = zi ^ zh
    if determined:
        r = reduce(np.logical_xor, r)
        xi_xh = reduce(np.logical_xor, xi_xh)
        zi_zh = reduce(np.logical_xor, zi_zh)
        symplectic_matrix[h[0], -1] = r
        symplectic_matrix[h[0], :dim] = xi_xh
        symplectic_matrix[h[0], dim:-1] = zi_zh
    else:
        symplectic_matrix[h, -1] = r
        symplectic_matrix[h, :dim] = xi_xh
        symplectic_matrix[h, dim:-1] = zi_zh
    return symplectic_matrix


def _determined_outcome(state, q, nqubits):
    state[-1, :] = 0
    idx = (state[:nqubits, q].nonzero()[0] + nqubits).astype(np.uint)
    state = _pack_for_measurements(state, nqubits)
    state = _rowsum(
        state,
        2 * nqubits * np.ones(idx.shape, dtype=np.uint),
        idx,
        _get_packed_dim(state.shape[1]),
        True,
    )
    state = _unpack_for_measurements(state, nqubits)
    return state, state[-1, -1]


def _random_outcome(state, p, q, nqubits):
    p = p[0] + nqubits
    tmp = state[p, q].copy()
    state[p, q] = 0
    h = state[:-1, q].nonzero()[0]
    state[p, q] = tmp
    if h.shape[0] > 0:
        state = _pack_for_measurements(state, nqubits)
        state = _rowsum(
            state,
            h.astype(np.uint),
            p * np.ones(h.shape[0], dtype=np.uint),
            _get_packed_dim(state.shape[1]),
            False,
        )
        state = _unpack_for_measurements(state, nqubits)
    state[p - nqubits, :] = state[p, :]
    outcome = np.random.randint(2, size=1).item()
    state[p, :] = 0
    state[p, -1] = outcome
    state[p, nqubits + q] = 1
    return state, outcome


@cache
def _get_dim(nqubits):
    return 2 * nqubits + 1


def _get_p(state, q, nqubits):
    return state[nqubits:-1, q].nonzero()[0]


def _packbits(array, axis):
    return np.packbits(array, axis=axis)


def _unpackbits(array, axis):
    return np.unpackbits(array, axis=axis)


def _pack_for_measurements(state, nqubits):
    r, x, z = _get_rxz(state, nqubits)
    x = _packbits(x, axis=1)
    z = _packbits(z, axis=1)
    return np.hstack((x, z, r[:, None]))


@cache
def _get_pad_size(shape, nqubits):
    return int((shape - _get_dim(nqubits)) / 2) + 1


@cache
def _get_packed_dim(shape):
    return int((shape - 1) / 2)


def _unpack_for_measurements(state, nqubits):
    xz = _unpackbits(state[:, :-1], axis=1)
    padding_size = _get_pad_size(xz.shape[1], nqubits)
    x, z = xz[:, :nqubits], xz[:, nqubits + padding_size : -padding_size]
    return np.hstack((x, z, state[:, -1][:, None]))


# valid for a standard basis measurement only
def M(state, qubits, nqubits, collapse=False):
    sample = []
    if collapse:
        state = _unpackbits(state, axis=0)[: _get_dim(nqubits)]
    else:
        state = state.copy()
    for q in qubits:
        p = _get_p(state, q, nqubits)
        # random outcome, affects the state
        if len(p) > 0:
            state, outcome = _random_outcome(state, p, q, nqubits)
        # determined outcome, state unchanged
        else:
            state, outcome = _determined_outcome(state, q, nqubits)
        sample.append(outcome)
    if collapse:
        state = _packbits(state, axis=0)
    return sample


def cast(x, dtype=None, copy=False):
    if dtype is None:
        dtype = "complex128"
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=copy)
    elif sparse.issparse(x):  # pragma: no cover
        return x.astype(dtype, copy=copy)
    return np.array(x, dtype=dtype, copy=copy)


def _clifford_pre_execution_reshape(state, pack=False):
    if pack:
        state = _packbits(state, axis=0)
    return state


def _clifford_post_execution_reshape(state, nqubits):
    state = _unpackbits(state, axis=0)[: _get_dim(nqubits)]
    return state


def identity_density_matrix(nqubits, normalize: bool = True):
    state = np.eye(2**nqubits, dtype="complex128")
    if normalize is True:  # pragma: no cover
        state /= 2**nqubits
    return state
