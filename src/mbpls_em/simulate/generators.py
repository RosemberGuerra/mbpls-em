# Generate synthetic data for the shared–specific model with enforced constraints:
#   X_k = T_k W^T + U_k P_k^T + E_k
#   Y_k = T_k beta_k^T + U_k phi_k^T + eps_k
# Constraints enforced:
#   W^T W = I_r,  P_k^T P_k = I_{q_k},  W^T P_k = 0
# and feasibility q_k <= d - r for all k.

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------------- small helpers ----------------

def orthonormal_cols(A: np.ndarray) -> np.ndarray:
    """
    Return an orthonormal basis for the column space of A (thin QR).
    If A has shape (d, m), the returned matrix has shape (d, rank).
    """
    if A.size == 0:
        return A
    Q, R = np.linalg.qr(A)
    # Ensure deterministic sign (optional): flip columns so diag(R) >= 0
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q[:, :A.shape[1]] * signs[:A.shape[1]]
    return Q

def project_orth(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Project columns of A onto the orthogonal complement of span(B).
    If B has orthonormal columns (B^T B = I), projection is A - B (B^T A).
    If B is empty (shape (d,0)), returns A unchanged.
    """
    if B.size == 0:
        return A
    return A - B @ (B.T @ A)

# ---------------- validation utilities ----------------

def assert_constraints(W: np.ndarray, P_list: List[np.ndarray],
                       rtol: float = 1e-6, atol: float = 1e-8) -> None:
    """
    Raise AssertionError if constraints are violated beyond tolerance.
    Checks: W^T W = I_r; P_k^T P_k = I_{q_k}; W^T P_k = 0
    """
    d, r = W.shape
    I_r = np.eye(r, dtype=W.dtype)
    assert np.allclose(W.T @ W, I_r, rtol=rtol, atol=atol), "W^T W != I_r"

    for k, Pk in enumerate(P_list):
        qk = Pk.shape[1]
        I_q = np.eye(qk, dtype=W.dtype)
        assert np.allclose(Pk.T @ Pk, I_q, rtol=rtol, atol=atol), f"P[{k}]^T P[{k}] != I"
        Z = W.T @ Pk
        assert np.allclose(Z, 0.0, rtol=rtol, atol=atol), f"W^T P[{k}] not ~ 0"

# ---------------- generator ----------------

def generate_multiblock_mbpls(
    K: int,
    N_list: List[int],
    d: int,
    r: int,
    q_list: List[int],
    sig2e: float | List[float] = 0.1,
    sig2eps: float | List[float] = 0.1,
    seed: Optional[int] = None,
    shared_beta: bool = False,
    shared_phi: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, object], Dict[str, object]]:
    """
    Generate synthetic data for the shared–specific model with orthogonality constraints.

    Parameters
    ----------
    K : int
        Number of blocks/datasets.
    N_list : list[int]
        Sample size per block (length K).
    d : int
        Number of variables per X_k.
    r : int
        Shared rank (columns of W).
    q_list : list[int]
        Specific ranks per block (length K). Must satisfy q_k <= d - r.
    sig2e : float or list[float]
        Noise variance(s) for X. Scalar applies to all blocks.
    sig2eps : float or list[float]
        Noise variance(s) for Y. Scalar applies to all blocks.
    seed : int or None
        RNG seed.
    shared_beta : bool
        If True, use the same beta across blocks (beta_k = beta for all k).
    shared_phi : bool
        If True, use the same phi across blocks (phi_k = phi for all k, with common q if all q_k equal).

    Returns
    -------
    data : list[dict]
        Per block: {'X': (N_k×d), 'Y': (N_k×1)}.
    params_true : dict
        Ground-truth parameters:
            - 'W': (d×r)
            - 'P': list of (d×q_k)
            - 'beta': list of (1×r)
            - 'phi': list of (1×q_k)
            - 'sig2e': list of float
            - 'sig2eps': list of float
    latents : dict
        Generated latent variables:
            - 'T': list of (N_k×r)
            - 'U': list of (N_k×q_k)
    """
    # --- checks ---
    assert len(N_list) == K, "N_list length must be K"
    assert len(q_list) == K, "q_list length must be K"
    for qk in q_list:
        assert qk <= d - r, f"Infeasible ranks: need q_k <= d - r, got q_k={qk}, d={d}, r={r}"

    # --- RNG and variance lists ---
    rng = np.random.default_rng(seed)
    if np.isscalar(sig2e):
        sig2e_list = [float(sig2e)] * K
    else:
        assert len(sig2e) == K
        sig2e_list = [float(v) for v in sig2e]
    if np.isscalar(sig2eps):
        sig2eps_list = [float(sig2eps)] * K
    else:
        assert len(sig2eps) == K
        sig2eps_list = [float(v) for v in sig2eps]

    # --- shared loadings W (orthonormal) ---
    W0 = rng.normal(0, 1, size=(d, r))
    W = orthonormal_cols(W0)

    # --- block-specific loadings P_k: orthogonal to W, orthonormal columns ---
    P_list: List[np.ndarray] = []
    for k in range(K):
        qk = q_list[k]
        if qk == 0:
            P_list.append(np.zeros((d, 0)))
            continue
        Rk = rng.normal(0, 1, size=(d, qk))
        Rk = project_orth(Rk, W)            # remove any component in span(W)
        Pk = orthonormal_cols(Rk)           # orthonormalize columns
        # Safety: in rare degenerate draws Rk might be rank-deficient; re-draw if needed
        if Pk.shape[1] != qk:
            # Try a few times
            for _ in range(5):
                Rk = rng.normal(0, 1, size=(d, qk))
                Rk = project_orth(Rk, W)
                Pk = orthonormal_cols(Rk)
                if Pk.shape[1] == qk:
                    break
            assert Pk.shape[1] == qk, "Failed to generate full-rank P_k"
        P_list.append(Pk)

    # --- beta and phi ---
    if shared_beta:
        beta0 = rng.normal(0, 1, size=(1, r))
        beta_list = [beta0.copy() for _ in range(K)]
    else:
        beta_list = [rng.normal(0, 1, size=(1, r)) for _ in range(K)]

    if shared_phi:
        # Only makes sense if all q_k are equal
        assert len(set(q_list)) == 1, "shared_phi=True requires all q_k equal"
        q_common = q_list[0]
        phi0 = rng.normal(0, 1, size=(1, q_common))
        phi_list = [phi0.copy() for _ in range(K)]
    else:
        phi_list = [rng.normal(0, 1, size=(1, qk)) if qk > 0 else np.zeros((1, 0))
                    for qk in q_list]

    # --- latents and observations ---
    data: List[Dict[str, np.ndarray]] = []
    T_list: List[np.ndarray] = []
    U_list: List[np.ndarray] = []

    for k in range(K):
        N_k = N_list[k]
        qk = q_list[k]
        T_k = rng.normal(0, 1, size=(N_k, r))
        U_k = rng.normal(0, 1, size=(N_k, qk)) if qk > 0 else np.zeros((N_k, 0))

        X_sig = T_k @ W.T + (U_k @ P_list[k].T if qk > 0 else 0.0)
        Y_sig = T_k @ beta_list[k].T + (U_k @ phi_list[k].T if qk > 0 else 0.0)

        E_k = rng.normal(0, np.sqrt(sig2e_list[k]), size=(N_k, d))
        eps_k = rng.normal(0, np.sqrt(sig2eps_list[k]), size=(N_k, 1))

        X_k = X_sig + E_k
        Y_k = Y_sig + eps_k

        data.append({'X': X_k, 'Y': Y_k})
        T_list.append(T_k)
        U_list.append(U_k)

    # --- final checks on constraints ---
    assert_constraints(W, P_list)

    params = dict(W=W, P=P_list, beta=beta_list, phi=phi_list,
                  sig2e=sig2e_list, sig2eps=sig2eps_list)
    latents = dict(T=T_list, U=U_list)
    return data, params, latents

def validate_params(data: List[Dict[str, np.ndarray]],
                    params: Dict[str, object],
                    rtol: float = 1e-6, atol: float = 1e-8) -> None:
    """
    Basic sanity checks on shapes and constraints against provided data.
    Raises AssertionError on mismatch.
    """
    K = len(data)
    d = data[0]['X'].shape[1]
    W: np.ndarray = params['W']
    P_list: List[np.ndarray] = params['P']
    beta_list: List[np.ndarray] = params['beta']
    phi_list: List[np.ndarray] = params['phi']
    sig2e: List[float] = params['sig2e']
    sig2eps: List[float] = params['sig2eps']

    # Shapes
    assert W.shape[0] == d, "W has wrong number of rows"
    for k in range(K):
        Xk = data[k]['X']; Yk = data[k]['Y']; Pk = P_list[k]
        assert Xk.shape[1] == d, "X_k has wrong number of columns"
        assert Yk.shape[1] == 1, "Y_k must be (N_k × 1)"
        assert beta_list[k].shape == (1, W.shape[1]), "beta_k shape mismatch"
        assert phi_list[k].shape == (1, Pk.shape[1]), "phi_k shape mismatch"
    assert len(sig2e) == K and len(sig2eps) == K, "variance lists must be length K"

    # Constraints
    assert_constraints(W, P_list, rtol=rtol, atol=atol)