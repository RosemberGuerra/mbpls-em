# E-step for the shared–specific model:
#   X_k = T_k W^T + U_k P_k^T + E_k
#   Y_k = T_k beta_k^T + U_k phi_k^T + eps_k
#
# For each block k, compute:
#   Σ_Zk = (I + Γ_k^T Σ_εk^{-1} Γ_k)^{-1}
#   μ_Zk = D_k Σ_εk^{-1} Γ_k Σ_Zk,  with D_k = [X_k  Y_k]
#   S_tt,k = N_k Σ_TT,k + μ_Tk^T μ_Tk, etc.
# and the stable marginal log-likelihood:
#   ℓ_k = -½[ N_k(d+1)log(2π) + N_k(d logσ_e^2 + logσ_ε^2 + log|I+B_k|)
#           + ||D_k Σ_εk^{-1/2}||_F^2 - ||A_k (I+B_k)^{-1/2}||_F^2 ]
# with B_k = Γ_k^T Σ_εk^{-1} Γ_k and A_k = D_k Σ_εk^{-1} Γ_k.

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple

LOG2PI = np.log(2.0 * np.pi)
def _chol_logdet(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Cholesky + log|A| for SPD A (adds no ridge; caller ensures SPD)."""
    L = np.linalg.cholesky(A)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return L, float(logdet)

def estep(
    data: List[Dict[str, np.ndarray]],
    params: Dict[str, object],
    ridge_small: float = 1e-12,
    var_floor_e: float = 1e-8,
    var_floor_eps: float = 1e-8,
) -> Dict[str, object]:
    """
    E-step: posterior means/covariances of [T_k U_k], sufficient statistics,
    and block log-likelihoods (stable).

    Parameters
    ----------
    data : list of blocks
        Each {'X': (N_k×d), 'Y': (N_k×1)}.
    params : dict
        {'W': (d×r), 'P': list(d×q_k), 'beta': list(1×r), 'phi': list(1×q_k),
         'sig2e': list float, 'sig2eps': list float }.
    ridge_small : float
        Tiny ridge added only to (I+B) to ensure Cholesky in borderline cases.
    var_floor_e, var_floor_eps : float
        Floors for variances to avoid division by tiny numbers.

    Returns
    -------
    E : dict with lists per block:
        mu_T[k]  : (N_k×r)
        mu_U[k]  : (N_k×q_k)
        Stt[k]   : (r×r)
        Suu[k]   : (q_k×q_k)
        Sut[k]   : (q_k×r)
        # (optional for debugging)
        var_T[k] : (r×r)
        var_U[k] : (q_k×q_k)
        var_UT[k]: (q_k×r)
        logl_blocks : list[float]
        logl_total  : float
    """
    W: np.ndarray = params["W"]  # (d×r)
    P_list: List[np.ndarray] = params["P"]  # (d×qk)
    beta_list: List[np.ndarray] = params["beta"]  # (1×r)
    phi_list: List[np.ndarray] = params["phi"]  # (1×qk)
    sig2e_list: List[float] = params["sig2e"]
    sig2eps_list: List[float] = params["sig2eps"]

    d, r = W.shape
    K = len(data)

    mu_T: List[np.ndarray] = []
    mu_U: List[np.ndarray] = []
    Stt: List[np.ndarray] = []
    Suu: List[np.ndarray] = []
    Sut: List[np.ndarray] = []

    var_T: List[np.ndarray] = []
    var_U: List[np.ndarray] = []
    var_UT: List[np.ndarray] = []
    logl_blocks: List[float] = []

    for k in range(K):
        Xk = data[k]["X"]  # (N×d)
        Yk = data[k]["Y"]  # (N×1)
        N = Xk.shape[0]
        Pk = P_list[k]  # (d×qk)
        qk = Pk.shape[1]
        betak = beta_list[k]  # (1×r)
        phik = phi_list[k]  # (1×qk)

        # Variances (floored)
        sig2e = float(max(sig2e_list[k], var_floor_e))
        sig2eps = float(max(sig2eps_list[k], var_floor_eps))

        # Γ_k and precision blocks
        Gamma = np.block([
            [W, Pk],
            [betak, phik],
        ])                  # (d+1)×(r+qk)

        inv_sig = np.diag(np.concatenate([
            np.full(d, 1.0 / sig2e, dtype=W.dtype),
            np.array([1.0 / sig2eps], dtype=W.dtype),
        ]))  # (d+1)×(d+1)

        # Small (r+qk) system
        Izu = np.eye(r + qk, dtype=W.dtype)
        B = Gamma.T @ inv_sig @ Gamma  # (r+qk)×(r+qk)

        # Cholesky of I+B (SPD)
        L_IB, logdet_IB = _chol_logdet(Izu + B + ridge_small * np.eye(r + qk, dtype=W.dtype))

        # Posterior covariance Σ_Z = (I+B)^{-1} via Cholesky solve of I
        I_m = np.eye(r + qk, dtype=W.dtype)
        # solve (I+B) Σ_Z = I  ⇒  first solve L y = I, then Lᵀ Σ_Z = y
        Ytmp = np.linalg.solve(L_IB, I_m)
        Sigma_Z = np.linalg.solve(L_IB.T, Ytmp)  # (r+qk)×(r+qk)

        # Posterior mean μ_Z = D Σ^{-1} Γ Σ_Z with D=[X Y]
        D_inv = np.hstack([Xk / sig2e, Yk / sig2eps])  # (N×(d+1))   uses 1/σ²
        A = D_inv @ Gamma  # (N×(r+qk))
        mu_Z = A @ Sigma_Z  # (N×(r+qk))

        mu_Tk = mu_Z[:, :r]
        mu_Uk = mu_Z[:, r:]

        # Blocks of Σ_Z
        var_Tk = Sigma_Z[:r, :r]
        var_Uk = Sigma_Z[r:, r:]
        var_UTk = Sigma_Z[r:, :r]

        # Sufficient statistics
        Stt_k = N * var_Tk + mu_Tk.T @ mu_Tk
        Suu_k = N * var_Uk + mu_Uk.T @ mu_Uk
        Sut_k = N * var_UTk + mu_Uk.T @ mu_Tk

        # Stable quadratic + determinant for LL
        D_sqrt = np.hstack([Xk / np.sqrt(sig2e), Yk / np.sqrt(sig2eps)])  # (N×(d+1)) uses 1/σ
        # ||A (I+B)^(-1/2)||_F^2 via triangular solve: Lᵀ Ũ = Aᵀ
        A_tilde = np.linalg.solve(L_IB.T, A.T).T
        quad = float(np.sum(D_sqrt * D_sqrt) - np.sum(A_tilde * A_tilde))
        det_term = d * np.log(sig2e) + np.log(sig2eps) + logdet_IB
        logl_k = -0.5 * (N * (d + 1) * LOG2PI + N * det_term + quad)

        # collect
        mu_T.append(mu_Tk)
        mu_U.append(mu_Uk)
        Stt.append(Stt_k)
        Suu.append(Suu_k)
        Sut.append(Sut_k)
        var_T.append(var_Tk)
        var_U.append(var_Uk)
        var_UT.append(var_UTk)
        logl_blocks.append(float(logl_k))

    return dict(
        mu_T=mu_T, mu_U=mu_U,
        Stt=Stt, Suu=Suu, Sut=Sut,
        var_T=var_T, var_U=var_U, var_UT=var_UT,
        logl_blocks=logl_blocks,
        logl_total=float(np.sum(logl_blocks)),
    )