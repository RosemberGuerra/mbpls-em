# Closed-form update for W using ORACLE latents (true T_k, U_k) and true P_k.
# Formula:
#  RHS = (Σ_k X_kᵀ T_k  −  Σ_k P_k (U_kᵀ T_k)) = UDVt (SVD)
#   W = U
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from mbpls_em.utils import orth

def _solve_spd(A: np.ndarray, B: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    d = A.shape[0]
    L = np.linalg.cholesky(A + ridge * np.eye(d, dtype=A.dtype))
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return X

# def orth(A: np.ndarray) -> np.ndarray:
#     """
#     Orthonormalize columns via your definition:
#       A = U D V^T,  R = V D,  orth(A) = A R^{-T} = U
#     Returns U with the same column count as A (handles rank-deficiency gracefully).
#     """
#     if A.size == 0:
#         return A
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)
#     # U has shape (d, a) if A is (d, a); columns are orthonormal.
#     return U

def update_W(
    data: List[Dict[str, np.ndarray]],
    P_list: List[np.ndarray],
    mu_T_list: List[np.ndarray] | None = None,
    S_tt_list: List[np.ndarray] | None = None,
    S_ut_list: List[np.ndarray] | None = None,
    T_list: List[np.ndarray] | None = None,   # oracle option
    U_list: List[np.ndarray] | None = None,   # oracle option
    orthonormal: bool = True,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Update W either:
      - with EM stats (mu_T, S_tt, S_ut), or
      - with oracle latents (T, U), if provided,
    and optionally enforce W^T W = I_r via orth().

    Returns
    -------
    W_hat : (d × r)
    kkt_res : float
        If orthonormal=False (LS), KKT residual of normal equation:
            || W S_tt - (Σ X^T mu_T - Σ P S_ut) ||_F / ||RHS||_F.
        If orthonormal=True (Procrustes), skew-symmetry residual:
            || (W^T M) - (W^T M)^T ||_F / (||M||_F + eps).
    """
    K = len(data)
    d = data[0]["X"].shape[1]

    # Determine r from inputs
    if mu_T_list is not None:
        r = mu_T_list[0].shape[1]
    elif T_list is not None:
        r = T_list[0].shape[1]
    else:
        raise ValueError("Provide either EM stats (mu_T, S_tt, S_ut) or oracle latents (T,U).")

    M =  np.zeros((d,r), dtype=float)
    if mu_T_list is not None:
        # EM stats route
        assert S_ut_list is not None, "S_ut list required with mu_T_list"
        # Build M = Σ X^T mu_T  −  Σ P S_ut
        for k in range(K):
            Xk = data[k]["X"]
            Pk= P_list[k]
            mu_Tk = mu_T_list[k]
            S_ut_k = S_ut_list[k]
            M += Xk.T @ mu_Tk - Pk @ S_ut_k
        if orthonormal:
            W_hat = orth(M)
            # KKT residual for Procrustes: W^T M must be symmetric
            WT_M = W_hat.T @ M
            kkt_res = float(np.linalg.norm(WT_M - WT_M.T, ord="fro") /
                            (np.linalg.norm(M, ord="fro") + 1e-12))
        else:
            # LS: W S_tt = M, with S_tt = Σ S_tt,k
            assert S_tt_list is not None, "S_tt_list required for LS mode"
            S_tt = sum(S_tt_list)
            W_hat = M @ _solve_spd(S_tt, np.eye(r), ridge=ridge)
            kkt_res = float(np.linalg.norm(W_hat @ S_tt - M, ord="fro") /
                            (np.linalg.norm(M, ord="fro") + 1e-12))
    else:
        # Oracle route (T, U provided)
        assert T_list is not None and U_list is not None
        # Build M = Σ X^T T  −  Σ P (U^T T); for LS we also need Σ T^T T
        Stt = np.zeros((r, r), dtype=float)
        for k in range(K):
            Xk = data[k]["X"];
            Pk = P_list[k]
            Tk = T_list[k];
            Uk = U_list[k]
            M += Xk.T @ Tk - Pk @ (Uk.T @ Tk)
            Stt += Tk.T @ Tk
        if orthonormal:
            W_hat = orth(M)
            WT_M = W_hat.T @ M
            kkt_res = float(np.linalg.norm(WT_M - WT_M.T, ord="fro") /
                            (np.linalg.norm(M, ord="fro") + 1e-12))
        else:
            W_hat = M @ _solve_spd(Stt, np.eye(r), ridge=ridge)
            kkt_res = float(np.linalg.norm(W_hat @ Stt - M, ord="fro") /
                            (np.linalg.norm(M, ord="fro") + 1e-12))

    return W_hat, kkt_res
