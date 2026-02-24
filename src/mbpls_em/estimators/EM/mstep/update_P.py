# mstep/update_P.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from mbpls_em.utils import orth

def _chol_solve(A: np.ndarray, B: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Solve (A + ridge I) X = B for SPD A via Cholesky."""
    n = A.shape[0]
    L = np.linalg.cholesky(A + ridge*np.eye(n, dtype=A.dtype))
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return X

# def orth(A: np.ndarray) -> np.ndarray:
#     """Return orthonormal columns of A (thin SVD -> U)."""
#     if A.size == 0:
#         return A
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)
#     return U  # columns orthonormal

def project_orth(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Project columns of A onto the orthogonal complement of span(W).
    Assumes W has orthonormal columns (W^T W = I).
    """
    if W.size == 0:
        return A
    return A - W @ (W.T @ A)
def update_P(
    data: List[Dict[str, np.ndarray]],
    W: np.ndarray,
    # --- E-step route ---
    mu_U_list: List[np.ndarray] | None = None,
    S_uu_list: List[np.ndarray] | None = None,
    S_tu_list: List[np.ndarray] | None = None,  # S_tu = S_ut^T
    # --- ORACLE route ---
    U_list: List[np.ndarray] | None = None,
    T_list: List[np.ndarray] | None = None,
    # --- options ---
    orthonormal: bool = True,
    enforce_W_orthogonality: bool = False,
    ridge: float = 1e-8,
) -> Tuple[List[np.ndarray], float]:
    """
    Update P_k for each block.

    Two modes:
      (A) EM stats: use mu_U, S_uu, S_tu.
          LS:   Pk = (X_k^T mu_Uk - W S_tu,k) S_uu,k^{-1}
          ORTH: Pk = orth( (I - W W^T) (X_k^T mu_Uk - W S_tu,k) )
      (B) Oracle: use U_k, T_k.
          LS:   Pk = (X_k^T U_k - W (T_k^T U_k)) (U_k^T U_k)^{-1}
          ORTH: Pk = orth( (I - W W^T) (X_k^T U_k - W (T_k^T U_k)) )

    Returns
    -------
    P_list_hat : list of (d × q_k)
    kkt_avg_res : float
        Average normalized KKT residual across blocks:
          LS:   || Pk S_uu - (X^T mu_U - W S_tu) ||_F / ||RHS||_F
          ORTH: || skew(Pk^T M_k) ||_F / (||M_k||_F + eps),  M_k defined above.
    """
    K = len(data)
    d = data[0]["X"].shape[1]

    use_em = mu_U_list is not None
    use_oracle = U_list is not None

    if not (use_em ^ use_oracle):
        raise ValueError("Provide either EM stats (mu_U,S_uu,S_tu) or ORACLE latents (U,T), but not both.")

    P_out: List[np.ndarray] = []
    kkt_residuals: List[float] = []
    for k in range(K):
        Xk = data[k]["X"]  # (N×d)
        if use_em:
            mu_Uk = mu_U_list[k]  # (N×qk)
            S_uu_k = S_uu_list[k]  # (qk×qk)
            S_tu_k = S_tu_list[k]  # (r×qk)
            M_k = Xk.T @ mu_Uk - W @ S_tu_k  # (d×qk)
        else:
            Uk = U_list[k]  # (N×qk)
            Tk = T_list[k]  # (N×r)
            M_k = Xk.T @ Uk - W @ (Tk.T @ Uk)  # (d×qk)
            S_uu_k = Uk.T @ Uk
        if orthonormal:
            # Optionally project off W first, then orthonormalize
            Mk_tilde = project_orth(M_k, W) if enforce_W_orthogonality else M_k
            Pk = orth(Mk_tilde)
            # Procrustes optimality: Pk^T M_k should be symmetric
            KKT = np.linalg.norm(Pk.T @ M_k - (Pk.T @ M_k).T, ord="fro") / (np.linalg.norm(M_k, ord="fro") + 1e-12)
        else:
            # LS normal equations: Pk S_uu = M_k
            Pk = M_k @ _chol_solve(S_uu_k, np.eye(S_uu_k.shape[0]), ridge=ridge)
            # KKT residual for LS
            KKT = np.linalg.norm(Pk @ S_uu_k - M_k, ord="fro") / (np.linalg.norm(M_k, ord="fro") + 1e-12)

        P_out.append(Pk)
        kkt_residuals.append(float(KKT))

    return P_out, float(np.mean(kkt_residuals))