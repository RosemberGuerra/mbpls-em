# mstep/update_beta_phi.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple

def _chol_solve(A: np.ndarray, B: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Solve (A + ridge I) X = B for SPD A via Cholesky."""
    n = A.shape[0]
    L = np.linalg.cholesky(A + ridge*np.eye(n, dtype=A.dtype))
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return X

def update_beta_phi(
    data: List[Dict[str, np.ndarray]],
    # --- E-step route ---
    mu_T_list: List[np.ndarray] | None = None,
    mu_U_list: List[np.ndarray] | None = None,
    Stt_list:  List[np.ndarray] | None = None,
    Suu_list:  List[np.ndarray] | None = None,
    Sut_list:  List[np.ndarray] | None = None,  # (q×r)
    # --- ORACLE route ---
    T_list: List[np.ndarray] | None = None,
    U_list: List[np.ndarray] | None = None,
    # --- options ---
    ridge: float = 1e-6,     # small ridge on the block system
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Joint update of (beta_k, phi_k) for each block k by solving the (r+q_k) system:
        [Stt  Stu] [beta^T] = [mu_T^T Y]
        [Sut  Suu] [phi^T ]   [mu_U^T Y]
    with Stu = Sut^T. Adds a small ridge for stability.

    Two modes (exclusive):
      (A) E-step stats: provide mu_T/U and S-blocks;
      (B) Oracle: provide T/U (then S-blocks are built from them).

    Returns
    -------
    beta_list_hat : list of (1×r)
    phi_list_hat  : list of (1×q_k)
    kkt_avg_res   : average normalized residual of the block system:
        || S @ coef - rhs ||_2 / (||rhs||_2 + eps)
    """
    use_em = (mu_T_list is not None)
    use_oracle = (T_list is not None)
    if not (use_em ^ use_oracle):
        raise ValueError("Provide either EM stats (mu_T/U, S-blocks) OR Oracle (T/U), but not both.")

    K = len(data)
    beta_out: List[np.ndarray] = []
    phi_out: List[np.ndarray] = []
    residuals: List[float] = []

    for k in range(K):
        Yk = data[k]["Y"]  # (N×1)

        if use_em:
            mu_Tk = mu_T_list[k]  # (N×r)
            mu_Uk = mu_U_list[k]  # (N×q)
            Stt_k = Stt_list[k]  # (r×r)
            Suu_k = Suu_list[k]  # (q×q)
            Sut_k = Sut_list[k]  # (q×r)
            Stu_k = Sut_k.T  # (r×q)

            rhs = np.vstack([mu_Tk.T @ Yk,  # (r×1)
                             mu_Uk.T @ Yk])  # (q×1)
        else:
            Tk = T_list[k]  # (N×r)
            Uk = U_list[k]  # (N×q)
            Stt_k = Tk.T @ Tk
            Suu_k = Uk.T @ Uk
            Sut_k = Uk.T @ Tk
            Stu_k = Sut_k.T
            rhs = np.vstack([Tk.T @ Yk,
                             Uk.T @ Yk])
        # Assemble S = [[Stt Stu],[Sut Suu]] and solve
        S = np.block([[Stt_k, Stu_k],
                      [Sut_k, Suu_k]])  # ((r+q)×(r+q))
        coef = _chol_solve(S, rhs, ridge=ridge)  # ((r+q)×1)

        r = Stt_k.shape[0]
        beta_k = coef[:r].T  # (1×r)
        phi_k = coef[r:].T  # (1×q)
        # KKT residual (normalized)
        res = float(np.linalg.norm(S @ coef - rhs) / (np.linalg.norm(rhs) + 1e-12))

        beta_out.append(beta_k)
        phi_out.append(phi_k)
        residuals.append(res)

    return beta_out, phi_out, float(np.mean(residuals))