# mstep/update_sigmas.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional

def _ssq(A: np.ndarray) -> float:
    return float(np.sum(A*A))

def update_sigmas(
    data: List[Dict[str, np.ndarray]],
    # model parameters (current W, P, beta, phi)
    W: np.ndarray,
    P_list: List[np.ndarray],
    beta_list: List[np.ndarray],
    phi_list: List[np.ndarray],
    # --- E-step route (preferred inside EM) ---
    mu_T_list: Optional[List[np.ndarray]] = None,
    mu_U_list: Optional[List[np.ndarray]] = None,
    Stt_list:  Optional[List[np.ndarray]] = None,
    Suu_list:  Optional[List[np.ndarray]] = None,
    Sut_list:  Optional[List[np.ndarray]] = None,   # (q×r)
    # --- ORACLE route (sanity check) ---
    T_list: Optional[List[np.ndarray]] = None,
    U_list: Optional[List[np.ndarray]] = None,
    # variance floors / caps
    min_var_e: Optional[float] = None,
    min_var_eps: Optional[float] = None,
    floor_factor: float = 1e-2,   # fallback: 1% of empirical variance
    max_var: float = 1e6
) -> Tuple[List[float], List[float]]:
    """
    Update noise variances per block.

    EM-stat route (expected residual MS):
      sigma^2_{e,k}   = (1 / (N_k d)) E[ ||X_k - T_k W^T - U_k P_k^T||_F^2 | data ]
      sigma^2_{eps,k} = (1 /  N_k   ) E[ ||Y_k - T_k beta_k^T - U_k phi_k^T||_2^2 | data ]
      where expectations use (mu_T, mu_U, Stt, Suu, Sut).

    Oracle route (point residual MS, for checks):
      same formulas but using true T_k, U_k (no expectations).

    Floors:
      If min_var_* is None, we use floor_factor * (empirical variance of X_k or Y_k).
      Results are clipped to [min_var_*, max_var].

    Returns
    -------
    sig2e_list, sig2eps_list : list[float]
    """
    K = len(data)
    d = data[0]["X"].shape[1]

    use_em = (mu_T_list is not None)
    use_oracle = (T_list is not None)
    if not (use_em ^ use_oracle):
        raise ValueError("Provide either EM stats (mu_T/U & S-blocks) or ORACLE latents (T/U), but not both.")

    sig2e_out: List[float] = []
    sig2eps_out: List[float] = []

    for k in range(K):
        Xk = data[k]["X"]    # (N×d)
        Yk = data[k]["Y"]    # (N×1)
        N  = Xk.shape[0]
        Pk = P_list[k]
        betak = beta_list[k]
        phik  = phi_list[k]

        # adaptive floors from data if not given
        floor_e   = min_var_e   if min_var_e   is not None else floor_factor * (_ssq(Xk) / (N*d))
        floor_eps = min_var_eps if min_var_eps is not None else floor_factor * (_ssq(Yk) /  N   )

        if use_em:
            mu_Tk = mu_T_list[k]      # (N×r)
            mu_Uk = mu_U_list[k]      # (N×q)
            Stt_k = Stt_list[k]       # (r×r)
            Suu_k = Suu_list[k]       # (q×q)
            Sut_k = Sut_list[k]       # (q×r)

            # Expected X residual sum of squares (expanded)
            term_X = (
                _ssq(Xk)
                - 2.0 * np.trace(W.T   @ (Xk.T @ mu_Tk))
                - 2.0 * np.trace(Pk.T  @ (Xk.T @ mu_Uk))
                + np.trace((W.T @ W)   @ Stt_k)
                + 2.0 * np.trace((W.T @ Pk) @ Sut_k)
                + np.trace((Pk.T @ Pk) @ Suu_k)
            )
            sig2e_k = term_X / (N * d)

            # Expected Y residual sum of squares
            term_Y = (
                _ssq(Yk)
                - 2.0 * np.trace(betak.T @ (Yk.T @ mu_Tk))
                - 2.0 * np.trace(phik.T  @ (Yk.T @ mu_Uk))
                + np.trace((betak.T @ betak) @ Stt_k)
                + 2.0 * np.trace((betak.T @ phik) @ Sut_k)  # careful: beta^T phi * S_tu
                + np.trace((phik.T  @ phik)  @ Suu_k)
            )

            sig2eps_k = term_Y / N

        else:
            Tk = T_list[k]
            Uk = U_list[k]
            X_hat = Tk @ W.T + Uk @ Pk.T
            Y_hat = Tk @ betak.T + Uk @ phik.T

            sig2e_k   = _ssq(Xk - X_hat) / (N * d)
            sig2eps_k = _ssq(Yk - Y_hat) / N

        # clip to safe interval
        sig2e_k   = float(np.clip(sig2e_k,   floor_e,   max_var))
        sig2eps_k = float(np.clip(sig2eps_k, floor_eps, max_var))

        sig2e_out.append(sig2e_k)
        sig2eps_out.append(sig2eps_k)

    return sig2e_out, sig2eps_out
