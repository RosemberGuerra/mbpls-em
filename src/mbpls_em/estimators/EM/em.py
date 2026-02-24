# em.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import accumulate

# mbpls #
from mbpls_em.estimators.EM.estep import estep
from mbpls_em.estimators.EM.mstep.update_W import update_W
from mbpls_em.estimators.EM.mstep.update_P import update_P
from mbpls_em.estimators.EM.mstep.update_beta_phi import update_beta_phi
from mbpls_em.estimators.EM.mstep.update_sigmas import update_sigmas

# ---------------------------
# simple initializer
# ---------------------------

def _orthonormal_cols(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return A
    Q, _ = np.linalg.qr(A)
    return Q[:, :A.shape[1]]

def _project_orth(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    if W.size == 0:
        return A
    return A - W @ (W.T @ A)

def initialize_params(
    data: List[Dict[str, np.ndarray]],
    r: int,
    q_list: List[int],
    seed: int = 0,
    orthonormal_W: bool = True,
    orthonormal_P: bool = True,
    enforce_WP_orth: bool = True,
) -> Dict[str, object]:
    """
    Minimal initializer from data scales + random small loadings.
    Produces W (d×r), P_k (d×q_k), beta_k (1×r), phi_k (1×q_k), and variances per block.
    """
    rng = np.random.default_rng(seed)
    d = data[0]["X"].shape[1]
    K = len(data)

    # W
    X_conc = np.concatenate(tuple(d["X"] for d in data), axis=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_conc)

    component_list =  [0,r] + q_list
    cumulative_list = list(accumulate(component_list))

    # total_components = r + sum(q_list)
    pca = PCA(n_components= cumulative_list[-1])
    pca_fit = pca.fit(X_scaled)

    # W
    W0 = pca_fit.components_[:r,:]
    W = W0.T

    # P
    P: List[np.ndarray] = []
    for k in range(2,K+2): # the 0, 1 are for W
        qk = cumulative_list[k] - cumulative_list[k-1]
        if qk == 0:
            P.append(np.zeros((d, 0)))
            continue
        P0 = pca_fit.components_[cumulative_list[k-1]:cumulative_list[k],:].T


        if enforce_WP_orth:
            P0 = _project_orth(P0, W)

        P.append(P0)

    # beta, phi,  sig2e, sig2eps
    beta = []
    phi = []
    sig2e = []
    sig2eps = []
    for k in range(K):
        X_k = data[k]["X"]
        Y_K = data[k]["Y"]
        Tk = X_k @ W
        Uk = X_k @ P[k]

        scores_k = np.concatenate((Tk,Uk),axis=1)
        regr = LinearRegression()
        regr.fit(X=scores_k, y=Y_K)
        beta_k = regr.coef_[:,:r]
        if q_list[k] == 0:
            phi_k = np.zeros((1, 0))
        else:
            phi_k = regr.coef_[:,-q_list[k]:]

        # sig2e
        N_k, d = X_k.shape
        X_k_hat = Tk @ W.T + Uk @ P[k].T
        sig2e_k = (np.linalg.norm(X_k-X_k_hat, ord="fro")**2)/(N_k * d)

        # sig2eps
        Y_k_hat = regr.predict(X=scores_k)
        sig2eps_k = mean_squared_error(y_true=Y_K, y_pred=Y_k_hat)

        # append #
        beta.append(beta_k)
        phi.append(phi_k)
        sig2e.append(sig2e_k)
        sig2eps.append(sig2eps_k)
    # variances from data empirical scales
    # sig2e   = []
    # sig2eps = []
    # for k in range(K):
    #     Xk = data[k]["X"]; Yk = data[k]["Y"]
    #     Nk, d = Xk.shape
    #     sig2e.append(float(np.sum(Xk*Xk) / (Nk * d)))
    #     sig2eps.append(float(np.sum(Yk*Yk) / Nk))
    return dict(W=W, P=P, beta=beta, phi=phi, sig2e=sig2e, sig2eps=sig2eps)


# ---------------------------
# blending / damping
# ---------------------------

def _blend(A, B, alpha: float):
    """Return (1-alpha)*A + alpha*B for arrays or lists of arrays."""
    if isinstance(A, list) and isinstance(B, list):
        return [ (1.0 - alpha) * a + alpha * b for a, b in zip(A, B) ]
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return (1.0 - alpha) * A + alpha * B
    else:
        raise TypeError("Unsupported types for blending")

def _blend_scalars(a_list: List[float], b_list: List[float], alpha: float) -> List[float]:
    return [ (1.0 - alpha) * a + alpha * b for a, b in zip(a_list, b_list) ]


# ---------------------------
# EM driver
# ---------------------------

def MBPLS_EM(
    data: List[Dict[str, np.ndarray]],
    r: int,
    q_list: List[int],
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
    # update styles
    orthonormal_W: bool = False,           # set False for strict EM monotonicity
    orthonormal_P: bool = False,           # set False for strict EM monotonicity
    enforce_WP_orth: bool = False,         # can break monotonicity if True
    # numerics
    ridge: float = 1e-6,
    damping: float = 1.0,                  # in (0,1]; 1.0 = full step
    var_floor_factor: float = 1e-3,        # adaptive floors: fraction of empirical var
    min_var_e: Optional[float] = None,
    min_var_eps: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Run EM until convergence.

    Notes on monotonicity:
      - With orthonormal=False for W and P (pure LS), this is a standard EM and
        log-likelihood should be non-decreasing (up to tiny numerical noise).
      - Enforcing orthonormal columns and W^T P ≈ 0 via projection is a constrained
        optimization step and may slightly break monotonicity. Use damping < 1 if needed.

    Returns
    -------
    params : dict
        Final parameters (W, P, beta, phi, sig2e, sig2eps)
    history : dict
        'loglik': list[float], total log-likelihood per iteration
        'loglik_blocks': list[list[float]], per-block log-likelihoods
        'iters': int, number of iterations performed
    """
    # init
    params = initialize_params(
        data, r, q_list, seed=seed,
        orthonormal_W=orthonormal_W,
        orthonormal_P=orthonormal_P,
        enforce_WP_orth=enforce_WP_orth,
    )

    loglik_hist: List[float] = []
    loglik_blocks_hist: List[List[float]] = []

    for it in range(max_iter):
        # --- E-step (also gives LL evaluated at current params)
        E = estep(data, params)
        loglik_hist.append(E["logl_total"])
        loglik_blocks_hist.append(E["logl_blocks"])

        if verbose:
            print(f"[iter {it:03d}] logL={E['logl_total']:.6e}")

        # --- M-step pieces (using E-step stats)
        # W
        W_new, _ = update_W(
            data=data,
            P_list=params["P"],
            mu_T_list=E["mu_T"],
            S_ut_list=E["Sut"],
            S_tt_list=E["Stt"],
            orthonormal=orthonormal_W,
            ridge=ridge,
        )

        # P
        P_new, _ = update_P(
            data=data,
            W=W_new if damping == 1.0 else params["W"],  # small subtlety for consistency
            mu_U_list=E["mu_U"],
            S_uu_list=E["Suu"],
            S_tu_list=[S.T for S in E["Sut"]],
            orthonormal=orthonormal_P,
            enforce_W_orthogonality=enforce_WP_orth,
            ridge=ridge,
        )

        # beta, phi
        beta_new, phi_new, _ = update_beta_phi(
            data=data,
            mu_T_list=E["mu_T"], mu_U_list=E["mu_U"],
            Stt_list=E["Stt"],   Suu_list=E["Suu"], Sut_list=E["Sut"],
            ridge=ridge,
        )

        # variances
        sig2e_new, sig2eps_new = update_sigmas(
            data=data,
            W=W_new, P_list=P_new,
            beta_list=beta_new, phi_list=phi_new,
            mu_T_list=E["mu_T"], mu_U_list=E["mu_U"],
            Stt_list=E["Stt"],   Suu_list=E["Suu"], Sut_list=E["Sut"],
            min_var_e=min_var_e, min_var_eps=min_var_eps,
            floor_factor=var_floor_factor,
        )

        # --- Damping / blending
        a = float(np.clip(damping, 1e-8, 1.0))
        params["W"]       = _blend(params["W"],       W_new,       a)
        params["P"]       = _blend(params["P"],       P_new,       a)
        params["beta"]    = _blend(params["beta"],    beta_new,    a)
        params["phi"]     = _blend(params["phi"],     phi_new,     a)
        params["sig2e"]   = _blend_scalars(params["sig2e"],   sig2e_new,   a)
        params["sig2eps"] = _blend_scalars(params["sig2eps"], sig2eps_new, a)

        # --- Convergence check (needs LL at next E-step; use relative improvement proxy)
        if it > 0:
            inc = loglik_hist[-1] - loglik_hist[-2]
            # allow tiny negative dip due to numerics
            if abs(inc) < tol:
                break

    history = dict(loglik=loglik_hist, loglik_blocks=loglik_blocks_hist, iters=len(loglik_hist))
    return params, history
