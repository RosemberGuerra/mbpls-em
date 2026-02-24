import numpy as np
from typing import  List, Dict, Tuple, Optional
from mbpls_em.metrics.reconstruction import relF

def orthogonal_procrustes(A:np.ndarray, B:np.ndarray)-> np.ndarray:
    """
    Find orthogonal R minimizing ||A R - B||_F, with A,B having same # columns.
    Returns R (square, orthogonal). If A or B has fewer cols, use thin SVD.
    """
    U,_,Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    R = U @ Vt
    return R

def coef_errors_rotation_invariant(
        true_params: Dict[str,object],
        est_params: Dict[str,object]
)-> Dict[str,object]:
    """
    Rotation-invariant coefficient errors:
      - align shared part via R_W from W_est → W_true
      - for each block, align specific via R_Pk from P_est → P_true
      - compute relative Frobenius error on coefficients:
          vector Y:  beta (1×r),  phi (1×q_k)
          matrix Y:  B (C×r),     Phi (C×q_k)
    Returns per-block errors and summary averages.
    """
    W_t = true_params["W"] # (dxr)
    W_h = est_params["W"]  # (dxr)
    P_t_list = true_params["P"] # (dxq_k)
    P_h_list = est_params["P"]  # (dxq_k)

    beta_t = true_params["beta"]
    beta_h = est_params["beta"]
    phi_t = true_params["phi"]
    phi_h = est_params["phi"]

    # align W
    Rw = orthogonal_procrustes(W_t,W_h) # (rxr)

    per_block = []
    relF_beta_list = []
    relF_phi_list  = []

    for k, (Pk_t,Pk_h) in enumerate(zip(P_t_list,P_h_list)):
        # align P
        qk = Pk_t.shape[1]
        if qk > 0:
            Rp = orthogonal_procrustes(Pk_t,Pk_h)
        else:
            Rp = np.eye(0)

        # rotate estimated coeff
        Bt_t = beta_t[k]
        Bt_h = beta_h[k] @ Rw

        Ph_t = phi_t[k]
        Ph_h = phi_h[k] @ Rp

        err_beta = relF(Bt_t,Bt_h)
        err_phi = relF(Ph_t,Ph_h)

        relF_beta_list.append(err_beta)
        relF_phi_list.append(err_phi)

        per_block.append(dict(
            k=k,
            relF_beta = err_beta,
            relF_phi = err_phi
        ))
    return dict(
        per_block = per_block,
        relF_beta_mean=float(np.mean(relF_beta_list)) if relF_beta_list else 0.0,
        relF_phi_mean=float(np.mean(relF_phi_list)) if relF_phi_list else 0.0,
    )



