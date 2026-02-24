# tests/test_update_W_emlatents.py
import numpy as np

from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators.EM.estep import estep
from mbpls_em.estimators.EM.mstep.update_W import update_W

def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column spaces of A and B."""
    if A.size == 0 or B.size == 0:
        return np.array([])
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)

def test_update_W_with_em_latents_orthonormal():
    # Problem size (moderate SNR via generator variances)
    K = 2
    N_list = [100, 80]
    d, r = 30, 3
    q_list = [2, 1]

    data, true_params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=0.15, sig2eps=0.10, seed=123
    )

    # E-step with true parameters
    E = estep(data, true_params)

    # Update W using EM stats (Procrustes/orthonormal version)
    W_hat, procrustes_res = update_W(
        data=data,
        P_list=true_params["P"],
        mu_T_list=E["mu_T"], S_ut_list=E["Sut"], S_tt_list=E["Stt"],
        orthonormal=True
    )

    # Checks
    # 1) Orthonormal columns
    I_r = np.eye(r)
    assert np.allclose(W_hat.T @ W_hat, I_r, atol=1e-6), "W_hat^T W_hat not ~ I"

    # 2) Procrustes KKT residual small
    assert procrustes_res < 1e-6, f"Procrustes KKT residual too large: {procrustes_res:.2e}"

    # 3) Subspace close to true W (angles small)
    angles = principal_angles(true_params["W"], W_hat)
    mean_deg = float(np.degrees(np.mean(angles)))
    # For these sizes/SNR, expect mean angle < ~15 degrees (tune as needed)
    assert mean_deg < 15.0, f"Shared subspace too far: mean angle {mean_deg:.1f}°"
