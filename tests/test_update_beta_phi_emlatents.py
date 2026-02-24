# tests/test_update_beta_phi_emlatents.py
import numpy as np
from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators.EM.estep import estep
from mbpls_em.estimators.EM.mstep.update_beta_phi import update_beta_phi


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    num = float(u @ v.T)  # both are (1×r) row vectors
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return abs(num / den)

def test_update_beta_phi_with_em_latents():
    # Moderate size/SNR
    K = 3
    N_list = [120, 100, 80]
    d, r = 25, 2
    q_list = [2, 1, 3]

    data, true_params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=0.15, sig2eps=0.12, seed=314
    )

    # E-step with true parameters
    E = estep(data, true_params)

    beta_hat, phi_hat, kkt_avg = update_beta_phi(
        data=data,
        mu_T_list=E["mu_T"], mu_U_list=E["mu_U"],
        Stt_list=E["Stt"],   Suu_list=E["Suu"], Sut_list=E["Sut"],
        ridge=1e-6
    )

    # Checks per block
    for k in range(K):
        b_true = true_params["beta"][k]  # (1×r)
        f_true = true_params["phi"][k]   # (1×q)
        b_hat  = beta_hat[k]
        f_hat  = phi_hat[k]

        # Cosine similarity should be reasonably high (tune threshold if needed)
        assert cosine_sim(b_true, b_hat) > 0.8
        if f_true.shape[1] > 0:
            assert cosine_sim(f_true, f_hat) > 0.8

    # KKT residual small
    assert kkt_avg < 1e-6
