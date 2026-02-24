# tests/test_update_P_emlatents.py
import numpy as np

from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators.EM.estep import estep
from mbpls_em.estimators.EM.mstep.update_P import update_P

def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.array([])
    QA, _ = np.linalg.qr(A); QB, _ = np.linalg.qr(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)

def test_update_P_with_em_latents_orthonormal():
    K = 3
    N_list = [90, 80, 70]
    d, r = 25, 2
    q_list = [2, 1, 3]

    data, true_params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=0.15, sig2eps=0.12, seed=202
    )

    E = estep(data, true_params)

    P_hat, kkt_avg = update_P(
        data=data,
        W=true_params["W"],
        mu_U_list=E["mu_U"],
        S_uu_list=E["Suu"],
        S_tu_list=[S.T for S in E["Sut"]],   # Sut is (q×r) → S_tu is (r×q)
        orthonormal=True,
        enforce_W_orthogonality=True
    )

    # checks
    for k, (Pk_hat, Pk_true) in enumerate(zip(P_hat, true_params["P"])):
        # shapes
        assert Pk_hat.shape == Pk_true.shape
        qk = Pk_true.shape[1]
        # orthonormal columns
        if qk > 0:
            Iq = np.eye(qk)
            assert np.allclose(Pk_hat.T @ Pk_hat, Iq, atol=1e-6)
            # orthogonality to W
            assert np.allclose(true_params["W"].T @ Pk_hat, 0.0, atol=1e-6)
            # subspace closeness (median angle)
            ang = principal_angles(Pk_true, Pk_hat)
            if ang.size:
                median_deg = float(np.degrees(np.median(ang)))
                assert median_deg < 20.0  # tune threshold as desired

    assert kkt_avg < 1e-6
