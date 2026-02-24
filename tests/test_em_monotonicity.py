# tests/test_em_monotonicity.py
import numpy as np

from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators import MBPLS_EM

def test_em_loglik_monotone_ls_updates():
    """
    With LS updates (orthonormal=False) and no W-P orth constraint,
    EM should be (near) monotone non-decreasing in log-likelihood.
    """
    K = 3
    N_list = [120, 100, 90]
    d, r = 25, 3
    q_list = [2, 1, 3]

    data, true_params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=[0.15, 0.12, 0.18],
        sig2eps=[0.10, 0.12, 0.11],
        seed=2024
    )

    # pure LS updates for monotonicity; no extra constraints
    params, hist = MBPLS_EM(
        data=data, r=r, q_list=q_list,
        max_iter=60, tol=1e-5, seed=7,
        orthonormal_W=False, orthonormal_P=False, enforce_WP_orth=False,
        ridge=1e-6, damping=1.0, var_floor_factor=1e-3, verbose=False
    )

    ll = np.array(hist["loglik"])
    # non-decreasing up to tiny numerical noise
    diffs = np.diff(ll)
    assert np.all(diffs >= -1e-6), f"Log-likelihood decreased: min delta={diffs.min():.2e}"
    # and should improve overall
    assert ll[-1] > ll[0], "No overall improvement in log-likelihood"

def test_em_runs_with_orthonormal_options():
    """
    Smoke test: EM runs with orthonormal W/P and W^T P orthogonality enforced.
    (Monotonicity isn't guaranteed here, so we only check it doesn't diverge or NaN.)
    """
    K = 2
    N_list = [100, 100]
    d, r = 20, 2
    q_list = [2, 2]

    data, _, _ = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=0.15, sig2eps=0.10, seed=123
    )

    params, hist = MBPLS_EM(
        data=data, r=r, q_list=q_list,
        max_iter=40, tol=1e-5, seed=3,
        orthonormal_W=True, orthonormal_P=True, enforce_WP_orth=True,
        ridge=1e-6, damping=0.8, var_floor_factor=1e-3, verbose=False
    )

    ll = np.array(hist["loglik"])
    assert np.isfinite(ll).all(), "Log-likelihood contains non-finite values"
    assert not np.isnan(params["W"]).any(), "Parameters contain NaNs"
