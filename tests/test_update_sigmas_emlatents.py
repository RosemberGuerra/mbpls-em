# tests/test_update_sigmas_emlatents.py

from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators.EM.estep import estep
from mbpls_em.estimators.EM.mstep.update_sigmas import update_sigmas

def test_update_sigmas_with_em_latents():
    K = 3
    N_list = [200, 150, 120]   # decent size → tighter variance recovery
    d, r = 25, 3
    q_list = [2, 1, 3]

    true_sig2e   = [0.20, 0.10, 0.15]
    true_sig2eps = [0.08, 0.12, 0.10]

    data, true_params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=true_sig2e, sig2eps=true_sig2eps, seed=99
    )

    # E-step with true parameters
    E = estep(data, true_params)

    # Variance updates using EM stats and true W,P,beta,phi
    sig2e_hat, sig2eps_hat = update_sigmas(
        data=data,
        W=true_params["W"], P_list=true_params["P"],
        beta_list=true_params["beta"], phi_list=true_params["phi"],
        mu_T_list=E["mu_T"], mu_U_list=E["mu_U"],
        Stt_list=E["Stt"], Suu_list=E["Suu"], Sut_list=E["Sut"],
        min_var_e=None, min_var_eps=None, floor_factor=1e-3   # mild floor
    )

    # Relative errors should be modest (tune thresholds as needed)
    for k in range(K):
        rel_e   = abs(sig2e_hat[k]   - true_sig2e[k])   / true_sig2e[k]
        rel_eps = abs(sig2eps_hat[k] - true_sig2eps[k]) / true_sig2eps[k]
        assert rel_e   < 0.25, f"sig2e block {k} off: {sig2e_hat[k]} vs {true_sig2e[k]}"
        assert rel_eps < 0.25, f"sig2eps block {k} off: {sig2eps_hat[k]} vs {true_sig2eps[k]}"
