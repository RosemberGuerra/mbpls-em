# tests/test_data_constraints.py
from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.simulate.generators import validate_params

def test_generator_constraints_hold():
    K = 3
    N_list = [50, 60, 40]
    d, r = 20, 3
    q_list = [2, 1, 4]  # all <= d - r (17)

    data, params, latents = generate_multiblock_mbpls(
        K=K, N_list=N_list, d=d, r=r, q_list=q_list,
        sig2e=0.2, sig2eps=0.1, seed=123
    )

    # This will raise AssertionError if constraints are violated
    validate_params(data, params)

    # sanity: shapes
    assert params['W'].shape == (d, r)
    assert len(params['P']) == K
    for k in range(K):
        assert params['P'][k].shape == (d, q_list[k])
