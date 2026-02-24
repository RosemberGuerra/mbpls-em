import numpy as np
from typing import  List, Dict, Tuple, Optional
from mbpls_em.estimators.EM.estep import estep

# relative Frobenius errors (handle zero norms)
def relF(A, B):
    denom = float(np.linalg.norm(A, ord='fro')) + 1e-12
    return float(np.linalg.norm(A - B, ord='fro') / denom)

def reconstruction_metrics(
        data: List[Dict[str, np.array]],
        params: Dict[str, object],
        estep_out: Optional[Dict[str,object]] = None
) -> Dict[str,object]:
    """
       Compute reconstruction MSE and R^2 for X and Y using current params.

       If estep_out is not provided, we call estep(data, params).

       Returns
       -------
       {
         "mse_X": float, "mse_Y": float,
         "R2_X_blocks": List[float], "R2_Y_blocks": List[float],
         "n_entries_X": int, "n_entries_Y": int
       }
       """
    # Modify this part when the estimates are included !!!
    if estep_out is None:
        E = estep(data,params)
    else:
        E = estep_out

    W = params["W"]
    P_list = params["P"]
    beta = params["beta"]
    phi = params["phi"]

    sse_x_total = 0
    sse_y_total = 0
    nx_total = 0
    ny_total = 0
    R2_x: list = []
    R2_y: list =  []

    K = len(data)
    for k in range(K):

        # define the parametes at k
        Xk = data[k]["X"]
        Yk = data[k]["Y"]
        Pk = P_list[k]
        Tk = E["mu_T"][k]
        Uk = E["mu_U"][k]
        beta_k = beta[k]
        phi_k = phi[k]

        assert Xk.shape[1] == W.shape[0] == Pk.shape[0]
        assert E["mu_T"][k].shape[1] == W.shape[1] == beta_k.shape[1]
        assert E["mu_U"][k].shape[1] == Pk.shape[1] == phi_k.shape[1]
        assert Yk.shape[1] == beta_k.shape[0] == phi_k.shape[0]

        Xk_hat = Tk @ W.T + Uk @ Pk.T
        Yk_hat = Tk @ beta_k.T + Uk @ phi_k.T

        # MSE
        sse_x_k =  float(np.linalg.norm(Xk-Xk_hat,ord="fro")**2)
        sse_y_k =  float(np.linalg.norm(Yk-Yk_hat,ord="fro")**2)
        sst_x_k = float(np.linalg.norm(Xk, ord= "fro")**2)
        sst_y_k = float(np.linalg.norm(Yk, ord= "fro")**2)


        # R^2

        R2_x.append(1 - sse_x_k/sst_x_k)
        R2_y.append( 1 - sse_y_k/sst_y_k)

        sse_x_total += sse_x_k
        sse_y_total += sse_y_k
        nx_total += Xk.size
        ny_total += Yk.size

    sse_x = sse_x_total/max(1,nx_total)
    sse_y = sse_y_total/max(1,ny_total)

    return {
        "mse_X":sse_x,
        "mse_Y":sse_y,
        "R2_X_blocks": R2_x, "R2_Y_blocks": R2_y,
        "n_entries_X": nx_total, "n_entries_Y": ny_total

    }