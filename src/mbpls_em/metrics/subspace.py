from __future__ import  annotations
import numpy as np
from typing import  List, Dict, Tuple, Optional

def principal_angles(A:np.dnarray, B:np.dnarray) -> np.ndarray:
    """
    Principal angles(radians) between the culumns space of A and B.
    return an array of lenght min(rank(A),rank(B))
    :param A:
    :param B:
    :return:
    """
    if A.size == 0 or B.size == 0:
        return np.array([])
    QA, _ = np.linalg.qr(A) # orthogonal bases
    QB, _ = np.linalg.qr(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s,0,1)
    return np.arccos(s)


def subspace_angle_stats(
        W_true:np.ndarray ,W_hat:np.ndarray,
        P_true_list: list[np.ndarray], P_hat_list: list[np.ndarray]
) -> Dict[str,object]:
    """
    Compute principal-angle summaries (in degrees) for shared W and each P_k.
    Returns mean and max angle for W, and lists per block for P_k.

    """
    # W
    ang_W = principal_angles(W_true,W_hat)
    ang_W_deg = np.degrees(ang_W) if ang_W.size else np.array([])
    mean_angle_W = float(np.mean(ang_W_deg)) if ang_W_deg.size else np.array([])
    max_angle_W = float(np.max(ang_W_deg)) if ang_W_deg.size else np.array([])

    # P
    mean_angle_Pk = []
    max_angle_Pk = []
    per_block_angles_deg = []
    for P_t, P_h in zip(P_true_list,P_hat_list):
        ang = principal_angles(P_t,P_h)
        ang_deg = np.degrees(ang) if ang.size else np.array([])
        per_block_angles_deg.append(ang_deg.tolist())
        if ang_deg.size:
            mean_angle_Pk.append(float(np.mean(ang_deg)))
            max_angle_Pk.append(float(np.max(ang_deg)))
        else:
            mean_angle_Pk.append(0.0)
            max_angle_Pk.append(0.0)
    return dict(
        W_mean_deg = mean_angle_W, W_max_deg = max_angle_W,
        P_mean_deg_list = mean_angle_Pk, P_max_deg_list = max_angle_Pk,
        P_angles_deg_per_block=per_block_angles_deg
    )