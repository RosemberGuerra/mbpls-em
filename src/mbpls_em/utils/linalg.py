import numpy as np

def orth(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Orthonormal Procrustes / polar factor.

    For thin SVD A = U S V^T, returns U V^T, which solves:
        min_W ||A - W||_F  subject to  W^T W = I.

    Works for rectangular A (d x r) with d >= r.
    Falls back to reduced QR if A is (near) rank-deficient.
    """
    if A.size == 0:
        return A

    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # If near rank-deficient, QR guarantees orthonormal columns
    if np.any(s < eps):
        Q, _ = np.linalg.qr(A, mode="reduced")
        return Q

    return U @ Vt