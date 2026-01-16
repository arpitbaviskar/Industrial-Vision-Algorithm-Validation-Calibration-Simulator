import numpy as np
import os


def load_camera_model(calib_dir):
    """
    Load camera intrinsic matrix and distortion coefficients
    produced by Day 1 calibration.

    Returns
    -------
    K : np.ndarray
        Camera intrinsic matrix
    D : np.ndarray
        Distortion coefficients
    """

    K_path = os.path.join(calib_dir, "camera_matrix.npy")
    D_path = os.path.join(calib_dir, "dist_coeffs.npy")

    if not os.path.exists(K_path) or not os.path.exists(D_path):
        raise FileNotFoundError(
            "Camera calibration files not found. Run Day 1 calibration first."
        )

    K = np.load(K_path)
    D = np.load(D_path)

    return K, D
