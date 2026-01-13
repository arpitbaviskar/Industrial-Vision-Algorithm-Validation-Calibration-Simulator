import numpy as np 
K = np.array([
    [652.3,   0.0, 251.7],
    [0.0,   653.7, 162.9],
    [0.0,     0.0,   1.0]
], dtype=np.float64)
#(k1,k2,p1,p2) distortion coefficients
D = np.array([0.023, 1.408 , -0.00058 ,-0.00265 , -8.708], dtype=np.float32)
