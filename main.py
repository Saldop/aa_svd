from numpy import linalg as LA
import numpy as np

matrix = np.matrix([[1, 2, 5], [3, 4, 6], [5, 6, 7]])


def nearest_eigen_value(B):
    # extract right bottom 2x2 submatrix
    fn2 = B[-3, -2]
    dn1 = B[-2, -2]
    fn1 = B[-2, -1]
    dn = B[-1, -1]

    df = dn ** 2 + fn1 ** 2
    T = np.matrix([[dn1 ** 2 + fn2 ** 2, dn1 * fn1], [dn1 * fn1, df]])
    # calculate eigen values
    w, v = LA.eig(T)
    # return eigen value closer to df
    lambd = w[0] if abs(df - w[0]) < abs(df - w[1]) else w[1]
    return lambd


def compuite_rotation_g1(B, lambd):
    d1, f1 = B[0:1, 0:2]
    np.array([d1**2-lambd, d1*f1])


print('result', nearest_eigen_value(matrix))

# (u*v)/(|u|*|v|) = cos alfa