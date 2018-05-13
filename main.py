from numpy import linalg as LA
import numpy as np
import math

def nearest_eigen_value(B):
    # 2 posledni hodnoty diagonala + hodnota nad tim
    fn2 = B[-3, -2]
    dn1 = B[-2, -2]
    fn1 = B[-2, -1]
    dn = B[-1, -1]

    # B[-3:-1,-2:-1]
    # B[-2::1,-1::1]

    df = dn ** 2 + fn1 ** 2
    T = np.matrix([[dn1 ** 2 + fn2 ** 2, dn1 * fn1], [dn1 * fn1, df]])
    # vlastni vektor
    w, v = LA.eig(T)
    # vratime to vlastni cislo, ktere je blize k df
    lambd = w[0] if abs(df - w[0]) < abs(df - w[1]) else w[1]
    return lambd


# http://drsfenner.org/blog/2016/03/givens-rotations-and-the-case-of-the-blemished-bidiagonal-matrix/
# https://web.stanford.edu/class/cme335/lecture6.pdf

# menim radek (zprava)
def compute_rotation_g1(B, lambd):
    d1, f1 = np.asarray(B[0:1, 0:2])[0]
    u = np.array([d1 ** 2 - lambd, d1 * f1])
    r = math.sqrt(u[0] ** 2 + u[1] ** 2)
    x1, x2 = u
    c = x1 / r
    s = x2 / r

    G1 = np.identity(B.shape[0])
    # v knizce to je transponovane
    G1[0, 0] = c
    G1[0, 1] = -s
    G1[1, 0] = s
    G1[1, 1] = c
    return G1


def get_u(B, i, j):
    # vytahnu sloupec [i,j]
    u = np.asarray(B[i:i + 2, j:j + 1].reshape(1, 2))[0]
    x1, x2 = u
    r = math.sqrt(x1 ** 2 + x2 ** 2)
    c = x1 / r
    s = x2 / r

    if math.isnan(c):
        c = 1
    if math.isnan(s):
        s = 0

    U = np.identity(B.shape[0])
    U[i, j] = c
    U[i, j + 1] = s
    U[i + 1, j] = -s
    U[i + 1, j + 1] = c

    return U


# menim sloupec (zprava)
def get_v(B, i, j):
    # vytahnu radek [i,j]
    u = np.asarray(B[i:i + 1, j:j + 2])[0]
    x1, x2 = u
    r = math.sqrt(u[0] ** 2 + u[1] ** 2)
    c = x1 / r
    s = x2 / r

    if math.isnan(c):
        c = 1
    if math.isnan(s):
        s = 0

    U = np.identity(B.shape[0])
    # o jedno dolu a o jedno doprava
    U[i + 1, j] = c
    U[i + 1, j + 1] = -s
    U[i + 2, j] = s
    U[i + 2, j + 1] = c

    return U


def remove_almost_zeros(M):
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            if math.fabs(M[x, y]) < 1e-12:
                M[x, y] = 0

def multiply_list_of_matrices(matrices):
    M = matrices[0]
    for i in range(1, len(matrices)):
        M = np.matmul(M, matrices[i])
    return M

def check_svd(B, U, B_, V):
    B_verify = np.matmul(U.T, B_)
    B_verify = np.matmul(B_verify, V.T)
    remove_almost_zeros(B_verify)
    # assert (np.allclose(B, B_verify))


def multiply_matrices_right_partially(m1, m2, col, n):
    for row_index in range(0, n):
        row = m1[row_index].copy()
        row[col] = np.dot(m1[row_index], m2[:,col])
        row[col+1] = np.dot(m1[row_index], m2[:,col+1])
        m1[row_index] = row

    return m1


def multiply_matrices_left_partially(m1, m2, row, n):
    for col_index in range(0, n):
        col = m2[:,col_index].copy()
        col[row] = np.dot(m1[row], m2[:,col_index])
        col[row+1] = np.dot(m1[row+1], m2[:,col_index])
        m2[:,col_index] = col

    return m2


def svd(B):
    if B.shape[0] is not B.shape[1]:
        raise Exception('B neni ctvercova matice.')

    Hl = np.identity(B.shape[0])
    Hr = np.identity(B.shape[0])

    M = B.copy()
    while True:
        original_sum_of_main_diag = sum(map(math.fabs, np.asarray(B.diagonal())))
        original_sum_of_second_diag = sum(map(math.fabs, np.asarray(B.diagonal(1))))

        Hl_s, B, Hr_s = _svd_step(B)
        Hl = np.matmul(Hl_s, Hl)
        Hr = np.matmul(Hr, Hr_s)

        check_svd(M, Hl, B, Hr)

        sum_of_main_diag = sum(map(math.fabs, np.asarray(B.diagonal())))
        sum_of_second_diag = sum(map(math.fabs, np.asarray(B.diagonal(1))))

        print(sum_of_second_diag)
        if sum_of_main_diag == original_sum_of_main_diag and \
            sum_of_second_diag == original_sum_of_second_diag or \
            sum_of_second_diag < 1e-14:
            return Hl, B, Hr

        # assert sum_of_main_diag >= original_sum_of_main_diag
        # assert sum_of_second_diag < original_sum_of_second_diag

def _svd_step(B):
    n = B.shape[0]
    lambd = nearest_eigen_value(B)
    G1 = compute_rotation_g1(B, lambd)

    print("B * G1=")
    print("{} \n*\n {}".format(B, G1))
    # B = np.matmul(B, G1)
    B = multiply_matrices_right_partially(B, G1, 0, n)
    print("B=\n{}\n".format(B))
    Hl = np.identity(B.shape[0])
    Hr = G1

    for x in range(0, n - 1):
        # print("B=\n{}".format(B))
        Un = get_u(B, x, x)
        print("U{} * B=".format(x + 1))
        print("{} \n*\n {}".format(Un, B))
        # B = np.matmul(Un, B)
        B = multiply_matrices_left_partially(Un, B, x, n)
        assert (math.fabs(B[x + 1, x]) < 1e-12)
        print("=\n{}\n".format(B))
        Hl = np.matmul(Un, Hl)

        if x == n - 2:
            break
        Vn = get_v(B, x, x + 1)
        print("B * V{}=".format(x + 2))
        print("{} \n*\n {}".format(B, Vn))
        # B = np.matmul(B, Vn)
        B = multiply_matrices_right_partially(B, Vn, x+1, n)
        # assert (math.fabs(B[x, x + 2]) < 1e-12)
        print("=\n{}\n".format(B))
        Hr = np.matmul(Hr, Vn)

    return (Hl, B, Hr)

N = 6
B = np.zeros((N, N))
np.fill_diagonal(B, range(1, N + 1))
np.fill_diagonal(B[:, 1:], range(N + 1, 2 * N))

U, B_, V = svd(B.copy())

print("B=\n{}\n".format(B))
print("B_=\n{}\n".format(B_))
print("U=\n{}\n".format(U))
print("V=\n{}".format(V))

check_svd(B, U, B_, V)

diag = sum(map(math.fabs, np.asarray(B.diagonal())))
diag_ = sum(map(math.fabs, np.asarray(B_.diagonal())))
assert (diag < diag_)

diag2 = sum(map(math.fabs, np.asarray(B.diagonal(1))))
diag2_ = sum(map(math.fabs, np.asarray(B_.diagonal(1))))
assert (diag2 > diag2_)
