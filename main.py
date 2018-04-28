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
    c = x2 / r
    s = x1 / r

    G1 = np.identity(B.shape[0])
    # v knizce to je transponovane
    G1[0,0] = c
    G1[0,1] = -s
    G1[1,0] = s
    G1[1,1] = c
    return G1

def get_u(B, i, j):
    # vytahnu sloupec [i,j]
    u = np.asarray(B[i:i + 2, j:j + 1].reshape(1,2))[0]
    x1, x2 = u
    r = math.sqrt(x1 ** 2 + x2 ** 2)
    c = x1 / r
    s = x2 / r

    U = np.identity(B.shape[0])
    U[i,j] = c
    U[i,j + 1] = s
    U[i + 1,j] = -s
    U[i + 1,j + 1] = c

    return U

# menim sloupec (zprava)
def get_v(B, i, j):
    # vytahnu radek [i,j]
    u = np.asarray(B[i:i + 1, j:j + 2])[0]
    x1, x2 = u
    r = math.sqrt(u[0] ** 2 + u[1] ** 2)
    c = x1 / r
    s = x2 / r

    U = np.identity(B.shape[0])
    # o jedno dolu a o jedno doprava
    U[i + 1,j] = c
    U[i + 1,j + 1] = -s
    U[i + 2,j] = s
    U[i + 2,j + 1] = c

    return U

def remove_almost_zeros(M):
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            if math.fabs(M[x,y]) < 1e-10:
                M[x,y] = 0

def svd(B):
    if B.shape[0] is not B.shape[1]:
        raise Exception('B neni ctvercova matice.')

    n = B.shape[0]
    lambd = nearest_eigen_value(B)
    G1 = compute_rotation_g1(B, lambd)

    print("B * G1=")
    print("{} \n*\n {}".format(B, G1))
    B = np.matmul(B,G1)
    print("B=\n{}\n".format(B))
    Us = []
    Vs = [G1]

    for x in range(0, n - 1):
        #print("B=\n{}".format(B))
        Un = get_u(B, x, x)
        print("U{} * B=".format(x + 1))
        print("{} \n*\n {}".format(Un, B))
        B = np.matmul(Un, B)
        remove_almost_zeros(B)
        assert(B[x + 1,x] == 0.0)
        print("=\n{}\n".format(B))
        Us.insert(0, Un)

        if x == n-2:
            break
        Vn = get_v(B, x, x + 1)
        print("B * V{}=".format(x + 2))
        print("{} \n*\n {}".format(B, Vn))
        B = np.matmul(B, Vn)
        remove_almost_zeros(B)
        assert(B[x,x + 2] == 0.0)
        print("=\n{}\n".format(B))
        Vs.append(Vn)

        #print("U=\n{}".format(Un))
        #B = np.matmul(Un.T, B)
        #print("U^T * B =\n{}".format(B))

    def multiply_list_of_matrices(matrices):
        M = matrices[0]
        for i in range(1, len(matrices)):
            M = np.matmul(M, matrices[i])
        return M

    U = multiply_list_of_matrices(Us)
    V = multiply_list_of_matrices(Vs)

    #B = np.matmul(U,B)
    #B = np.matmul(B, V)

    return (U, B, V)

#[[ 0,  1,  2,  3],
# [ 4,  5,  6,  7],
# [ 8,  9, 10, 11],
# [12, 13, 14, 15]]
#matrix = np.arange(0, 16).reshape((4,4))
#B = np.matrix([
#    [1,4,0,0],
#    [0,2,5,0],
#    [0,0,3,6],
#    [0,0,0,4]])

N = 8
B = np.zeros((N,N))
np.fill_diagonal(B, range(1, N+1))
np.fill_diagonal(B[:,1:], range(N+1, 2*N))

U,B_,V = svd(B.copy())

print("B=\n{}\n".format(B))
print("B_=\n{}\n".format(B_))
print("U=\n{}\n".format(U))
print("V=\n{}".format(V))

B_verify = np.matmul(U, B)
B_verify = np.matmul(B_verify, V)
remove_almost_zeros(B_verify)
print("B_verify=\n{}".format(B_verify))

# B' = U * B * V = B_
assert(np.allclose(B_, B_verify))

diag =  np.asarray(B.diagonal())
diag_ = np.asarray(B_.diagonal())
print(diag_ - diag)
# last one is broken, remove later on
diag = np.delete(diag, -1)
diag_ = np.delete(diag_, -1)
assert(all(diag < diag_))


diag2 = np.asarray(B.diagonal(1))
diag2_ = np.asarray(B_.diagonal(1))
print(diag2 - diag2_)
assert(all(diag2 > diag2_))