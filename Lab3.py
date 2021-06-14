import time
from operator import inv
from numpy import zeros, diag, diagflat, dot, allclose, zeros_like
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import random
from scipy.sparse import csr_matrix

iteration = 100
N = 10
NUM_RANGE = 10

#A = array([[10, -1, 2, 0],
         #  [-1, 11, -1, 3],
        #   [2, -1, 10, -1],
         #  [0, 3, -1, 8]])

#B = array([6.0, 25.0, -11.0, 15.0])


def createArrayA():
    massA = np.random.randint(-NUM_RANGE, NUM_RANGE, (N, N))
    return massA


def createArrayB():
    massB = np.random.randint(-NUM_RANGE, NUM_RANGE, N)
    return massB


A = createArrayA().astype('float32')
B = createArrayB().astype('float32')

def Saydel(a, b):
    count = 0

    x = zeros_like(B)
    for it_count in range(1, iteration):
        x_new = zeros_like(x)
        count += 1
        print("Step {0}: {1}".format(it_count, x))
        for i in range(A.shape[0]):
            s1 = dot(A[i, :i], x_new[:i])
            s2 = dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        if allclose(x, x_new, rtol=1e-8):
            break
        x = x_new
    print("Решение: {0}".format(x))
    print("Количество итераций:", str(count))
    return x

def Yakobi(a, b):
    count = 0
    #A = createArrayA()
    #print("Mass A:")
    #print(A)
    #B = createArrayB()
    #print("Mass B:")
    #print(B)
    x = zeros_like(B)
    for it_count in range(iteration):
        if it_count != 0:
            print("Step {0}: {1}".format(it_count, x))
            count += 1
        x_new = zeros_like(x)
        for i in range(A.shape[0]):
            s1 = dot(A[i, :i], x[:i])
            s2 = dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
            if x_new[i] == x_new[i - 1]:
                break
        if allclose(x, x_new, atol=1e-10, rtol=0.):
            break
        x = x_new
    print("Solution:")
    print(x)
    print("Количество итераций:", str(count))
    return x



def Yacobifor6(A, B, n=25, x=None):
    print("A = ", A)
    print("B = ", B)
    if x is None:
        x = zeros(len(A[0]))
    D = diag(A)
    R = A - diagflat(D)
    for i in range(n):
        x = (B - dot(R, x)) / D
    print("Solution:")
    print(x)
    return x


def generateMatrix(n: int):
    s = (n, n)
    M = np.zeros(s)
    for i in range(n):
        for j in range(n):
            M[i][j] = random.uniform(0, 10)
    conditionNumber(M)
    return csr_matrix(M)
    #return M


def generateGilbertMatrix(n: int):
    s = (n, n)
    M = np.zeros(s)
    for i in range(n):
        for j in range(n):
            M[i][j] = 1/((i+1) + (j+1) - 1)
    #conditionNumber(M)
    return csr_matrix(M)


def generateDiagonalDomination(k: int):
    s = (k, k)
    M = np.zeros(s)
    for i in range(k):
        for j in range(k):
            if (i != j):
                M[i][j] = random.uniform(0, 10)
    for i in range(k):
        sum = 0
        for j in range(k):
            sum += M[i][j]
        M[i][i] = -(sum - M[i][i]) + 10 ** (-k)
    conditionNumber(M)
    return csr_matrix(M)


def conditionNumber(matrix):
    condNumber = np.linalg.cond(matrix)
    print("\nCondition number:" + str(condNumber))
    return condNumber


def showMatrix(matrix):
    for row in matrix:
        for x in row:
            print(str("{:4}".format(round(x, 3)))+"\t", end="")
        print()

def showCSRMatrix(matrix):
    m = matrix.A
    for row in m:
        for x in row:
            print(str("{:4}".format(round(x, 3)))+"\t", end="")
        print()


def decompose_to_LU(matrix):
    n = matrix.shape[0]
    U = matrix.toarray().copy()
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            L[j][i] = U[j][i] / U[i][i]

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    for i in range(n):
        L[i][i] = U[i][i]
        U[i][i] = 1

    return csr_matrix(L), csr_matrix(U)

def create_f(A, x):
    length = len(x)
    F = [0 for i in range(length)]
    for i in range(length):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            index = A.indices[j]
            a_j = A.data[j]
            F[i] += a_j * x[index]
    return F

def gauss_method(matrix, E):
    n = matrix.shape[0]
    L, U = decompose_to_LU(matrix)
    #showCSRMatrix(L)
    #showCSRMatrix(U)
    y = up_to_down_gauss(L, E)
    #print(y)
    x = down_to_up_gauss(U, y)
    #print(x)
    return x


def up_to_down_gauss(initial, result):
    length = initial.shape[0]
    x = [0 for i in range(length)]
    for i in range(length):
        tmp = result[i]
        for j in range(initial.indptr[i], initial.indptr[i + 1]):
            index = initial.indices[j]
            i_j = initial.data[j]
            tmp -= x[index] * i_j
            if j == initial.indptr[i + 1] - 1:
                x[index] = tmp / i_j
    return x


def down_to_up_gauss(initial, result):
    length = initial.shape[0]
    x = [0 for i in range(length)]
    for i in range(length, 0, -1):
        tmp = result[i - 1]
        for j in range(initial.indptr[i] - 1, initial.indptr[i - 1] - 1, -1):
            index = initial.indices[j]
            i_j = initial.data[j]
            tmp -= x[index] * i_j
            if j == initial.indptr[i - 1]:
                x[index] = tmp / i_j
    return x

def findInvMatrix(sL, sU):
    arrL = {}
    arrU = {}

    L = sL.toarray().copy()
    U = sU.toarray().copy()
    #L = [[2, 0, 0, 0],[ 1, 2, 0,0],[ 3, 2, 3,0], [2,1,2,4]]
    #U = [[1, 2, -2, 3], [0, 1, 2, -1], [0, 0, 1, -2], [0, 0, 0, 1]]

    for k in range(n):
        A = np.eye(n)
        for i in range(n):
            A[i][k] = L[i][k]
        # print("\nA:")
        # showMatrix(A)
        arrL[k] = A

        A = np.eye(n)
        for i in range(k):
            A[i][k] = U[i][k]
        # print("\nB:")
        # showMatrix(A)
        arrU[k] = A

    invU = inv(arrU[0])
    invL = inv(arrL[0])

    for k in range(1, n):
        invL = np.dot(inv(arrL[k]), invL)
        invU = np.dot(invU, inv(arrU[k]))

    #print("\n inverted L: ")
    #showMatrix(invL)

    #print("\n inverted U:")
    #showMatrix(invU)

    invMatrix = np.dot(inv(U), inv(L))

    return csr_matrix(invMatrix)

def solve_LU(matrix, b):
    n = matrix.shape[0]
        # вычисляем вектор y, который получается при замене Ux=y
    y = np.zeros([n, 1])

    L, U = decompose_to_LU(matrix)
    lu = np.zeros((n, n))
    L = L.toarray()
    U = U.toarray()

    for i in range(n):
        for j in range(0, i+1):
            lu[i][j] = L[i][j]

    for i in range(n):
        for j in range(i, n):
            lu[i][j] = U[i][j]

    #print(lu)
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - np.dot(lu[i, :i], y[:i])

        # вычисляем вектор ответов x
    x = np.zeros([n, 1])

    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - np.dot(lu[-i, -i:], x[-i:, 0])) / lu[-i, -i]

    return x

def getF(matrix):
    n = matrix.shape[0]
    X = np.zeros([n, 1])
    m = matrix.toarray()
    for i in range(n):
        X[i, 0] = i + 1
    return np.dot(m, X)

def getBcof(A):
    n = A.shape[0]
    y = np.array([])
    for i in range(1, n + 1):
        y = np.append(y, i)

    return np.array(A.dot(y))

n = 3

#matrix = generateMatrix(n)
#matrix = generateGilbertMatrix(n)
matrix = generateDiagonalDomination(n)

print("\nMatrix:")
showCSRMatrix(matrix)

L, U = decompose_to_LU(matrix)
print("\nL:")
showCSRMatrix(L)
print("\nU:")
showCSRMatrix(U)

invM = findInvMatrix(L, U)
print("\ninverted Matrix:")
showCSRMatrix(invM)

E = np.eye(n)
F = getF(matrix)
B = getBcof(matrix)
print("\nF:")
print(F)

print("\nGauss:")
#matrix = csr_matrix(np.array([[1, 0, 2], [3, -1, 0], [4, -1, 3]]))
print(gauss_method(matrix, F))

print("\nB:")
B = getBcof(matrix)
print(B)
print("\nGauss:")
print(gauss_method(matrix, B))

print("\nsolve LU:")
print(solve_LU(matrix, F))
#invM = findInvMatrix(LU[0], LU[1])

mass = [10, 50, 100, 1000]
arr = []
#for i in mass:
   # buildLUTime = time.time()
    #Yacobifor6(A, B, i)
   # buildLUTime = time.time() - buildLUTime
   # print("Operation time: ", buildLUTime)
   #arr.append(buildLUTime)

for i in mass:
    buildLUTime = time.time()
    gauss_method(matrix, B)
    buildLUTime = time.time() - buildLUTime
    print("Operation time: ", buildLUTime)
    arr.append(buildLUTime)

print(arr)
print(mass)
plt.axis([0, 1000, 0, 0.02])
plt.plot(mass, arr)
plt.title('График')
plt.xlabel('Размерность матрицы')
plt.ylabel('Эффективность')
plt.show()
#buildLUTime = time.time()
#jacobi(A, B, N=25, x=None)
#buildLUTime = time.time() - buildLUTime
#print("Operation time: ", buildLUTime)
#Saydel(A, B)
#Yakobi(A, B)
