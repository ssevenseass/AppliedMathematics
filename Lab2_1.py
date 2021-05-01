from numpy.linalg import norm, inv, det
from numpy import array
from functools import reduce
from random import random
import numpy as np
import math

n = 10
s = (n, n)
A= np.zeros(s)

def generate_matrix(n: int):
    s = (n, n)
    Y = np.zeros(s)
    for i in range(0, n):
        for j in range(i, n):
            Y[i][i] = random()
            A[i][i] = Y[i][i]
    return Y

def generate_func(n: int):
    def line_mapper(line, line_number, *args):
        return reduce(lambda prev, cur: prev + args[line_number] * args[cur[0]] * cur[1], enumerate(line), 0)
    matrix = generate_matrix(n)
    print(reduce(lambda prev, line: prev + ' '.join(map(lambda v: '%.3f' % v, line)) + '\n', matrix, ''), end='')
    ShowF(matrix, n)
    print("\nCondition number:", np.linalg.cond(matrix))
    return lambda *args: reduce(lambda val, line: val + line_mapper(line[1], line[0], *args), enumerate(matrix), 0)

def ShowF(S, n):
    str_z = ' '
    for i in range(n):
        str_z += str(S[i,i]) + ' * x['+str(i)+'] ** 2'
        if (i != n-1):
            str_z += ' + '
    print(str(str_z))
    return str_z

F = generate_func(n)

def gradfunc(x):
    y = [0] * len(x)
    for i in range(len(x)):
        y[i] = x[i] * A[i][i]
    return y

def func(x):
    y = [0] * len(x)
    for i in range(len(x)):
        y[i] = x[i] ** 2 * A[i][i]
    return sum(y)

def ConstStep():
    x = [0] * n
    for i in range(n):
        x[i] = 1
    epsilon = 0.01
    steps = 1e-1
    count = 0
    while True:
        grad = gradfunc(x)
        count += 1

        if norm(grad) < epsilon:
            print("Count of steps: " + str(count))
            #print("Size of step: " + str(alpha))
            return count
        if count > 100000 / steps:
            print("Difference between method and size of steps: " + str(steps))
            for i in range(0, len(x) - 1):
                x[i] = 9e100
            return x
        alpha = steps
        for i in range(n):
            x[i]= x[i] - alpha * grad[i]

def CrashStep():
    L = 1.0
    x = [0] * n
    nx = [0] * n

    for i in range(n):
        x[i] = 1
    count = 0
    epsilon = 0.001
    fv = func(x)

    while L > 0.005:
        grad = gradfunc(x)
        count += 1

        for i in range(n):
            nx[i] = x[i] - L * grad[i]
        nf = func(nx)
        if norm(grad) < epsilon:
            print("Count of steps: " + str(count))
            return count

        exitP = True
        for i in range(n):
            if (math.fabs(nx[i] - x[i]) > epsilon):
                exitP = False

        if exitP:
            break

        s = 0
        for i in range(n):
            s += grad[i] ** 2

        if nf <= (fv - 0.5 * L * s):
            x = nx
            fv = nf
        else:
            L = L * 0.5
    #print(x[0], "   ", x[1])
    print("Count of steps: " + str(count))
    return 0

print("\nAfter optimisation:")
ConstStep()