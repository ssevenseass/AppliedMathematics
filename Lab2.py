from numpy.linalg import norm, inv, det
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import sin
import math


#def func(x):
    #return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 2 * x[0] + x[1]

def func(x):
    return x[0]**2 + x[1]**2 - sin(x[0]) + sin(x[1])

#def gradfunc(x):
    #return array([2 * x[0] + x[1] - 2, x[0] + 2 * x[1] + 1])

def gradfunc(x):
    return array([2 * x[0] - cos(x[0]), 2 * x[1] + cos(x[1])])

#def hessian():
    #return array([[2, 1], [1, 2]])

def hessian(x):
    return array([[sin(x[0]) + 2, 0], [0, 2 - sin(x[1])]])

def golden_ratio(function):
    l = -1
    r = 1
    epsilon = 10 ** (-8)
    x1 = l + (3 - math.sqrt(5)) / 2 * (r - l)
    x2 = l + (math.sqrt(5) - 1) / 2 * (r - l)
    f1 = function(x1)
    f2 = function(x2)
    iters = 0
    interval = r - l
    while (interval > epsilon):
        if function(x1) > function(x2):
            l = x1
            x1 = x2
            x2 = l + (math.sqrt(5) - 1) / 2 * (r - l)
            f1 = f2
            f2 = function(x2)
        else:
            r = x2
            x2 = x1
            x1 = l + (3 - math.sqrt(5)) / 2 * (r - l)
            f2 = f1
            f1 = function(x1)
        prev_leng = interval
        interval = r - l
        iters += 1
    return (l + r) / 2


def get_fibonacci(n):
    return round(1 / math.sqrt(5) * (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n))


def get_n(left_border, right_border, epsilon):
    argument = math.sqrt(5) * (((right_border - left_border) / epsilon) - 0.5)
    return math.ceil(math.log(argument, ((1 + math.sqrt(5)) / 2)))


def fibonacci(function):
    l = -1
    r = 1
    epsilon = 10 ** (-8)
    iteration = 0
    count = 4
    n = get_n(l, r, epsilon)
    left_border = l
    right_border = r
    interval = right_border - left_border
    x1 = left_border + get_fibonacci(n) / get_fibonacci(n + 2) * interval
    x2 = left_border + get_fibonacci(n + 1) / get_fibonacci(n + 2) * interval
    for k in range(int(n)):
        k += 1
        count += 2
        fx1 = function(x1)
        fx2 = function(x2)
        interval = right_border - left_border

        if function(x1) <= function(x2):
            right_border = x2
            x2 = x1
            x1 = left_border + get_fibonacci(n - k - 1) * (right_border - left_border) / get_fibonacci(n - k + 1)
            fx2 = fx1
            fx1 = function(x1)

        else:
            left_border = x1
            x1 = x2
            x2 = left_border + get_fibonacci(n - k) * (right_border - left_border) / get_fibonacci(n - k + 1)
            fx1 = fx2
            fx2 = function(x2)

        if function(x1) == function(x2):
            return x1
        iteration += 1
    return left_border + right_border / 2


def SteepestDescent():
    start = (-1, 1)
    epsilon = 0.1
    x = array(start)
    step = -1
    trajectory = []
    while True:
        step += 1
        grad = gradfunc(x)
        trajectory.append([x[0], x[1]])
        if (norm(grad) < epsilon):
            print('closest solution: ' + str(x[0]) + ', ' + str(x[1]))
            print("Count: " + str(step))
            return x
            #return trajectory
        alpha = golden_ratio(lambda al: func(x - al * grad))
        #alpha = fibonacci(lambda al: func(x - al * grad))
        nextX = x - alpha * grad
        norm_ = norm(x - nextX)
        fprev = func(x)
        fnext = func(nextX)
        fAbs = abs(fnext - fprev)
        if (norm_.all() < epsilon) & (fAbs < epsilon):
            print('closest solution: ' + str(x[0]) + ', ' + str(x[1]))
            print("f(x,y): " + str(func(x)))
            break
        x = nextX
        print(x[0], x[1])


def ConjugateGrad():
    start = (0, 0)
    epsilon = 0.1
    seq = [2]
    x = array(start)
    step_count = 0
    k = 0
    trajectory = []
    while True:
        Nseq = seq
        p = - gradfunc(x)
        while True:
            #alpha = golden_ratio(lambda al: func(x + al * p))
            alpha = fibonacci(lambda al: func(x + al * p))
            xk = x
            x = xk + alpha * p
            step_count += 1
            print(str(step_count) + " " + str(x[0]) + " " + str(x[1]) + " " + str(func(x)))
            trajectory.append([x[0], x[1]])
            if norm(p) < epsilon:
                print("Колво шагов:" + str(step_count))
                #return x
                return trajectory
            if k + 1 == Nseq:
                k = 0
                break
            beta = ((norm(gradfunc(x))) ** 2) / ((norm(gradfunc(xk))) ** 2)
            pk = p
            p = - gradfunc(x) + beta * pk
            k += 1
            print(x[0], " ", x[1])


def Conjugate():
    start = (0, 0)
    x = array(start)
    count = 0
    epsilon = 1e-8
    num = 2
    k = 0
    trajectory = []
    while True:
        seq = num
        grad = gradfunc(x)
        while True:
            alpha = np.dot(grad, gradfunc(x)) / (norm(grad) ** 2 * det(hessian(x)))
            xk = x
            x = xk - alpha * grad
            count += 1
            trajectory.append([x[0], x[1]])
            if norm(grad) < epsilon:
                print("Count = " + str(count))
                return trajectory
                #return x
            if k == seq:
                k = 0
                break
            beta = ((norm(gradfunc(x))) ** 2) / ((norm(gradfunc(xk))) ** 2)
            gradk = grad
            grad = gradfunc(x) + beta * gradk
            k += 1
            print(x[0], x[1])



def NewtonsMethod():
    start = (0, 0)
    x = array(start)
    count = 0
    epsilon = 0.001
    trajectory = []
    while True:
        grad = gradfunc(x)
        trajectory.append([x[0], x[1]])
        if norm(grad) < epsilon:
            print("Count = " + str(count))
            #return x
            return trajectory
        G = hessian(x)
        alpha = 1
        invH = inv(G)
        x = x - alpha * np.dot(invH, grad)
        count += 1
        print(x[0], x[1])


def ConstStep():
    start = (-1, 1)
    x = array(start)
    epsilon = 0.01
    steps = 0.001
    count = 0
    trajectory = []
    while True:
        grad = gradfunc(x)
        count += 1
        trajectory.append([x[0], x[1]])
        if norm(grad) < epsilon:
            print("Count of steps: " + str(count))
            print("Size of step: " + str(alpha))
            return trajectory
        if count > 10 / steps:
            print("Difference between method and size of steps: " + str(steps))
            for i in range(0, len(x) - 1):
                x[i] = 9e100
            #return x
        alpha = steps
        x = x - alpha * grad
        print("res: ", x[0], x[1])


def CrashStep():
    L = 1.0
    x = [-1, 1]
    epsilon = 0.01
    fv = func(x)
    trajectory = []
    count = -1
    while L > 0.005:
        grad = gradfunc(x)
        gx = gradfunc(x)[0]
        gy = gradfunc(x)[1]
        nx = x[0] - L * gx
        ny = x[1] - L * gy
        nf = func([nx, ny])
        trajectory.append([x[0], x[1]])
        count += 1
        if norm(grad) < epsilon:
            print("Count of steps: " + str(count))
            #return x
            return trajectory
        #if math.fabs(nx - x[0]) < epsilon and math.fabs(ny - x[1]) < epsilon:
            #break
        if nf <= (fv - 0.5 * L * (gx * gx + gy * gy)):
            x[0] = nx
            x[1] = ny
            fv = nf
        else:
            L = L * 0.5
            #return trajectory
        print(x[0], "   ", x[1])




def create_plot(f, traj):
    bounds = np.mgrid[-2:3, -2:3]
    x, y = bounds
    z = f([x, y])
    fig, ax = plt.subplots()
    c = ax.contour(x, y, z)
    ax.clabel(c)
    ax.plot(array(traj)[:, 0], array(traj)[:, 1], color='r')   #marker='.'
    ax.grid()
    plt.show()
    pass

#trajectory = ConjugateGrad()
trajectory = ConstStep()
#trajectory = CrashStep()
#trajectory = NewtonsMethod()
#trajectory = Conjugate()
#trajectory = SteepestDescent()
create_plot(func, trajectory)

#ConjugateGrad()
#SteepestDescent()
#Conjugate()
#NewtonsMethod()
#CrashStep()
#ConstStep()
#CrashStep()
#SteepestDescent()