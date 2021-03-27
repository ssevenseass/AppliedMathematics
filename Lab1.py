import math

NUMBER_OF_EPSILON = 1000


def dichotomy(function,l, r, epsilon):
    interval = r - l
    iteration = 0
    delta = epsilon / 2
    left_border = l
    right_border = r

    # print('start_point', 'end_point', 'length', 'lenght/prev_lenght', 'x1', 'f(x1)', 'x2', 'f(x2)')
    while abs(right_border - left_border) >= epsilon:
        iteration += 1
        middle = (left_border + right_border) / 2

        if function(middle - delta) > function(middle + delta):
            left_border = middle

        else:
            right_border = middle
        # print(left_border, right_border, right_border - left_border, (right_border - left_border) / interval)
        prev_leng = interval
        interval = right_border - left_border
    return left_border + right_border / 2, iteration, right_border - left_border, (interval) / prev_leng


def golden_ratio(function, l, r, epsilon):
    x1 = l + (3 - math.sqrt(5)) / 2 * (r - l)
    x2 = l + (math.sqrt(5) - 1) / 2 * (r - l)
    f1 = function(x1)
    f2 = function(x2)
    iters = 0
    interval = r - l
    # print('start_point', 'end_point', 'length', 'lenght/prev_lenght', 'x1', 'f(x1)', 'x2', 'f(x2)')
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
        # print(l, r, r - l, (r - l) / prev_leng, x1, x2, f1, f2)
    return (l + r) / 2, epsilon, iters, r - l, (r - l) / prev_leng


def fibonacci(function, l, r, epsilon):
    iteration = 0
    count = 4
    n = get_n(l, r, epsilon)
    left_border = l
    right_border = r
    interval = right_border - left_border
    x1 = left_border + get_fibonacci(n) / get_fibonacci(n + 2) * interval
    x2 = left_border + get_fibonacci(n + 1) / get_fibonacci(n + 2) * interval
    # print('start_point', 'end_point', 'length', 'lenght/prev_lenght', 'x1', 'f(x1)', 'x2', 'f(x2)')
    for k in range(n):
        k += 1
        count +=2
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
        # print(left_border, right_border, right_border - left_border, (right_border - left_border) / interval, x1, x2, fx1, fx2)
        prev_leng = interval
        interval = r - l
        iteration += 1
    return left_border + right_border / 2, iteration, right_border - left_border, (interval) / prev_leng


def get_fibonacci(n):
    return round(1 / math.sqrt(5) * (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n))


def get_n(left_border, right_border, epsilon):
    argument = math.sqrt(5) * (((right_border - left_border) / epsilon) - 0.5)
    return math.ceil(math.log(argument, ((1 + math.sqrt(5)) / 2)))


def combined_brent(function,l, r, epsilon):
    a = l
    c = r
    iteration = 0
    x = w = v = a + 0.381966011 * (c - a)
    fx = fw = fv = function(x)
    d = 0
    e = d
    interval = c - a
    t = 1e-4
    # print('start_point', 'end_point', 'length', 'lenght/prev_lenght', 'x1', 'f(x1)', 'x2', 'f(x2)')
    while True:
        tol = epsilon * abs(x) + t

        if abs(x - (a + c) / 2) <= 2 * tol - (c - a) / 2:
            break

        r = q = p = 0

        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)

            if q > 0:
                p = -p

            q = abs(q)
            r, e = e, d

        if (abs(p) < abs(0.5 * q * r)) and (q * (a - x) < p) and (p < q * (c - x)):
            d = p / q
            u = x + d

            if (u - a < 2 * tol) and (c - u < 2 * tol):
                d = tol if x < (a + c) / 2 else -tol

        else:
            if x < (c + a) / 2:
                e = c - x
                d = 0.381966011 * e
            else:
                e = a - x
                d = 0.381966011 * e

        if tol <= abs(d):
            u = x + d
        elif d > 0:
            u = x + tol
        else:
            u = x - tol
        iteration += 1
        fu = function(u)
        last = [a, c]
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x

            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        else:
            if u >= x:
                c = u
            else:
                a = u

            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w < epsilon:
                v = u
                fv = fu
            prev_len = interval
            interval = c - a
        # print(last[0], last[1], interval, (c - a) / interval, a, c, function(a), function(c))
    return a + c / 2, iteration, interval, (c - a) / prev_len


def parabolic(function, l, m, r, epsilon):
    iteration = 0
    x1 = l
    x3 = r
    x2 = m
    prev_len = r - l
    xi = 0
    interval = r - l
    print('start_point', 'end_point', 'length', 'lenght/prev_lenght', 'x1', 'f(x1)', 'x2', 'f(x2)')
    while function(x2) < function(x1) and function(x2) < function(x3):
        u = x2 - 0.5 * ((x2 - x1) ** 2 * (function(x2) - function(x3)) - (x2 - x3) ** 2 * (function(x2) - function(x1))) / (
            (x2 - x1) * (function(x2) - function(x3)) - (x2 - x3) * (function(x2) - function(x1)))
        fu = function(u)

        if function(u) <= function(x2):
            if u < x2:
                x3 = x2
            else:
                x1 = x2
                x2 = u
        else:
            if x2 < u:
                x3 = u
            else:
                x1 = u
        if ((iteration > 0) and (abs(xi - u) < epsilon)):
            break
        xi = u
        print(x1, x3, abs(x3 - x1), (r - l) / prev_len, x1, x2,
              f1, f2)
        iteration += 1
        prev_len = interval
        interval = r - l
    return u, iteration, r - l, (r-l) / prev_len


def f1(x):
    return math.sin(x) * math.pow(x, 3)


def f2(x):
    return 0.007 * x ** 6 - 0.15 * x ** 5 + 1.14 * x ** 4 - 3.5 * x ** 3 + 2.9 * x ** 2 + 2.95 * x + 2.25


l = -1
r = 1
e = 0.001
# f = f1

# res = golden_ratio(f1, l, r, e)
# print(golden_ratio(f1, -1,  1, 0.001))
a = 1
for i in range(5):
    a /= 10
    res1 = combined_brent(f1, l, r, a)
    print(res1)