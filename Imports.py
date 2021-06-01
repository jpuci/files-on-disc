import numpy as np
import matplotlib.pyplot as plt
import math


def drawdemp(random_sample, sub=None, title=None,  xlim_min=None, xlim_max=None, line_style=""):
    """Draws CDF for the given random_sample."""
    xs = np.array(random_sample, dtype=np.float)
    xs.sort()
    step = (xs[-1] - xs[0]) / xs.size
    xs = np.insert(xs, 0, xs[0] - step)
    xs = np.insert(xs, xs.size, xs[-1] + step)
    ys = np.linspace(0, 1, xs.size - 1)

    fig, ax = plt.subplots()
    ax.hlines(y=ys, xmin=xs[:-1], xmax=xs[1:])
    ax.vlines(x=xs[1:-1], ymin=ys[:-1], ymax=ys[1:], linestyle='dashed')
    ax.set_xlim(min(xs), max(xs))
    plt.xlabel("x")
    plt.ylabel("F(x)")
    if xlim_min:
        ax.set_xlim(xlim_min, max(xs))
    if xlim_max:
        ax.set_xlim(min(xs), xlim_max)
    if xlim_min and xlim_max:
        ax.set_xlim(xlim_min, xlim_max)
    if sub:
        plt.plot(xs, [sub(i) for i in xs], line_style)
    if title:
        plt.title(title)
    plt.show()


def drawdemp_mult(data, title=None, label=None):
    """Draws poisson process trajectory for the given random_sample."""
    fig, ax = plt.subplots()
    if type(data) == list:
        colors = plt.cm.get_cmap('hsv', len(data))
        for i, random_sample in enumerate(data):
            random_sample.sort()
            step = (random_sample[-1] - random_sample[0]) / len(random_sample)
            xs = [random_sample[0] - step] + random_sample + [random_sample[-1] + step]
            ys = np.arange(0, len(random_sample) + 1, 1)
            if any(label):
                ax.hlines(y=ys, xmin=xs[:-1], xmax=xs[1:], color=colors(i), label=label[i])
            else:
                ax.hlines(y=ys, xmin=xs[:-1], xmax=xs[1:], color=colors(i))
            ax.vlines(x=xs[1:-1], ymin=ys[:-1], ymax=ys[1:], linestyle='dashed', color=colors(i))
            ax.set_xlim(xs[0], xs[-1])
    else:
        random_sample = data
        random_sample.sort()
        step = (random_sample[-1] - random_sample[0]) / len(random_sample)
        xs = [random_sample[0] - step] + random_sample + [random_sample[-1] + step]
        ys = np.arange(0, len(random_sample) + 1, 1)

        fig, ax = plt.subplots()
        ax.hlines(y=ys, xmin=xs[:-1], xmax=xs[1:])
        ax.vlines(x=xs[1:-1], ymin=ys[:-1], ymax=ys[1:], linestyle='dashed')
        ax.set_xlim(xs[0], xs[-1])

    if title:
        plt.title(title)
    if any(label):
        plt.legend()

def finv(f, dx=0.01, ys=None):
    a = -1
    b = 1
    while float(f(a)) > dx:
        a *= 2
    while f(b) < 1 - dx:
        b *= 2
    xs = np.arange(a, b, dx)
    if not ys:
        ys = np.arange(0, 1, dx)
    dic = {}
    for y in ys:
        T = [x for x in xs if f(x) > y]
        dic[y] = min(T, default=b)
    return dic, list(dic.keys()), list(dic.values())


def drawqq(X, distr, title=None):
    """Draws QQ-Plot"""
    X.sort()
    N = len(X)
    q_dic, x, y = finv(distr, dx=1 / N)
    plt.scatter(y, X)
    plt.plot(X, X)
    if title:
        plt.title(title)
    plt.show()


def empchar(X, t):
    X_exp = [np.exp(complex(0, 1) * t * j) for j in X]
    return sum(X_exp) / len(X_exp)


def uniform_char(t, a=0, b=1):
    if t != 0:
        return (np.exp(b * t * complex(0, 1)) - np.exp(a * t * complex(0, 1))) / (complex(0, 1) * t * (b - a))
    else:
        return 1


def norm_char(t, mu=0, s=1):
    return np.exp(mu * t * complex(0, 1) - (((s * t) ** 2) / 2))


def cauchy_char(t, m=0, l=1):
    return np.exp(m * t * complex(0, 1) - l * abs(t))


def stable_char(t, a=1 / 2, b=0, m=0, s=1):
    if a != 1:
        return np.exp(
            m * t * complex(0, 1) - (abs(t * s) ** a) * (1 - b * complex(0, 1) * (np.sign(t) * np.tan(np.pi / 2 * a))))
    else:
        return np.exp(m * t * complex(0, 1) - (abs(t * s) ** a) * (
                    1 - b * complex(0, 1) * (-np.sign(t) * (2 / np.pi) * np.log(abs(t)))))


def stable_char1(t, a=1 / 2, b=1 / 2, m=0, s=1):
    if a != 1:
        return np.exp(
            m * t * complex(0, 1) - (abs(t * s) ** a) * (1 - b * complex(0, 1) * (np.sign(t) * np.tan(np.pi / 2 * a))))
    else:
        return np.exp(m * t * complex(0, 1) - (abs(t * s) ** a) * (
                    1 - b * complex(0, 1) * (-np.sign(t) * (2 / np.pi) * np.log(abs(t)))))


def stable_char2(t, a=np.sqrt(2), b=0, m=0, s=1):
    if a != 1:
        return np.exp(
            m * t * complex(0, 1) - (abs(t * s) ** a) * (1 - b * complex(0, 1) * (np.sign(t) * np.tan(np.pi / 2 * a))))
    else:
        return np.exp(m * t * complex(0, 1) - (abs(t * s) ** a) * (
                    1 - b * complex(0, 1) * (-np.sign(t) * (2 / np.pi) * np.log(abs(t)))))


def plot_3d_complex(fun, xs=np.arange(0, 1, 0.01)):
    X_c = [fun(i) for i in xs]
    ys = [i.real for i in X_c]
    zs = [i.imag for i in X_c]
    ys, zs = np.meshgrid(ys, zs)
    fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    ax2.plot_surface(xs, ys, zs)
    plt.show()


def plot_2d_complex(fun, xs=np.arange(0, 1, 0.01)):
    X_c = [fun(i) for i in xs]
    ys = [i.real for i in X_c]
    zs = [i.imag for i in X_c]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs, ys)
    ax1.set_title("real")
    ax2.plot(xs, zs)
    ax2.set_title("imag")
    plt.show()

def plot_2d_complex2(fun, xs=np.arange(0, 1, 0.01)):
    X_c = [fun(i) for i in xs]
    ys = [i.real for i in X_c]
    zs = [i.imag for i in X_c]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys)
    plt.title("real")
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.plot(xs, zs)
    plt.title("imag")
    plt.show()


def norm_pdf(x, mu, s):
    return (1 / np.sqrt(2 * np.pi * s ** 2)) * np.exp((-(x - mu) ** 2) / (2 * s ** 2))


def norm_cdf(x, mu, s):
    return (1 / 2) * (1 + math.erf((x - mu) / (s * np.sqrt(2))))


def cauchy_pdf(x, m, l):
    return (l / np.pi) * (1 / (1 + (l * (x - m)) ** 2))


def cauchy_distr(x, x_0, g):
    return 1 / 2 + 1 / np.pi * np.arctan((x - x_0) / g)
