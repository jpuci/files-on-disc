import decimal
import time
import numpy as np


class RandomGen:
    def __init__(self, c=21*2**30+3, m=2**38, seed=int((time.time()%1)*10**16)):
        self.c = decimal.Decimal(c)
        self.m = decimal.Decimal(m)
        self.seed = decimal.Decimal(seed)
    
    def change_seed(self, val):
        """Sets seed value equat to val. """
        val = decimal.Decimal(float(val))
        self.seed = val

    def random_seed(self):
        """Sets value of seed equat to random number."""
        val = int((time.time()%1)*10**16)
        self.change_seed(val)

    def random_int(self, n=1, a=0, b=1):
        """Linear congruential generator returns list of n pseudo random numbers(int) from [a, b].
        a and be should be integers."""
        a = decimal.Decimal(int(a))
        b = decimal.Decimal(int(b))
        l = np.zeros(n, dtype = int)
        for i in range(n):
            x_i = (self.c * self.seed + 1) % self.m
            if b - a == 1:
                l[i] = (x_i / self.m + a).quantize(0)
            else:
                l[i] = x_i % (b + 1 - a) + a
            self.change_seed(x_i)
        if n==1:
            return l[0]
        return l

    def random_float(self, n=1, a=0, b=1, prec=28, to_float=True):
        """Linear congruential generator. Returns float number from U(a,b), where a < b,
        or array of n floats with precision prec, by default 28 (cannot be too small). 
        By default returns float64 but if to_float is False returns Decimal objects. """
        decimal.getcontext().prec = prec
        a = decimal.Decimal(a)
        b = decimal.Decimal(b)
        l = np.zeros(n, dtype=np.float64)
        if not to_float:
            l = np.zeros(n, dtype=np.dtype(decimal.Decimal))
        for i in range(n):
            x_i = (self.c * self.seed + 1) % self.m
            l[i] = x_i / self.m * (b - a) + a
            self.change_seed(x_i)
        if n==1:
            return l[0]
        return l

    def random_float_union(self, n=1, prec=28, a=2**64-1, b=2**64, same_prec=True):
        """Returns list of n pseudo random numbers (Decimal objects) 
        from U((0, 1)U(a, b)). Prec means number of digits after comma, 
        by default 28(cannot be too small). If same_prec is True all numbers 
        have the same number of digits after coma."""
        l = np.zeros(n, dtype=np.dtype(decimal.Decimal))
        dp = 0
        if same_prec:
            s = str(self.random_float(a=a, b=b, prec=prec, to_float=False))
            k = s.index('.')
            dp = len(str(self.random_float(prec=prec, to_float=False))) - 1 - len(s[k:])
        for i in range(n):
            U = self.random_float()
            if U < 0.5:
                l[i] = self.random_float(prec=prec, to_float=False)
            else:
                l[i] = self.random_float(a=a, b=b, prec=prec+dp, to_float=False)
        if n==1:
            return l[0]
        return l

    def random_poisson(self, l=1):
        i = 0
        p = np.exp(-l)
        F = p
        U = self.random_float()
        while U > F:
            p  = p * l/(i+1)
            F += p
            i += 1
        return i

    def exp_gen(self, n, l):
        U = self.random_float(n) 
        return -np.log(U)/l
    