#!/usr/bin/env python
#coding: utf-8

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("agg")

def draw_weight():
    X = list()
    W1 = list()
    W2 = list()
    W3 = list()
    for i in xrange(-100, 101):
        x = 0.01 * i
        
        """
        # method1
        alpha = max(0.0, x)
        beta = abs(2.0 * alpha - 1.0)
        w1 = beta*(alpha<0.5)
        w2 = (1-beta)
        w3 = beta*(alpha>0.5)
        """

        # method2
        alpha = max(0.0, x)
        m = 20
        beta = 2 - 1 / (1 + math.e ** (m*(0.25-alpha))) - 1 / (1 + math.e ** (m*(alpha-0.75)))
        w1 = beta*(alpha<0.5)
        w2 = (1-beta)
        w3 = beta*(alpha>0.5)

        X.append(x)
        W1.append(w1)
        W2.append(w2)
        W3.append(w3)

    #plt.figure("Figure")
    #plt.title("Title")
    #plt.xlabel("x")
    #plt.ylabel("y")
    plt.grid(True)
    plt.xlim(-1.0, 1.0)
    plt.ylim(0.0, 1.0)
    #plt.yticks(color="w")
    #plt.xticks(color="w")
    plt.plot(X, W1, "r-", X, W2, "g--", X, W3, "b-.", linewidth=2)
    plt.savefig("../figures/W.jpg")
    plt.close()


if __name__ == "__main__":
    draw_weight()
