import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.dpi"] = 600

X = [1e-4, 1e-2, 1e1, 1e4, 1e7, 1e10, 1e13]
Y_CPD = [100, 100, 0, 0, 0, 73, 100]
Y_TT = [1, 3, 4,0,0,0,0]

plt.plot(X, Y_CPD, label="CPD")
plt.plot(X, Y_TT, label="TT")
plt.xscale('log')
plt.xlabel("variance of gound truth data model")
plt.ylabel("number of over/underflow issues")
plt.legend()
plt.show()