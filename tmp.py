import matplotlib.pyplot as plt
import numpy as np

from utils import unfold
A = np.array([[['a', 'e'], ['c', 'g']],[['b','f'],['d','h']]])
print(unfold(A, 2))