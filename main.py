import torch
from utils import *
from model import *

X, Y = generate_toy_dataset(6, 10, 0.1, linear=True)

test = BTTKM(3, [1,2,2,1],[6,6,6])
test.train(X, Y)