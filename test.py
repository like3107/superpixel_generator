import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np

a =[]
for i in range(1, 100000):
    a.append(51201 % i == 0)


print np.sum(np.array(a) == 1)