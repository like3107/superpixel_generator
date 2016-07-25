import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import utils as u


b= np.random.randint(0, 3, size=(100, 100))-1
plt.imshow(b, cmap=u.random_color_map())
plt.show()
