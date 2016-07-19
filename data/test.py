import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


b= np.random.randint(0, 256, size=(1000, 1000))
x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)

# line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

im = ax.imshow(b)

for phase in np.linspace(0, 10*np.pi, 500):

    b = np.random.randint(0, 256, size=(1000, 1000))
    im.set_data(b)

    # fig.canvas.draw()

