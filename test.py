import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils as u
from matplotlib import pyplot as plt
# Create the initial black and white image
import numpy as np
from scipy import ndimage

fig, ax = plt.subplots(1,2)
a = np.zeros((512, 512)).astype(np.uint8) #unsigned integer type needed by watershed
y, x = np.ogrid[0:512, 0:512]
m1 = ((y-200)**2 + (x-100)**2 < 30**2)
m2 = ((y-350)**2 + (x-400)**2 < 20**2)
m3 = ((y-260)**2 + (x-200)**2 < 20**2)
a[m1+m2+m3]=1
ax[0].imshow(a, cmap ='gray')# left plot in the image above


xm, ym = np.ogrid[0:512:10, 0:512:10]
markers = np.zeros_like(a).astype(np.int16)
markers[xm, ym]= np.arange(xm.size*ym.size).reshape((xm.size,ym.size))
res2 = ndimage.watershed_ift(a.astype(np.uint8), markers)
# res2[xm, ym] = res2[xm-1, ym-1] # remove the isolate seeds
ax[1].imshow(res2)
plt.show()
# <matplotlib.image.AxesImage object at 0xf1fd1ac>