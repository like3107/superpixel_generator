cimport numpy as np
import numpy as np

ctypedef np.float32_t dtype_t
ctypedef unsigned long ULong

def get_hmin_array(np.ndarray[dtype_t, ndim=2] image, np.ndarray[ULong, ndim=2] label, np.ndarray[dtype_t, ndim=2] out, int max_height = 34):
    cdef int x, y, sx, sy, sy_from, sy_to, sx_from, sx_to, o_label
    cdef float hmin
    for y in range(image.shape[1]):
        sy_from = max(0,y-max_height)
        sy_to =  min(image.shape[1], y+max_height)
        for x in range(image.shape[0]):
            sx_from = max(0,x-max_height)
            sx_to = min(image.shape[0], x+max_height)
            hmin = max_height;
            o_label = label[y,x]

            for sy in range(sy_from, sy_to):
                for sx in range(sx_from, sx_to):
                    if label[sy,sx] == o_label and image[sy, sx] < hmin:
                        hmin = image[sy, sx]
            if hmin > 0 and hmin < max_height:
                out[y,x] = (image[y, x] - hmin) * (max_height/(float(max_height-hmin)) )
            else:
                out[y,x] = image[y, x]