import h5py
import numpy as np
f = h5py.File('../data/volumes/label_CREMI_noz_test.h5','r+')

s=f['data'][102]
print np.sum(s==18527)
s[s==18527]=16870
f['data'][102] = s

s = f['data'][142]
print np.sum(s==12009)
s[s==12009] = 11010
f['data'][142] = s

s = f['data'][141]
print np.sum(s==13734)
s[s==13734] = 11010
f['data'][141] = s

s = f['data'][147]
print np.sum(s==13551)
s[s==13551] = 11010
print np.sum(s==13268)
s[s==13268] = 11010
f['data'][147] = s

s =  f['data'][133]
print np.sum(s==13268)
s[s==11916] = 11010
f['data'][133] = s

f.close()
