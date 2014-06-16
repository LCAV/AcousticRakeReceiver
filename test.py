import numpy as np
from beamforming import *
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


R = np.array([np.linspace(0, .01, 4), np.zeros((4))])
# print R

bf = Beamformer(R)

# x = np.array([[2], [3]])

# print np.linalg.norm(A-x, axis=0).reshape(2,1)

# print bf.steering_vector_2D(np.array([1000]), np.array([0,np.pi/2]),
# np.array([10,10]))

A_good = bf.steering_vector_2D(np.array([1000]), np.array([0]), np.array([10]))
A_bad = bf.steering_vector_2D(
    np.array([1000]), np.array([1, 2]), np.array([10, 10]))

bf.add_weights([1000], [echo_beamformer(A_good, A_bad)])


# for phi in phi_list: print [phi]

phi_list = np.arange(-np.pi, np.pi, .05)

print 'bfresp'
bfresp = bf.response(phi_list, 1000)
plt.plot(phi_list, np.abs(bfresp)[0, :])
plt.show()

print
