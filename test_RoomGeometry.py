
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [5, 4]
source1 = [0.5, 1.5]
mic1 = [2, 3]
Fs = 44000
absorption = [0.95, 0.2, 0.8, 0.75]
max_order = 7

room1 = rg.Room.shoeBox2D(p1, p2, max_order=max_order, absorption=absorption)
room1.addSource(source1)
fig, ax = room1.plot()
ax.scatter(mic1[0], mic1[1], marker='x')

h = room1.sampleImpulseResponse(mic1, Fs)
plt.figure()
plt.plot(np.real(h[0]))
plt.plot(np.imag(h[0]), 'r')

# room 2 : convex polygon
source2 = [3,3]
corners = [[0,0],[4,0],[6,4],[3,7],[-1,6]]
max_order = 4

room2 = rg.Room(corners, max_order=max_order, absorption=0.5)
room2.addSource(source1)
room2.addSource(source2)
room2.plot()

plt.show()
