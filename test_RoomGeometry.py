
import matplotlib
import matplotlib.pyplot as plt

import RoomGeometry as rg

p1 = [0, 0]
p2 = [2, 2]
source = [0.5, 1.5]

rg = rg.RoomGeometry.shoeBox2D(source, 3, p1, p2)

rg.plot()
plt.show()
