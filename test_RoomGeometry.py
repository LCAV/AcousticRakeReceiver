
import matplotlib
import matplotlib.pyplot as plt

import Room as rg

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [3, 2]
source1 = [0.5, 1.5]
max_order = 7

room1 = rg.Room.shoeBox2D(p1, p2, max_order)
room1.addSource(source1)
room1.plot()

# room 2 : convex polygon
source2 = [3,3]
corners = [[0,0],[4,0],[6,4],[3,7],[-1,6]]
max_order = 4

room2 = rg.Room(corners, max_order)
room2.addSource(source1)
room2.addSource(source2)
room2.plot()

plt.show()
