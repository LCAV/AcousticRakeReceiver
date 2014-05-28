
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [5, 4]
source1 = [0.5, 1.5]
source2 = [3,3]
mic1 = [2, 3]
Fs = 44000
absorption = 0.8
max_order = 4

# create a microphone array
M = 10
d = 0.05
f = 1000.
phi = -np.pi/3
mics = bf.Beamformer.circular2D(mic1, M, radius=0.05)

# create the room
room1 = rg.Room.shoeBox2D(p1, p2, max_order=max_order, absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)

mics.echoBeamformerWeights(room1.sources[0].getImages(max_order=1),room1.sources[1].getImages(max_order=1), f)

room1.addMicrophoneArray(mics)
fig, ax = room1.plot(freq=f, img_order=1)

plt.show()
