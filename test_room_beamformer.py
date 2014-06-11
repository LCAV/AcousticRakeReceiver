
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [5, 4]

# The first signal is Homer
source1 = [0.5, 1.5]

# the second signal is some speech
source2 = [3,3]

# Some simulation parameters
Fs = 44000
absorption = 0.8
max_order = 4

# create a microphone array
mic1 = [2, 3]
M = 5
d = 0.1
f = 1000.
phi = -np.pi/3
mics = bf.Beamformer.linear2D(mic1, M, d=d) + bf.Beamformer.linear2D(mic1, 4, d=d, phi=np.pi/2)

# create the room with sources
room1 = rg.Room.shoeBox2D(p1, p2, Fs, max_order=max_order, absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)

# create the echo beamformer and add to the room
mics.echoBeamformerWeights(room1.sources[0].getImages(max_order=1),room1.sources[1].getImages(max_order=1), f)
room1.addMicrophoneArray(mics)

# plot the result
room1.plot(freq=f, img_order=1)

plt.show()
