
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [5, 4]
Fs = 44000
source1 = [0.5, 1.5]
source2 = [3, 3]
mic1 = bf.MicrophoneArray(np.array([[2, 3]]).T, Fs)
absorption = 0.8
max_order = 15

# set t0 so that decay of sinc band-pass filter is less than tol at t=0
tol = 1e-5
t0 = 1. / (tol * Fs)

# create the room
room1 = rg.Room.shoeBox2D(p1, p2, Fs, t0=t0, max_order=max_order, absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)
room1.addMicrophoneArray(mic1)
fig, ax = room1.plot()

# compute room impulse response for mic1
RIRs = room1.compute_RIR()
M = room1.micArray.M
S = len(room1.sources)
plt.figure()
for r in xrange(M):
    for s in xrange(S):
        h = room1.rir[r][s]
        plt.subplot(M, S, r*S + s + 1)
        plt.plot(np.arange(len(h)) / float(Fs), h)

# room 2 : convex polygon
source2 = [3, 3]
corners = np.array([[0, 0], [4, 0], [6, 4], [3, 7], [-1, 6]]).T
absorption = [0.5, 0.6, 0.45, 0.75, 0.63]
max_order = 7

room2 = rg.Room(corners, Fs, max_order=max_order, absorption=absorption)
room2.addSource(source1)
room2.addSource(source2)
room2.plot()

plt.show()
