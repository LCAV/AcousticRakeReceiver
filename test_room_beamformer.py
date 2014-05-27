
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
mics = bf.Beamformer.linear2D(mic1, 4, d=0.05, phi=np.pi/6.)

# create the room
room1 = rg.Room.shoeBox2D(p1, p2, max_order=max_order, absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)
room1.addMicrophoneArray(mics)
fig, ax = room1.plot()

# set t0 so that decay of sinc band-pass filter is less than tol at t=0
tol = 1e-5
t0 = 1./(tol*Fs)

# compute room impulse response for mic1
RIRs = room1.impulseResponse(mic1, Fs, t0=0.3)
plt.figure()
for i,h in enumerate(RIRs):
  plt.subplot(len(RIRs),1,i+1)
  plt.plot(np.arange(len(h))/float(Fs), np.real(h))

plt.show()
