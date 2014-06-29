
import numpy as np
import matplotlib
import constants
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile

# Room 1 : Shoe box
p1 = [-1, 0]
p2 = [5, 4]

# The first signal is Homer
source1 = [1.2, 1.5]

# the second signal is some speech
source2 = [2.5, 2]

# Some simulation parameters
Fs = 44000
absorption = 0.8
max_order = 4

# create a microphone array
mic1 = [2, 3]
M = 6
d = 0.1
phi = -np.pi / 3
mics = bf.Beamformer.linear2D(mic1, M, 0, d, Fs, 'TimeDomain', 1024)

# create the room with sources
room1 = rg.Room.shoeBox2D(
    p1,
    p2,
    Fs,
    max_order=max_order,
    absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)

# create the echo beamformer and add to the room1
mics.echoBeamformerWeights(
                        room1.sources[0].getImages(max_order=3), 
                        room1.sources[1].getImages(max_order=3), 
                        0.001 * np.eye(M))
room1.addMicrophoneArray(mics)

i = 60
print mics.frequencies[i]

print 'SNRs'
print mics.SNR(room1.sources[0].getImages(max_order=0), room1.sources[1].getImages(max_order=0), np.eye(M), i)
print mics.SNR(room1.sources[0].getImages(max_order=1), room1.sources[1].getImages(max_order=1), np.eye(M), i)
print mics.SNR(room1.sources[0].getImages(max_order=2), room1.sources[1].getImages(max_order=2), np.eye(M), i)
print mics.SNR(room1.sources[0].getImages(max_order=3), room1.sources[1].getImages(max_order=3), np.eye(M), i)

print 'UDRs'
print mics.UDR(room1.sources[0].getImages(max_order=0), room1.sources[1].getImages(max_order=0), np.eye(M), i)
print mics.UDR(room1.sources[0].getImages(max_order=1), room1.sources[1].getImages(max_order=1), np.eye(M), i)
print mics.UDR(room1.sources[0].getImages(max_order=2), room1.sources[1].getImages(max_order=2), np.eye(M), i)
print mics.UDR(room1.sources[0].getImages(max_order=3), room1.sources[1].getImages(max_order=3), np.eye(M), i)


# plot the result
f = [1000, 1050, 2000]
print mics.frequencies[2] - mics.frequencies[1]
room1.plot(freq=f, img_order=1)

plt.show()
