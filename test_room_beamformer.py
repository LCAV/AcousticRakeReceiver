
import numpy as np
import matplotlib
import constants
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile

# Room 1 : Shoe box
p1 = [0, 0]
p2 = [4, 6]

# The first signal is Homer
source1 = [1.2, 1.5]

# the second signal is some speech
source2 = [2.5, 2]

# Some simulation parameters
Fs = 44100
absorption = 0.8
max_order = 4

# create a microphone array
mic1 = [2, 3]
M = 12
d = 0.1
phi = -np.pi / 3
f = 1100

mics = bf.Beamformer.linear2D(Fs, mic1, M, 0, d)
mics.frequencies = np.array([f])

# create the room with sources
room1 = rg.Room.shoeBox2D(
    p1,
    p2,
    Fs,
    max_order=max_order,
    absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)

# create the echo beamformer and add to the room
mics.rakeOneForcingWeights(room1.sources[0].getImages(n_nearest=3, ref_point=mics.center), 
                        None, # room1.sources[1].getImages(n_nearest=2, ref_point=mics.center), 
                        R_n=0.0001 * np.eye(mics.M))
room1.addMicrophoneArray(mics)

print np.abs(mics.response_from_point(np.array(source1)[:,np.newaxis], f)[1])
print np.abs(mics.response_from_point(np.array(room1.sources[0].getImages(n_nearest=5, ref_point=mics.center)[:,4])[:,np.newaxis], f)[1])

print 'SNRs'
print mics.SNR(room1.sources[0].getImages(max_order=0), room1.sources[1].getImages(max_order=0), f, R_n=np.eye(mics.M))
print mics.SNR(room1.sources[0].getImages(max_order=1), room1.sources[1].getImages(max_order=1), f, R_n=np.eye(mics.M))
print mics.SNR(room1.sources[0].getImages(max_order=2), room1.sources[1].getImages(max_order=2), f, R_n=np.eye(mics.M))
print mics.SNR(room1.sources[0].getImages(max_order=3), room1.sources[1].getImages(max_order=3), f, R_n=np.eye(mics.M))

print 'UDRs'
print mics.UDR(room1.sources[0].getImages(max_order=0), room1.sources[1].getImages(max_order=0), f, R_n=np.eye(mics.M))
print mics.UDR(room1.sources[0].getImages(max_order=1), room1.sources[1].getImages(max_order=1), f, R_n=np.eye(mics.M))
print mics.UDR(room1.sources[0].getImages(max_order=2), room1.sources[1].getImages(max_order=2), f, R_n=np.eye(mics.M))
print mics.UDR(room1.sources[0].getImages(max_order=3), room1.sources[1].getImages(max_order=3), f, R_n=np.eye(mics.M))

# plot the result
room1.plot(freq=f, img_order=1)

plt.show()
