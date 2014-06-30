    
import numpy as np
import matplotlib
import constants
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile

# Room 1 : Shoe box
p1 = np.array([-1, 0])
p2 = np.array([5, 4])

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
freqs = np.arange(200, 2000, 50)
f = 1000
sigma2 = 1e-2

source1 = p1 + np.random.rand(2) * (p2 - p1)
source2 = p1 + np.random.rand(2) * (p2 - p1)

# create the room with sources
room1 = rg.Room.shoeBox2D(
    p1,
    p2,
    Fs,
    max_order=max_order,
    absorption=absorption)
room1.addSource(source1)
room1.addSource(source2)

mics = bf.Beamformer.linear2D(Fs, mic1, M, 0, d)
mics.frequencies = freqs

max_K = 30

for K in range(1+max_K)[1::2]:

    # create the echo beamformer and add to the room
    mics.rakeMaxSINRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                            room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                            R_n=sigma2 * np.eye(mics.M),
                            ff=False,
                            attn=True)

    room1.addMicrophoneArray(mics)

    SNR = mics.SNR(room1.sources[0].getImages(n_nearest=max_K, ref_point=mics.center), 
                             room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                             f, 
                             R_n=sigma2 * np.eye(mics.M),
                             dB=True)
    UDR = mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                             room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                             f, 
                             R_n=0.001 * np.eye(mics.M),
                             ff=False,
                             attn=True)

    print 'K = ', K, '| SNR =', SNR, '| UDR = ', UDR
