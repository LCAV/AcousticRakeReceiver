    
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
M = 12
d = 0.3
freqs = np.array([1000])
f = 1000
sigma2 = 1e-2

mics = bf.Beamformer.circular2D(Fs, mic1, M, 0, d)
mics.frequencies = freqs

max_K = 20
n_monte_carlo = 1000

SNR = np.zeros((max_K, 1))
UDR = np.zeros((max_K, 1))

for K in range(1, 1+max_K):
    for n in xrange(n_monte_carlo):

        # create the room with sources
        room1 = rg.Room.shoeBox2D(
            p1,
            p2,
            Fs,
            max_order=max_order,
            absorption=absorption)

        source1 = p1 + np.random.rand(2) * (p2 - p1)
        source2 = p1 + np.random.rand(2) * (p2 - p1)

        room1.addSource(source1)
        room1.addSource(source2)

        # create the echo beamformer and add to the room
        mics.rakeMaxSINRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=False,
                                attn=True)

        room1.addMicrophoneArray(mics)

        SNR[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M),
                                 dB=True)
        UDR[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 None, # room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=0.001 * np.eye(mics.M))

        # print 'K = ', K, '| SNR =', SNR, '| UDR = ', UDR



    SNR[K-1] /= float(n_monte_carlo)
    UDR[K-1] /= float(n_monte_carlo)
    print 'Computed for K = ', K, '| SNR =', SNR[K-1], '| UDR =', UDR[K-1]


# Plot the results
plt.figure
plt.plot(range(1, 1+max_K), SNR)
plt.show()
