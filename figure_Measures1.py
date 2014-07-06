    
import numpy as np
import matplotlib
import constants
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile

# Room 1 : Shoe box
p1 = np.array([0, 0])
p2 = np.array([4, 6])

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

# How much to simulate?
max_K = 20
n_monte_carlo = 100

SNR_DS                = np.zeros((max_K, 1))
SNR_MaxSINR           = np.zeros((max_K, 1))
SNR_MaxSINR_FF        = np.zeros((max_K, 1))
SNR_MaxSINR_FF_noattn = np.zeros((max_K, 1))
SNR_MaxUDR            = np.zeros((max_K, 1))
SNR_OneForcing        = np.zeros((max_K, 1))

UDR_DS                = np.zeros((max_K, 1))
UDR_MaxSINR           = np.zeros((max_K, 1))
UDR_MaxSINR_FF        = np.zeros((max_K, 1))
UDR_MaxSINR_FF_noattn = np.zeros((max_K, 1))
UDR_MaxUDR            = np.zeros((max_K, 1))
UDR_OneForcing        = np.zeros((max_K, 1))


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

        # Create different beamformers and evaluate corresponding performance measures

        #--------------------------------------------------------------------
        # Rake Delay-and-Sum
        #--------------------------------------------------------------------

        mics.rakeDelayAndSumWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center),
                                attn=False,
                                ff=False)
        room1.addMicrophoneArray(mics)

        SNR_DS[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                f, 
                                R_n=sigma2 * np.eye(mics.M),
                                dB=True)
        UDR_DS[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                f, 
                                R_n=sigma2 * np.eye(mics.M))


        #--------------------------------------------------------------------
        # Rake Max-SINR
        #--------------------------------------------------------------------

        mics.rakeMaxSINRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=False,
                                attn=True)
        room1.addMicrophoneArray(mics)

        SNR_MaxSINR[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M),
                                 dB=True)
        UDR_MaxSINR[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M))


        #--------------------------------------------------------------------
        # Rake Max-SINR-FF
        #--------------------------------------------------------------------

        mics.rakeMaxSINRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=True,
                                attn=True)
        room1.addMicrophoneArray(mics)

        SNR_MaxSINR_FF[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M),
                                 dB=True)
        UDR_MaxSINR_FF[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M))

        #--------------------------------------------------------------------
        # Rake Max-SINR-FF without attenuating the steering vectors
        #--------------------------------------------------------------------

        mics.rakeMaxSINRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=True,
                                attn=False)
        room1.addMicrophoneArray(mics)

        SNR_MaxSINR_FF_noattn[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                         room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                         f, 
                                         R_n=sigma2 * np.eye(mics.M),
                                         dB=True)
        UDR_MaxSINR_FF_noattn[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                         room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                         f, 
                                         R_n=sigma2 * np.eye(mics.M))


        #--------------------------------------------------------------------
        # Rake Max-UDR
        #--------------------------------------------------------------------

        mics.rakeMaxUDRWeights(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=False,
                                attn=True)
        room1.addMicrophoneArray(mics)

        SNR_MaxUDR[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M),
                                 dB=True)
        UDR_MaxUDR[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M))

        #--------------------------------------------------------------------
        # One-Forcing
        #--------------------------------------------------------------------

        mics.rakeOneForcingWeights(room1.sources[0].getImages(n_nearest=2, ref_point=mics.center), 
                                room1.sources[1].getImages(n_nearest=K, ref_point=mics.center), 
                                R_n=sigma2 * np.eye(mics.M),
                                ff=False,
                                attn=True)
        room1.addMicrophoneArray(mics)

        SNR_OneForcing[K-1] += mics.SNR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2*np.eye(mics.M),
                                 dB=True)
        UDR_OneForcing[K-1] += mics.UDR(room1.sources[0].getImages(n_nearest=K, ref_point=mics.center), 
                                 room1.sources[1].getImages(n_nearest=max_K, ref_point=mics.center), 
                                 f, 
                                 R_n=sigma2 * np.eye(mics.M))


    SNR_DS[K-1] /= float(n_monte_carlo)
    SNR_MaxSINR[K-1] /= float(n_monte_carlo)
    SNR_MaxSINR_FF[K-1] /= float(n_monte_carlo)
    SNR_MaxSINR_FF_noattn[K-1] /= float(n_monte_carlo)
    SNR_MaxUDR[K-1] /= float(n_monte_carlo)
    SNR_OneForcing[K-1] /= float(n_monte_carlo)

    UDR_DS[K-1] /= float(n_monte_carlo)
    UDR_MaxSINR[K-1] /= float(n_monte_carlo)
    UDR_MaxSINR_FF[K-1] /= float(n_monte_carlo)
    UDR_MaxSINR_FF_noattn[K-1] /= float(n_monte_carlo)
    UDR_MaxUDR[K-1] /= float(n_monte_carlo)
    UDR_OneForcing[K-1] /= float(n_monte_carlo)

    print 'Computed for K = ', K, '| SNR =', SNR_MaxSINR[K-1], '| UDR =', UDR_MaxSINR[K-1]


# Plot the results
#
# Make SublimeText use iPython, right? currently it uses python... at least make sure that it uses the correct one.
#
plt.figure()
plt.plot(range(1, 1+max_K), np.concatenate((SNR_DS, 
                                            SNR_MaxSINR,
                                            SNR_MaxUDR,
                                            SNR_OneForcing), axis=1))

plt.figure()
plt.plot(range(1, 1+max_K), np.concatenate((UDR_DS, 
                                            UDR_MaxSINR, 
                                            UDR_MaxUDR,
                                            UDR_OneForcing), axis=1))

plt.show()













p