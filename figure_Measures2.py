import numpy as np
import matplotlib
import constants
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

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
Fs = 44100
absorption = 0.8
max_order = 4

# create a microphone array
mic1 = [2, 3]
M = 12
d = 0.3
freqs = np.arange(100,4000,200)
sigma2 = 1e-3

mics = bf.Beamformer.circular2D(Fs, mic1, M, 0, d)
mics.frequencies = freqs

# How much to simulate?
n_monte_carlo = 1000

beamformer_names = ['DS',
                    'MVDR/Max-SINR',
                    'Rake-DS',
                    'Rake-MaxSINR',
                    'Rake-MaxUDR']
                    # 'Rake-OF']
bf_weights_fun   = [mics.rakeDelayAndSumWeights,
                    mics.rakeMaxSINRWeights,
                    mics.rakeDelayAndSumWeights,
                    mics.rakeMaxSINRWeights,
                    mics.rakeMaxUDRWeights]
                    # mics.rakeOneForcingWeights]

SNR = {}
UDR = {}
for bf in beamformer_names:
    SNR.update({bf: np.zeros((freqs.size, n_monte_carlo))})
    UDR.update({bf: np.zeros((freqs.size, n_monte_carlo))})

K = 10
max_K = 1000


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
    for i_bf, bf in enumerate(beamformer_names):

        if (bf is 'DS') or (bf is 'MVDR/Max-SINR'):
            n_nearest = 1
        else:
            n_nearest = K+1

        bf_weights_fun[i_bf](room1.sources[0].getImages(n_nearest=n_nearest, ref_point=mics.center), 
                        None, # room1.sources[1].getImages(n_nearest=n_nearest, ref_point=mics.center), 
                        R_n=sigma2 * np.eye(mics.M),
                        ff=False,
                        attn=True)

        room1.addMicrophoneArray(mics)

        # TO DO: Average in dB or in the linear scale?
        for i_f, f in enumerate(freqs):
            SNR[bf][i_f][n] = mics.SNR(room1.sources[0].getImages(n_nearest=K+1, ref_point=mics.center), 
                                   None, # room1.sources[1].getImages(n_nearest=max_K+1, ref_point=mics.center), 
                                   f, 
                                   R_n=sigma2 * np.eye(mics.M),
                                   dB=True)
            UDR[bf][i_f][n] = mics.UDR(room1.sources[0].getImages(n_nearest=K+1, ref_point=mics.center), 
                                   room1.sources[1].getImages(n_nearest=max_K+1, ref_point=mics.center), 
                                   f, 
                                   R_n=sigma2 * np.eye(mics.M))

    print 'Computed for n =', n

# Plot the results
#
# Make SublimeText use iPython, right? currently it uses python... at least make sure that it uses the correct one.
#
plt.figure(figsize=(4, 3))

n_curves = len(beamformer_names)
values = range(n_curves)
cmap = plt.get_cmap('gist_heat')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1]+1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

for i, bf in enumerate(beamformer_names):
    color_val = scalarMap.to_rgba(values[-i])

    p, = plt.plot(freqs,
                np.mean(SNR[bf], axis=1), 
                linewidth=1,
                color=color_val)

# Hide right and top axes
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 5))
ax1.spines['left'].set_position(('outward', 10))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# Set ticks
plt.xticks(size=9)
plt.yticks(size=9)

# Set labels
plt.xlabel(r'Frequency $f$', fontsize=10)
plt.ylabel('Output SINR', fontsize=10)
plt.tight_layout()


plt.legend(beamformer_names, fontsize=7, loc='lower right', frameon=False, labelspacing=0)

plt.savefig('SINR_vs_freq.png')
plt.savefig('SINR_vs_freq.pdf')



