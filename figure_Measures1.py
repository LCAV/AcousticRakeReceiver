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
freqs = np.array([1000])
f = 1000
sigma2 = 1e-3

mics = bf.Beamformer.circular2D(Fs, mic1, M, 0, d)
mics.frequencies = freqs

# How much to simulate?
max_K = 21
n_monte_carlo = 20000

beamformer_names = ['DS',
                    'Max-SINR',
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
SNR_ci = {}
SNR_ci_minus = {}
SNR_ci_plus = {}

UDR = {}
UDR_ci = {}

for bf in beamformer_names:
    SNR.update({bf: np.zeros((max_K, n_monte_carlo))})
    SNR_ci.update({bf: np.float(0)})
    UDR.update({bf: np.zeros((max_K, n_monte_carlo))})
    UDR_ci.update({bf: np.float(0)})

SNR_ci_minus = SNR_ci.copy()
SNR_ci_plus = SNR_ci.copy()

for K in range(0, max_K):
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
        for i, bf in enumerate(beamformer_names):
    
            if (bf is 'DS') or (bf is 'Max-SINR'):
                n_nearest = 1
            else:
                n_nearest = K+1


            bf_weights_fun[i](room1.sources[0].getImages(n_nearest=n_nearest, ref_point=mics.center), 
                              room1.sources[1].getImages(n_nearest=n_nearest, ref_point=mics.center), 
                              R_n=sigma2 * np.eye(mics.M),
                              ff=False,
                              attn=True)
    
            room1.addMicrophoneArray(mics)

            # TO DO: Average in dB or in the linear scale?
            SNR[bf][K][n] = mics.SNR(room1.sources[0].getImages(n_nearest=K+1, ref_point=mics.center), 
                                     room1.sources[1].getImages(n_nearest=max_K+1, ref_point=mics.center), 
                                     f, 
                                     R_n=sigma2 * np.eye(mics.M),
                                     dB=True)
            UDR[bf][K][n] = mics.UDR(room1.sources[0].getImages(n_nearest=K+1, ref_point=mics.center), 
                                     room1.sources[1].getImages(n_nearest=max_K+1, ref_point=mics.center), 
                                     f, 
                                     R_n=sigma2 * np.eye(mics.M),
                                     dB=True)

    print 'Computed for K =', K


# Compute the confidence regions, symmetrically, and then separately for
# positive and for negative differences
p = 0.5
for bf in beamformer_names:
    err_SNR = SNR[bf][K] - np.median(SNR[bf][K])
    n_plus = np.sum(err_SNR >= 0)
    n_minus = np.sum(err_SNR < 0)
    SNR_ci[bf] = np.sort(np.abs(err_SNR))[np.floor(p*n_monte_carlo)]
    SNR_ci_plus[bf] = np.sort(err_SNR[err_SNR >= 0])[np.floor(p*n_plus)]
    SNR_ci_minus[bf] = np.sort(-err_SNR[err_SNR < 0])[np.floor(p*n_minus)]

    err_UDR = UDR[bf][K] - np.median(UDR[bf][K])
    UDR_ci[bf] = np.sort(np.abs(err_UDR))[np.floor(p*n_monte_carlo)]


#---------------------------------------------------------------------
# Export the SNR figure
#---------------------------------------------------------------------

plt.figure(figsize=(4, 3))

newmap = plt.get_cmap('gist_heat')
ax1 = plt.gca()
ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.9,len(beamformer_names))])

from itertools import cycle
lines = ['-s','-o','-v','-D','->']
linecycler = cycle(lines)

for i, bf in enumerate(beamformer_names):
    p, = plt.plot(range(0, max_K), 
            np.median(SNR[bf], axis=1), 
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)

plt.fill_between(range(0, max_K),
                 np.median(SNR['Rake-MaxSINR'], axis=1) - SNR_ci['Rake-MaxSINR'],
                 np.median(SNR['Rake-MaxSINR'], axis=1) + SNR_ci['Rake-MaxSINR'],
                 color='grey',
                 linewidth=0.3,
                 edgecolor='k',
                 alpha=0.7)

# Hide right and top axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 10))
ax1.spines['left'].set_position(('outward', 15))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# Make ticks nicer
ax1.xaxis.set_tick_params(width=.3, length=3)
ax1.yaxis.set_tick_params(width=.3, length=3)

# Make axis lines thinner
for axis in ['bottom','left']:
  ax1.spines[axis].set_linewidth(0.3)

# Set ticks fontsize
plt.xticks(size=9)
plt.yticks(size=9)

# Set labels
plt.xlabel(r'Number of images $K$', fontsize=10)
plt.ylabel('Output SINR [dB]', fontsize=10)
plt.tight_layout()


plt.legend(beamformer_names, fontsize=7, loc='upper left', frameon=False, labelspacing=0)

plt.savefig('figures/SINR_vs_K.png')
plt.savefig('figures/SINR_vs_K.pdf')

plt.close()

#---------------------------------------------------------------------
# Export the UDR figure
#---------------------------------------------------------------------

plt.figure(figsize=(4, 3))

newmap = plt.get_cmap('gist_heat')
ax1 = plt.gca()
ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.9,len(beamformer_names))])

for i, bf in enumerate(beamformer_names):
    p, = plt.plot(range(0, max_K), 
            np.median(UDR[bf], axis=1), 
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)

plt.fill_between(range(0, max_K),
                 np.median(UDR['Rake-MaxUDR'], axis=1) - UDR_ci['Rake-MaxUDR'],
                 np.median(UDR['Rake-MaxUDR'], axis=1) + UDR_ci['Rake-MaxUDR'],
                 color='grey',
                 linewidth=0.3,
                 edgecolor='k',
                 alpha=0.7)

# Hide right and top axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 10))
ax1.spines['left'].set_position(('outward', 15))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# Make ticks nicer
ax1.xaxis.set_tick_params(width=.3, length=3)
ax1.yaxis.set_tick_params(width=.3, length=3)

# Make axis lines thinner
for axis in ['bottom','left']:
  ax1.spines[axis].set_linewidth(0.3)

# Set ticks fontsize
plt.xticks(size=9)
plt.yticks(size=9)

# Set labels
plt.xlabel(r'Number of images $K$', fontsize=10)
plt.ylabel('Output UDR [dB]', fontsize=10)
plt.tight_layout()


plt.legend(beamformer_names, fontsize=7, loc='upper left', frameon=False, labelspacing=0)

plt.savefig('figures/UDR_vs_K.png')
plt.savefig('figures/UDR_vs_K.pdf')
