
import numpy as np
import scipy.special as spfun

import matplotlib
import constants

import matplotlib.colors as colors
import matplotlib.cm as cmx

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf

# Room 1 : Shoe box
p1 = np.array([0, 0])
p2 = np.array([4, 6])
mic1 = [2, 3]
Fs = 44100
absorption = 0.8
max_order = 4

# Parameters for the theoretical curve
a = 5
b = 10
Delta = b-a

# Create a microphone array
M = 12
d = 0.2
frequencies = np.arange(25, 600, 5)

mics = bf.Beamformer.linear2D(Fs, mic1, M, 0, d)

K_list = [16, 8]
n_monte_carlo = 1000

SNR_gain = np.zeros((len(K_list), frequencies.size))
SNR_gain_theory = np.zeros((len(K_list), frequencies.size))

for i_K, K in enumerate(K_list):
    for i, f in enumerate(frequencies):
        print 'Simulating for the frequency', f
        for n in range(0, n_monte_carlo):

            # Generate a source at a random location. TO DO: Add a bounding box for
            # sources!
            source1 = p1 + np.random.rand(2) * (p2 - p1)

            # Create the room
            room1 = rg.Room.shoeBox2D(
                p1,
                p2,
                Fs,
                max_order=max_order,
                absorption=absorption)
            room1.addSource(source1)
            room1.addMicrophoneArray(mics)

            A = mics.steering_vector_2D_from_point(f, room1.sources[0].getImages(n_nearest=K+1, ref_point=mics.center), attn=False)
            SNR_gain[i_K][i] += np.linalg.norm(np.sum(A, axis=1))**2 / np.linalg.norm(A[:, 0])**2

        SNR_gain[i_K][i] /= n_monte_carlo

        m = np.arange(M)
        kappa = 2*np.pi*f / constants.c
        SNR_gain_theory[i_K][i] = np.sum(np.abs(A[0,:]))*np.sum(1 + 2*spfun.jv(0, m*d*kappa)**2 * (1-np.cos(Delta * kappa)) / (Delta * kappa)**2)/np.linalg.norm(A[:, 0])**2

# Plot the results
plt.figure(figsize=(4, 2.5))
ax1 = plt.gca()

newmap = plt.get_cmap('gist_heat')
ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.8,2)])

plt.plot(frequencies, 10*np.log10(SNR_gain.T))
plt.plot(frequencies, 10*np.log10(SNR_gain_theory.T), 'o', markersize=2.5, markeredgewidth=.3)

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

# Set ticks
plt.xticks(size=9)
plt.yticks(size=9)

# Do the legend
plt.legend([r'Simulation, $K=16$',
            r'Simulation, $K=8$',
            r'Theorem, $K=16$',
            r'Theorem, $K=8$'], fontsize=7, loc='upper right', frameon=False, labelspacing=0)

# Set labels
plt.xlabel(r'Frequency [Hz]', fontsize=10)
plt.ylabel('SNR gain [dB]', fontsize=10)
plt.tight_layout()

plt.savefig('figures/SNR_gain.png')
plt.savefig('figures/SNR_gain.pdf')

