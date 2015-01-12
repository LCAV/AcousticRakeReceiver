
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

import Room as rg
import beamforming as bf
import windows
import utilities as u

# Beam pattern figure properties
freq=[800, 1600]
figsize=(4*1.88,2.24)
xlim=[-4,8]
ylim=[-5.2,10]

# Some simulation parameters
Fs = 8000
t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
absorption = 0.90
max_order_sim = 10
sigma2_n = 1e-7

# Room 1 : Shoe box
room_dim = [4, 6]

# the good source is fixed for all 
good_source = [1, 4.5]       # good source
normal_interferer = [2.8, 4.3]   # interferer
hard_interferer = [1.5, 3]   # interferer in direct path

# microphone array design parameters
mic1 = [2, 1.5]         # position
M = 8                    # number of microphones
d = 0.08                # distance between microphones
phi = 0.                # angle from horizontal
max_order_design = 1    # maximum image generation used in design
shape = 'Linear'        # array shape

# create a microphone array
if shape is 'Circular':
    mics = bf.Beamformer.circular2D(Fs, mic1, M, phi, d*M/(2*np.pi)) 
else:
    mics = bf.Beamformer.linear2D(Fs, mic1, M, phi, d) 

# define the array processing type
L = 4096                # frame length
hop = 2048              # hop between frames
zp = 2048               # zero padding (front + back)
mics.setProcessing('FrequencyDomain', L, hop, zp, zp)

# The first signal (of interest) is singing
rate1, signal1 = wavfile.read('samples/singing_'+str(Fs)+'.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = u.normalize(signal1)
signal1 = u.highpass(signal1, Fs)
delay1 = 0.

# the second signal (interferer) is some german speech
rate2, signal2 = wavfile.read('samples/german_speech_'+str(Fs)+'.wav')
signal2 = np.array(signal2, dtype=float)
signal2 = u.normalize(signal2)
signal2 = u.highpass(signal2, Fs)
delay2 = 1.

# create the room with sources and mics
room1 = rg.Room.shoeBox2D(
    [0,0],
    room_dim,
    Fs,
    t0 = t0,
    max_order=max_order_sim,
    absorption=absorption,
    sigma2_awgn=sigma2_n)

# add mic and good source to room
room1.addSource(good_source, signal=signal1, delay=delay1)
room1.addMicrophoneArray(mics)

# start a figure
fig = plt.figure(figsize=figsize)

#rect = fig.patch
#rect.set_facecolor('white')
#rect.set_alpha(0.15)

def nice_room_plot(label, leg=None):
    ax = plt.gca()

    room1.plot(img_order=np.minimum(room1.max_order, 1), 
            freq=freq,
            xlim=xlim, ylim=ylim,
            autoscale_on=False)

    if leg is not None:
        l = ax.legend(leg, loc=(0.005,0.85), fontsize=7, frameon=False)

    ax.text(xlim[1]-1.1, ylim[1]-1.1, label, weight='bold')

    ax.axis('on')
    ax.tick_params(\
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left='off',
        right='off',
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        labelleft='off') # 

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.15)
    ax.patch.edgecolor = 'none'
    ax.patch.linewidth = 0
    ax.edgecolor = 'none'
    ax.linewidth = 0


''' 
SCENARIO 1
Only one source of interest
Max-SINR
'''
print 'Scenario1...'

# Compute the beamforming weights depending on room geometry
good_sources = room1.sources[0].getImages(max_order=max_order_design)
mics.rakeMaxSINRWeights(good_sources, None,
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

# plot the room and beamformer
ax = plt.subplot(1,4,1)
nice_room_plot('A', leg=('800 Hz', '1600 Hz'))

'''
SCENARIO 2
One source or interest and one interefer (easy)
Max-SINR
'''
print 'Scenario2...'

room1.addSource(normal_interferer, signal=signal2, delay=delay2)

# Compute the beamforming weights depending on room geometry
bad_sources = room1.sources[1].getImages(max_order=max_order_design)
mics.rakeMaxSINRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

# plot the room and beamformer
ax = plt.subplot(1,4,2)
nice_room_plot('B')


'''
SCENARIO 3
One source or interest and one interefer (easy)
Max-UDR (eSNR)
'''
print 'Scenario3...'

# Compute the beamforming weights depending on room geometry
mics.rakeMaxUDRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        attn=True, ff=False)

# plot the room and beamformer
plt.subplot(1,4,3)
nice_room_plot('C')

'''
SCENARIO 4
One source and one interferer in the direct path (hard)
Max-SINR
'''
print 'Scenario4...'

room1.sources.pop()
room1.addSource(hard_interferer, signal=signal2, delay=delay2)

# Compute the beamforming weights depending on room geometry
bad_sources = room1.sources[1].getImages(max_order=max_order_design)
mics.rakeMaxSINRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

# plot the room and beamformer
ax = plt.subplot(1,4,4)
nice_room_plot('D')

plt.subplots_adjust(left=0.0, right=1., bottom=0., top=1., wspace=0.05, hspace=0.02)

fig.savefig('figures/beam_scenarios.pdf')
fig.savefig('figures/beam_scenarios.png',dpi=300)

plt.show()

