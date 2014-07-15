
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
figsize=(1.88,2.24)
xlim=[-4,8]
ylim=[-4.9,9.4]

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
normal_interferer = [3, 4]   # interferer
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
fig, ax = room1.plot(img_order=np.minimum(room1.max_order, 1), 
        freq=freq,
        figsize=figsize, no_axis=True,
        xlim=xlim, ylim=ylim,
        autoscale_on=False)
fig.savefig('figures/scenario_no_interferer_MaxSINR.pdf',
            facecolor=fig.get_facecolor(), edgecolor='none')

plt.figure()
mics.plot_beam_response()

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
fig, ax = room1.plot(img_order=np.minimum(room1.max_order, 1), 
        freq=freq,
        figsize=figsize, no_axis=True,
        xlim=xlim, ylim=ylim,
        autoscale_on=False)
fig.savefig('figures/scenario_interferer_MaxSINR.pdf',
            facecolor=fig.get_facecolor(), edgecolor='none')

plt.figure()
mics.plot_beam_response()

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
fig, ax = room1.plot(img_order=np.minimum(room1.max_order, 1), 
        freq=freq,
        figsize=figsize, no_axis=True,
        xlim=xlim, ylim=ylim,
        autoscale_on=False)
fig.savefig('figures/scenario_interferer_MaxUDR.pdf',
            facecolor=fig.get_facecolor(), edgecolor='none')

plt.figure()
mics.plot_beam_response()

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
fig, ax = room1.plot(img_order=np.minimum(room1.max_order, 1), 
        freq=freq,
        figsize=figsize, no_axis=True,
        xlim=xlim, ylim=ylim,
        autoscale_on=False)
fig.savefig('figures/scenario_interferer_in_direct_path_MaxSINR.pdf',
            facecolor=fig.get_facecolor(), edgecolor='none')

plt.figure()
mics.plot_beam_response()

# show all plots
plt.show()
