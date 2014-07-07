
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

import Room as rg
import beamforming as bf
import windows
import utilities as u

# Some simulation parameters
Fs = 8000
t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
absorption = 0.90        # absorption of walls
max_order_sim = 10       # number of generations of image sources in simulation
sigma2_n = 1e-7          # Noise power (should replace with proper SNR)

# Room 1 : Shoe box
room_dim = [4, 6]

# the sources
source1 = [1, 4.5]        # good source

# microphone array design parameters
mic1 = [2, 1.5]         # position
M = 8                   # number of microphones
d = 0.08                # distance between microphones
phi = 0.                # angle from horizontal
max_order_design = 1    # maximum image generation used in design
shape = 'Linear'        # array shape

# create a microphone array
if shape is 'Circular':
    mics = bf.Beamformer.circular2D(Fs, mic1, M, phi, d*M/(2*np.pi)) 
elif shape is 'Poisson':
    mics = bf.Beamformer.poisson(Fs, mic1, M, d) 
else:
    mics = bf.Beamformer.linear2D(Fs, mic1, M, phi, d) 

# define the array processing type
L = 8192                # frame length
hop = 4096              # hop between frames
zp = 4096               # zero padding (front + back)
mics.setProcessing('FrequencyDomain', L, hop, zp, zp)

# The first signal (of interest) is singing
rate1, signal1 = wavfile.read('samples/singing_'+str(Fs)+'.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = u.normalize(signal1)
signal1 = u.highpass(signal1, Fs)
delay1 = 0.

# create the room with sources and mics
room1 = rg.Room.shoeBox2D(
    [0,0],
    room_dim,
    Fs,
    t0 = t0,
    max_order=max_order_sim,
    absorption=absorption,
    sigma2_awgn=sigma2_n)
room1.addSource(source1, signal=signal1, delay=delay1)
room1.addMicrophoneArray(mics)

# compute RIRs
room1.compute_RIR()

# simulate the microphone signals
room1.simulate()

# Compute the beamforming weights depending on room geometry
good_source = room1.sources[0].getImages(max_order=max_order_design)
mics.rakeMaxSINRWeights(good_source, None, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

# process the signal through the beamformer
output = mics.process()

# save the array signal to file
mics.to_wav('raw_output.wav', mono=True, norm=True, type=float)
wavfile.write('proc_output.wav', Fs, u.normalize(output))

# plot the room and beamformer
# make the picutre 6x10cm ~ 2.4x3.9in
fig, ax = room1.plot(img_order=np.minimum(room1.max_order, 1), 
        freq=[500, 1500, 2500],
        figsize=(2.4,3.9), no_axis=True,
        xlim=[-4,8], ylim=[-8,12],
        autoscale_on=False)
fig.savefig('figures/room_no_interferer.pdf')

# Plot the TF of beamformer as seen from source and interferer
plt.figure()
mics.plot_response_from_point(np.array([source1]).T, 
                              legend=('source','interferer'))

# open and plot the two signals
plt.figure()
u.comparePlot(mics.signals[mics.M/2], output, Fs, fft_size=400, 
        norm=True, equal=True, title1='Mic input', title2='Beamformer output')

# show all plots
plt.show()
