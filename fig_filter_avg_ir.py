
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
N = int(1.5*Fs)                # frame length
zero_padding_factor = 2
mics.setProcessing('TimeDomain', N)

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
room1.addSource(normal_interferer, signal=signal2, delay=delay2)
room1.addMicrophoneArray(mics)

# plot the room and beamformer
fig = plt.figure(figsize=(4,3))

# define a new set of colors for the beam patterns
newmap = plt.get_cmap('autumn')
desat = 0.7
plt.gca().set_color_cycle([newmap(k) for k in desat*np.linspace(0,1,3)])


'''
BEAMFORMER 1
Rake-MaxSINR
'''
print 'Beamformer 1...'

# Compute the beamforming weights depending on room geometry
good_sources = room1.sources[0].getImages(max_order=max_order_design)
bad_sources = room1.sources[1].getImages(max_order=max_order_design)
mics.rakeMaxSINRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

mics.plot_IR(sum_ir=True, norm=1., zp=zero_padding_factor, linewidth=0.5)

'''
BEAMFORMER 2
Rake-MaxUDR (eSNR)
'''
print 'Beamformer 2...'

# Compute the beamforming weights depending on room geometry
mics.rakeMaxUDRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        attn=True, ff=False)

mics.plot_IR(sum_ir=True, norm=1., zp=zero_padding_factor, linewidth=0.5)

'''
BEAMFORMER 3
MaxSINR (MVDR)
'''
print 'Beamformer 3...'

# Compute the beamforming weights depending on room geometry
mics.rakeMaxSINRWeights(room1.sources[0].getImages(max_order=0),
                       room1.sources[1].getImages(max_order=0),
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0.,
                        attn=True, ff=False)

mics.plot_IR(sum_ir=True, norm=1., zp=zero_padding_factor, linewidth=0.5)

'''
FINISH PLOT
'''


leg = ('Rake-MaxSINR', 'Rake-MaxUDR', 'MaxSINR')
plt.legend(leg, fontsize=7, loc='upper left', frameon=False, labelspacing=0)

# Hide right and top axes
ax1 = plt.gca()

# prepare axis
#ax1.autoscale(tight=True, axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 5))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# set x axis limit
#ax1.set_xlim(0.5, 1.5)

# Set ticks
plt.xticks(np.arange(0, float(N)/Fs+1, 0.5), size=9)
plt.xlim(0, 1.5)
plt.yticks([])

# Set labels
plt.xlabel(r'Time [s]', fontsize=10)
plt.ylabel('')
plt.tight_layout()

fig.savefig('figures/AvgIR.pdf')

# show all plots
plt.show()
