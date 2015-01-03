
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

import Room as rg
import beamforming as bf

from constants import eps
from stft import stft, spectroplot
import windows
import utilities as u

# Spectrogram figure properties
figsize=(7.87, 1.65)        # figure size
figsize2=(7.87, 1.5*1.65)        # figure size
fft_size = 512              # fft size for analysis
fft_hop  = 8               # hop between analysis frame
fft_zp = 512
analysis_window = np.concatenate((windows.hann(fft_size), np.zeros(fft_zp)))
t_cut = 0.83                # length in [s] to remove at end of signal (no sound)

# Some simulation parameters
Fs = 8000
t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
absorption = 0.90
max_order_sim = 10
SNR_at_mic = 20          # SNR at center of microphone array in dB

# Room 1 : Shoe box
room_dim = [4, 6]

# the good source is fixed for all 
good_source = [1, 4.5]       # good source
normal_interferer = [2.8, 4.3]   # interferer

# microphone array design parameters
mic1 = [2, 1.5]         # position
M = 8                   # number of microphones
d = 0.08                # distance between microphones
phi = 0.                # angle from horizontal
design_order_good = 3   # maximum image generation used in design
design_order_bad  = 3   # maximum image generation used in design
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

# compute the noise variance at center of array wrt signal1 and SNR
sigma2_signal1 = np.mean(signal1**2)
distance = np.linalg.norm(mics.center[:,0] - np.array(good_source))
sigma2_n = sigma2_signal1/(4*np.pi*distance)**2/10**(SNR_at_mic/10)

# create the room with sources and mics
room1 = rg.Room.shoeBox2D(
    [0,0],
    room_dim,
    Fs,
    t0 = t0,
    max_order=max_order_sim,
    absorption=absorption,
    sigma2_awgn=sigma2_n)

# add mic and sources to room
room1.addSource(good_source, signal=signal1, delay=delay1)
room1.addSource(normal_interferer, signal=signal2, delay=delay2)
room1.addMicrophoneArray(mics)

# Compute RIR and simulate propagation of signals
room1.compute_RIR()
room1.simulate()

''' 
BEAMFORMER 1: Max SINR
'''
print 'Max SINR...'

# Compute the beamforming weights depending on room geometry
good_sources = room1.sources[0].getImages(max_order=0)
bad_sources = room1.sources[1].getImages(max_order=0)
mics.rakeMaxSINRWeights(good_sources, bad_sources,
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

output_mvdr = mics.process()

# high-pass and normalize
output_mvdr = u.highpass(output_mvdr, Fs)
output_mvdr = u.normalize(output_mvdr)

'''
BEAMFORMER 2: Rake MaxSINR
'''
print 'Rake MaxSINR...'


# Compute the beamforming weights depending on room geometry
good_sources = room1.sources[0].getImages(max_order=design_order_good)
bad_sources = room1.sources[1].getImages(max_order=design_order_bad)
mics.rakeMaxSINRWeights(good_sources, bad_sources, 
                        R_n = sigma2_n*np.eye(mics.M), 
                        rcond=0., 
                        attn=True, ff=False)

output_maxsinr = mics.process()

# high-pass and normalize
output_maxsinr = u.highpass(output_maxsinr, Fs)
output_maxsinr = u.normalize(output_maxsinr)

'''
PLOT SPECTROGRAM
'''

dSNR = u.dB(room1.dSNR(mics.center[:,0], source=0), power=True)
print 'The direct SNR for good source is ' + str(dSNR)

# as comparison pic central mic signal
input_mic = mics.signals[mics.M/2]

# high-pass and normalize
input_mic = u.highpass(input_mic, Fs)
input_mic = u.normalize(input_mic)

# remove a bit of signal at the end and time-align all signals.
# the delays were visually measured by plotting the signals
n_lim = np.ceil(len(input_mic) - t_cut*Fs)
input_clean = signal1[:n_lim]
input_mic = input_mic[105:n_lim+105]
output_mvdr = output_mvdr[31:n_lim+31]
output_maxsinr = output_maxsinr[31:n_lim+31]

# save all files for listening test
wavfile.write('output_samples/input_mic.wav', Fs, input_mic)
wavfile.write('output_samples/output_maxsinr.wav', Fs, output_mvdr)
wavfile.write('output_samples/output_rake-maxsinr.wav', Fs, output_maxsinr)

# compute time-frequency planes
F0 = stft(input_clean, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F1 = stft(input_mic, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F2 = stft(output_mvdr, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F3 = stft(output_maxsinr, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)

# (not so) fancy way to set the scale to avoid having the spectrum
# dominated by a few outliers
p_min = 7
p_max = 100
all_vals = np.concatenate((u.dB(F1+eps), 
                           u.dB(F2+eps), 
                           u.dB(F3+eps),
                           u.dB(F0+eps))).flatten()
vmin, vmax = np.percentile(all_vals, [p_min, p_max])

cmap = 'afmhot'
interpolation='sinc'

fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=4)

ax = plt.subplot(1,4,1)
spectroplot(F0.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(1,4,2)
spectroplot(F1.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(1,4,3)
spectroplot(F2.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax, 
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(1,4,4)
spectroplot(F3.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax, 
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

plt.subplots_adjust(left=0.0, right=1., bottom=0., top=1., wspace=0.02)

fig.savefig('figures/spectrograms.pdf', dpi=600)


# Second figure with difference spectrograms

fig, ax = plt.subplots(figsize=figsize2, nrows=1, ncols=4)

ax = plt.subplot(2,4,1)
spectroplot(F0.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(2,4,2)
spectroplot(F1.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(2,4,3)
spectroplot(F2.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax, 
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(2,4,4)
spectroplot(F3.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax, 
        cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

# now the difference spectrograms
N2 = (fft_size+fft_zp)/2 + 1
E1 = np.abs(u.dB(F0.T) - u.dB(F1.T))[:N2,:]
E2 = np.abs(u.dB(F0.T) - u.dB(F2.T))[:N2,:]
E3 = np.abs(u.dB(F0.T) - u.dB(F3.T))[:N2,:]
p_min = 0
p_max = 100
all_vals = np.concatenate((E1, E2, E3)).flatten()
vmin, vmax = np.percentile(all_vals, [p_min, p_max])

ax = plt.subplot(2,4,6)
plt.imshow(E1,
        aspect='auto', 
        origin='lower',
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(2,4,7)
plt.imshow(E2,
        aspect='auto', 
        origin='lower',
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

ax = plt.subplot(2,4,8)
plt.imshow(E3,
        aspect='auto', 
        origin='lower',
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axis('off')

plt.subplots_adjust(left=0.0, right=1., bottom=0., top=1., wspace=0.02, hspace=0.02)

fig.savefig('figures/spectrograms2.pdf', dpi=600)

plt.show()
