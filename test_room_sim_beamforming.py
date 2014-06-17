
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile
from scipy.signal import resample

import utilities as u

# Room 1 : Shoe box
p1 = [0, 0]
p2 = [5, 4]

# Some simulation parameters
Fs = 8000
absorption = 0.9
max_order = 15

# The first signal (of interest) is singing
source1 = [4, 0.8]
rate1, signal1 = wavfile.read('samples/singing_8000.wav')
signal1 = np.array(signal1, dtype=float)
delay1 = 1.

if (rate1 - Fs is not 0):
    print 'resampling signal 1'
    signal1 = resample(signal1, np.ceil(len(signal1) / float(rate1) * Fs))

signal1 = u.normalize(signal1)
signal1 = u.highpass(signal1, Fs)

# the second signal (interferer) is some german speech
source2 = [3, 3]
rate2, signal2 = wavfile.read('samples/german_speech_8000.wav')
signal2 = np.array(signal2, dtype=float)
delay2 = 0.5

if (rate2 - Fs is not 0):
    print 'resampling signal 2'
    signal2 = resample(signal2, np.ceil(len(signal2) / float(rate2) * Fs))

signal2 = u.normalize(signal2)
signal2 = u.highpass(signal2, Fs)

# create a microphone array
mic1 = [2, 1.5]
M = 9
d = 0.10
phi = -np.pi / 3

L = 128
hop = 64
zp = 64

#mics = bf.Beamformer.linear2D(mic1, M, phi, d, Fs, 'FrequencyDomain', L, hop, zp, zp)
mics = bf.Beamformer.linear2D(mic1, M, phi, d, Fs, 'TimeDomain', 4096)

# create the room with sources and mics
room1 = rg.Room.shoeBox2D(
    p1,
    p2,
    Fs,
    max_order=max_order,
    absorption=absorption)
room1.addSource(source1, signal=signal1)
room1.addSource(source2, signal=signal2, delay=1.)
room1.addMicrophoneArray(mics)

# compute RIRs
room1.compute_RIR()

# simulate the microphone signals
room1.simulate()

# save the array signal to file
mics.to_wav('raw_output.wav', Fs)

# create the echo beamformer and add to the room
max_order = 3
good_source = room1.sources[0].getImages(max_order=max_order)
bad_source = room1.sources[1].getImages(max_order=max_order)
mics.echoBeamformerWeights(good_source, bad_source, rcond=1e-3)
#mics.rakeDelayAndSumWeights(good_source)

processed = mics.process()

# clip the signal over 16 bit precision
clipped = u.clip(processed, 2 ** 15 - 1, -2 ** 15)

input_signal = mics.signals[mics.M / 2]

hp_sig = u.highpass(clipped, Fs, fc=200)
output = np.array(hp_sig/hp_sig.max(), dtype=float)

wavfile.write('proc_output.wav', Fs, output)

# plot the room and beamformer
room1.plot(img_order=1, freq=[500, 1000, 2000])

# plot the weights
plt.figure()
mics.plot()

# open and plot the two signals
plt.figure()
plt.subplot(2, 2, 1)
u.time_dB(input_signal, Fs)
plt.subplot(2, 2, 2)
u.time_dB(clipped, Fs)
plt.subplot(2, 2, 3)
u.spectrum(input_signal, Fs, 1024)
plt.subplot(2, 2, 4)
u.spectrum(clipped, Fs, 1024)

# show all plots
plt.show()
