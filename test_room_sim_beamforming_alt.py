
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
M = 12
d = 0.08
phi = -np.pi / 2.2
mics = bf.Beamformer.linear2D(Fs, mic1, M, phi, d) 

# define the processing type
L = 2048
hop = 1024
zp = 1024
mics.setProcessing('FrequencyDomain', L, hop, zp, zp)

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
mics.to_wav('raw_output.wav', mono=True, norm=True, type=float)

# create the echo beamformer and add to the room
max_order = 3
good_source = room1.sources[0].getImages(max_order=1)
bad_source = room1.sources[1].getImages(max_order=5)
mics.rakeOneForcingWeights(good_source, bad_source, R_n=1e-5*np.eye(mics.M), attn=True, ff=False)

# process the signal through the beamformer
processed = mics.process()

input_signal = u.normalize(mics.signals[mics.M / 2])

output = np.array(u.normalize(processed), dtype=float)

wavfile.write('proc_output.wav', Fs, output)

# plot the room and beamformer
room1.plot(img_order=1, freq=[500, 1000, 2000])

# plot the weights
plt.figure()
mics.plot()

# open and plot the two signals
plt.figure()
u.comparePlot(input_signal, output, Fs, 400, 
        title1='Mic input', title2='Beamformer output')

# show all plots
plt.show()
