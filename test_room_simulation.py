
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf
from scipy.io import wavfile
from scipy.signal import resample

# Room 1 : Shoe box
p1 = [0, 0]
p2 = [20, 10]

# Some simulation parameters
Fs = 44000
absorption = 0.8
max_order = 15

# The first signal is Homer
source1 = [5, 3]
rate1, signal1 = wavfile.read('samples/singing.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = signal1 / signal1.max() * 32767
delay1 = 1.

if (rate1 - Fs is not 0):
    signal1 = resample(signal1, np.ceil(len(signal1) / float(rate1) * Fs))

# the second signal is some speech
source2 = [14, 7.5]
rate2, signal2 = wavfile.read('samples/german_speech.wav')
signal2 = np.array(signal2, dtype=float)
signal2 = signal2 / signal2.max() * 32767
delay2 = 0.5

if (rate2 - Fs is not 0):
    signal2 = resample(signal2, np.ceil(len(signal2) / float(rate2) * Fs))

# create a microphone array
mic1 = [18, 6]
M = 9
d = 0.2
f = 1000.
phi = -np.pi / 3
mics = bf.Beamformer.linear2D(mic1, M, d=d)

# create the room with sources
room1 = rg.Room.shoeBox2D(
    p1,
    p2,
    Fs,
    max_order=max_order,
    absorption=absorption)
room1.addSource(source1, signal=signal1)
room1.addSource(source2, signal=signal2, delay=1.)

# create the echo beamformer and add to the room
room1.addMicrophoneArray(mics)

# compute RIRs
room1.compute_RIR()

# simulate the microphone signals
room1.simulate()

# save the array signal to file
mics.to_wav('output.wav', Fs)

rate, signal = wavfile.read('output.wav')

# plot signals to see if we get something meaningful
room1.plot(img_order=2)

# plot all the RIR
S = len(room1.sources)
i = 1
plt.figure()
for m in np.arange(mics.M):
    for s in np.arange(S):
        plt.subplot(M, S, i)
        plt.plot(room1.rir[m][s])
        i += 1


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(signal1)
plt.subplot(2, 1, 2)
plt.plot(mics.signals[0])
plt.plot(mics.signals[1], 'r')
plt.show()
