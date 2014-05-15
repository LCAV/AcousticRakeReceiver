'''
STFT processing test
====================

This script tests the STFT processing chain.
'''


import numpy as np
import scipy as s

import matplotlib.pyplot as plt

import stft
import windows

import scipy.io.wavfile as wio

filename = 'samples/sputnk1b.wav'

Fs, data = wio.read(filename)
print 'Fs: ',Fs,'Hz'

# prepare a noisy version of the sputnk signal
signal = data.astype('float64')
signal -= np.mean(signal)
SNR_dB = -10.
sigma2_x = np.mean(signal**2)
sigma2_n = sigma2_x*10**(-SNR_dB/10.)
signal += np.random.normal(0., np.sqrt(sigma2_n), signal.shape)

# STFT parameters
L = 512
hop = 256
zp_back = 128
zp_front = 128
N = zp_front + L + zp_back

# create the appropriate window
win = np.concatenate((np.zeros(zp_front), windows.hann(L), np.zeros(zp_back)))

# STFT and inverse
X = stft.stft(signal, L, hop, win=win, zp_back=zp_back, zp_front=zp_front)
iX = stft.istft(X, L, hop, win=None, zp_back=zp_back, zp_front=zp_front)

# plot result
plt.figure()
plt.subplot(1,2,1)
plt.plot(signal)

plt.subplot(1,2,2)
plt.plot(abs(signal[:len(iX)-zp_front-zp_back] - iX[zp_front:-zp_back]))
plt.title('Difference of original and reconstruction from STFT')

plt.figure()
stft.spectroplot(X.T, N, hop, L-hop, Fs, 1000, 1)
plt.title('Orignal spectrum')

# Create an optimal bandpass filter
ntaps = 17
bands = np.array([0., 1000., 1500., 2000., 2500, Fs/2.])/float(Fs)
desired = [0., 1., 0.]
h = s.signal.remez(72, bands, desired)

# plot filter response
freq, response = s.signal.freqz(h)
ampl = np.abs(response)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.semilogy(freq/(2*np.pi), ampl, 'b-')  # freq in Hz
plt.title('Response of the filter')

# Time-domain processing
td_filter = s.signal.convolve(signal, h)
Ztd = stft.stft(td_filter, L, hop, win=windows.hann(L))
plt.figure()
stft.spectroplot(Ztd.T, L, hop, L-hop, Fs, 1000, 1)
plt.title('TD filtered spectrum')

# STFT-domain processing
H = np.fft.fft(h, N)
fd_filter = stft.istft(H*X, L, hop, win=None, zp_back=zp_back, zp_front=zp_front)
Zfd = stft.stft(fd_filter, L, hop, win=windows.hann(L))
plt.figure()
stft.spectroplot(Zfd.T, L, hop, L-hop, Fs, 1000, 1)
plt.title('STFT filtered spectrum')

# plot the difference between the two signals
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.abs(fd_filter[zp_front+hop+len(h):-L+hop-zp_back] - td_filter[hop+len(h):len(fd_filter)-zp_front-L+hop-zp_back]))
plt.title('Difference of TD and STFT filtered signals')

plt.subplot(1,2,2)
a, b = 100, 2000
plt.plot(td_filter[a:b], 'b')
plt.plot(fd_filter[a:b], 'r')
plt.legend(('TD filtering', 'STFT filtering'))

plt.show()

