
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import sys

Fs = int(sys.argv[1])
filename = sys.argv[2]

base, suffix = filename.split('.')

rate, signal = wavfile.read(filename)

if (rate == Fs):
    print 'Sampling rate is already matching.'
    sys.exit(1)

signal = resample(
    np.array(
        signal, dtype=float), np.ceil(
            len(signal) / float(rate) * Fs))

wavfile.write(
    base +
    '_' +
    str(Fs) +
    '.' +
    suffix,
    Fs,
    np.array(
        signal,
        dtype=np.float))
