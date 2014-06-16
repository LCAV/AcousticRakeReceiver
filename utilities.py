
import numpy as np


def clip(signal, high, low):
    '''
    Clip a signal from above at high and from below at low.
    '''
    s = signal.copy()

    s[np.where(s > high)] = high
    s[np.where(s < low)] = low

    return s


def normalize(signal, bits=16):
    '''
    normalize a signal to be in the range of signed integers with a given number of bits
    default number of bits is 16
    '''

    signal /= np.abs(signal).max()
    signal *= 2 ** (bits - 1)
    signal = clip(signal, 2 ** (bits - 1) - 1, -2 ** (bits - 1))

    return signal


def normalize_pwr(sig1, sig2):
    '''
    normalize sig1 to have the same power as sig2
    '''

    # average power per sample
    p1 = np.mean(sig1 ** 2)
    p2 = np.mean(sig2 ** 2)

    # normalize
    return sig1 * np.sqrt(p2 / p1)


def highpass(signal, Fs, fc=50, plot=False):
    '''
    Filter out the really low frequencies, default is below 50Hz
    '''

    # have some predefined parameters
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = 4    # order of the filter
    type = 'butter'

    # normalized cut-off frequency
    wc = 2. * fc / Fs

    # design the filter
    from scipy.signal import iirfilter, lfilter, freqz
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype=type)

    # plot frequency response of filter if requested
    if (plot):
        import matplotlib.pyplot as plt
        w, h = freqz(b, a)

        plt.figure()
        plt.title('Digital filter frequency response')
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title('Digital filter frequency response')
        plt.ylabel('Amplitude Response [dB]')
        plt.xlabel('Frequency (rad/sample)')
        plt.grid()

    # apply the filter
    signal = lfilter(b, a, signal)

    return signal


def time_dB(signal, Fs, bits=16):
    '''
    Compute the signed dB amplitude of the oscillating signal
    normalized wrt the number of bits used for the signal
    '''

    import matplotlib.pyplot as plt

    # min dB
    lsb = -20 * np.log10(2.) * (bits - 1)

    # magnitude in dB (clipped)
    pos = clip(signal, 2. ** (bits - 1) - 1, 1.) / 2. ** (bits - 1)
    neg = -clip(signal, -1., -2. ** (bits - 1)) / 2. ** (bits - 1)

    mag_pos = np.zeros(signal.shape)
    Ip = np.where(pos > 0)
    mag_pos[Ip] = 20 * np.log10(pos[Ip]) + lsb + 1

    mag_neg = np.zeros(signal.shape)
    In = np.where(neg > 0)
    mag_neg[In] = 20 * np.log10(neg[In]) + lsb + 1

    plt.plot(np.arange(len(signal)) / float(Fs), mag_pos - mag_neg)
    yticks = plt.getp(plt.gca(), 'yticks')

    newticks = yticks.copy()
    Ip = np.where(yticks > 0)
    In = np.where(yticks < 0)
    newticks[Ip] = newticks[Ip] + np.floor(lsb) + 1
    newticks[In] = newticks[In] - np.floor(lsb) - 1
    plt.setp(plt.gca(), 'yticklabels', -newticks)


def spectrum(signal, Fs, N):

    import stft
    import windows

    F = stft.stft(signal, N, N / 2, win=windows.hann(N))
    stft.spectroplot(F.T, N, N / 2, N / 2, Fs)
