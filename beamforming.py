import numpy as np
import cvxpy as cp
from time import sleep

import constants

import windows
import stft


#=========================================================================
# Free (non-class-member) functions related to beamformer design
#=========================================================================


def H(A, **kwargs):
    '''Returns the conjugate (Hermitian) transpose of a matrix.'''

    return np.transpose(A, **kwargs).conj()

def sumcols(A): 
    '''Sums the columns of a matrix (np.array). The output is a 2D np.array
        of dimensions M x 1.'''

    return np.sum(A, axis=1, keepdims=1)
    

def mdot(*args):
    '''Left-to-right associative matrix multiplication of multiple 2D
    ndarrays'''

    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret,a)

    return ret


def complex_to_real_matrix(A):

    A_real = np.real(A)
    A_imag = np.imag(A)

    A_ctr = np.vstack([np.hstack([A_real, -A_imag]),
                       np.hstack([A_imag, A_real])])

    return A_ctr


def real_to_complex_vector(b):

    n = b.shape[0] / 2
    return b[0:n] + 1j * b[n:]


def echo_beamformer_cvx(A_good, A_bad):

    # Expand complex matrices and vectors to real matrices and vectors
    A_good_ctr_H = complex_to_real_matrix(H(A_good))
    A_bad_ctr_H = complex_to_real_matrix(H(A_bad))

    M = A_good.shape[0]
    K = A_good.shape[1]

    h = cp.Variable(2 * M)

    # Objective: minimize(norm(h.H * A_good)^2)

    objective = cp.Minimize(cp.sum_entries(cp.square(A_bad_ctr_H * h)))

    # Constraint: sum(h.H * A_good) = 1 + 0*1j
    constraints = [cp.sum_entries((A_good_ctr_H * h)[0:K]) == 1,
                   cp.sum_entries((A_good_ctr_H * h)[K:]) == 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return np.array(real_to_complex_vector(h.value))


def distance(X, Y):
    # Assume X, Y are arrays, *not* matrices
    X = np.array(X)
    Y = np.array(Y)

    XX, YY = [np.sum(A ** 2, axis=0, keepdims=True) for A in (X, Y)]

    return np.sqrt(np.abs((XX.T + YY) - 2 * np.dot(X.T, Y)))


def unit_vec2D(phi):
    return np.array([[np.cos(phi), np.sin(phi)]]).T


def linear2DArray(center, M, phi, d):
    u = unit_vec2D(phi)
    return np.array(center)[:, np.newaxis] + d * \
        (np.arange(M)[np.newaxis, :] - (M - 1.) / 2.) * u


def circular2DArray(center, M, phi0, radius):
    phi = np.arange(M) * 2. * np.pi / M
    return np.array(center)[:, np.newaxis] + radius * \
        np.vstack((np.cos(phi + phi0), np.sin(phi + phi0)))


def fir_approximation_ls(weights, T, n1, n2):

    freqs_plus = np.array(weights.keys())[:, np.newaxis]
    freqs = np.vstack([freqs_plus,
                       -freqs_plus])
    omega = 2 * np.pi * freqs
    omega_discrete = omega * T

    n = np.arange(n1, n2)

    # Create the DTFT transform matrix corresponding to a discrete set of
    # frequencies and the FIR filter indices
    F = np.exp(-1j * omega_discrete * n)
    print np.linalg.pinv(F)

    w_plus = np.array(weights.values())[:, :, 0]
    w = np.vstack([w_plus,
                   w_plus.conj()])

    return np.linalg.pinv(F).dot(w)


#=========================================================================
# Classes (microphone array and beamformer related)
#=========================================================================


class MicrophoneArray(object):

    """Microphone array class."""

    def __init__(self, R, Fs):
        self.dim = R.shape[0]   # are we in 2D or in 3D
        self.M = R.shape[1]     # number of microphones
        self.R = R              # array geometry

        self.Fs = Fs            # sampling frequency of microphones

        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)


    def to_wav(self, filename, mono=False, norm=False, type=float):
        '''
        Save all the signals to wav files
        '''
        from scipy.io import wavfile

        if mono is True:
            signal = self.signals[self.M/2]
        else:
            signal = self.signals.T  # each column is a channel

        if type is float:
            bits = None
        elif type is np.int8:
            bits = 8
        elif type is np.int16:
            bits = 16
        elif type is np.int32:
            bits = 32
        elif type is np.int64:
            bits = 64
        else:
            raise NameError('No such type.')

        if norm is True:
            from utilities import normalize
            signal = normalize(signal, bits=bits)

        signal = np.array(signal, dtype=type)

        wavfile.write(filename, self.Fs, signal)

    @classmethod
    def linear2D(cls, Fs, center, M, phi, d):
        return MicrophoneArray(linear2DArray(center, M, phi, d), Fs)

    @classmethod
    def circular2D(cls, Fs, center, M, phi, radius):
        return MicrophoneArray(circular2DArray(center, M, phi, radius), Fs)


class Beamformer(MicrophoneArray):

    """Beamformer class. At some point, in some nice way, the design methods
    should also go here. Probably with generic arguments."""

    def __init__(self, R, Fs):
        MicrophoneArray.__init__(self, R, Fs)

        # All of these will be defined in setProcessing
        self.processing = None          # Time or frequency domain
        self.N = None
        self.L = None
        self.hop = None
        self.zpf = None
        self.zpb = None

        self.frequencies = None         # frequencies of weights are defined in processing

        # weights will be computed later, the array is of shape (M, N/2+1)
        self.weights = None


    def setProcessing(self, processing, *args):
        """ Setup the processing type and parameters """

        self.processing = processing

        if processing == 'FrequencyDomain':
            self.L  = args[0]    # frame size
            if self.L % 2 is not 0: self.L += 1     # ensure even length
            self.hop = args[1]   # hop between two successive frames
            self.zpf = args[2]   # zero-padding front
            self.zpb = args[3]   # zero-padding back
            self.N = self.L + self.zpf + self.zpb
            if self.N % 2 is not 0:  # ensure even length
                self.N += 1
                self.zpb += 1
        elif processing == 'TimeDomain':
            self.N = args[0]                    # filter length
            if self.N % 2 is not 0: self.N += 1     # ensure even length
        elif processing == 'Total':
            self.N = self.signals.shape[1]
        else:
            raise NameError(processing + ': No such type of processing')

        # for now only support equally spaced frequencies
        self.frequencies = np.arange(0, self.N/2+1)/float(self.N)*float(self.Fs)
        
    def __add__(self, y):
        """ Concatenates two beamformers together """

        return Beamformer(np.concatenate((self.R, y.R), axis=1), self.Fs)


    # def steering_vector_2D_ff(self, frequency, phi, attn=False):
    #     phi = np.array([phi]).reshape(phi.size)
    #     omega = 2*np.pi*frequency

    #     return np.exp(-1j*omega*)


    def steering_vector_2D(self, frequency, phi, dist, attn=False):

        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + self.center

        D = distance(self.R, X)
        omega = 2 * np.pi * frequency

        if attn:
            # TO DO 1: This will mean slightly different absolute value for
            # every entry, even within the same steering vector. Perhaps a
            # better paradigm is far-field with phase carrier.
            return 1. / (4 * np.pi) / D * np.exp(-1j * omega * D / constants.c)
        else:
            return np.exp(-1j * omega * D / constants.c)


    def steering_vector_2D_from_point(self, frequency, source, attn=True, ff=False):
        """ Creates a steering vector for a particular frequency and source

        Args:
            frequency
            source: location in cartesian coordinates
            attn: include attenuation factor if True
            ff:   uses far-field distance if true

        Return: 
            A 2x1 ndarray containing the steering vector
        """
        phi = np.angle(         (source[0] - self.center[0, 0]) 
                         + 1j * (source[1] - self.center[1, 0]))
        if (not ff):
            dist = np.sqrt(np.sum((source - self.center) ** 2, axis=0))
        else:
            dist = constants.ffdist
        return self.steering_vector_2D(frequency, phi, dist, attn=attn)


    def response(self, phi_list, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[:,i_freq]), self.steering_vector_2D(
            self.frequencies[i_freq], phi_list, constants.ffdist))

        return self.frequencies[i_freq], bfresp


    def response_from_point(self, x, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[:,i_freq]), self.steering_vector_2D_from_point(
            self.frequencies[i_freq], x, attn=True, ff=False))

        return self.frequencies[i_freq], bfresp


    def plot_response_from_point(self, x, legend=None):

        if np.rank(x) == 0:
            x = np.array([x])

        import matplotlib.pyplot as plt

        HF = np.zeros((x.shape[1], self.frequencies.shape[0]), dtype=complex)
        for k,p in enumerate(x.T):
            for i,f in enumerate(self.frequencies):
                r = np.dot(H(self.weights[:,i]), 
                        self.steering_vector_2D_from_point(f, p, attn=True, ff=False))
                HF[k,i] = r[0]


        plt.subplot(2,1,1)
        plt.title('Beamformer response')
        for hf in HF:
            plt.plot(self.frequencies, np.abs(hf))
        plt.ylabel('Modulus')
        plt.axis('tight')
        plt.legend(legend)

        plt.subplot(2,1,2)
        for hf in HF:
            plt.plot(self.frequencies, np.unwrap(np.angle(hf)))
        plt.ylabel('Phase')
        plt.xlabel('Frequency [Hz]')
        plt.axis('tight')
        plt.legend(legend)


    def farFieldWeights(self, phi):
        '''
        This method computes weight for a far field at infinity
        
        phi: direction of beam
        '''

        u = unit_vec2D(phi)
        proj = np.dot(u.T, self.R - self.center)[0]

        # normalize the first arriving signal to ensure a causal filter
        proj -= proj.max()

        self.weights = np.exp(2j * np.pi * 
        self.frequencies[:, np.newaxis] * proj / constants.c).T


    def rakeDelayAndSumWeights(self, source, attn=False, ff=False):

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        K = source.shape[1] - 1

        for i, f in enumerate(self.frequencies):
            W = self.steering_vector_2D_from_point(f, source, attn=False, ff=False)
            self.weights[:,i] = 1.0/self.M/(K+1) * np.sum(W, axis=1)



    def rakeOneForcingWeights(self, source, interferer, R_n=None, ff=False, attn=True):

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer, attn=attn, ff=ff)

            R_nq     = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            A_s      = self.steering_vector_2D_from_point(f, source, attn=attn, ff=ff)
            R_nq_inv = np.linalg.pinv(R_nq)
            D        = np.linalg.pinv(mdot(H(A_s), R_nq_inv, A_s))

            self.weights[:,i] = sumcols( mdot( R_nq_inv, A_s, D ) )[:,0]

    def rakeMaxSINRWeights(self, source, interferer, R_n=None, 
            rcond=0., ff=False, attn=True):
        '''
        This method computes a beamformer focusing on a number of specific sources
        and ignoring a number of interferers.

        INPUTS
          * source     : source locations
          * interferer : interferer locations
        '''

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i,f in enumerate(self.frequencies):

            A_good = self.steering_vector_2D_from_point(f, source, attn=attn, ff=ff)

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer, attn=attn, ff=ff)

            a_good = sumcols(A_good)
            a_bad = sumcols(A_bad)

            # TO DO: Fix this (check for numerical rank, use the low rank approximation)
            K_inv = np.linalg.pinv(a_bad.dot(H(a_bad)) + R_n + rcond * np.eye(A_bad.shape[0]))
            self.weights[:,i] = (K_inv.dot(a_good) / mdot(H(a_good), K_inv, a_good))[:,0]


    def rakeMaxUDRWeights(self, source, interferer, R_n=None, ff=False, attn=True):
        
        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            A_good = self.steering_vector_2D_from_point(f, source, attn=attn, ff=ff)

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer, attn=attn, ff=ff)

            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            C = np.linalg.cholesky(R_nq)
            l, v = np.linalg.eig( mdot( H(np.linalg.inv(C)), A_good, H(A_good), np.linalg.inv(C) ) )

            self.weights[:,i] = v[:,0]


    def SNR(self, source, interferer, f, R_n=None, dB=False):

        i_f = np.argmin(np.abs(self.frequencies - f))

        # This works at a single frequency because otherwise we need to pass
        # many many covariance matrices. Easy to change though (you can also
        # have frequency independent R_n).

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        # To compute the SNR, we /must/ use the real steering vectors, so no
        # far field, and attn=True
        A_good = self.steering_vector_2D_from_point(self.frequencies[i_f], source, attn=True, ff=False)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(self.frequencies[i_f], interferer, attn=True, ff=False)
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:,i_f]
        a_1 = sumcols(A_good)

        SNR = np.real(mdot(H(w), a_1, H(a_1), w) / mdot(H(w), R_nq, w))

        if dB is True:
            SNR = 10 * np.log10(SNR)

        return SNR


    def UDR(self, source, interferer, f, R_n=None):

        i_f = np.argmin(np.abs(self.frequencies - f))

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        A_good = self.steering_vector_2D_from_point(self.frequencies[i_f], source, attn=True, ff=False)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(self.frequencies[i_f], interferer, attn=True, ff=False)
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:,i_f]
        return np.real(mdot(H(w), A_good, H(A_good), w) / mdot(H(w), R_nq, w))


    def process(self):

        if (self.signals is None or len(self.signals) == 0):
            raise NameError('No signal to beamform')

        if self.processing is 'FrequencyDomain':

            # create window function
            win = np.concatenate((np.zeros(self.zpf),
                                  windows.hann(self.L), 
                                  np.zeros(self.zpb)))

            # do real STFT of first signal
            tfd_sig = stft.stft(self.signals[0], 
                                self.L, 
                                self.hop, 
                                zp_back=self.zpb, 
                                zp_front=self.zpf,
                                transform=np.fft.rfft, 
                                win=win) * np.conj(self.weights[0])
            for i in xrange(1, self.M):
                tfd_sig += stft.stft(self.signals[i],
                                     self.L,
                                     self.hop,
                                     zp_back=self.zpb,
                                     zp_front=self.zpf,
                                     transform=np.fft.rfft,
                                     win=win) * np.conj(self.weights[i])

            #  now reconstruct the signal
            output = stft.istft(
                tfd_sig,
                self.L,
                self.hop,
                zp_back=self.zpb,
                zp_front=self.zpf,
                transform=np.fft.irfft)

            # remove the zero padding from output signal
            if self.zpb is 0:
                output = output[self.zpf:]
            else:
                output = output[self.zpf:-self.zpb]

        elif self.processing is 'TimeDomain':

            # go back to time domain and shift DC to center
            tw = np.sqrt(self.weights.shape[1])*np.fft.irfft(np.conj(self.weights), axis=1)
            tw = np.concatenate((tw[:, self.N/2:], tw[:, :self.N/2]), axis=1)

            from scipy.signal import fftconvolve

            # do real STFT of first signal
            output = fftconvolve(tw[0], self.signals[0])
            for i in xrange(1, len(self.signals)):
                output += fftconvolve(tw[i], self.signals[i])

        elif self.processing is 'Total':

            W = np.concatenate((self.weights, np.conj(self.weights[:,-2:0:-1])), axis=1)
            W[:,0] = np.real(W[:,0])
            W[:,self.N/2] = np.real(W[:,self.N/2])

            F_sig = np.zeros(self.signals.shape[1], dtype=complex)
            for i in xrange(self.M):
                F_sig += np.fft.fft(self.signals[i])*np.conj(W[i,:])

            f_sig = np.fft.ifft(F_sig)
            print np.abs(np.imag(f_sig)).mean()
            print np.abs(np.real(f_sig)).mean()

            output = np.real(np.fft.ifft(F_sig))

        return output


    def plot(self):

        import matplotlib.pyplot as plt

        plt.subplot(2, 2, 1)
        plt.plot(self.frequencies, np.abs(self.weights.T))
        plt.title('Beamforming weights [modulus]')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Weight modulus')

        plt.subplot(2, 2, 2)
        plt.plot(self.frequencies, np.unwrap(np.angle(self.weights.T), axis=0))
        plt.title('Beamforming weights [phase]')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Unwrapped phase')

        # go back to time domain and shift DC to center
        tw = np.fft.irfft(np.conj(self.weights), axis=1, n=self.N)
        tw = np.concatenate((tw[:,self.N/2:], tw[:, :self.N/2]), axis=1)

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(self.N)/float(self.Fs), tw.T)
        plt.title('Beamforming filters')
        plt.xlabel('Time [s]')
        plt.ylabel('Filter amplitude')
        plt.axis('tight')


    @classmethod
    def linear2D(cls, Fs, center, M, phi, d):
        ''' Create linear beamformer '''
        return Beamformer(linear2DArray(center, M, phi, d), Fs)

    @classmethod
    def circular2D(cls, Fs, center, M, phi, radius):
        ''' Create circular beamformer'''
        return Beamformer(circular2DArray(center, M, phi, radius), Fs)

    @classmethod
    def poisson(cls, Fs, center, M, d):
        ''' Create beamformer with microphone positions drawn from Poisson process '''

        from numpy.random import standard_exponential, randint

        R = d*standard_exponential((2, M))*(2*randint(0,2, (2,M)) - 1)
        R = R.cumsum(axis=1)
        R -= R.mean(axis=1)[:,np.newaxis]
        R += np.array([center]).T

        return Beamformer(R, Fs)

