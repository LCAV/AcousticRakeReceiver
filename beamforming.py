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


def echo_beamformer(A_good, A_bad, R_n=None, rcond=1e-15):

    # Use the fact that this is just MaxSINR with a particular covariance
    # matrix, and a particular steering vector. TODO: For more microphones
    # than steering vectors, K is rank-deficient. Is the solution still fine?
    # The answer seems to be yes.

    a_1 = sumcols(A_good)
    a_bad = sumcols(A_bad)

    if R_n is None:
        R_n = np.zeros(A_good.shape[0])

    K_inv = np.linalg.pinv(a_bad.dot(H(a_bad)) + R_n + rcond * np.eye(A_bad.shape[0]))
    return K_inv.dot(a_1) / mdot(H(a_1), K_inv, a_1)


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


def circular2DArray(center, M, radius, phi0):
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

    def __init__(self, R):
        self.dim = R.shape[0]  # are we in 2D or in 3D
        self.M = R.shape[1]   # number of microphones
        self.R = R            # array geometry

        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)

    def to_wav(self, filename, Fs):
        '''
        Save all the signals to wav files
        '''
        from scipy.io import wavfile
        #scaled = np.array(self.signals.T, dtype=np.int16)
        scaled = np.array(self.signals.T/self.signals.max(), dtype=float)
        wavfile.write(filename, Fs, scaled)

    @classmethod
    def linear2D(cls, center, M, phi=0.0, d=1.):
        return MicrophoneArray(linear2Darray(center, M, phi, d))

    @classmethod
    def circular2D(cls, center, M, radius=1., phi=0.):
        return MicrophoneArray(circular2DArray(center, M, radius, phi))


class Beamformer(MicrophoneArray):

    """Beamformer class. At some point, in some nice way, the design methods
    should also go here. Probably with generic arguments."""

    def __init__(self, R, Fs, processing, *args):
        MicrophoneArray.__init__(self, R)

        self.Fs = Fs                    # sampling frequency
        self.processing = processing    # Time or frequency domain

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
        else:
            raise NameError(processing + ': No such type of processing')

        # for now only support equally spaced frequencies
        self.frequencies = np.arange(0, self.N/2+1)/float(self.N)*float(Fs)
        
        # weights will be computed later, the array is of shape (M, N/2+1)
        self.weights = None


    def __add__(self, y):

        if self.processing is 'FrequencyDomain':
            return Beamformer(np.concatenate((self.R, y.R), axis=1), self.Fs, self.processing, self.L, self.hop, self.zpf, self.zpb)
        elif self.processing is 'TimeDomain':
            return Beamformer(np.concatenate((self.R, y.R), axis=1), self.Fs, self.processing, self.N)
        else:
            raise NameError('Unknown processing type.')

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
            return 1 / (4 * np.pi) / D * np.exp(-1j * omega * D / constants.c)
        else:
            return np.exp(-1j * omega * D / constants.c)


    def steering_vector_2D_from_point_ff(self, frequency, source, attn=False):
        phi = np.angle(         (source[0] - self.center[0, 0]) 
                         + 1j * (source[1] - self.center[1, 0]))
        return self.steering_vector_2D(
            frequency,
            phi,
            constants.ffdist,
            attn=attn)


    def steering_vector_2D_from_point(self, frequency, source, attn=False):
        phi = np.angle(         (source[0] - self.center[0, 0]) 
                         + 1j * (source[1] - self.center[1, 0]))
        dist = np.sqrt(np.sum((source - self.center) ** 2, axis=0))
        return self.steering_vector_2D(frequency, phi, dist, attn=attn)


    def response(self, phi_list, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[:,i_freq]), self.steering_vector_2D(
            self.frequencies[i_freq], phi_list, constants.ffdist))

        return self.frequencies[i_freq], bfresp


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


    def echoBeamformerWeights(self, source, interferer, R_n=None, rcond=1e-15):
        '''
        This method computes a beamformer focusing on a number of specific sources
        and ignoring a number of interferers
        source: source locations
        interferer: interferers locations
        '''

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i,f in enumerate(self.frequencies):

            A_good = self.steering_vector_2D_from_point(f, source, attn=True)
            A_bad = self.steering_vector_2D_from_point(f, interferer, attn=True)
            self.weights[:,i] = echo_beamformer(A_good, A_bad, R_n, rcond=rcond)[:,0]


    def rakeMaxSINRWeightsFF(self, source, interferer, R_n):
        pass


    def rakeDelayAndSumWeights(self, source):

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        K = source.shape[1] - 1

        for i, f in enumerate(self.frequencies):
            W = self.steering_vector_2D_from_point(f, source, attn=False)
            self.weights[:,i] = 1.0/self.M/(K+1) * np.sum(W, axis=1)


    def rakeOneForcingWeights(self, source, interferer, R_n):

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            A_bad    = self.steering_vector_2D_from_point(f, interferer, attn=True)
            R_nq     = R_n + H(sumcols(A_bad)).dot(sumcols(A_bad))

            A_s      = self.steering_vector_2D_from_point(f, source, attn=True)
            R_nq_inv = np.linalg.pinv(R_nq)
            D        = np.linalg.pinv(mdot(H(A_s), R_nq_inv, A_s))

            self.weights[:, i] = sumcols( mdot( R_nq_inv, A_s, D ) )
 

    def rakeMaxUDRWeights(self, source, interferer, R_n):
        
        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            A_good = self.steering_vector_2D_from_point(f, source, attn=True)
            A_bad = self.steering_vector_2D_from_point(f, interferer, attn=True)

            R_nq = R_n + H(sumcols(A_bad)).dot(sumcols(A_bad))
            C    = np.linalg.cholesky(R_nq)
            l, v = np.linalg.eig( mdot( H(np.linalg.inv(C)), A_good, H(A_good), np.linalg.inv(C) ) )

            self.weights[:, i] = v[:, 0]



    def SNR(self, source, interferer, R_n, i):

        # This works at a single frequency because otherwise we need to pass
        # many many covariance matrices. Easy to change though (you can also
        # have frequency independent R_n).


        # To compute the SNR, we /must/ use the real steering vectors, so no
        # far field, and attn=True

        f = self.frequencies[i]       
        A_good = self.steering_vector_2D_from_point(f, source, attn=True)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(f, interferer, attn=True)
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:, i]

        a_1 = sumcols(A_good)
        return np.real(mdot(H(w), a_1, H(a_1), w) / mdot(H(w), np.linalg.pinv(R_nq), w))


    def UDR(self, source, interferer, R_n, i):

        f = self.frequencies[i]
        A_good = self.steering_vector_2D_from_point(f, source, attn=True)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(f, interferer, attn=True)
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:, i]
        return np.real(mdot(H(w), A_good, H(A_good), w) / mdot(H(w), np.linalg.pinv(R_nq), w))


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

        elif self.processing is 'TimeDomain':

            # go back to time domain and shift DC to center
            tw = np.fft.irfft(np.conj(self.weights), axis=1)
            tw = np.concatenate((tw[:, self.N/2:], tw[:, :self.N/2]), axis=1)

            from scipy.signal import fftconvolve

            # do real STFT of first signal
            output = fftconvolve(tw[0], self.signals[0])
            for i in xrange(1, len(self.signals)):
                output += fftconvolve(tw[i], self.signals[i])

        return output


    def plot(self):

        import matplotlib.pyplot as plt

        plt.subplot(2, 1, 1)
        plt.plot(self.frequencies, np.abs(self.weights.T))
        plt.title('Beamforming weights')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Weight modulus')

        # go back to time domain and shift DC to center
        tw = np.fft.irfft(np.conj(self.weights), axis=1, n=self.N)
        tw = np.concatenate((tw[:,self.N/2:], tw[:, :self.N/2]), axis=1)

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(self.N)/float(self.Fs), tw.T)
        plt.title('Beamforming filters')
        plt.xlabel('Time [s]')
        plt.ylabel('Filter amplitude')


    @classmethod
    def linear2D(cls, center, M, phi, d, Fs, proc, *args):
        return Beamformer(linear2DArray(center, M, phi, d), Fs, proc, *args)

    @classmethod
    def circular2D(cls, center, M, radius, phi, Fs, proc, *args):
        return Beamformer(circular2DArray(center, M, radius, phi), Fs, proc, *args)

