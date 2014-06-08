import numpy as np
import cvxpy as cp
from time import sleep

import constants


#===============================================================================
# Free (non-class-member) functions related to beamformer design
#===============================================================================


def H(A, **kwargs):
    """Returns the conjugate (Hermitian) transpose of a matrix."""
    return np.transpose(A, **kwargs).conj()


def complex_to_real_matrix(A):

    A_real = np.real(A)
    A_imag = np.imag(A)

    A_ctr = np.vstack([np.hstack([A_real, -A_imag]),
                       np.hstack([A_imag,  A_real])])

    return A_ctr


def real_to_complex_vector(b):

    n = b.shape[0]/2
    return b[0:n] + 1j*b[n:]


def echo_beamformer_cvx(A_good, A_bad):

    # Expand complex matrices and vectors to real matrices and vectors
    A_good_ctr_H = complex_to_real_matrix(H(A_good))
    A_bad_ctr_H = complex_to_real_matrix(H(A_bad))

    M = A_good.shape[0]
    K = A_good.shape[1]

    h = cp.Variable(2*M)

    # Objective: minimize(norm(h.H * A_good)^2)
 
    objective = cp.Minimize(cp.sum_entries(cp.square(A_bad_ctr_H * h)))

    # Constraint: sum(h.H * A_good) = 1 + 0*1j
    constraints = [cp.sum_entries((A_good_ctr_H*h)[0:K]) == 1, 
                   cp.sum_entries((A_good_ctr_H*h)[K:]) == 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return np.array(real_to_complex_vector(h.value))

def echo_beamformer(A_good, A_bad):

    # Use the fact that this is just MaxSINR with a particular covariance
    # matrix, and a particular steering vector

    a = np.sum(A_good, axis=1, keepdims=1)
    K_inv = np.linalg.inv(A_bad.dot(H(A_bad)))
    return K_inv.dot(a) / ( H(a).dot(K_inv.dot(a)) )

def distance(X, Y):
    # Assume X, Y are arrays, *not* matrices
    X = np.array(X)
    Y = np.array(Y)

    XX, YY = [np.sum(A**2, axis=0, keepdims=True) for A in X, Y]

    return np.sqrt(np.abs((XX.T + YY) - 2*np.dot(X.T, Y)))

def unit_vec2D(phi):
    return np.array([[np.cos(phi), np.sin(phi)]]).T


def linear2DArray(center, M, phi, d):
    u = unit_vec2D(phi)
    return np.array(center)[:,np.newaxis] + d*(np.arange(M)[np.newaxis,:] - (M-1.)/2.)*u


def circular2DArray(center, M, radius, phi0):
    phi = np.arange(M)*2.*np.pi/M
    return np.array(center)[:,np.newaxis] + radius*np.vstack((np.cos(phi+phi0), np.sin(phi+phi0)))


def fir_approximation_ls(weights, T, n1, n2):

    freqs_plus = np.array(weights.keys())[:,np.newaxis]
    freqs = np.vstack([freqs_plus, 
                      -freqs_plus])
    omega = 2*np.pi*freqs
    omega_discrete = omega * T

    n = np.arange(n1, n2)

    # Create the DTFT transform matrix corresponding to a discrete set of
    # frequencies and the FIR filter indices
    F = np.exp(-1j*omega_discrete*n)
    print np.linalg.pinv(F)

    w_plus = np.array(weights.values())[:,:,0]
    w = np.vstack([w_plus, 
                   w_plus.conj()])

    return np.linalg.pinv(F).dot(w)



#===============================================================================
# Classes (microphone array and beamformer related)
#===============================================================================


class MicrophoneArray(object):
    """Microphone array class."""
    def __init__(self, R):
        self.dim = R.shape[0] # are we in 2D or in 3D
        self.M = R.shape[1]   # number of microphones
        self.R = R            # array geometry

        self.center = np.mean(R, axis=1, keepdims=True)


    @classmethod
    def linear2D(cls, center, M, phi=0.0, d=1.):
      return MicrophoneArray(linear2Darray(center, M, phi, d))


    @classmethod
    def circular2D(cls, center, M, radius=1., phi0=0.):
      return MicrophoneArray(circular2DArray(center, M, radius, phi0))


class Beamformer(MicrophoneArray):
    """Beamformer class. At some point, in some nice way, the design methods 
    should also go here. Probably with generic arguments."""


    def __init__(self, R):
        MicrophoneArray.__init__(self, R)

        # 
        self.weights = {} # weigths at different frequencies


    def __add__(self, y):

       return Beamformer(np.concatenate((self.R, y.R), axis=1)) 


    def steering_vector_2D(self, frequency, phi, dist):

        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + self.center

        D = distance(self.R, X)
        omega = 2*np.pi*frequency

        return np.exp(-1j*omega*D/constants.c)


    def steering_vector_2D_from_point(self, frequency, source):

        phi = np.angle((source[0]-self.center[0,0]) + 1j*(source[1]-self.center[1,0]))

        return self.steering_vector_2D(frequency, phi, constants.ffdist)


    def response(self, phi_list, frequency):
        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[frequency]), self.steering_vector_2D(frequency, phi_list, constants.ffdist))
        return bfresp


    def farFieldWeights(self, phi, f):
      if (np.rank(f) == 0):
        f = np.array([f])
      else:
        f = np.array(f)
      u = unit_vec2D(phi)
      proj = np.dot(u.T, self.R - self.center)[0]
      proj -= proj.max()
      w = np.exp(2j*np.pi*f[:,np.newaxis]*proj/constants.c)
      self.weights.update(zip(f, w[:,:,np.newaxis]))


    def echoBeamformerWeights(self, source, interferer, frequencies):
      if (np.rank(frequencies) == 0):
        frequencies = np.array([frequencies])

      for f in frequencies:

        A_good = self.steering_vector_2D_from_point(f, source)
        A_bad = self.steering_vector_2D_from_point(f, interferer)
        w = echo_beamformer(A_good, A_bad)

        # print np.linalg.norm(A_good[:,0]), np.linalg.norm(np.sum(A_good, axis=1))

        self.weights.update({f: w})

    def add_weights(self, new_frequency_list, new_weights):
        self.weights.update(zip(new_frequency_list, new_weights))


    @classmethod
    def linear2D(cls, center, M, phi=0.0, d=1.):
      return Beamformer(linear2DArray(center, M, phi, d))


    @classmethod
    def circular2D(cls, center, M, radius=1., phi0=0.):
      return Beamformer(circular2DArray(center, M, radius, phi0))
