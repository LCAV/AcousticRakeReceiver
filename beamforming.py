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
                       np.hstack([A_imag, A_real])])

    return A_ctr


def real_to_complex_vector(b):

    n = b.shape[0]/2
    return b[0:n] + 1j*b[n:]


def echo_beamformer(A_good, A_bad):

    # Expand complex matrices and vectors to real matrices and vectors
    A_good_ctr_H = complex_to_real_matrix(H(A_good))
    A_bad_ctr_H = complex_to_real_matrix(H(A_bad))

    M = A_good.shape[0]
    K = A_good.shape[1]

    h = cp.Variable(2*M)

    # Objective: minimize(norm(h.H * A_good)^2)
 

    objective = cp.Minimize(sum(cp.square(A_bad_ctr_H * h)))

    # Constraint: sum(h.H * A_good) = 1 + 0*1j
    constraints = [sum((A_good_ctr_H*h)[0:K]) == 1, 
                   sum((A_good_ctr_H*h)[K:]) == 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return np.array(real_to_complex_vector(h.value))


def linear2DArray(center, M, phi, d):
  u = np.array([[np.cos(phi), np.sin(phi)]]).T
  return np.array(center)[:,np.newaxis] + d*(np.arange(M)[np.newaxis,:] - (M-1.)/2.)*u


def circular2DArray(center, M, radius, phi0):
  phi = np.arange(M)*2.*np.pi/M
  return np.array(center)[:,np.newaxis] + radius*np.vstack((np.cos(phi+phi0), np.sin(phi+phi0)))


def distance(X, Y):
    # Assume X, Y are arrays, *not* matrices
    X = np.array(X)
    Y = np.array(Y)

    XX, YY = [np.sum(A**2, axis=0, keepdims=True) for A in X, Y]

    return np.sqrt(np.abs((XX.T + YY) - 2*np.dot(X.T, Y)))


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


    def steering_vector_2D(self, frequency, phi, dist):

        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + self.center

        # print np.array([np.cos(phi), np.sin(phi)])
        # sleep(1)

        D = distance(self.R, X)
        omega = 2*np.pi*frequency
        return np.exp(-1j*omega*D/constants.c)


    def response(self, phi_list, frequency):
        # For the moment assume that we are in 2D

        bfresp = np.dot(H(self.weights[frequency]), self.steering_vector_2D(frequency, phi_list, constants.ffdist))
        return bfresp


    def add_weights(self, new_frequency_list, new_weights):
        self.weights.update(zip(new_frequency_list, new_weights))


    @classmethod
    def linear2D(cls, center, M, phi=0.0, d=1.):
      return Beamformer(linear2DArray(center, M, phi, d))


    @classmethod
    def circular2D(cls, center, M, radius=1., phi0=0.):
      return Beamformer(circular2DArray(center, M, radius, phi0))


