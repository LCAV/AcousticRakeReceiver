'''A collection of windowing functions.'''

import numpy as np

# cosine window function
def cosine(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = np.cos(np.pi*(t - 0.5))**2

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    w **= 2
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d

  # compute window
  return w


# root triangular window function
def triang(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = 1. - np.abs(2.*t - 1.)

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d

  # compute window
  return w


# root hann window function
def hann(N, flag='asymmetric', length='full'):

  # first choose the indexes of points to compute
  if (length == 'left'):     # left side of window
    t = np.arange(0,N/2)
  elif(length == 'right'):   # right side of window
    t = np.arange(N/2,N)
  else:                   # full window by default
    t = np.arange(0,N)

  # if asymmetric window, denominator is N, if symmetric it is N-1
  if (flag == 'symmetric' or flag == 'mdct'):
    t = t/float(N-1)
  else:
    t = t/float(N)

  w = 0.5*(1-np.cos(2*np.pi*t))

  # make the window respect MDCT condition
  if (flag == 'mdct'):
    d = w[:N/2]+w[N/2:]
    w[:N/2] *= 1./d
    w[N/2:] *= 1./d

  # compute window
  return w


# Rectangular window function
def rect(N):
  return np.ones(N)


