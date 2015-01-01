
import numpy as np
import os
from stft import stft

def median(x):
    '''
    m, ci = median(x)
    computes median and 0.95% confidence interval.
    x: 1D ndarray
    m: median
    ci: [le, ue]
    The confidence interval is [m-le, m+ue]
    '''
    x = np.sort(x);
    n = x.shape[0]

    if n % 2 == 1:
        # if n is odd, take central element
        m = x[(n+1)/2];
    else:
        # if n is even, average the two central elements
        m = 0.5*(x[n/2] + x[n/2+1]);

    # This table is taken from the Performance Evaluation lecture notes by J-Y Le Boudec
    # available at: http://perfeval.epfl.ch/lectureNotes.htm
    CI = [[1,6],  [1,7],  [1,7],  [2,8],  [2,9],  [2,10], [3,10], [3,11], [3,11],[4,12], \
          [4,12], [5,13], [5,14], [5,15], [6,15], [6,16], [6,16], [7,17], [7,17],[8,18], \
          [8,19], [8,20], [9,20], [9,21], [10,21],[10,22],[10,22],[11,23],[11,23], \
          [12,24],[12,24],[13,25],[13,26],[13,27],[14,27],[14,28],[15,28],[15,29], \
          [16,29],[16,30],[16,30],[17,31],[17,31],[18,32],[18,32],[19,33],[19,34], \
          [19,35],[20,35],[20,36],[21,36],[21,37],[22,37],[22,38],[23,39],[23,39], \
          [24,40],[24,40],[24,40],[25,41],[25,41],[26,42],[26,43],[26,44],[27,44]];
    CI = np.array(CI)

    # adjust to indexing from 0
    CI -= 1

    if n < 6:
        # If we have less than 6 samples, we cannot have a confidence interval
        ci = np.array([0,0])
    elif n <= 70:
        # For 6 <= n <= 70, we use exact values from the table
        j = CI[n-6,0]
        k = CI[n-6,1]
        ci = np.array([x[j]-m,x[k]-m])
    else:
        # For 70 < n, we use the approximation for large sets
        j = np.floor(0.5*n - 0.98*np.sqrt(n))
        k = np.ceil(0.5*n + 1 + 0.98*np.sqrt(n))
        ci = np.array([x[j]-m,x[k]-m])

    return m, ci

# Simple mean squared error function
def mse(x1, x2):
  return (np.abs(x1-x2)**2).sum()/len(x1)


# Itakura-Saito distance function
def itakura_saito(x1, x2, sigma2_n, stft_L=128, stft_hop=128):

  P1 = np.abs(stft(x1, stft_L, stft_hop))**2
  P2 = np.abs(stft(x2, stft_L, stft_hop))**2

  VAD1 = P1.mean(axis=1) > 2*stft_L**2*sigma2_n
  VAD2 = P2.mean(axis=1) > 2*stft_L**2*sigma2_n
  VAD = np.logical_or(VAD1, VAD2)

  if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
    raise ValueError("Error: Itakura-Saito requires both array to have same length")

  R = P1[VAD,:]/P2[VAD,:]

  IS = (R - np.log(R) - 1.).mean(axis=1)

  '''
  import matplotlib.pyplot as plt
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(x1)
  plt.plot(x2)
  plt.legend(('x1','x2'))
  plt.subplot(2,1,2)
  plt.plot(np.arange(P1.shape[0])[VAD],IS)
  plt.plot(VAD)
  plt.show()
  '''

  return np.median(IS)

def snr(ref, deg):

    return np.sum(ref**2)/np.sum((ref-deg)**2)

# Perceptual Evaluation of Speech Quality
def pesq(ref, deg, Fs=8000, swap=False, wb=False, bin='./bin/pesq'):
    '''
    pesq_mos, mos_lqo = pesq(ref, deg, sample_rate=None, bin='./pesq'):
    Uses the utility obtained from ITU P.862
    http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en
    '''

    if not os.path.isfile(ref) or not os.path.isfile(deg):
        raise ValueError('Some file did not exist')

    if Fs not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')

    args = [ bin, '+%d' % int(Fs) ]

    if swap is True:
        args.append('+swap')

    if wb is True:
        args.append('+wb')

    args.append(ref)
    args.append(deg)

    import subprocess
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    last_line = out.split('\n')[-2]


    if wb is True:
        if not last_line.startswith('P.862.2 Prediction'):
            raise ValueError(last_line)
        return 0, float(last_line.split()[-1])
    else:
        if not last_line.startswith('P.862 Prediction'):
            raise ValueError(last_line)
        return tuple(map(float, last_line.split()[-2:]))

