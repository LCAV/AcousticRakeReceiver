
import numpy as np

from SoundSource import SoundSource

import constants

'''
Room
A room geometry is defined by all the source and all its images
'''

class Room(object):

  def __init__(self, corners, absorption=1., max_order=1, sources=None):

    # make sure we have an ndarray of the right size
    corners = np.array(corners)
    if (np.rank(corners) != 2):
      raise NameError('Room corners is a 2D array.')

    # make sure the corners are anti-clockwise
    if (self.area(corners) <= 0):
      raise NameError('Room corners must be anti-clockwise')

    self.corners = corners
    self.dim = len(corners[0])

    # circular wall vectors (counter clockwise)
    self.walls = self.corners - self.corners[xrange(-1,corners.shape[0]-1), :]

    # compute normals (outward pointing)
    self.normals = self.walls[:,[1,0]]/np.linalg.norm(self.walls, axis=1)[:,np.newaxis]
    self.normals[:,1] *= -1;

    # list of attenuation factors for the wall reflections
    absorption = np.array(absorption, dtype='float64')
    if (np.rank(absorption) == 0):
      self.absorption = absorption*np.ones(self.corners.shape[0])
    elif (np.rank(absorption) > 1 or self.corners.shape[0] != len(absorption)):
      raise NameError('Absorption and corner must be the same size')
    else:
      self.absorption = absorption
  
    # a list of sources
    if (sources is None):
      self.sources = []
    elif (sources is list):
      self.sources = sources
    else:
      raise NameError('Room needs a source or list of sources.')

    # a maximum orders for image source computation
    self.max_order = max_order


  def addSource(self, position, signal=None):

    # generate first order images
    i,d = self.firstOrderImages(np.array(position))
    images = [i]
    damping = [d]

    # generate all higher order images up to max_order
    o = 1
    while o < self.max_order:
      # generate all images of images of previous order
      img = np.zeros((0,self.dim))
      dmp = np.array([])
      for si, sd in zip(images[o-1], damping[o-1]):
        i,d = self.firstOrderImages(si)
        img = np.concatenate((img, i), axis=0)
        dmp = np.concatenate((dmp, d*sd), axis=1)

      # remove duplicates
      ordering = np.lexsort(img.T)
      img = img[ordering]
      dmp = dmp[ordering]
      diff = np.diff(img, axis=0)
      ui = np.ones(len(img), 'bool')
      ui[1:] = (diff != 0).any(axis=1)

      # add to array of images
      images.append(img[ui])
      damping.append(dmp[ui])

      # next order
      o += 1

    # add a new source to the source list
    self.sources.append(SoundSource(position, images=images, \
        damping=damping, signal=signal))



  def plot(self, ord=None):

    import matplotlib
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # draw room
    polygon = Polygon(self.corners, True)
    p = PatchCollection([polygon], cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)

    # define some markers for different sources and colormap for damping
    markers = ['o','s', 'v','x','.']
    cmap = plt.get_cmap('YlGnBu')
    # draw the scatter of images
    for i, source in enumerate(self.sources):
      # draw source
      ax.scatter(source.position[0], source.position[1], c=cmap(1.), s=20, \
          marker=markers[i%len(markers)], edgecolor=cmap(1.))

      # draw images
      if (ord == None):
        ord = self.max_order
      for o in xrange(ord):
        # map the damping to a log scale (mapping 1 to 1)
        val = (np.log2(source.damping[o])+10.)/10.
        # plot the images
        ax.scatter(source.images[o][:,0], source.images[o][:,1], \
            c=cmap(val), s=20, \
            marker=markers[i%len(markers)], edgecolor=cmap(val))

    # keep axis equal, or the symmetry is lost
    ax.axis('equal')

    return fig, ax


  def firstOrderImages(self, source_position):

    # projected length onto normal
    ip = np.sum(self.normals*(self.corners - source_position), axis=1)

    # projected vector from source to wall
    d = ip[:,np.newaxis]*self.normals

    # compute images points, positivity is to get only the reflections outside the room
    images = source_position + 2*d[ip > 0,:]

    # collect absorption factors of reflecting walls
    damping = self.absorption[ip > 0]

    return images, damping


  def impulseResponse(self, mic_pos, Fs, t0=0., max_order=None, c=constants.c, window=False):
    '''
    Return sampled room impulse response based on images list
    '''

    mic = np.array(mic_pos)
    h = []

    for source in self.sources:

      # use all images available by default
      if (max_order == None):
        max_order = len(source.images)

      # stack source and all images
      img = np.array([source.position])
      dmp = np.array([1.])
      for o in xrange(max_order):
        img = np.concatenate((img, source.images[o]), axis=0)
        dmp = np.concatenate((dmp, source.damping[o]), axis=0)

      # compute the distance
      dist = np.sqrt(np.sum((img - mic[np.newaxis,:])**2, axis=1))
      time = dist/c + t0

      # the minimum length needed is the maximum time of flight multiplied by Fs
      # make it twice that amount to minimize aliasing
      N = 2*np.ceil(time.max()*Fs)

      # compute the discrete band-limited spectrum
      index = np.arange(0, N/2+1)
      F = np.exp(-2j*np.pi*index[:,np.newaxis]*time[np.newaxis,:]*float(Fs)/N)
      H = np.dot(F, dmp/(4*np.pi*dist))

      # window if required
      if (window == True):
        H *= np.hanning(N+1)[N/2:]

      # inverse the spectrum to get the band-limited impulse response
      h.append(np.fft.irfft(H))

    return h


  @classmethod
  def shoeBox2D(cls, p1, p2, max_order=1, absorption=1.):
    '''
    Create a new Shoe Box room geometry.
    Arguments:
    p1: the lower left corner of the room
    p2: the upper right corner of the room
    max_order: the maximum order of image sources desired.
    '''

    # compute room characteristics
    corners = np.array([[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]])

    return Room(corners, absorption=absorption, max_order=max_order)


  @classmethod
  def area(cls, corners):
    '''
    Compute the area of a room represented by its corners
    '''
    x = corners[:,0] - corners[xrange(-1,corners.shape[0]-1),0]
    y = corners[:,1] + corners[xrange(-1,corners.shape[0]-1),1]
    return -0.5*(x*y).sum()


  @classmethod
  def isAntiClockwise(cls, corners):
    '''
    Return true if the corners of the room are arranged anti-clockwise
    '''
    return (cls.area(corners) > 0)


  @classmethod
  def ccw3p(cls, p):
    '''
    Argument: p, a (3,2)-ndarray whose rows are the vertices of a 2D triangle
    Returns
    1: if triangle vertices are counter-clockwise
    -1: if triangle vertices are clock-wise
    0: if vertices are colinear

    Ref: https://en.wikipedia.org/wiki/Curve_orientation
    '''
    if (p.shape != (3,2)):
      raise NameError('Room.ccw3p is for three 2D points, input is 3x2 ndarray')
    D = (p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1])
    
    if (np.abs(D) < constants.eps):
      return 0
    elif (D > 0):
      return 1
    else:
      return -1


