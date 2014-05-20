import numpy as np

class SoundSource(object):

  def __init__(self, position, images=None):
    self.position = np.array(position)
    if (images == None):
      self.images = []
    else:
      self.images = images


'''
Room
A room geometry is defined by all the source and all its images
'''

class Room(object):

  def __init__(self, corners, max_order=1, sources=None):

    # make sure we have an ndarray of the right size
    corners = np.array(corners)
    if (np.rank(corners) != 2):
      raise NameError('Room corners is a 2D array.')

    # make sure the corners are anti-clockwise
    if (self.area(corners) <= 0):
      corners = corners[::-1]

    self.corners = corners
    self.dim = len(corners[0])

    # circular wall vectors (counter clockwise)
    self.walls = self.corners - self.corners[xrange(-1,corners.shape[0]-1), :]

    # compute normals (outward pointing)
    self.normals = self.walls[:,[1,0]]/np.linalg.norm(self.walls, axis=1)[:,np.newaxis]
    self.normals[:,1] *= -1;
  
    # a list of sources
    if (sources is None):
      self.sources = []
    elif (sources is list):
      self.sources = sources
    else:
      raise NameError('Room needs a list or sources.')

    # a maximum orders for image source computation
    self.max_order = max_order


  def addSource(self, position):

    # add a new source in the room
    self.sources.append(SoundSource(position))
    new_source = self.sources[-1]

    # generate first order images
    new_source.images = [self.firstOrderImages(position)]

    # generate all higher order images up to max_order
    o = 1
    while o < self.max_order:
      # generate all images of images of previous order
      images = np.zeros((0,self.dim))
      for s in new_source.images[o-1]:
        images = np.concatenate((images, self.firstOrderImages(s)), axis=0)

      # remove duplicates
      images = images[np.lexsort(images.T)]
      diff = np.diff(images, axis=0)
      ui = np.ones(len(images), 'bool')
      ui[1:] = (diff != 0).any(axis=1)

      # add to array of images
      new_source.images.append(images[ui])

      # next order
      o += 1


  def sampleImpulseResponse(self, Fs, mic_pos):
    '''
    Return sampled room impulse response for every source in the room
    '''


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

    # draw the scatter of images
    colors = ['b','g','r','c','m','y','k']
    markers = ['o','s', 'v','.']
    for i, source in enumerate(self.sources):
      # draw source
      ax.scatter(source.position[0], source.position[1], c=colors[-1], s=20, \
          marker=markers[i%len(markers)], edgecolor='none')

      # draw images
      if (ord == None):
        ord = self.max_order
      for o in xrange(ord):
        ax.scatter(source.images[o][:,0], source.images[o][:,1], \
            c=colors[o%len(colors)], s=20, \
            marker=markers[i%len(markers)], edgecolor='none')

    ax.axis('equal')


  def firstOrderImages(self, source_position):

    # projected length onto normal
    ip = np.sum(self.normals*(self.corners - source_position), axis=1)

    # projected vector from source to wall
    d = ip[:,np.newaxis]*self.normals

    # compute images points, positivity is to get only the reflections outside the room
    images = source_position + 2*d[ip > 0,:]

    return images


  @classmethod
  def shoeBox2D(cls, p1, p2, max_order=1):
    '''
    Create a new Shoe Box room geometry.
    Arguments:
    p1: the lower left corner of the room
    p2: the upper right corner of the room
    max_order: the maximum order of image sources desired.
    '''

    # compute room characteristics
    corners = np.array([[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]])

    return Room(corners, max_order)


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


