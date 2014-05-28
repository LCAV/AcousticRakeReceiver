
import numpy as np

'''
A class to represent sound sources
'''

class SoundSource(object):

  def __init__(self, position, images=None, damping=None, signal=None):

    self.position = np.array(position)

    if (images == None):
      # set to empty list if nothing provided
      self.images = []
      self.damping = []

    else:
      # save list if provided
      self.images = images

      # we need to have damping factors for every image
      if (damping == None):
        # set to one if not set
        self.damping = []
        for o in images:
          self.damping.append(np.ones(o.shape))
      else:
        # check damping is the same size as images
        if (len(damping) != len(images)):
            raise NameError('Images and damping must have same shape')
        for i in range(len(damping)):
          if (damping[i].shape[0] != images[i].shape[1]):
            raise NameError('Images and damping must have same shape')

        # copy over if correct
        self.damping = damping

    # The sound signal of the source
    self.signal = signal


  def addSignal(signal):

    self.signal = signal


  def getImages(self, max_order=None):

    if (max_order is None):
      max_order = len(images)
      
    # stack source and all images
    img = np.array([self.position]).T
    for o in xrange(max_order):
      img = np.concatenate((img, self.images[o]), axis=1)

    return img

  def getDamping(self, max_order=None):
    if (max_order is None):
      max_order = len(images)
      
    # stack source and all images
    dmp = np.array([1.])
    for o in xrange(max_order):
      dmp = np.concatenate((dmp, self.damping[o]))

    return dmp



