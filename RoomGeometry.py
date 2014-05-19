import numpy as np

'''
RoomGeometry
A room geometry is defined by all the source and all its images
'''

class RoomGeometry:

  def __init__(self, images):
    self.images = images
    self.max_order = len(images)-1
    self.dim = len(images[0][0])
    self.room = None

  def plot(self, ord=None):

    import matplotlib
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # draw room
    polygon = Polygon(self.room, True)
    p = PatchCollection([polygon], cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)

    # draw the scatter of images
    colors = ['b','g','r','c','m','y','k','w']
    if (ord == None):
      ord = self.max_order+1
    for o in xrange(ord):
      ax.scatter(self.images[o][:,0], self.images[o][:,1], c=colors[o%len(colors)], s=20, marker='o', edgecolor='none')


  @classmethod
  def shoeBox2D(cls, source, max_order, p1, p2):
    '''
    Create a new Shoe Box room geometry.
    Arguments:
    source: the sound source position.
    max_order: the maximum order of image sources desired.
    p1: the lower left corner of the room
    p2: the upper right corner of the room
    '''

    # compute room characteristics
    corners = np.array([[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]])
    walls = cls.walls(corners)
    normals = cls.normals(walls)

    # order zero is the source itself
    all_images = [np.array([source])]

    # generate all images up to max order
    o = 1
    while o <= max_order:
      images = np.zeros((0,2))
      for s in all_images[o-1]:
        images = np.concatenate((images, cls.firstOrderImages(s, corners, normals)), axis=0)

      # remove duplicates
      images = images[np.lexsort(images.T)]
      diff = np.diff(images, axis=0)
      ui = np.ones(len(images), 'bool')
      ui[1:] = (diff != 0).any(axis=1)

      # add to array of images
      all_images.append(images[ui])

      # next order
      o += 1

    # create new object
    new_rg = RoomGeometry(all_images)
    new_rg.room = corners

    return new_rg

  @classmethod
  def firstOrderImages(cls, source, corners, normals):

    # projected length onto normal
    ip = np.sum(normals*(corners - source), axis=1)

    # projected vector from source to wall
    d = ip[:,np.newaxis]*normals

    # compute images points, positivity is to get only the reflections outside the room
    images = source + 2*d[ip > 0,:]

    return images

  @classmethod
  def walls(cls, corners):
    # circular wall vectors (counter clockwise)
    return corners - corners[xrange(-1,corners.shape[0]-1), :]

  @classmethod
  def normals(cls, walls):
    # compute normals (outward pointing)
    nrmls = walls[:,[1,0]]/np.linalg.norm(walls, axis=1)[:,np.newaxis]
    nrmls[:,1] *= -1;
    return nrmls


