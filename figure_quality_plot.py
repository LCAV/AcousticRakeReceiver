import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import metrics as metrics

import os
import fnmatch

max_sources = 11
sim_data_dir = './sim_data/'

beamformer_names = ['Rake-DS',
                    'Rake-MaxSINR',
                    'Rake-MaxUDR']
NBF = len(beamformer_names)

loops = 0

# Specify all files to use with a pattern
name_pattern = 'quality_2015*.npz'
files = [file for file in os.listdir(sim_data_dir) if fnmatch.fnmatch(file, name_pattern)]

# Or just specify a list of file names
files = [ '20150109_10000Loops/quality_20150109-070951z.npz', 
          '20150109_10000Loops/quality_20150109-095429z.npz', ]
#files = ['20150109_10000Loops/quality_20150109-095429z.npz', ]

# Empty data containers
good_source = np.zeros((0,2))
bad_source = np.zeros((0,2))
ipesq = np.zeros((0,2))
opesq_tri = np.zeros((0,2,2))
opesq_bf = np.zeros((0,2,NBF,max_sources))
isinr = np.zeros((0))
osinr_tri = np.zeros((0,2))
osinr_bf = np.zeros((0,NBF,max_sources))

# Read in all the data
for fname in files:
    print 'Loading from',fname

    a = np.load(sim_data_dir + fname)

    print good_source.shape

    good_source = np.concatenate((good_source, a['good_source']), axis=0)
    bad_source = np.concatenate((bad_source, a['bad_source']), axis=0)

    isinr = np.concatenate((isinr,a['isinr']), axis=0)
    osinr_bf = np.concatenate((osinr_bf,a['osinr_bf']), axis=0)
    osinr_tri = np.concatenate((osinr_tri,a['osinr_trinicon']), axis=0)
    ipesq = np.concatenate((ipesq,a['pesq_input']), axis=0)
    opesq_bf = np.concatenate((opesq_bf,a['pesq_bf']), axis=0)
    opesq_tri = np.concatenate((opesq_tri,a['pesq_trinicon']), axis=0)

loops = good_source.shape[0]

opesq_bf_win = opesq_bf[:4800]
opesq_bf_lin = opesq_bf[4800:]

m_win = np.median(opesq_bf_win[:,0,:,:], axis=0)
m_lin = np.median(opesq_bf_lin[:,0,:,:], axis=0)
print m_win - m_lin

print 'Number of loops:',loops
print 'Median input Raw MOS',np.median(ipesq[:,0])
print 'Median input MOS LQO',np.median(ipesq[:,1])

# Trinicon is blind so we have PESQ for both output channels
# Select the channel that has highest Raw MOS for evaluation
I_tri = np.argmax(opesq_tri[:,0,:], axis=1)
opesq_tri_max = np.array([opesq_tri[i,:,I_tri[i]] for i in xrange(opesq_tri.shape[0])])
osinr_tri_max = np.array([osinr_tri[i,I_tri[i]] for i in xrange(osinr_tri.shape[0])])

print 'Median Trinicon Raw MOS',np.median(opesq_tri_max[:,0])
print 'Median Trinicon MOS LQO',np.median(opesq_tri_max[:,1])
print 'Median Trinicon SINR',np.median(osinr_tri_max[:])

plt.figure()
plt.plot(good_source[:,0], good_source[:,1], 'o')
plt.plot(bad_source[:,0], bad_source[:,1], '*')

plt.figure(figsize=(18,9))

newmap = plt.get_cmap('gist_heat')
from itertools import cycle
lines = ['-s','-o','-v','-D','->']
linecycler = cycle(lines)

def nice_plot(x):
    ax1 = plt.gca()
    ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.9,len(beamformer_names))])

    for i, bf in enumerate(beamformer_names):
        p, = plt.plot(range(0, max_sources), 
                np.median(x[:,i,:], axis=0),
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)

        if bf == 'Rake-MaxSINR':
            plt.fill_between(range(0, max_sources),
                np.percentile(x[:,i,:], 25, axis=0),
                np.percentile(x[:,i,:], 75, axis=0),
                color='grey',
                linewidth=0.3,
                edgecolor='k',
                alpha=0.7)


plt.subplot(2,3,1)
nice_plot(opesq_bf[:,0,:,:])
plt.xlabel('Number of sources')
plt.ylabel('Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,2)
nice_plot(opesq_bf[:,1,:,:])
plt.xlabel('Number of sources')
plt.ylabel('MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,3)
nice_plot(osinr_bf)
plt.xlabel('Number of sources')
plt.ylabel('output SINR')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,4)
nice_plot(opesq_bf[:,0,:,:] - ipesq[:,0,np.newaxis,np.newaxis])
plt.xlabel('Number of sources')
plt.ylabel('Improvement Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,5)
nice_plot(opesq_bf[:,1,:,:] - ipesq[:,1,np.newaxis,np.newaxis])
plt.xlabel('Number of sources')
plt.ylabel('Improvement MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,6)
nice_plot(osinr_bf[:,:,:] - isinr[:,np.newaxis,np.newaxis])
plt.xlabel('Number of sources')
plt.ylabel('Improvement SINR')
plt.legend(beamformer_names, loc=2)

plt.savefig('figures/perceptual_quality.pdf')

plt.show()
