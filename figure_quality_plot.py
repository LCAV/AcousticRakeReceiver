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
name_pattern = 'quality_NSOURCES' + str(max_sources) + '_LOOPS*_PID*.npz'
files = [file for file in os.listdir(sim_data_dir) if fnmatch.fnmatch(file, name_pattern)]

ipesq = np.zeros((2,max_sources,0))
opesq = np.zeros((2,NBF,max_sources,0))
isinr = np.zeros((max_sources,0))
osinr = np.zeros((NBF,max_sources,0))

for fname in files:

    a = np.load(sim_data_dir + fname)

    #print osinr.shape
    #print a['osinr'].shape

    isinr = np.concatenate((isinr,a['isinr']), axis=-1)
    osinr = np.concatenate((osinr,a['osinr']), axis=-1)
    ipesq = np.concatenate((ipesq,a['pesq_input']), axis=-1)
    opesq = np.concatenate((opesq,a['pesq']), axis=-1)

print 'Median input Raw MOS',np.median(ipesq[0,:,:])
print 'Median input MOS LQO',np.median(ipesq[1,:,:])

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
            np.median(x[i,:,:], axis=1),
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)

        plt.fill_between(range(0, max_sources),
            np.percentile(x[i,:,:], 25, axis=1),
            np.percentile(x[i,:,:], 75, axis=1),
            color='grey',
            linewidth=0.3,
            edgecolor='k',
            alpha=0.7)


plt.subplot(2,3,1)
nice_plot(opesq[0,:,:,:])
plt.xlabel('Number of sources')
plt.ylabel('Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,2)
nice_plot(opesq[1,:,:,:])
plt.xlabel('Number of sources')
plt.ylabel('MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,3)
nice_plot(osinr)
plt.xlabel('Number of sources')
plt.ylabel('output SINR')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,4)
nice_plot(opesq[0,:,:,:] - ipesq[0,:,:])
plt.xlabel('Number of sources')
plt.ylabel('Improvement Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,5)
nice_plot(opesq[1,:,:,:] - ipesq[1,:,:])
plt.xlabel('Number of sources')
plt.ylabel('Improvement MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,6)
nice_plot(osinr[:,:,:] - isinr[np.newaxis,:,:])
plt.xlabel('Number of sources')
plt.ylabel('Improvement SINR')
plt.legend(beamformer_names, loc=2)

plt.savefig('figures/perceptual_quality.pdf')

plt.show()
