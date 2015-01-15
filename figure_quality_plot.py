import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import metrics as metrics

import sys
import os
import fnmatch

max_sources = 11
sim_data_dir = './sim_data/'

beamformer_names = ['Rake-DS',
                    'Rake-MaxSINR',
                    'Rake-MaxUDR']
bf_dict = dict(zip(beamformer_names, 
               range(len(beamformer_names))))
NBF = len(beamformer_names)

loops = 0

if len(sys.argv) == 0:
    # if no argument is specified, use all available files
    name_pattern = './sim_data/quality_2015*.npz'
    files = [file for file in os.listdir(sim_data_dir) if fnmatch.fnmatch(file, name_pattern)]
else:
    files = sys.argv[1:]

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

    a = np.load(fname)

    good_source = np.concatenate((good_source, a['good_source']), axis=0)
    bad_source = np.concatenate((bad_source, a['bad_source']), axis=0)

    isinr = np.concatenate((isinr,u.dB(a['isinr'])), axis=0)
    osinr_bf = np.concatenate((osinr_bf,u.dB(a['osinr_bf'])), axis=0)
    osinr_tri = np.concatenate((osinr_tri,u.dB(a['osinr_trinicon'])), axis=0)
    ipesq = np.concatenate((ipesq,a['pesq_input']), axis=0)
    opesq_bf = np.concatenate((opesq_bf,a['pesq_bf']), axis=0)
    opesq_tri = np.concatenate((opesq_tri,a['pesq_trinicon']), axis=0)

loops = good_source.shape[0]

print 'Number of loops:',loops
print 'Median input Raw MOS',np.median(ipesq[:,0])
print 'Median input MOS LQO',np.median(ipesq[:,1])
print 'Median input SINR',np.median(isinr[:])

# Trinicon is blind so we have PESQ for both output channels
# Select the channel that has highest Raw MOS for evaluation
I_tri = np.argmax(opesq_tri[:,0,:], axis=1)
opesq_tri_max = np.array([opesq_tri[i,:,I_tri[i]] for i in xrange(opesq_tri.shape[0])])
osinr_tri_max = np.array([osinr_tri[i,I_tri[i]] for i in xrange(osinr_tri.shape[0])])

print 'Median Trinicon Raw MOS',np.median(opesq_tri_max[:,0])
print 'Median Trinicon MOS LQO',np.median(opesq_tri_max[:,1])
print 'Median Trinicon SINR',np.median(osinr_tri_max[:])

def nice_plot(x, ylabel, bf_order=None):
    '''
    Define a function to plot consistently the data
    '''

    if bf_order is None:
        bf_order = beamformer_names

    ax1 = plt.gca()

    newmap = plt.get_cmap('gist_heat')
    from itertools import cycle

    # totally a hack to get the same line styles as Fig6/7
    lines = ['-D','-v','->','-s','-o']
    linecycler = cycle(lines)

    # totally a hack to get the same line styles as Fig6/7
    map1 = [newmap( k ) for k in np.linspace(0.25,0.9,5)]
    map2 = [map1[3],map1[2],map1[4],map1[0],map1[1]]

    ax1.set_color_cycle(map2)

    # no clipping of the beautiful markers
    plt.setp(ax1,'clip_on',False)

    for bf in bf_order:
        i = bf_dict[bf]
        p, = plt.plot(range(0, max_sources), 
                np.median(x[:,i,:], axis=0),
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5,
            clip_on=False)

        if bf == 'Rake-MaxSINR':
            plt.fill_between(range(0, max_sources),
                np.percentile(x[:,i,:], 25, axis=0),
                np.percentile(x[:,i,:], 75, axis=0),
                color='grey',
                linewidth=0.3,
                edgecolor='k',
                alpha=0.7)

        # Hide right and top axes
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position(('outward', 10))
        ax1.spines['left'].set_position(('outward', 15))
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        # Make ticks nicer
        ax1.xaxis.set_tick_params(width=.3, length=3)
        ax1.yaxis.set_tick_params(width=.3, length=3)

        # Make axis lines thinner
        for axis in ['bottom','left']:
          ax1.spines[axis].set_linewidth(0.3)

        # Set ticks fontsize
        plt.xticks(size=9)
        plt.yticks(size=9)

        # Set labels
        plt.xlabel(r'Number of images $K$', fontsize=10)
        plt.ylabel(ylabel, fontsize=10)

        plt.legend(bf_order, fontsize=7, loc='upper left', frameon=False, labelspacing=0)


'''
# Here is a larger figure with all performance measures.
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
nice_plot(opesq_bf[:,0,:,:], 'PESQ [Raw MOS]')
plt.xlabel('Number of sources')
plt.ylabel('Raw MOS')

plt.subplot(2,3,2)
nice_plot(opesq_bf[:,1,:,:], 'PESQ [MOS LQO]')

plt.subplot(2,3,3)
nice_plot(osinr_bf, 'SINR [dB]')
plt.xlabel('Number of sources')
plt.ylabel('output SINR')

plt.subplot(2,3,4)
nice_plot(opesq_bf[:,0,:,:] - ipesq[:,0,np.newaxis,np.newaxis], 'Improvement PESQ [Raw MOS]')
plt.xlabel('Number of sources')
plt.ylabel('Improvement Raw MOS')

plt.subplot(2,3,5)
nice_plot(opesq_bf[:,1,:,:] - ipesq[:,1,np.newaxis,np.newaxis], 'Improvement PESQ [MOS LQO]')
plt.xlabel('Number of sources')
plt.ylabel('Improvement MOS LQO')

plt.subplot(2,3,6)
nice_plot(osinr_bf[:,:,:] - isinr[:,np.newaxis,np.newaxis], 'Improvement SINR [dB]')
plt.xlabel('Number of sources')
plt.ylabel('Improvement SINR')

plt.tight_layout(pad=0.2)
'''

# Here we plot the figure used in the paper (Fig. 10)
plt.figure(figsize=(4,3))
nice_plot(opesq_bf[:,0,:,:], 'PESQ [MOS]', 
        bf_order=['Rake-MaxSINR','Rake-DS','Rake-MaxUDR'])
#plt.plot(np.arange(max_sources), np.median(ipesq[:,0])*np.ones(max_sources))
#plt.plot(np.arange(max_sources), np.median(opesq_tri_max[:,0])*np.ones(max_sources))
plt.tight_layout()
plt.savefig('figures/perceptual_quality.pdf')

