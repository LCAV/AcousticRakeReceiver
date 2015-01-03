import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import metrics as metrics

max_sources = 11
loops = 1000
sim_data_dir = './sim_data/'

beamformer_names = ['Rake-DS',
                    'Rake-MaxSINR',
                    'Rake-MaxUDR']
NBF = len(beamformer_names)

pesq_improv_rawmos = np.zeros((NBF, max_sources))
pesq_improv_rawmos_ci = np.zeros((2*NBF, max_sources))
pesq_rawmos_med = np.zeros((NBF,max_sources))
pesq_rawmos_ci = np.zeros((2*NBF,max_sources))

all_pesq_rawmos_input = np.array([])
all_pesq_moslqo_input = np.array([])

pesq_improv_moslqo = np.zeros((NBF, max_sources))
pesq_improv_moslqo_ci = np.zeros((2*NBF, max_sources))
pesq_moslqo_med = np.zeros((NBF,max_sources))
pesq_moslqo_ci = np.zeros((2*NBF,max_sources))

sinr_improv = np.zeros((NBF, max_sources))
sinr_improv_ci = np.zeros((2*NBF, max_sources))
osinr_med = np.zeros((NBF,max_sources))
osinr_ci = np.zeros((2*NBF,max_sources))

for ns in np.arange(max_sources):

    fname = 'quality_NSOURCES' + str(ns+1) + '_LOOPS' + str(loops) + '.npz'
    a = np.load(sim_data_dir + fname)

    isinr = a['isinr']
    osinr = a['osinr']

    pesq_input_rawmos = a['pesq_input_rawmos']
    pesq_rawmos = a['pesq_rawmos']
    all_pesq_rawmos_input = np.concatenate((all_pesq_rawmos_input, pesq_input_rawmos))

    pesq_input_moslqo = a['pesq_input_moslqo']
    pesq_moslqo = a['pesq_moslqo']
    all_pesq_moslqo_input = np.concatenate((all_pesq_moslqo_input, pesq_input_moslqo))

    for i in xrange(NBF):
        pesq_rawmos_med[i,ns], pesq_rawmos_ci[2*i:2*i+2,ns] = \
                metrics.median(pesq_rawmos[i,:])
        pesq_moslqo_med[i,ns], pesq_moslqo_ci[2*i:2*i+2,ns] = \
                metrics.median(pesq_moslqo[i,:])
        osinr_med[i,ns], osinr_ci[2*i:2*i+2,ns] = metrics.median(osinr[i,:])

        pesq_improv_rawmos[i,ns], pesq_improv_rawmos_ci[2*i:2*i+2,ns] = \
                metrics.median(pesq_rawmos[i,:] - pesq_input_rawmos[:])
        pesq_improv_moslqo[i,ns], pesq_improv_moslqo_ci[2*i:2*i+2,ns] = \
                metrics.median(pesq_moslqo[i,:] - pesq_input_moslqo[:])
        sinr_improv[i,ns], sinr_improv_ci[2*i:2*i+2,ns] = \
                metrics.median(u.dB(osinr[i,:]) - u.dB(isinr[:]))
    

print 'Median input Raw MOS',np.median(all_pesq_rawmos_input)
print 'Median input MOS LQO',np.median(all_pesq_moslqo_input)

plt.figure(figsize=(18,9))

newmap = plt.get_cmap('gist_heat')
from itertools import cycle
lines = ['-s','-o','-v','-D','->']
linecycler = cycle(lines)

plt.subplot(2,3,1)

ax1 = plt.gca()
ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.9,len(beamformer_names))])

for i, bf in enumerate(beamformer_names):
    p, = plt.plot(range(0, max_sources), 
            pesq_rawmos_med[i,:],
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)
    plt.fill_between(range(0, max_sources),
            pesq_rawmos_med[i,:] + pesq_rawmos_ci[2*i,:],
            pesq_rawmos_med[i,:] + pesq_rawmos_ci[2*i+1,:],
            color='grey',
            linewidth=0.3,
            edgecolor='k',
            alpha=0.7)

plt.xlabel('Number of sources')
plt.ylabel('Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,2)
plt.plot(pesq_moslqo_med.T)
plt.xlabel('Number of sources')
plt.ylabel('MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,3)
plt.plot(u.dB(osinr_med.T))
plt.xlabel('Number of sources')
plt.ylabel('output SINR')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,4)

ax1 = plt.gca()
ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.25,0.9,len(beamformer_names))])

for i, bf in enumerate(beamformer_names):
    p, = plt.plot(range(0, max_sources), 
            pesq_improv_rawmos[i,:],
            next(linecycler),
            linewidth=1,
            markersize=4,
            markeredgewidth=.5)
    plt.fill_between(range(0, max_sources),
            pesq_improv_rawmos[i,:] + pesq_improv_rawmos_ci[2*i,:],
            pesq_improv_rawmos[i,:] + pesq_improv_rawmos_ci[2*i+1,:],
            color='grey',
            linewidth=0.3,
            edgecolor='k',
            alpha=0.7)

plt.xlabel('Number of sources')
plt.ylabel('Improvement Raw MOS')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,5)
plt.plot(pesq_improv_moslqo.T)
plt.xlabel('Number of sources')
plt.ylabel('Improvement MOS LQO')
plt.legend(beamformer_names, loc=2)

plt.subplot(2,3,6)
plt.plot(sinr_improv.T)
plt.xlabel('Number of sources')
plt.ylabel('Improvement SINR')
plt.legend(beamformer_names, loc=2)

plt.savefig('figures/perceptual_quality.pdf')

# some histograms
fname = 'quality_NSOURCES' + str(1) + '_LOOPS' + str(loops) + '.npz'
a = np.load(sim_data_dir + fname)
prm0 = a['pesq_rawmos'][1,:]
pml0 = a['pesq_moslqo'][1,:]
fname = 'quality_NSOURCES' + str(5) + '_LOOPS' + str(loops) + '.npz'
a = np.load(sim_data_dir + fname)
prm5 = a['pesq_rawmos'][1,:]
pml5 = a['pesq_moslqo'][1,:]

plt.figure()
plt.subplot(2,2,1)
plt.hist(prm0, bins=30)
plt.title('MVDR')
plt.xlabel('PESQ [Raw MOS]')

plt.subplot(2,2,2)
plt.hist(pml0, bins=30)
plt.title('MVDR')
plt.xlabel('PESQ [MOS LQO]')

plt.subplot(2,2,3)
plt.hist(prm5, bins=30)
plt.title('RAKE')
plt.xlabel('PESQ [Raw MOS]')

plt.subplot(2,2,4)
plt.hist(pml5, bins=30)
plt.title('RAKE')
plt.xlabel('PESQ [MOS LQO]')

plt.show()
