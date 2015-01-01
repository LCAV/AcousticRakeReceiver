import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import metrics as metrics

max_sources = 11
loops = 20
sim_data_dir = './sim_data/'

beamformer_names = ['Rake-DS',
                    'Rake-MaxSINR',
                    'Rake-MaxUDR']
NBF = len(beamformer_names)

pesq_improv_rawmos = np.zeros((NBF, max_sources))
pesq_rawmos_med = np.zeros((NBF,max_sources))

pesq_improv_moslqo = np.zeros((NBF, max_sources))
pesq_moslqo_med = np.zeros((NBF,max_sources))

sinr_improv = np.zeros((NBF, max_sources))
osinr_med = np.zeros((NBF,max_sources))

for ns in np.arange(max_sources):

    fname = 'quality_NSOURCES' + str(ns+1) + '_LOOPS' + str(loops) + '.npz'
    a = np.load(sim_data_dir + fname)

    isinr = a['isinr']
    osinr = a['osinr']

    pesq_input_rawmos = a['pesq_input_rawmos']
    pesq_rawmos = a['pesq_rawmos']

    pesq_input_moslqo = a['pesq_input_moslqo']
    pesq_moslqo = a['pesq_moslqo']

    pesq_rawmos_med[:,ns] = np.median(pesq_rawmos, axis=1)
    pesq_moslqo_med[:,ns] = np.median(pesq_moslqo, axis=1)
    osinr_med[:,ns] = np.median(osinr, axis=1)

    pesq_improv_rawmos[:,ns] = np.median(pesq_rawmos - pesq_input_rawmos, axis=1)
    pesq_improv_moslqo[:,ns] = np.median(pesq_moslqo - pesq_input_moslqo, axis=1)
    sinr_improv[:,ns] = np.median(u.dB(osinr) - u.dB(isinr), axis=1)
    

plt.figure()

plt.subplot(2,3,1)
plt.plot(pesq_rawmos_med.T)
plt.xlabel('Number of sources')
plt.ylabel('Raw MOS')
plt.legend(beamformer_names)

plt.subplot(2,3,2)
plt.plot(pesq_moslqo_med.T)
plt.xlabel('Number of sources')
plt.ylabel('MOS LQO')
plt.legend(beamformer_names)

plt.subplot(2,3,3)
plt.plot(u.dB(osinr_med.T))
plt.xlabel('Number of sources')
plt.ylabel('output SINR')
plt.legend(beamformer_names)

plt.subplot(2,3,4)
plt.plot(pesq_improv_rawmos.T)
plt.xlabel('Number of sources')
plt.ylabel('Improvement Raw MOS')
plt.legend(beamformer_names)

plt.subplot(2,3,5)
plt.plot(pesq_improv_moslqo.T)
plt.xlabel('Number of sources')
plt.ylabel('Improvement MOS LQO')
plt.legend(beamformer_names)

plt.subplot(2,3,6)
plt.plot(sinr_improv.T)
plt.xlabel('Number of sources')
plt.ylabel('Improvement SINR')
plt.legend(beamformer_names)

'''
plt.figure()
plt.subplot(2,3,1)
plt.hist(mvdr_pesq_rawmos, bins=30)
plt.title('MVDR')
plt.xlabel('PESQ [Raw MOS]')

plt.subplot(2,3,2)
plt.hist(mvdr_pesq_moslqo, bins=30)
plt.title('MVDR')
plt.xlabel('PESQ [MOS LQO]')

plt.subplot(2,3,3)
plt.hist(mvdr_snr, bins=30)
plt.title('MVDR')
plt.xlabel('SNR [dB]')

plt.subplot(2,3,4)
plt.hist(rake_pesq_rawmos, bins=30)
plt.title('RAKE')
plt.xlabel('PESQ [Raw MOS]')

plt.subplot(2,3,5)
plt.hist(rake_pesq_moslqo, bins=30)
plt.title('RAKE')
plt.xlabel('PESQ [MOS LQO]')

plt.subplot(2,3,6)
plt.hist(rake_snr, bins=30)
plt.title('RAKE')
plt.xlabel('SNR [dB]')
'''

plt.show()
