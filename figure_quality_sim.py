
def perceptual_quality_evaluation(good_source, bad_source):
    '''
    Perceputal Quality evaluation simulation
    Inner Loop
    '''

    # Imports are done in the function so that it can be easily
    # parallelized
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample
    from os import getpid

    from Room import Room
    from beamforming import Beamformer, MicrophoneArray
    from trinicon import trinicon

    from utilities import normalize, to_16b, highpass
    from phat import time_align
    from metrics import snr, pesq
    
    # number of number of sources
    n_sources = np.arange(1,12)
    S = n_sources.shape[0]

    # we the speech samples used
    speech_sample1 = 'samples/fq_sample1_8000.wav'
    speech_sample2 = 'samples/fq_sample2_8000.wav'

    # Some simulation parameters
    Fs = 8000
    t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
    absorption = 0.90
    max_order_sim = 10
    SNR_at_mic = 20          # SNR at center of microphone array in dB

    # Room 1 : Shoe box
    room_dim = [4, 6]

    # microphone array design parameters
    mic1 = [2, 1.5]         # position
    M = 8                   # number of microphones
    d = 0.08                # distance between microphones
    phi = 0.                # angle from horizontal
    shape = 'Linear'        # array shape

    # create a microphone array
    if shape is 'Circular':
        mics = Beamformer.circular2D(Fs, mic1, M, phi, d*M/(2*np.pi)) 
    else:
        mics = Beamformer.linear2D(Fs, mic1, M, phi, d) 

    # create a single reference mic at center of array
    ref_mic = MicrophoneArray(mics.center, Fs)

    # define the array processing type
    L = 4096                # frame length
    hop = 2048              # hop between frames
    zp = 2048               # zero padding (front + back)
    mics.setProcessing('FrequencyDomain', L, hop, zp, zp)

    # data receptacles
    beamformer_names = ['Rake-DS',
                        'Rake-MaxSINR',
                        'Rake-MaxUDR']
    bf_weights_fun   = [mics.rakeDelayAndSumWeights,
                        mics.rakeMaxSINRWeights,
                        mics.rakeMaxUDRWeights]
    bf_fnames = ['1','2','3']
    NBF = len(beamformer_names)

    # receptacle arrays
    pesq_input = np.zeros(2)
    pesq_trinicon = np.zeros(2)
    pesq_bf = np.zeros((2,NBF,S))
    isinr = 0
    osinr_trinicon = 0
    osinr_bf = np.zeros((NBF,S))

    # since we run multiple thread, we need to uniquely identify filenames
    pid = str(getpid())

    file_ref  = 'output_samples/fqref' + pid + '.wav'
    file_suffix = '-' + pid + '.wav'
    files_tri = ['output_samples/fqt' + str(i+1) + file_suffix for i in xrange(2)]
    files_bf = ['output_samples/fq' + str(i+1) + file_suffix for i in xrange(NBF)]
    file_raw  = 'output_samples/fqraw' + pid + '.wav'

    # Read the two speech samples used
    rate, good_signal = wavfile.read(speech_sample1)
    good_signal = np.array(good_signal, dtype=float)
    good_signal = normalize(good_signal)
    good_signal = highpass(good_signal, rate)
    good_len = good_signal.shape[0]/float(Fs)

    rate, bad_signal = wavfile.read(speech_sample2)
    bad_signal = np.array(bad_signal, dtype=float)
    bad_signal = normalize(bad_signal)
    bad_signal = highpass(bad_signal, rate)
    bad_len = bad_signal.shape[0]/float(Fs)

    # variance of good signal
    good_sigma2 = np.mean(good_signal**2)

    # normalize interference signal to have equal power with desired signal
    bad_signal *= good_sigma2/np.mean(bad_signal**2)

    # pick good source position at random
    good_distance = np.linalg.norm(mics.center[:,0] - np.array(good_source))
    
    # pick bad source position at random
    bad_distance = np.linalg.norm(mics.center[:,0] - np.array(bad_source))

    if good_len > bad_len:
        good_delay = 0
        bad_delay = (good_len - bad_len)/2.
    else:
        bad_delay = 0
        good_delay = (bad_len - good_len)/2.

    # compute the noise variance at center of array wrt good signal and SNR
    sigma2_n = good_sigma2/(4*np.pi*good_distance)**2/10**(SNR_at_mic/10)

    # create the reference room for freespace, noisless, no interference simulation
    ref_room = Room.shoeBox2D(
        [0,0],
        room_dim,
        Fs,
        t0 = t0,
        max_order=0,
        absorption=absorption,
        sigma2_awgn=0.)
    ref_room.addSource(good_source, signal=good_signal, delay=good_delay)
    ref_room.addMicrophoneArray(ref_mic)
    ref_room.compute_RIR()
    ref_room.simulate()
    reference = ref_mic.signals[0]
    reference_n = normalize(reference)

    # save the reference desired signal
    wavfile.write(file_ref, Fs, to_16b(reference_n))

    # create the 'real' room with sources and mics
    room1 = Room.shoeBox2D(
        [0,0],
        room_dim,
        Fs,
        t0 = t0,
        max_order=max_order_sim,
        absorption=absorption,
        sigma2_awgn=sigma2_n)

    # add sources to room
    room1.addSource(good_source, signal=good_signal, delay=good_delay)
    room1.addSource(bad_source, signal=bad_signal, delay=bad_delay)

    # Record first the degraded signal at reference mic (center of array)
    room1.addMicrophoneArray(ref_mic)
    room1.compute_RIR()
    room1.simulate()
    raw_n = normalize(highpass(ref_mic.signals[0], Fs))

    # save degraded reference signal
    wavfile.write(file_raw, Fs, to_16b(raw_n))

    # Compute PESQ and SINR of raw degraded reference signal
    isinr = snr(reference_n, raw_n[:reference_n.shape[0]])
    pesq_input[:] = pesq(file_ref, file_raw, Fs=Fs).T
        
    # Now record input of microphone array
    room1.addMicrophoneArray(mics)
    room1.compute_RIR()
    room1.simulate()

    # Run the Trinicon algorithm
    double_sig = mics.signals.copy()
    for i in xrange(2):
        double_sig = np.concatenate((double_sig, mics.signals), axis=1)
    sig_len = mics.signals.shape[1]
    output_trinicon = trinicon(double_sig)[:,-sig_len:]

    # normalize time-align and save to file
    output_tri1 = normalize(highpass(output_trinicon[0,:], Fs))
    output_tri1 = time_align(reference_n, output_tri1)
    wavfile.write(files_tri[0], Fs, to_16b(output_tri1))
    output_tri2 = normalize(highpass(output_trinicon[1,:], Fs))
    output_tri2 = time_align(reference_n, output_tri2)
    wavfile.write(files_tri[1], Fs, to_16b(output_tri2))

    # evaluate
    # we consider the signal with highest PESQ Raw MOS as target signal
    pesq_val = pesq(file_ref, files_tri, Fs=Fs)
    i_m = np.argmax(pesq_val[0,:])
    pesq_trinicon = pesq_val[:,i_m]
    if i_m == 0:
        osinr_trinicon = snr(reference_n, output_tri1)
    else:
        osinr_trinicon = snr(reference_n, output_tri2)

    # Run all the beamformers
    for k,s in enumerate(n_sources):

        ''' 
        BEAMFORMING PART
        '''
        # Extract image sources locations and create noise covariance matrix
        good_sources = room1.sources[0].getImages(n_nearest=s, 
                                                    ref_point=mics.center)
        bad_sources = room1.sources[1].getImages(n_nearest=s,
                                                    ref_point=mics.center)
        Rn = sigma2_n*np.eye(mics.M)

        # run for all beamformers considered
        for i, bfr in enumerate(beamformer_names):

            # compute the beamforming weights
            bf_weights_fun[i](good_sources, bad_sources,
                                    R_n = sigma2_n*np.eye(mics.M), 
                                    attn=True, ff=False)

            output = mics.process()
            output = normalize(highpass(output, Fs))
            output = time_align(reference_n, output)

            # save files for PESQ evaluation
            wavfile.write(files_bf[i], Fs, to_16b(output))

            # compute output SINR
            osinr_bf[i,k] = snr(reference_n, output)

            # compute PESQ
            pesq_bf[:,i,k] = pesq(file_ref, files_bf[i], Fs=Fs).T

            # end of beamformers loop

        # end of number of sources loop

    return pesq_input, pesq_trinicon, pesq_bf, isinr, osinr_trinicon, osinr_bf



if __name__ == '__main__':

    import numpy as np
    import sys
    import time
    from IPython import parallel

    # setup parallel computation env
    c = parallel.Client()
    print c.ids
    c.blocks = True
    view = c.load_balanced_view()

    Loops = int(sys.argv[1])

    # number of image sources to consider
    n_src = np.arange(l_src, u_src)

    # we restrict sources to be in a square 1m away from every wall and from the array
    bbox_size = np.array([[2.,2.5]])
    bbox_origin = np.array([[1.,2.5]])

    # draw all target and interferer at random
    good_source = np.random.random((Loops,2))*bbox_size + bbox_origin
    bad_source = np.random.random((Loops,2))*bbox_size + bbox_origin

    start = time.time()
    out = view.map_sync(perceptual_quality_evaluation, good_source, bad_source)
    #perceptual_quality_evaluation(good_source[:,0], bad_source[:,0])
    ellapsed = time.time() - start

    print('Time ellapsed: ' + str(ellapsed))

    # recover all the data
    pesq_input = np.array([o[0] for o in out])
    pesq_trinicon = np.array([o[1] for o in out])
    pesq_bf = np.array([o[2] for o in out])
    isinr = np.array([o[3] for o in out])
    osinr_trinicon = np.array([o[4] for o in out])
    osinr_bf = np.array([o[5] for o in out])

    # save the simulation results to file
    filename = 'sim_data/quality_' + time.strftime('%Y%m%d-%H%M%Sz') + '.npz'
    np.savez_compressed(filename, good_source=good_source, bad_source=bad_source,
            isinr=isinr, osinr_bf=osinr_bf, osinr_trinicon=osinr_trinicon,
            pesq_bf=pesq_bf, pesq_input=pesq_input, pesq_trinicon=pesq_trinicon)

