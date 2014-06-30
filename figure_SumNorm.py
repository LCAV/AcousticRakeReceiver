
import numpy as np
import matplotlib
import constants

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Room as rg
import beamforming as bf

# Room 1 : Shoe box
p1 = np.array([-1, 0])
p2 = np.array([5, 4])
mic1 = [2, 3]
Fs = 44000
absorption = 0.8
max_order = 4

# Create a microphone array
M = 12
d = 0.2
frequencies = np.arange(100, 4000, 100)
mics = bf.Beamformer.linear2D(Fs, mic1, M, 0, d)


n_monte_carlo = 100

SNR_gain = np.zeros(frequencies.shape)
for i, f in enumerate(frequencies):
    print 'Simulating for the frequency', f
    for n in range(0, n_monte_carlo):

        # Generate a source at a random location. TO DO: Add a bounding box for
        # sources!
        source1 = p1 + np.random.rand(2) * (p2 - p1)

        # Create the room
        room1 = rg.Room.shoeBox2D(
            p1,
            p2,
            Fs,
            max_order=max_order,
            absorption=absorption)
        room1.addSource(source1)
        room1.addMicrophoneArray(mics)

        A = mics.steering_vector_2D_from_point(f, room1.sources[0].getImages(max_order=2), attn=True)
        SNR_gain[i] += np.linalg.norm(np.sum(A, axis=1)) ** 2 / np.linalg.norm(A[:, 0]) ** 2

    SNR_gain[i] /= n_monte_carlo


# Plot the results
plt.figure
plt.plot(frequencies, SNR_gain)
plt.show()
