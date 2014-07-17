Acoustic Rake Receiver
======================

This repository contains all the code to reproduce the results of the paper
*Raking the Cocktail Party*.

We created a simple framework for simulation of room acoustics in object
oriented python and apply it to perform numerical experiments related to
this paper. All the figures and sound samples can be recreated by calling
simple scripts leveraging this framework. We strongly hope that this code
will be useful beyond the scope of this paper and plan to develop it into
a standalone python package in the future.

We are available for any question or request relating to either the code
or the theory behind it. Just ask!

Abstract
--------

We present the concept of an acoustic rake receiver (ARR) — a microphone
beamformer that uses echoes to improve the noise and interference suppression.
The rake idea is well-known in wireless communications. It involves
constructively combining different multipath components that arrive at the
receiver antennas. Unlike typical spread-spectrum signals used in wireless
communications, speech signals are not orthogonal to their shifts, which makes
acoustic raking a more challenging problem. That is why the correct way to
think about it is spatial. Instead of explicitly estimating the channel, we
create correspondences between early echoes in time and image sources in space.
These multiple sources of the desired and interfering signals offer additional
spatial diversity that we can exploit in the beamformer design.

We present several "intuitive" and optimal formulations of ARRs, and show
theoretically and numerically that the rake formulation of the maximum
signal-to-interference-and-noise beamformer offers significant performance
boosts in terms of noise suppression and interference cancellation. We
accompany the paper by the complete simulation and processing chain written in
Python.


Authors
-------

Ivan Dokmanić, Robin Scheibler, and Martin Vetterli are with 
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).

<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contact

[Ivan Dokmanić](mailto:ivan[dot]dokmanic[at]epfl[dot]ch) <br>
EPFL-IC-LCAV <br>
BC Building <br>
Station 14 <br>
1015 Lausanne


Sound Samples
-------------

* [sample1](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/input_mic.wav) Simulated microphone input signal.
* [sample2](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/output_maxsinr.wav) Output of conventional Max-SINR beamformer.
* [sample3](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/output_rake-maxsinr.wav) Output of proposed  Rake-Max-SINR beamformer.


Dependencies
------------

* A working distribution of [Python 2.7](https://www.python.org/downloads/).

* The code relies heavily on [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/), and [matplotlib](http://matplotlib.org).

* We use the distribution [anaconda](https://store.continuum.io/cshop/anaconda/) to simplify the setup of the environment.


Recreate the figures and sound samples
--------------------------------------

In a UNIX terminal, run the following script.

    ./make_all_figures.sh

Alternatively, type in the following commands in an ipython shell.

    run figure_spectrograms.py
    run figure_beam_scenarios.py
    run figure_Measures1.py
    run figure_Measures2.py
    run figure_SumNorm.py

The figures and sound samples generated are collected in `figures` and
`output_samples`, respectively.

License
-------

Copyright (c) 2014, Ivan Dokmanić, Robin Scheibler, Martin Vetterli

This code is free to reuse for non-commercial purpose such as academic or
educational. For any other use, please contact the authors.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Acoustic Rake Receiver</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://lcav.epfl.ch" property="cc:attributionName" rel="cc:attributionURL">Ivan Dokmanić, Robin Scheibler, Martin Vetterli</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/LCAV/AcousticRakeReceiver" rel="dct:source">https://github.com/LCAV/AcousticRakeReceiver</a>.

