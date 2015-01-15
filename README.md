Raking the Cocktail Party
=========================

This repository contains all the code to reproduce the results of the paper
[*Raking the Cocktail Party*](http://infoscience.epfl.ch/record/200336).

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


Selected results from the paper
-------------------------------

### Spectrograms and Sound Samples

<img src="https://raw.githubusercontent.com/LCAV/AcousticRakeReceiver/master/figures/spectrograms.png" width=800>

Comparison of the conventional Max-SINR and Rake-Max-SINR beamformer on a real
speech sample.  Spectrograms of (A) clean signal of interest, (B) signal
corrupted by an interferer and additive white Gaussian noise at the microphone
input, outputs of (C) conventional Max-SINR and (D) Rake-Max- SINR beamformers.
Time naturally goes from left to right, and frequency increases from zero at
the bottom up to Fs/2. To highlight the improvement of Rake-Max-SINR over
Max-SINR, we blow-up three parts of the spectrograms in the lower part of the
figure. The boxes and the corresponding part of the original spectrogram are
numbered in (A). The numbering is the same but omitted in the rest of the
figure for clarity.

The corresponding sound samples:

* [A](https://github.com/LCAV/AcousticRakeReceiver/raw/master/samples/singing_8000.wav) Desired signal.
* [B](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/input_mic.wav) Simulated microphone input signal.
* [C](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/output_maxsinr.wav) Output of conventional Max-SINR beamformer.
* [D](https://github.com/LCAV/AcousticRakeReceiver/raw/master/output_samples/output_rake-maxsinr.wav) Output of proposed  Rake-Max-SINR beamformer.

### Beam Patterns

<img src="https://raw.githubusercontent.com/LCAV/AcousticRakeReceiver/master/figures/beam_scenarios.png" width=800>

Beam patterns in different scenarios. The rectangular room is 4 by 6 metres and
contains a source of interest (•) and an interferer (✭) ((B), (C), (D) only).
The first order image sources are also displayed. The weight computation of the
beamformer includes the direct source and the first order image sources of both
desired source and interferer (when applicable). (A) Rake-Max-SINR, no
interferer, (B) Rake-Max-SINR, one interferer, (C) Rake-Max-UDR, one
interferer, (D) Rake-Max-SINR, interferer is in direct path.

Dependencies
------------

* A working distribution of [Python 2.7](https://www.python.org/downloads/).
* The code relies heavily on [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/), and [matplotlib](http://matplotlib.org).
* We use the distribution [anaconda](https://store.continuum.io/cshop/anaconda/) to simplify the setup of the environment.

### PESQ Tool

Download the [source files](http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en) of the ITU P.862
compliance tool from the ITU website.

#### Unix compilation (Linux/Mac OS X)

Execute the following sequence of commands to get to the source code.

    mkdir PESQ
    cd PESQ
    wget 'https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items'
    unzip dologin_pub.asp\?lang\=e\&id\=T-REC-P.862-200511-I\!Amd2\!SOFT-ZST-E\&type\=items
    cd Software
    unzip 'P862_annex_A_2005_CD  wav final.zip'
    cd P862_annex_A_2005_CD/source/

In the `Software/P862_annex_A_2005_CD/source/` directory, create a file called `Makefile` and copy
the following into it.

    CC=gcc
    CFLAGS=-O2

    OBJS=dsp.o pesqdsp.o pesqio.o pesqmod.o pesqmain.o
    DEPS=dsp.h pesq.h pesqpar.h

    %.o: %.c $(DEPS)
      $(CC) -c -o $@ $< $(CFLAGS)

    pesq: $(OBJS)
      $(CC) -o $@ $^ $(CFLAGS)

    .PHONY : clean
    clean :
      -rm pesq $(OBJS)

Execute compilation by typing this.

    make pesq

Finally move the `pesq` binary to `<repo_root>/bin/`.

Notes:
* The files input to the pesq utility must be 16 bit PCM wav files.
* File names longer than 14 characters (suffix included) cause the utility to
  crash with the message `Abort trap(6)` or similar.

#### Windows compilation

1. Open visual studio, create a new project from existing files and select the directory
  containing the source code of PESQ (`Software\P862_annex_A_2005_CD\source\`).

          FILE -> New -> Project From Existing Code...

2. Select `Visual C++` from the dropdown menu, then next.
    * *Project file location* : directory containing source code of pesq (`Software\P862_annex_A_2005_CD\source\`).
    * *Project Name* : pesq
    * Then next.
    * As *project type*, select `Console application` project.
    * Then finish.

3. Go to

          BUILD -> Configuration Manager...

    and change active solution configuration from `Debug` to `Release`. Then Close.

4. Then 

          BUILD -> Build Solution

5. Copy the executable `Release\pesq.exe` to the bin folder.

*(tested with Microsoft Windows Server 2012)*

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
    run figure_quality_sim.py -s 10000
    run figure_quality_plot.py

The figures and sound samples generated are collected in `figures` and
`output_samples`, respectively.

The script `figure_quality_sim.py` is very heavy computationally. Above, 10000
is the number of loops. This number can be decreased when testing the code.
It is possible to run it also in parallel in the following way. Open a shell
and type in the following.

    ipcluster start -n <number_of_workers>
    ipython figure_quality_sim.py 10000

On the first line, we start the ipython workers. Notice that we omit the `-s`
option on the second line.  This will run `<number_of_workers>` parallel jobs.
Be sure to *deactivate* the MKL extensions if you have them enabled to make sure
you have maximum efficiency.

License
-------

Copyright (c) 2014, Ivan Dokmanić, Robin Scheibler, Martin Vetterli

This code is free to reuse for non-commercial purpose such as academic or
educational. For any other use, please contact the authors.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Acoustic Rake Receiver</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://lcav.epfl.ch" property="cc:attributionName" rel="cc:attributionURL">Ivan Dokmanić, Robin Scheibler, Martin Vetterli</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/LCAV/AcousticRakeReceiver" rel="dct:source">https://github.com/LCAV/AcousticRakeReceiver</a>.

