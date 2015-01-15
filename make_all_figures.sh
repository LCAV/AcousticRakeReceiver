#!/bin/bash

# Create all figures and sound samples

ipython figure_spectrograms.py

ipython figure_beam_scenarios.py

ipython figure_Measures1.py

ipython figure_Measures2.py

ipython figure_SumNorm.py

# Here one can launch a cluster of ipython
# workers and remove the '-s' option for a larg
# speed gain.
ipython figure_quality_sim.py -- -s 10000

ipython figure_quality_plot.py

