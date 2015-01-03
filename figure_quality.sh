#!/bin/bash

# This script will dispatch the perceptual quality evaluation
# to multiple process to use most of the computer resource available.

LOOPS=1000

# simulate for 1 source to 21 sources
for i in {1..11}
do
  echo python figure_quality_sim.py ${i} ${LOOPS}
  screen -d -m python figure_quality_sim.py ${i} ${LOOPS}
done
