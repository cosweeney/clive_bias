""" Plotting functions for examining Mass bias from fits. 
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.integrate import simpson
from scipy.stats import t
from tqdm import tqdm 
sys.path.append('/home/cosweeney/code/Fits/')

from JoCond.jocond.hgcf import *
from JoCond.jocond.myconfig import*

from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck13')
a = 0.83760
redshift=1/a-1
H = cosmo.Hz(redshift)
h = cosmo.h