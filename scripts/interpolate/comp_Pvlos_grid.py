#%%
import numpy as np
import sys
sys.path.append('/home/cosweeney/code/Fits/')

sys.path.append('/home/cosweeney/code/Fits/Jocond')

from JoCond.jocond.losvd import *

from tqdm import tqdm
import psutil
psutil.Process().nice(9)

#%%

# Compute P_vlos on grid of M 
Mvals = np.logspace(np.log10(0.5), np.log10(11), 100) # grid of Mvals (200 pts for latest, up-to-date Jocond)

# define vel and radial bins
bin_num = 50
vedges = np.linspace(-2500, 2500, bin_num+1) 
vcens = 0.5*(vedges[1:]+vedges[:-1])

r_edges = np.linspace(5, 15, (15-5)+1)
r_cens = (r_edges[1:]+r_edges[:-1])/2.0

M_p = 1e14  # h^-1 M_sun

#%%

dists = np.zeros((len(vcens), len(r_cens), len(Mvals) ))

for j in tqdm(range( len(Mvals) )): 
    for i in range( len( r_cens ) ):
        dist = P_vphys_R_OG(vcens, r_cens[i], Mvals[j]*M_p) 
        
        dists[:, i, j] = dist

# save results
# path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_M_grid_R_5_15'
path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_M_grid_OG_R_5_15'

np.save(path+'_gal', dists) 
