
# imports
import h5py as h5
import numpy as np
from corrfunc.tpcf import *
from corrfunc.bins import *

import psutil
psutil.Process().nice(19)


# define radial bins (subject to change. For now, choosing 20 bins per decade.)
r_cens, r_edges = generate_bins(0.1, 150, 70+1) 

# path to MDPL2 halos, galaxies, data
halo_path = '/spiff/cosweeney/simulations/MDPL2/hlists/hlist_0.83760_update.hdf5'
hsb_path = '/spiffball/cosweeney/simulations/MDPL2/halos/SBp/'

gal_path = '/spiff/cosweeney/simulations/MDPL2/galaxies/UM_full_gal_table.hdf5'
save_path = '/spiffball/cosweeney/simulations/MDPL2/data/CorrFuncs/xi_hg_150.hdf5'

# sim quantities
L_box = 1_000 # h^-1 Mpc
g_box = L_box // 4 # grid size

with h5.File(halo_path, 'r') as hdf:
    hm_cut = hdf['mvir'][()] >= 8e13 # halo mass, h^-1 M_sun

    hpos = np.array([
        hdf['x'][()][hm_cut],
        hdf['y'][()][hm_cut], 
        hdf['z'][()][hm_cut],
    ], dtype=np.float64).T 

    with h5.File(gal_path, 'r') as gdf:

        sm_cut = gdf['sm'][()] > 1e10 # stellar mass, h^-1 M_sun

        gal_pos = np.array([
            gdf['pos'][:, 0][()][sm_cut],
            gdf['pos'][:, 1][()][sm_cut], 
            gdf['pos'][:, 2][()][sm_cut],
        ], dtype=np.float64).T 

        #print(hpos.dtype, gal_pos.dtype)


        print('Computing hg correlation function...')
        xi, xi_samples, xi_mean, xi_cov = cross_tpcf_jk(hpos, gal_pos, r_edges, boxsize=L_box, gridsize=g_box, nthreads=84)

        print('Saving results to: '+save_path)
        with h5.File(save_path, 'a') as f:
            f.create_dataset(name=f'xi', data=xi, dtype=np.float64)
            f.create_dataset(name=f'xi_samples', data=xi_samples, dtype=np.float64)
            f.create_dataset(name=f'xi_mean', data=xi_mean, dtype=np.float64)
            f.create_dataset(name=f'xi_cov', data=xi_cov, dtype=np.float64)
            f.create_dataset(name=f'r_cens', data=r_cens, dtype=np.float64)
            f.create_dataset(name=f'r_edges', data=r_edges, dtype=np.float64)

print('Done.')