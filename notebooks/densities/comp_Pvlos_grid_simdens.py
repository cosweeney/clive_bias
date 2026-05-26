#%%
import numpy as np
import sys
sys.path.append('/home/cosweeney/code/Fits/')

sys.path.append('/home/cosweeney/code/Fits/Jocond')

import h5py as h5
from scipy.interpolate import RegularGridInterpolator

from JoCond.jocond.losvd import *
from JoCond.jocond.cat import *

from tqdm import tqdm
import psutil
psutil.Process().nice(9)

#%%
vel_path = '/spiff/cosweeney/simulations/MDPL2/data/r_R_rlos_vpeclos_vphyslos_data_mass_100_cyl_6K_indmass.hdf5'

Rvirs, Mvirs, mbins = compute_bin_virial_R_M('/spiff/cosweeney/Tables/halo_pair_cat_final2_HD.hdf5')

# Compute P_vlos on grid of M 
M_edges = np.logspace(np.log10(0.80), np.log10(11), 50) # grid of Mvals
Mvals = np.sqrt(M_edges[1:] * M_edges[:-1])
M_p = 1e14  # h^-1 M_sun

# define vel and radial bins
bin_num = 50
vedges = np.linspace(-2500, 2500, bin_num+1) 
vcens = 0.5*(vedges[1:]+vedges[:-1])

r_edges = np.linspace(5, 15, (15-5)+1)
r_cens = (r_edges[1:]+r_edges[:-1])/2.0

# compute grid of densities 

n_gal = 0.014 # (Mpc/h)^-3
rmax = 40
rlos_edges = np.linspace(-rmax, rmax, int(rmax)+1)
rlos_cens = (rlos_edges[1:]+rlos_edges[:-1])/2.0

# N_R_rlos = np.zeros((len(Mvals), len(r_cens), len(rlos_cens)))

# R_widths = r_edges[1:] - r_edges[:-1]
# rlos_widths = rlos_edges[1:] - rlos_edges[:-1]
# R_grid, rlos_grid = np.meshgrid(r_cens, rlos_cens, indexing='ij')
# bin_vols = 2 * np.pi * R_grid * R_widths[:, None] * rlos_widths[None, :]

# with h5.File(vel_path, 'r') as hdf:
#     for k, M in tqdm(enumerate(Mvirs)):
#         rel_R = hdf[f'R/{str(k)}'][()]
#         rel_rlos = hdf[f'rlos/{str(k)}'][()]
#         bin_Mvir = hdf[f'Mvir/{str(k)}'][()]

#         for j, m in enumerate(Mvals):
#             mask = (bin_Mvir >= M_edges[j]*M_p) & (bin_Mvir < M_edges[j+1]*M_p)

#             if np.sum(mask) > 0:
#                 N_R_rlos[j], _, _ = np.histogram2d(rel_R[mask], rel_rlos[mask], bins=[r_edges, rlos_edges], density=False)

# compute # of halos in each mass bin
halo_path = '/spiff/cosweeney/simulations/MDPL2/hlists/hlist_0.83760_update.hdf5'

N_halos = np.zeros(len(Mvirs))
with h5.File(halo_path, 'r') as hcat:
    for k in range(len(mbins)-1):
        mcut = (hcat['mvir'] > mbins[k]) & (hcat['mvir'] <= mbins[k+1])

        N_bin = mcut.sum()

        N_halos[k] = N_bin

r_fine_edges = np.linspace(5, 50, 100)
r_fine_cens = 0.5*(r_fine_edges[1:]+r_fine_edges[:-1])

rho_RM = np.zeros((6, len(r_fine_cens)))

with h5.File(vel_path, 'r') as hdf:

    for k in range(len(Mvirs)):
        
        rs = hdf[f'r/{str(k)}'][()]

        for j in range(len(r_fine_cens)):
            
            r_mask = (rs >= r_fine_edges[j])&(rs <= r_fine_edges[j+1])

            N_pairs = r_mask.sum()

            V_shell = 4/3 *  np.pi * ( r_fine_edges[j+1]**3 - r_fine_edges[j]**3 ) 

            rho = N_pairs / N_halos[k] / V_shell 

            rho_RM[k, j] = rho

rho_interp = RegularGridInterpolator((Mvirs, r_fine_cens), rho_RM, method='cubic', bounds_error=False, fill_value=None)

#%%

def P_vphys_R_simdens(vphys, R, M, pars=jocond.myconfig.joint_MAP, H=74.719, a=0.83760, h=0.6777):

    #R = r_cens[i]
    #M = Mvals[k]*1e14 

    # rlos integration range
    rlos_max = 40 
    rlos_edges = np.linspace(-rlos_max, rlos_max, int(rlos_max + 1))
    rlos_cens = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])

    rlos_vals = rlos_cens

    vpec = vphys[:, None] - a * H * rlos_vals[None, :] / h

    vr_e = np.linspace(-4000, 4000, 100)
    vr_grid = 0.5 * (vr_e[1:] + vr_e[:-1])

    Pvlos_grid = np.array(
        [P_vlos_R_rlos(vr_grid, R, rlos, M, pars) for rlos in rlos_vals]
    )

    Pvlos_interp_list = []

    for j in range(len(rlos_vals)):
        interp_func = interp1d(
            vr_grid, Pvlos_grid[j], kind="linear", bounds_error=False, fill_value=0.0
        )
        interpolated = interp_func(vpec[:, j])

        Pvlos_interp_list.append(interpolated)

    Pvlos_interp = np.array(Pvlos_interp_list)
    Pvlos_interp = Pvlos_interp.T

    r3d = np.sqrt(R**2 + rlos_vals**2)
    #k = np.where(M/1e14 == Mvals)[0][0]
    #i = np.where(r_cens == R)[0][0]
    weight = np.array([rho_interp((M, ri)) for ri in r3d]) #N_R_rlos[k, i] / bin_vols[i] #rho_g * (cf_inf(r3d, M, jocond.myconfig.cf_pars) + 1)

    integrand = Pvlos_interp * weight[None, :]
    P_out = simpson(integrand, x=rlos_vals, axis=1)

    sigma_R = Sigma_inf(R, M, jocond.myconfig.cf_pars)
    P_out /= sigma_R

    return P_out / np.sum(P_out * np.diff(vphys)[0])

dists = np.zeros((len(vcens), len(r_cens), len(Mvals) ))

for j in tqdm(range( len(Mvals) )): 
    for i in range( len( r_cens ) ):
        dist = P_vphys_R_simdens(vcens, r_cens[i], Mvals[j]*M_p) 
        
        dists[:, i, j] = dist

# save results
path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_simdens_M_grid_R_5_15'

np.save(path+'_gal', dists) 
