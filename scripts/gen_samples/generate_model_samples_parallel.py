"""Generates LOS velocity samples using Sweeney and Rozo 2026 Model, assuming all galaxies in a bin of R+M have that exact R and M. 
    TODO: update to sample from halos' individual R and M, once we get the data for that (waiting on mass data).
"""
#Imports _______________________________________________
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
sys.path.append('/home/cosweeney/code/Fits/')

from scipy.stats import t, johnsonsu
from scipy.optimize import curve_fit
import pickle

from vel_bias_utils import *
from clusters_jove.jove.cat import *
from clusters_jove.jove.joint import *
from clusters_jove.jove.hgcf import *
from clusters_jove.jove.losvd import *
from clusters_jove.jove.myconfig import *

Rvirs, Mvirs, mbins = compute_bin_virial_R_M('/spiff/cosweeney/Tables/halo_pair_cat_final2_HD.hdf5')


from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck13')
a = 0.83760
redshift=1/a-1
h = cosmo.h 
a = 0.83760
redshift=1/a-1
H = h*100*cosmo.Ez(redshift)

import warnings
warnings.filterwarnings("ignore") 

sys.path.append('/home/cosweeney/code/Fits/')

import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})

from tqdm import tqdm

import psutil
psutil.Process().nice(10)

# radial bins
r_min = 4
r_max = 15
r_edges = np.linspace(r_min, r_max, (r_max-r_min)+1)
r_cens = (r_edges[1:]+r_edges[:-1])/2.0

#%%
# bin data 
vel_path = '/spiff/cosweeney/simulations/MDPL2/data/r_R_rlos_vpeclos_vphyslos_data_mass_100_cyl_6K_indmass.hdf5'

vpeclos_ls_exs  = []
vphyslos_ls_exs = []
rlos_ls_exs = []
R_ls_exs = []
M_ls_exs = []

#for k, M in enumerate(Mvirs):
for k, M in enumerate(Mvirs):
    for i, R in enumerate(r_cens):
        with h5.File(vel_path, 'r') as hdf:
            rel_R = hdf[f'R/{str(k)}'][()]
            rel_rlos = hdf[f'rlos/{str(k)}'][()]
            vpec = hdf[f'vpeclos/{str(k)}'][()]
            vphys = hdf[f'vphyslos/{str(k)}'][()]
            Mvir = hdf[f'Mvir/{str(k)}'][()]  

        mask = (rel_R >= r_edges[i]) & (rel_R <= r_edges[i+1])

        vpeclos_ls_exs.append(vpec[mask])
        vphyslos_ls_exs.append(vphys[mask])
        rlos_ls_exs.append(rel_rlos[mask])
        R_ls_exs.append(rel_R[mask])
        M_ls_exs.append(Mvir[mask])

#%%
#define model fcn

def joint_dist_samples(N, r, M, pars):
    """Optimized version with vectorization and pre-computation.
    
    Args:
        N: Number of samples to generate (should equal len(r) for vectorized case)
        r: Radial distance(s) - scalar or array
        M: Mass
        pars: Model parameters
    """
    
    vrpp, vrps, vrsp, vrss, srpp, srps, srsp, srss, delta_m, delta_b, \
        A, Bp, Bs, mu0p, mu0c, mu1p, mu1c, C1p, C1c = pars

    # Vectorize if r is an array
    r = np.atleast_1d(r)
    n_r = r.shape[0]
    
    # For proper vectorization, N should equal n_r
    if N != n_r:
        # Fall back to generating N samples for a single r value
        if n_r == 1:
            r = np.repeat(r, N)
            n_r = N
        else:
            raise ValueError(f"N ({N}) must equal len(r) ({n_r}) for vectorized operation")
    
    # Mass dependence (vectorized)
    M_pivot = 1e14
    m = M / M_pivot
    
    vr_p = -plaw(m, vrpp, vrps)
    vr_s = -plaw(m, vrsp, vrss)
    sr_p = plaw(m, srpp, srps)
    sr_s = -plaw(m, srsp, srss)
    
    r_pivot = 10
    sr_0 = 213027
    
    # Vectorized radial dependence
    r_ratio = r / r_pivot
    peak = plaw(r_ratio, vr_p, vr_s)
    variance = plaw(r_ratio, sr_p, sr_s) + sr_0
    
    # Vectorized interpolation - handle output correctly
    stacked_input = np.column_stack([peak, variance])
    xi_output = xi_RBF(stacked_input)
    lam_output = lam_RBF(stacked_input)
    
    # Handle different possible output formats from RBF
    if isinstance(xi_output, tuple):
        xi = xi_output[0]
    else:
        xi = xi_output
    
    if isinstance(lam_output, tuple):
        lam = lam_output[0]
    else:
        lam = lam_output
    
    # Ensure 1D arrays
    xi = np.atleast_1d(xi).ravel()
    lam = np.atleast_1d(lam).ravel()
    
    # Vectorized JSU parameters
    gam = 0.00111 * xi + 0.71
    delt = 0.0022 * lam + line(m, delta_m, delta_b)
    
    # Generate one radial sample per r value
    rad_samples = np.array([
        johnsonsu.rvs(a=gam[i], b=delt[i], loc=xi[i], scale=lam[i], size=1)[0]
        for i in range(n_r)
    ])
    
    mask = np.abs(rad_samples) < 3000
    
    # Pre-compute constants
    dof = 5
    r_C = 48
    C0 = 105_000
    
    # Vectorized mass dependence
    B = plaw(m, Bp, Bs)
    mu0 = plaw(m, mu0p, 1) + mu0c
    mu1 = plaw(m, mu1p, 1) + mu1c
    C1 = plaw(m, C1p, 1) + C1c
    
    # Vectorized mu and C calculation
    mu = mu0 / r + mu1  # Shape: (n_r,)
    C = C1 * (r - r_C) + C0  # Shape: (n_r,)
    
    # Vectorized variance calculation - NO LOOPS!
    threshold = mu - 1000  # Shape: (n_r,)
    
    # Element-wise comparison
    above_threshold = rad_samples > threshold  # Shape: (n_r,)
    
    # Calculate variance for all samples at once
    diff = rad_samples - mu  # Shape: (n_r,)
    variance_above = A * diff**3 + B * diff**2 + C
    
    # Pre-compute slope and intercept for linear extrapolation (vectorized)
    vr0 = mu - 1000
    vr2 = mu - 998  # Small offset for numerical stability
    
    sig0 = A * (vr0 - mu)**3 + B * (vr0 - mu)**2 + C
    sig2 = A * (vr2 - mu)**3 + B * (vr2 - mu)**2 + C
    
    slope = (sig0 - sig2) / (vr0 - vr2)  # Shape: (n_r,)
    intercept = sig0 - slope * vr0  # Shape: (n_r,)
    
    # Linear extrapolation for below threshold
    variance_below = slope * rad_samples + intercept
    
    # Combine using boolean indexing
    variance = np.where(above_threshold, variance_above, variance_below)
    
    # Ensure non-negative variance
    variance = np.maximum(variance, 1e-6)
    scale = np.sqrt(variance)
    
    # Generate tangential samples (one per r value)
    tan_samples = t.rvs(df=dof, scale=scale, size=n_r)
    
    return rad_samples, tan_samples, mask

def P_vphys_R_samples(ind, m_ind, R_cens, rlos_vals):
    index = ind + m_ind*len(R_cens)

    samp_rlos = rlos_vals[index]

    r_distances = np.sqrt(R_cens[ind]**2 + samp_rlos**2)

    rad_samps, tan_samps, _ = joint_dist_samples(
        N=len(r_distances), 
        r=r_distances, 
        M=Mvirs[m_ind], 
        pars=joint_MAP
    )

    # Stack 
    samp_vr_vt = np.stack([rad_samps, tan_samps], axis=1)[:, :, np.newaxis]

    h = cosmo.h 
    a = 0.83760
    redshift=1/a-1
    H = h*100*cosmo.Ez(redshift)
    theta = np.arctan2(samp_rlos, R_cens[ind]) 

    cos = np.cos(theta)
    sin = np.sin(theta)

    vpeclos_samp =  samp_vr_vt[:, 0, 0]*sin + samp_vr_vt[:, 1, 0]*cos 

    vphyslos_samp = vpeclos_samp + a*H*samp_rlos / h 

    return vphyslos_samp

# generate samples in each bin 

from multiprocessing import Pool
from functools import partial
import os

samp_path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_models/'


def compute_single_combination(args, R_cens, rlos_vals, Mvirs, joint_MAP, cosmo):
    """
    Compute LOS velocity samples for a single (M, R) combination.
    
    Args:
        args: Tuple of (k, i) where k is mass bin index, i is R bin index
        R_cens: Array of R center values
        rlos_vals: Array of rlos values
        Mvirs: Array of mass values
        joint_MAP: Model parameters
        cosmo: Cosmology object
    
    Returns:
        Tuple of (k, i, vphyslos_samples)
    """
    k, i = args
    
    # Use the existing P_vphys_R_samples function
    vphyslos_samp = P_vphys_R_samples(i, k, R_cens, rlos_vals)
    
    return (k, i, vphyslos_samp)


def compute_all_los_velocities_parallel(R_cens, rlos_vals, Mvirs, joint_MAP, cosmo, 
                                       output_file= samp_path+'gal_model_bin_mass_samples_r_4_15.hdf5', n_processes=None):
    """
    Compute LOS velocity samples for all (M, R) combinations using multiprocessing.
    
    Args:
        R_cens: Array of R center values
        rlos_vals: Array of rlos values
        Mvirs: Array of mass values 
        joint_MAP: Model parameters
        cosmo: Cosmology object
        output_file: Path to output HDF5 file
        n_processes: Number of processes to use (None = use all available cores)
    """
    n_M = len(Mvirs)
    n_R = len(R_cens)
    
    # Create all (k, i) combinations
    combinations = [(k, i) for k in range(n_M) for i in range(n_R)]
    
    print(f"Computing LOS velocities for {len(combinations)} combinations...")
    print(f"Mass bins: {n_M}, R bins: {n_R}")
    
    # Create partial function with fixed parameters
    compute_func = partial(
        compute_single_combination,
        R_cens=R_cens,
        rlos_vals=rlos_vals,
        Mvirs=Mvirs,
        joint_MAP=joint_MAP,
        cosmo=cosmo
    )
    
    # Use multiprocessing to compute all combinations
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(compute_func, combinations),
            total=len(combinations),
            desc="Computing samples"
        ))
    
    # Save results to HDF5 file
    print(f"Saving results to {output_file}...")
    with h5.File(output_file, 'w') as f:
        # Create vphyslos group
        vphyslos_group = f.create_group('vphyslos')
        
        for k, i, vphyslos_samp in results:
            # Create mass bin group if it doesn't exist
            if str(k) not in vphyslos_group:
                mass_group = vphyslos_group.create_group(str(k))
            else:
                mass_group = vphyslos_group[str(k)]
            
            # Save samples under the specified key
            mass_group.create_dataset(str(i), data=vphyslos_samp)
        
        # Optionally save metadata
        f.attrs['n_M'] = n_M
        f.attrs['n_R'] = n_R
        f.attrs['M_values'] = Mvirs
        f.attrs['R_values'] = R_cens
    
    print(f"Successfully saved {len(results)} sample sets to {output_file}")
    return output_file


if __name__ == '__main__':
    
    # Run parallel computation
    results = compute_all_los_velocities_parallel(
        Mvirs=Mvirs,
        R_cens=r_cens,
        rlos_vals=rlos_ls_exs,
        joint_MAP=joint_MAP,
        cosmo=cosmo,
        n_processes=None  
    )



# def process_bin(k, M, i, R, r_cens, vphyslos_ls_exs, R_ls_exs, rlos_ls_exs, gal_MAP, a, H, h):
#     """Process a single (k, i) combination"""
#     index = i + k * len(r_cens)
#     bin_vlos = vphyslos_ls_exs[index]
#     bin_R = R_ls_exs[index]
#     bin_rlos = rlos_ls_exs[index]
    
#     # Create samples
#     samp_vr_vt = np.array([
#         joint_dist_samples(1, r=np.sqrt(R**2 + bin_rlos[j]**2), M=M, pars=gal_MAP) 
#         for j in range(len(bin_vlos))
#     ])
    
#     # Calculate velocities
#     theta_bin = np.arctan(bin_R / bin_rlos)
#     cos_bin = np.cos(theta_bin)
#     sin_bin = np.sin(theta_bin)
#     vpeclos_bin = samp_vr_vt[:, 0, 0] * sin_bin + samp_vr_vt[:, 1, 0] * cos_bin
#     vphyslos_bin = vpeclos_bin + a * H * bin_rlos / h
    
#     return k, i, bin_R, bin_rlos, vpeclos_bin, vphyslos_bin

# from multiprocessing import Pool
# from functools import partial

# def process_bin_wrapper(args):
#     """Wrapper to unpack arguments"""
#     return process_bin(*args)

# # Prepare arguments
# args_list = [
#     (k, M, i+16, R, r_cens, vphyslos_ls_exs, R_ls_exs, rlos_ls_exs, gal_MAP, a, H, h)
#     for k, M in enumerate([Mvirs[0]]) # now, run only for lowest M, remaining R bins
#     for i, R in enumerate(r_cens[16:])
# ]

# # Parallel execution
# with Pool() as pool:
#     results = list(tqdm(
#         pool.imap(process_bin_wrapper, args_list),
#         total=len(args_list),
#         desc="Processing bins"
#     ))

# # Save results
# with h5.File(samp_path+'gal_model_bin_mass_samples_r_4_30.hdf5', 'a') as hdf:
#     for k, i, bin_R, bin_rlos, vpeclos_bin, vphyslos_bin in results:
#         hdf.create_dataset(name=f'R/{str(k)}/{str(i)}', data=bin_R, dtype=np.float64)
#         hdf.create_dataset(name=f'rlos/{str(k)}/{str(i)}', data=bin_rlos, dtype=np.float64)
#         hdf.create_dataset(name=f'vpeclos/{str(k)}/{str(i)}', data=vpeclos_bin, dtype=np.float64)
#         hdf.create_dataset(name=f'vphyslos/{str(k)}/{str(i)}', data=vphyslos_bin, dtype=np.float64) 

# # for k, M in tqdm(enumerate(Mvirs)):
# #     for i, R in tqdm(enumerate(r_cens)):
# #         index = i + k*len(r_cens)

# #         bin_vlos = vphyslos_ls_exs[index] 
# #         bin_R = R_ls_exs[index]
# #         bin_rlos = rlos_ls_exs[index] 
# #         #bin_Mvir = M_ls_exs[index]

# #         samp_vr_vt = np.array([joint_dist_samples(1, r=np.sqrt(R**2 + bin_rlos[i]**2), M=M, pars=gal_MAP) for i in range(len(bin_vlos))]) 

# #         theta_bin = np.arctan(bin_R/bin_rlos) #np.arctan2(test_R, test!_rlos)

# #         cos_bin = np.cos(theta_bin)
# #         sin_bin = np.sin(theta_bin)

# #         vpeclos_bin = ( samp_vr_vt[:, 0, 0]*sin_bin + samp_vr_vt[:, 1, 0]*cos_bin )

# #         vphyslos_bin = vpeclos_bin + a*H*bin_rlos / h

# #         # save data
# #         with h5.File(samp_path+'gal_model_bin_mass_samples_r_4_30.hdf5', 'a') as hdf:
# #             hdf.create_dataset(name=f'R/{str(k)}/{str(i)}', data=bin_R, dtype=np.float64)
# #             hdf.create_dataset(name=f'rlos/{str(k)}/{str(i)}', data=bin_rlos, dtype=np.float64)
# #             hdf.create_dataset(name=f'vpeclos/{str(k)}/{str(i)}', data=vpeclos_bin, dtype=np.float64)
# #             hdf.create_dataset(name=f'vphyslos/{str(k)}/{str(i)}', data=vphyslos_bin, dtype=np.float64) 
#             #hdf.create_dataset(name=f'Mvir/{str(k)}/{str(i)}', data=bin_Mvir, dtype=np.float64) 
