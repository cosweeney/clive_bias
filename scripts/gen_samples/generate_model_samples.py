"""Generates LOS velocity samples using Sweeney and Rozo 2025 Model, assuming all galaxies in a bin of R+M have that exact R and M. 
    TODO: update to sample from halos' individual R and M, once we get the data for that (waiting on mass data).
"""
#Imports _______________________________________________
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
sys.path.append('/home/cosweeney/code/Fits/')

from fit_utils import bin_velocities_abs
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad_vec
from scipy.stats import t, johnsonsu
from scipy.optimize import curve_fit
import pickle
import scipy

from vel_bias_utils import *

from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck13')
a = 0.83760
redshift=1/a-1
h = cosmo.h 
a = 0.83760
redshift=1/a-1
H = h*100*cosmo.Ez(redshift)
print(cosmo.h, cosmo.Hz(redshift), cosmo.H0*cosmo.Ez(redshift))

import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/cosweeney/code/Fits/')

import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})

from tqdm import tqdm

import psutil
psutil.Process().nice(9)

with h5.File('/spiff/cosweeney/Tables/halo_pair_cat_final2_HD.hdf5', 'r') as pc:
    print(pc.keys(), end=' ')

    tag = np.array(pc['tag'])
    orbit_mask = (tag == 0)
    infall_mask = (tag == 1) 

    hostmvir = np.array(pc['hostmvir'])
    hostrvir = np.array(pc['hostrvir'])

    #Massbins
    print('Minimum M:', hostmvir.min()/1e14, '$10^{14}M_\odot$')
    print('Maximum M:', hostmvir.max()/1e14, '$10^{14}M_\odot$')

    mini=hostmvir.min()
    maxi=1.6e15#pc['hostmvir'].max()

    #Choose Bins: Log-spaced bins between min and max
    massbins = np.logspace( np.log10(mini), np.log10(maxi), 7+1)
    mbins = np.delete(massbins, -2)
    print('massbins:', mbins)

    Rvirs=[]
    Mvirs=[]
    for i in range(len(mbins)-1):
        mmask = (hostmvir>=mbins[i])&(hostmvir<=mbins[i+1])
        orbi_M = hostmvir[mmask&orbit_mask]
        orbi_R = hostrvir[mmask&orbit_mask]
        print(mbins[i]/1e14, mbins[i+1]/1e14)

        Rvirs.append(np.median(orbi_R))
        Mvirs.append(np.median(orbi_M))
    Rvirs=np.array(Rvirs)
    Mvirs=np.array(Mvirs)

# radial bins
r_edges = np.linspace(4, 30, 26+1)
r_cens = (r_edges[1:]+r_edges[:-1])/2.0

zel_xhg = np.fromfile('/spiff/cosweeney/simulations/MDPL2/data/CorrFuncs/zel_xhg.bin')
xhg = h5.File("/spiff/cosweeney/simulations/MDPL2/data/CorrFuncs/xihg_fit_fnl_2.h5")
rxhg = np.array(xhg['rbins'])
exhg = np.array(xhg['r_edges']) 
xihg = np.array(xhg['xi'])

path = "/spiff/cosweeney/simulations/MDPL2/data/models"

with open(path+'RBFint_xi.pkl', 'rb') as f:
    xi_RBF = pickle.load(f)

with open(path+'RBFint_lam.pkl', 'rb') as f: 
    lam_RBF = pickle.load(f) 
#%%
# bin data 
vel_path = '/spiff/cosweeney/simulations/MDPL2/data/r_R_rlos_vpeclos_vphyslos_data_mass_80_cyl_6K.hdf5'

vpeclos_ls_exs  = []
vphyslos_ls_exs = []
rlos_ls_exs = []
R_ls_exs = []
#M_ls_exs = []

#for k, M in enumerate(Mvirs):
for k, M in enumerate(Mvirs):
    for i, R in enumerate(r_cens):
        with h5.File(vel_path, 'r') as hdf:
            rel_R = hdf[f'R/{str(k)}'][()]
            rel_rlos = hdf[f'rlos/{str(k)}'][()]
            vpec = hdf[f'vpeclos/{str(k)}'][()]
            vphys = hdf[f'vphyslos/{str(k)}'][()]
            #Mvir = hdf[f'Mvir/{str(k)}'][()]  #TODO: add when we have this.

        mask = (rel_R >= r_edges[i]) & (rel_R <= r_edges[i+1])

        vpeclos_ls_exs.append(vpec[mask])
        vphyslos_ls_exs.append(vphys[mask])
        rlos_ls_exs.append(rel_rlos[mask])
        R_ls_exs.append(rel_R[mask])
        #M_ls_exs.append(Mvir[mask])

#%%
#define model fcn

def joint_dist_samples(N, r, M, pars):
    """Samples of the smooth model for 
    the joint distribution
    P(v_r, v_t | r, M). 

    This is the product of the marginal 
    radial velocity distribution and the 
    conditional tangential velocity 
    distribution:
    
    P(v_r, v_t | r, M) = P(v_r | r, M)P(v_t | v_r, r, M)  

    Args:
        v_r (np.ndarray): radial velocity bin centers. 
        v_t (np.ndarray): tangential velocity bin centers.
        r (float): radial distance bin center.
        pars (np.ndarray): parameters defining radial dependence.
    """

    vrpp, vrps, vrsp, vrss, srpp, srps, srsp, srss, delta_m, delta_b, \
        s_c, s_qm, s_qb, s_mmm, s_mmb, s_mbm, s_mbb, s_0mm, s_0mb = pars

    # Mass dependence
    M_pivot = 1e14 # M_sun
    vr_p = -plaw( M / M_pivot, vrpp, vrps)
    vr_s = -plaw( M / M_pivot, vrsp, vrss)

    sr_p = plaw( M / M_pivot, srpp, srps)
    sr_s = -plaw( M / M_pivot, srsp, srss)

    r_pivot = 10 # radial pivot, 10 h^-1 Mpc

    sr_0 = 213027 # variance limit as r -> 00, km^2 s^-2

    # Radial dependence
    peak = plaw(r / r_pivot , vr_p, vr_s)
    variance = plaw(r / r_pivot, sr_p, sr_s) + sr_0  

    xi = xi_RBF(np.column_stack([peak, variance]))[0] # interpolators
    lam = lam_RBF(np.column_stack([peak, variance]))[0]

    gam = 0.00078 * xi + 0.58599 # JSU skew, kurtosis 
    delt = 0.00209 * lam + 0.28703 + line(M / M_pivot, delta_m, delta_b)

    rad_samples = johnsonsu.rvs(a=gam, b=delt, loc=xi, scale=lam, size=N)

    mask1 = rad_samples > -2_000
    mask2 = ~mask1 #(np.abs(rad_samples) < 3000)

    dof = 5 # degrees of freedom, "shape" parameter
    s_0p = 48 # h^-1 Mpc, transition of sigma_vt^2 minimum to a constant
    s_0c = 105_000 # km^2 s^-2, minimum value as r -> 00 
    
    # Mass dependence 
    s_q = plaw( M / M_pivot, s_qm, s_qb)
    s_mm = line( M / M_pivot, s_mmm, s_mmb)
    s_mb = line( M / M_pivot, s_mbm, s_mbb)
    s_0m = line( M / M_pivot, s_0mm, s_0mb)


    s_m = s_mm / r + s_mb
    s_0 = s_0m * np.log( 1 + np.exp( - (r - s_0p))) + s_0c 

    variance = np.zeros_like(rad_samples)
    variance[mask1] = s_c * ( rad_samples[mask1] - s_m ) ** 3 + s_q * ( rad_samples[mask1] - s_m ) ** 2 + s_0

    test_vr = np.linspace(-2000, -1000, 100)
    p, _ = curve_fit(line, test_vr, s_c * ( test_vr - s_m ) ** 3 + s_q * ( test_vr - s_m ) ** 2 + s_0)
    variance[mask2] = line(rad_samples[mask2], p[0], p[1])

    scale = np.sqrt( variance )


    tan_samples = t.rvs(df=dof, scale=scale, size=N) 


    return rad_samples, tan_samples, mask1 

# generate samples in each bin 
samp_path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_models/'

for k, M in tqdm(enumerate(Mvirs)):
    for i, R in tqdm(enumerate(r_cens)):
        index = i + k*len(r_cens)

        bin_vlos = vphyslos_ls_exs[index] 
        bin_R = R_ls_exs[index]
        bin_rlos = rlos_ls_exs[index] 
        #bin_Mvir = M_ls_exs[index]

        samp_vr_vt = np.array([joint_dist_samples(1, r=np.sqrt(R**2 + bin_rlos[i]**2), M=M, pars=gal_MAP) for i in range(len(bin_vlos))]) 

        theta_bin = np.arctan(bin_R/bin_rlos) #np.arctan2(test_R, test_rlos)

        cos_bin = np.cos(theta_bin)
        sin_bin = np.sin(theta_bin)

        vpeclos_bin = ( samp_vr_vt[:, 0, 0]*sin_bin + samp_vr_vt[:, 1, 0]*cos_bin )

        vphyslos_bin = vpeclos_bin + a*H*bin_rlos / h

        # save data
        with h5.File(samp_path+'gal_model_bin_mass_samples_r_4_30.hdf5', 'a') as hdf:
            hdf.create_dataset(name=f'R/{str(k)}/{str(i)}', data=bin_R, dtype=np.float64)
            hdf.create_dataset(name=f'rlos/{str(k)}/{str(i)}', data=bin_rlos, dtype=np.float64)
            hdf.create_dataset(name=f'vpeclos/{str(k)}/{str(i)}', data=vpeclos_bin, dtype=np.float64)
            hdf.create_dataset(name=f'vphyslos/{str(k)}/{str(i)}', data=vphyslos_bin, dtype=np.float64) 
            #hdf.create_dataset(name=f'Mvir/{str(k)}/{str(i)}', data=bin_Mvir, dtype=np.float64) 
