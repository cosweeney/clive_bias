#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
sys.path.append('/home/cosweeney/code/Fits/')

from fit_utils import bin_velocities_abs
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import quad_vec
from scipy.stats import t, johnsonsu
from scipy.optimize import curve_fit
import pickle
import scipy

from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck13')
a = 0.83760
redshift=1/a-1
print(cosmo.h, cosmo.Hz(redshift), cosmo.H0*cosmo.Ez(redshift))

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


# define models, BF parameters


# pars
gal_MAP = np.array([ 1.98245745e+02,  5.53864658e-01,  5.72599594e-01,  3.94992883e-02,
        1.59549554e+05, -1.12290582e+00,  1.17453011e+00, -4.15201839e-01,
       -3.47438030e-02,  1.25418141e-01,  4.14120554e-05,  1.75603684e-01,
       -1.05049987e-01, -5.53158259e+02, -8.88081399e+02, -1.94343568e+01,
        4.77159076e+01,  2.77450696e+02, -1.14129239e+03])

part_MAP = np.array([ 1.89274827e+02,  5.83981117e-01,  7.47863822e-01, -1.33316764e-02,
  1.99611683e+05, -9.80959118e-01,  1.45001450e+00, -2.62303590e-01,
 -3.54686051e-02,  1.14008827e-01,  3.20841835e-05,  1.50276564e-01,
 -8.29957439e-02, -6.42615329e+02, -1.41167223e+03, -1.79471080e+01,
  1.19204199e+02,  2.82882901e+02, -1.07836598e+03])

cf_pars = [5.37569671e+00, 4.23001652e-01, 2.52267001e+00, 6.32218826e-02,
           2.89276487e+00, 4.40667100e-01, 3.83029496e-01, 2.98108140e+00,
           1.00202194e+00,]

# models

def plaw(x, p, s):
    """Power law function. 

    Args:
        x (any): power law argument 
        p (any): power law pivot
        s (any): power law slope
    """
    return p * x ** s


def marginal_radial_dist(vr, r, M, pars):
    """The smooth model for the marginal radial 
    distribution P(v_r|r, M). 

    This is a Johnson's SU distribution 
    (https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution)
    which is reparameterized to take the 
    distribution peak and variance as 
    input, using an RBF interpolator. 

    Args:
        vr (np.ndarray): radial velocity bin centers. 
        r (float): radial distance bin center.
        M (float): Mass bin median.
    """

    vrpp, vrps, vrsp, vrss, srpp, srps, srsp, srss, delta_m, delta_b = pars

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

    dist = johnsonsu.pdf(vr, a=gam, b=delt, loc=xi, scale=lam)

    return dist


def line(x,m,b):
    return m*x+b

def conditional_tangential_dist(vt, vr, r, M, pars):
    """The smooth model for the conditional tangential 
    velocity distribution P(v_t|v_r, r, M). 

    This is a Student's t distribution with a fixed 
    degree of freedom (dof) parameter characterizing 
    the kurtosis of the distribution. The remaining 
    dependences are captured in the variance. 

    Args:
        v_t (np.ndarray): tangential velocity bin centers.
        v_r (_type_): radial velocity bin center.
        r (_type_): radial distance bin center.
        M (float): Mass bin median.
        pars (_type_): parameters defining the radial dependence.
    """

    s_c, s_qm, s_qb, s_mmm, s_mmb, s_mbm, s_mbb, s_0mm, s_0mb = pars

    dof = 5 # degrees of freedom, "shape" parameter
    s_0p = 48 # h^-1 Mpc, transition of sigma_vt^2 minimum to a constant
    s_0c = 105_000 # km^2 s^-2, minimum value as r -> 00 
    
    M_pivot = 1e14 # M_sun

    # Mass dependence 
    s_q = plaw( M / M_pivot, s_qm, s_qb)
    s_mm = line( M / M_pivot, s_mmm, s_mmb)
    s_mb = line( M / M_pivot, s_mbm, s_mbb)
    s_0m = line( M / M_pivot, s_0mm, s_0mb)

    s_m = s_mm / r + s_mb
    s_0 = s_0m * np.log( 1 + np.exp( - (r - s_0p))) + s_0c 

    if vr > -2_000:
        variance = s_c * ( vr - s_m ) ** 3 + s_q * ( vr - s_m ) ** 2 + s_0
    else:
        test_vr = np.linspace(-2000, -1000, 100)
        p, _ = curve_fit(line, test_vr, s_c * ( test_vr - s_m ) ** 3 + s_q * ( test_vr - s_m ) ** 2 + s_0)
        variance = line(vr, p[0], p[1]) #s_c * ( vr - s_m ) ** 3 + s_q * ( vr - s_m ) ** 2 + s_0 #s_q * ( vr - s_m ) ** 2 + s_0
        #plt.plot(test_vr, line(test_vr, p[0], p[1])) 
        #plt.plot(test_vr, s_c * ( test_vr - s_m ) ** 3 + s_q * ( test_vr - s_m ) ** 2 + s_0)
    scale = np.sqrt( variance )

    dist = t.pdf(vt, df=dof, scale=scale)

    return dist

def joint_dist(vr, vt, r, M, pars):
    """The smooth model for the joint distribution
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


    marg_rad = marginal_radial_dist(vr, r, M, [vrpp, vrps, vrsp, vrss, srpp, srps, srsp, srss, delta_m, delta_b])

    if vr.size <= 1 or np.isscalar(vr):
        con_tan = conditional_tangential_dist(vt, vr, r, M, [s_c, s_qm, s_qb, s_mmm, s_mmb, s_mbm, s_mbb, s_0mm, s_0mb])

        jd = marg_rad * con_tan.T 

        return jd 

    else:
        con_tan = np.array([conditional_tangential_dist(vt, v, r, M, [s_c, s_qm, s_qb, s_mmm, s_mmb, s_mbm, s_mbb, s_0mm, s_0mb]) for v in vr ])

        jd = marg_rad * con_tan.T 

        norm = np.trapezoid(np.trapezoid(jd, vr, dx=np.diff(vr)[0], axis=-2), vt, dx=np.diff(vt)[0], axis=-1)

        return jd / norm
    
def P_vlos_R_rlos(vlos, R, r_los, M, pars):
    """
    Compute P(v_los | R, r_los) by numerically integrating over the joint distribution.

    Args:
        vlos (np.ndarray): 
            Line-of-sight velocity bin centers.
        R (float): 
            Projected radius.
        r_los (float): 
            Line-of-sight distance.
        M (float): 
            Halo mass.
    
    Returns:
        P_vlos (np.ndarray): 
            Probability density evaluated at each v_los bin center. 
    """
    vr = np.copy(vlos)
    vt = np.copy(vlos) 

    # distance relations
    theta = np.arctan2(R, r_los)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r = np.sqrt(R**2 + r_los**2) 

    JD = joint_dist(vr, vt, r, M, pars)
    JD = np.nan_to_num(JD, nan=0.0) 

    # Interpolation function for fast lookups
    JD_interp = RectBivariateSpline(vr, vt, JD)

    # Create output array
    P_out = np.zeros_like(vlos)

    # Integral: for each vlos, integrate over vr
    for i, v in enumerate(vlos):
        # Compute vt value corresponding to each vr
        vt_eval = (v - vr * sin_theta) / cos_theta
        integrand = JD_interp.ev(vr, vt_eval) / cos_theta
        P_out[i] = np.trapezoid(integrand, vr)

    return P_out / P_out.sum() / np.diff(vlos)[0]


full_r2d = np.logspace(np.log10(min(rxhg)), np.log10(max(rxhg)), 1000)
zel2d = np.fromfile('/spiff/cosweeney/simulations/MDPL2/data/CorrFuncs/zel2d')

r_dict = {value: index for index, value in enumerate(rxhg)}

def find_closest_zel_hg(r):
    r_points = full_r2d
    zel_hg_points = zel2d
    # Ensure r is within the range of r_points
    if r > r_points[-1]:
        return 0.0

    # Find the index of the closest point in r_points
    closest_index = np.argmin(np.abs(r_points - r))

    # Get the corresponding zel_hg value
    closest_zel_hg = zel_hg_points[closest_index]

    return closest_zel_hg

def cf_inf(r, M, params):
    b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu = params

    # constants
    M_vp = 1e14 # h^-1 M_sun
    B = 0.948 # offset
    r_h = (7.35305605e-01)*(M/1e14)**(2.80615822e-01) # From total hg fit

    # mass dependence
    b = b_p*(M/M_vp)**b_s
    g = g_p*(M/M_vp)**g_s
    c = c0*(1 - scipy.special.erf( (np.log10(M/M_vp) - c_m) / c_sig )  ) / 2

    x = r / r_h

    # Interpolate zel_xhg to match r
    zel_interp = np.interp(r, rxhg, zel_xhg)

    if np.isscalar(r):
        zel = find_closest_zel_hg(r)
        corr_inf1 = b*( ( 1 + ( r_inf / ( mu*r_h + r ) )**g ) / ( 1 + c * x * np.exp( -x ) ) )*B*zel 
        return corr_inf1
    else:
        corr_inf = b*( ( 1 + ( r_inf / ( mu*r_h + r ) )**g ) / ( 1 + c * x * np.exp( -x ) ) )*B*zel_interp
        return corr_inf
    
def surf_int_inf(rlos, R, M, h, params):
    b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu = params
    r3d   = np.sqrt(rlos**2 + R**2)

    a = 0.83760
    redshift=1/a-1
    H = h*100*cosmo.Ez(redshift)

    v_max = 4_000
    sigma = 418#532 

    return 0.014*(cf_inf(r3d, M, [b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu]) + 1)*( scipy.special.erf( (v_max - a*H*rlos/h)/ np.sqrt(2) / sigma ) + 1)

def Sigma_inf(R, M, h, params):
    b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu = params

    if np.isscalar(R): #np.sqrt(max(rxhg)**2 - R**2)
        return quad_vec(surf_int_inf, 0, max(rxhg), args=(R, M, h, [ b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu ]), workers=-1)[0]
    else:
        inte = np.array([np.array([quad_vec(surf_int_inf, 0, max(rxhg), args=(R[i], M, h, [ b_p, b_s, g_p, g_s, c0, c_m, c_sig, r_inf, mu ]), workers=-1)[0] ]) for i in range(len(R))]).T
        return inte[0]


def P_vphys_R_vec(v_phys, R, M, pars):
    M = M * 1e14

    h = 0.6777 #cosmo.h
    a = 0.83760
    redshift=1/a-1
    H = h*100*cosmo.Ez(redshift)

    # r_LOS integration range
    rlos_max = 4000/a/H*h #max(rxhg) #100 #np.sqrt(max(rxhg)**2 - R**2)
    rlos_vals = np.linspace(0.0, rlos_max, 100)

    # velocity grid for interpolation
    vr_grid = np.linspace(-4000, 4000, 100) 

    # Shape (n_v, n_r)
    vpec = v_phys[:, None] - a * H * rlos_vals[None, :] / h 
    
    # Compute P(vlos | R, rlos) for all rlos on vr_grid
    Pvlos_grid = np.array([P_vlos_R_rlos(vr_grid, R, rlos, M, pars) for rlos in rlos_vals])  
    # shape (n_r, n_v)

    # Interpolate along vr_grid axis for each rlos
    Pvlos_interp = np.array([
        np.interp(vpec[:, j], vr_grid, Pvlos_grid[j], left=None, right=None)
        for j in range(len(rlos_vals))
    ])  # shape (n_r, n_v)

    # Now shape (n_v, n_r)
    Pvlos_interp = Pvlos_interp.T  

    # Weight factor
    weight = 0.014 * (cf_inf(np.sqrt(R**2 + rlos_vals**2), M, cf_pars) + 1)  
    # shape (n_r,)

    # Multiply and integrate over rlos
    integrand = Pvlos_interp * weight[None, :]  
    P_out = np.trapezoid(integrand, rlos_vals, axis=1)  # shape (n_v,)

    # Normalize by Sigma(R)
    sigma_R = Sigma_inf(R, M, h, cf_pars)
    P_out /= sigma_R

    # Symmetrize
    dist = P_out + np.flip(P_out)

    return dist / dist.sum() / np.diff(v_phys)[0] 

rlos_edges = np.linspace(-50, 50, 1000+1)  #np.max(rxhg)
rlos_cens = 0.5*(rlos_edges[1:]+rlos_edges[:-1]) 

def P_vphys_R_vec(v_phys, R, M, pars): #newest, with updated integration, normalization. 
    M = M * 1e14

    h = cosmo.h 
    a = 0.83760
    redshift=1/a-1
    H = h*100*cosmo.Ez(redshift)

    rlos_vals = rlos_cens

    vpec_data_max = 4000.0

    v_buffer = 500
    vr_e = np.linspace(-vpec_data_max - v_buffer, vpec_data_max + v_buffer, 50+1)
    vr_grid = 0.5*(vr_e[1:]+vr_e[:-1]) 

    vpec = v_phys[:, None] - a * H * rlos_vals[None, :] / h 

    valid_mask = np.zeros((len(v_phys), len(rlos_vals)), dtype=bool)
    for j, rlos in enumerate(rlos_vals):
        v_phys_min_valid = -vpec_data_max + a * H * rlos / h
        v_phys_max_valid = vpec_data_max + a * H * rlos / h
        valid_mask[:, j] = (v_phys >= v_phys_min_valid) & (v_phys <= v_phys_max_valid)

    Pvlos_grid = np.array([P_vlos_R_rlos(vr_grid, R, rlos, M, pars) for rlos in rlos_vals])  


    from scipy.interpolate import interp1d

    Pvlos_interp_list = []

    for j in range(len(rlos_vals)):
        interp_func = interp1d(vr_grid, Pvlos_grid[j], kind='cubic', 
                            bounds_error=False, fill_value=0.0)
        interpolated = interp_func(vpec[:, j])

        Pvlos_interp_list.append(interpolated)

    Pvlos_interp = np.array(Pvlos_interp_list)
    Pvlos_interp = Pvlos_interp.T 

    r3d = np.sqrt(R**2 + rlos_vals**2)
    weight = 0.014 * (cf_inf(r3d, M, cf_pars) + 1)

    integrand = Pvlos_interp * weight[None, :]
    from scipy.integrate import simpson
    P_out = simpson(integrand, x=rlos_vals, axis=1)

    sigma_R = Sigma_inf(R, M, h, cf_pars)
    P_out /= sigma_R

    return P_out
#%%
# #Bin velocities
# vphyslos, R_cens2, R_data1, rlos_data1 = bin_velocities_abs('vlos', 'R', r_edges, massbins=mbins, return_rvals=True)


# # Large scales
# vel_path = '/spiff/cosweeney/simulations/MDPL2/data/r_R_rlos_vpeclos_vphyslos_data_mass_100_cyl_update_p.hdf5'

# vpeclos_ls  = []
# vphyslos_ls = []
# rlos_ls = []
# for k, M in enumerate(Mvirs):
#     for i, R in enumerate(r_cens):
#         with h5.File(vel_path, 'r') as hdf:
#             rel_R = hdf[f'R/{str(k)}'][()]
#             rel_rlos = hdf[f'rlos/{str(k)}'][()]
#             vpec = hdf[f'vpeclos/{str(k)}'][()]
#             vphys = hdf[f'vphyslos/{str(k)}'][()]

#         mask = (rel_R >= r_edges[i]) & (rel_R <= r_edges[i+1])

#         vpeclos_ls.append(vpec[mask])
#         vphyslos_ls.append(vphys[mask])
#         rlos_ls.append(rel_rlos[mask])

# #%%
# #check with plot


# ls_edges = np.linspace(30, 51, 51-30+1)
# ls_cens = (ls_edges[1:]+ls_edges[:-1])/2.0

# M_select_indices = [0, 5]
# M_select_edges = [mbins[i:i+2] for i in M_select_indices] 
# M_select_cens = [Mvirs[i] for i in M_select_indices] 

# # bins for plots
# ls_select_indices = [0, 10, 20]
# ls_select_edges = [ls_edges[i:i+2] for i in ls_select_indices] #[edges[2:4], edges[7:9], edges[20:22]]
# ls_select_cens = [ls_cens[i] for i in ls_select_indices] #[cens[2], cens[7], cens[20]]

# # radial bins
# edges = np.linspace(4, 30, 26+1)
# cens = (edges[1:]+edges[:-1])/2.0

# # bins for plots
# select_indices = [0, 10, 25]
# select_edges = [edges[i:i+2] for i in select_indices] #[edges[2:4], edges[7:9], edges[20:22]]
# select_cens = [cens[i] for i in select_indices] #[cens[2], cens[7], cens[20]]


# fig, ax = plt.subplots(2, 3, figsize=(20, 10), layout='constrained', sharex=True, sharey=True)

# a = 0.83760
# H = 74.71913613009617
# h = 0.6777 # damn you, little h!

# for k in range(len(M_select_cens)):
#     for i in range(len(select_cens)):
#         if i < 2:
#             vxs = np.linspace(-2_500, 2_500, 150) 
#             R = cens[select_indices[i]] #np.sqrt(np.mean(Rdata[nbins*select_indices[i]+select_indices[j]]**2))

#             index2 = select_indices[i] + M_select_indices[k]*len(cens)

#             #n, bins = np.histogram(vpeclos[index2], bins=60, range=(-2_500, 2_500), density = True)
#             nphys, bins = np.histogram(vphyslos[index2], bins=75, range=(-2_500, 2_500), density = True)

#             cent = (bins[:-1] + bins[1:])/2. 

#             print(cens[select_indices[i]], Mvirs[M_select_indices[k]]/1e14)
#             #model = P_vlos_R_romb(vxs, cens[select_indices[i]], Mvirs[M_select_indices[k]])
#             model_phys = P_vphys_R_vec(vxs, cens[select_indices[i]], Mvirs[M_select_indices[k]]/1e14, pars=gal_MAP)

#             #ax[k, i].hist(cent, bins=bins, weights=n, color="C0", histtype='stepfilled', alpha=0.3, label=r'$v_{\rm pec, LOS}$') 
#             #ax[k, i].hist(cent, bins=bins, weights=n, color="C0", histtype='step') 

#             ax[k, i].hist(cent, bins=bins, weights=nphys, color="C1", histtype='stepfilled', alpha=0.3, label=r"$\begin{array}{rl} v_{\rm LOS} = &v_{\rm pec, LOS} \\ &+ aHr_{\rm LOS} \end{array}$")
#             ax[k, i].hist(cent, bins=bins, weights=nphys, color="C1", histtype='step')

#             #ax[k, i].plot(vxs, model, color="C0")
#             ax[k, i].plot(vxs, model_phys, color="C1")
#             ax[1, i].set_xlabel(r'$v_{\rm LOS}$ [km/s]')
#             ax[k, 0].set_ylabel(r'$M \in [{:.2f}, {:.2f}] \cdot 10^{{14}} M_\odot$'.format(M_select_edges[k][0]/1e14, M_select_edges[k][1]/1e14), fontsize=25)
#             ax[k, i].set_yticks([])
#             ax[1, i].set_xticks([-2000, -1000, 0, 1000, 2000])
#             ax[0, i].set_title(r'$R \in [{:.2f}, {:.2f}] h^{{-1}}$ Mpc'.format(select_edges[i][0], select_edges[i][1]))
#             ax[0, i].margins(x=0)

#         if i == 2:
#             index2 = ls_select_indices[i] + M_select_indices[k]*len(ls_cens)
#             #n, bins = np.histogram(vpeclos_ls[index2], bins=60, range=(-2_500, 2_500), density = True)
#             nphys, bins = np.histogram(vphyslos_ls[index2], bins=bins, range=(-2_500, 2_500), density = True)

#             print(cens[select_indices[i]], Mvirs[M_select_indices[k]]/1e14)
#             #model = P_vlos_R_romb(vxs, cens[select_indices[i]], Mvirs[M_select_indices[k]])
#             model_phys = P_vphys_R_vec(vxs, cens[select_indices[i]], Mvirs[M_select_indices[k]]/1e14, pars=gal_MAP) 

#             cent = (bins[:-1] + bins[1:])/2. 

#             #ax[k, i].hist(cent, bins=bins, weights=n, color="C0", histtype='stepfilled', alpha=0.3) 
#             #ax[k, i].hist(cent, bins=bins, weights=n, color="C0", histtype='step', label=r'$v_{\rm pec, LOS}$') 

#             ax[k, i].hist(cent, bins=bins, weights=nphys, color="C1", histtype='stepfilled', alpha=0.3)
#             ax[k, i].hist(cent, bins=bins, weights=nphys, color="C1", histtype='step', label=r"$\begin{array}{rl} v_{\rm LOS} = &v_{\rm pec, LOS} \\ &+ aHr_{\rm LOS} \end{array}$")

#             #ax[k, i].plot(vxs, model, color="C0") 
#             ax[k, i].plot(vxs, model_phys, color="C1")
#             ax[k, i].set_yticks([])
#             ax[0, i].set_title(r'$R \in [{:.2f}, {:.2f}] h^{{-1}}$ Mpc'.format(select_edges[i][0], select_edges[i][1]))
#             ax[1, i].set_xlabel(r'$v_{\rm LOS}$ [km/s]')
#             ax[0, 2].legend(fontsize=14, loc='upper right')
# #ax[0, 0].legend(fontsize=14, loc='upper right')
# fig.suptitle(r"$P(v_{\rm LOS}|R, M)$")  


#%%

# Compute P_vlos on grid of M 
Mvals = np.logspace(np.log10(0.1), np.log10(20), 100) # grid of Mvals

bin_num = 50
vedges = np.linspace(-4000, 4000, 51) # across entire viable range
 #np.linspace(-2500, 2500, bin_num+1)
vcens = 0.5*(vedges[1:]+vedges[:-1])

dists = np.zeros((len(vcens), len(r_cens), len(Mvals) ))
which_pars = [gal_MAP, part_MAP]

for k in range(len(which_pars)):
    for j in tqdm(range( len(Mvals) )): 
        for i in range( len( r_cens ) ):
            dist = P_vphys_R_vec(vcens, r_cens[i], Mvals[j], which_pars[k]) # model using galaxy/particle best fit 
            dists[:, i, j] = dist

    # save results
    suff = ['_gal', '_part']  
    path = '/spiff/cosweeney/simulations/MDPL2/data/Pvlos_M_grid_norm_update'

    np.save(path+suff[k], dists) 

# from joblib import Parallel, delayed

# def compute_for_j(j):
#     """Compute all r_cens for a single Mval index j"""
#     result = np.zeros((len(vcens), len(r_cens)))
#     for i in range(len(r_cens)):
#         dist = P_vphys_R_vec(vcens, r_cens[i], Mvals[j], gal_MAP)
#         result[:, i] = dist
#     return j, result

# # This works even in Jupyter notebooks
# results = Parallel(n_jobs=-1, backend='loky')(
#     delayed(compute_for_j)(j) for j in tqdm(range(len(Mvals)))
# )

# # Reassemble results
# for j, result in results:
#     dists[:, :, j] = result



#%%
#interpolate

# interpolator = interp1d(
#     Mvals, 
#     dists, 
#     axis=2,  # interpolate along the M dimension
#     kind='cubic',  # or 'linear', 'quadratic'
#     bounds_error=False,
#     fill_value='extrapolate' 
# )

# # save 
# path = '/spiff/cosweeney/simulations/MDPL2/data/models/'

# with open(path+'Pvlos_R_M_interpolator.pkl', 'wb') as f:
#     pickle.dump(interpolator, f)