""" Models for simplified versions of the Projected LOS velocity distributions.
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

def plaw(x, a, b):
    return a*x**b


def skewed_t_pdf(x, xi, omega, alpha, nu):
    """
    Azzalini & Capitanio skewed t-distribution PDF
    
    Parameters:
    xi: location parameter
    omega: scale parameter (> 0)
    alpha: skewness parameter
    nu: degrees of freedom (> 0)
    """
    z = (x - xi) / omega
    
    t_pdf = t.pdf(z, df=nu)
    
    t_cdf = t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    
    pdf = (2 / omega) * t_pdf * t_cdf
    
    return pdf


def P_R_rlos(vlos, R, rlos, M, pars):
    vrpp, vrps, vrsp, vrss, sigma_p, sigma_s, alpha_p, alpha_s = pars

    M_p = 1e14 # h^-1 M_sun
    r_p = 20 # h^-1 Mpc
    R_p = 10 # h^-1 Mpc
    r = np.sqrt( R**2 + rlos**2 )
    theta = np.arctan2(rlos, R)
    sin = np.sin(theta)

    vrp = -plaw(M/M_p, vrpp, vrps)
    vrs = -plaw(M/M_p, vrsp, vrss)

    sigma = plaw(R/R_p, sigma_p, sigma_s)
    alpha = plaw(R/R_p, alpha_p, alpha_s)

    peak_vr = plaw(r/r_p, vrp, vrs)

    mean = peak_vr * sin + a*H*rlos/h

    dist = skewed_t_pdf(vlos, xi = mean, omega=sigma, alpha=-np.sign(rlos)*alpha, nu=4)

    return dist


def P_R(vlos, R, M, pars):

    rho_g = 0.014  # h^3 Mpc^-3 mean galaxy number density in MDPL2

    # rlos integration range
    rlos_max = 60
    rlos_edges = np.linspace(-rlos_max, rlos_max, 100+1)
    rlos_cens = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])

    r3d = np.sqrt(R**2 + rlos_cens**2)
    weight = rho_g * (cf_inf(r3d, M, jocond.myconfig.cf_pars) + 1)  

    Pvlos_grid = np.array(
        [P_R_rlos(vlos, R, rlos, M, pars) for rlos in rlos_cens]
    )

    integrand = Pvlos_grid.T * weight[None, :] 

    P_out = simpson(integrand, x=rlos_cens, axis=1)

    return P_out / np.sum(P_out * np.diff(vlos)[0])



def P_R_rlos_vec(vlos, R, rlos, M, pars):
    """
    vlos:  (Nv,)
    R:     (NR,)
    rlos:  (Nrlos,)
    M:     (NM,)
    returns: (Nv, NR, NM, Nrlos)
    """
    vrpp, vrps, vrsp, vrss, sigma_p, sigma_s, alpha_p, alpha_s = pars
    M_p = 1e14
    r_p = 20
    R_p = 10

    R_    = R[None, :, None, None]
    rlos_ = rlos[None, None, None, :]
    M_    = M[None, None, :, None]
    vlos_ = vlos[:, None, None, None]

    r     = np.sqrt(R_**2 + rlos_**2)
    theta = np.arctan2(rlos_, R_)
    sin   = np.sin(theta)

    vrp   = -plaw(M_ / M_p, vrpp, vrps)
    vrs   = -plaw(M_ / M_p, vrsp, vrss)
    sigma = plaw(R_ / R_p, sigma_p, sigma_s)
    alpha = plaw(R_ / R_p, alpha_p, alpha_s)

    peak_vr = plaw(r / r_p, vrp, vrs)
    mean    = peak_vr * sin + a * H * rlos_ / h

    dist = skewed_t_pdf(vlos_, xi=mean, omega=sigma,
                        alpha=-np.sign(rlos_) * alpha, nu=4)
    return dist  


def P_R_vec(vlos, R, M, pars):
    """
    vlos: (Nv,)
    R:    (NR,)
    M:    (NM,)
    returns: (Nv, NR, NM)
    """
    rho_g    = 0.014
    rlos_max = 60
    rlos_edges = np.linspace(-rlos_max, rlos_max, 101)
    rlos_cens  = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])  

    r3d    = np.sqrt(R[:, None]**2 + rlos_cens[None, :]**2)
    weight = rho_g * (cf_inf(r3d[:, None, :], M[None, :, None],
                             jocond.myconfig.cf_pars) + 1)

    Pvlos_grid = P_R_rlos_vec(vlos, R, rlos_cens, M, pars)

    integrand = Pvlos_grid * weight[None, :, :, :]

    P_out = simpson(integrand, x=rlos_cens, axis=-1)

    dv = np.diff(vlos)[0]
    return P_out / np.sum(P_out * dv, axis=0, keepdims=True)


# alternative model, relating skewness to vr_peak


def P_R_rlos_alt(vlos, R, rlos, M, pars):
    vrpp, vrps, vrsp, vrss, sigma_p, sigma_s, alpha_p, alpha_s = pars

    M_p = 1e14 # h^-1 M_sun
    r_p = 20 # h^-1 Mpc
    R_p = 10 # h^-1 Mpc
    r = np.sqrt( R**2 + rlos**2 )
    theta = np.arctan2(rlos, R)
    sin = np.sin(theta)

    vrp = -plaw(M/M_p, vrpp, vrps)
    vrs = -plaw(M/M_p, vrsp, vrss)

    sigma = plaw(R/R_p, sigma_p, sigma_s)
    alpha = plaw(R/R_p, alpha_p, alpha_s)

    peak_vr = plaw(r/r_p, vrp, vrs)

    mean = peak_vr * sin + a*H*rlos/h

    dist = skewed_t_pdf(vlos, xi = mean, omega=sigma, alpha=-np.sign(rlos)*alpha, nu=4)

    return dist


def P_R(vlos, R, M, pars):

    rho_g = 0.014  # h^3 Mpc^-3 mean galaxy number density in MDPL2

    # rlos integration range
    rlos_max = 60
    rlos_edges = np.linspace(-rlos_max, rlos_max, 100+1)
    rlos_cens = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])

    r3d = np.sqrt(R**2 + rlos_cens**2)
    weight = rho_g * (cf_inf(r3d, M, jocond.myconfig.cf_pars) + 1)  

    Pvlos_grid = np.array(
        [P_R_rlos(vlos, R, rlos, M, pars) for rlos in rlos_cens]
    )

    integrand = Pvlos_grid.T * weight[None, :] 

    P_out = simpson(integrand, x=rlos_cens, axis=1)

    return P_out / np.sum(P_out * np.diff(vlos)[0])



def P_R_rlos_vec(vlos, R, rlos, M, pars):
    """
    vlos:  (Nv,)
    R:     (NR,)
    rlos:  (Nrlos,)
    M:     (NM,)
    returns: (Nv, NR, NM, Nrlos)
    """
    vrpp, vrps, vrsp, vrss, sigma_p, sigma_s, alpha_p, alpha_s = pars
    M_p = 1e14
    r_p = 20
    R_p = 10

    R_    = R[None, :, None, None]
    rlos_ = rlos[None, None, None, :]
    M_    = M[None, None, :, None]
    vlos_ = vlos[:, None, None, None]

    r     = np.sqrt(R_**2 + rlos_**2)
    theta = np.arctan2(rlos_, R_)
    sin   = np.sin(theta)

    vrp   = -plaw(M_ / M_p, vrpp, vrps)
    vrs   = -plaw(M_ / M_p, vrsp, vrss)
    sigma = plaw(R_ / R_p, sigma_p, sigma_s)
    alpha = plaw(R_ / R_p, alpha_p, alpha_s)

    peak_vr = plaw(r / r_p, vrp, vrs)
    mean    = peak_vr * sin + a * H * rlos_ / h

    dist = skewed_t_pdf(vlos_, xi=mean, omega=sigma,
                        alpha=-np.sign(rlos_) * alpha, nu=4)
    return dist  


def P_R_vec(vlos, R, M, pars):
    """
    vlos: (Nv,)
    R:    (NR,)
    M:    (NM,)
    returns: (Nv, NR, NM)
    """
    rho_g    = 0.014
    rlos_max = 60
    rlos_edges = np.linspace(-rlos_max, rlos_max, 101)
    rlos_cens  = 0.5 * (rlos_edges[1:] + rlos_edges[:-1])  

    r3d    = np.sqrt(R[:, None]**2 + rlos_cens[None, :]**2)
    weight = rho_g * (cf_inf(r3d[:, None, :], M[None, :, None],
                             jocond.myconfig.cf_pars) + 1)

    Pvlos_grid = P_R_rlos_vec(vlos, R, rlos_cens, M, pars)

    integrand = Pvlos_grid * weight[None, :, :, :]

    P_out = simpson(integrand, x=rlos_cens, axis=-1)

    dv = np.diff(vlos)[0]
    return P_out / np.sum(P_out * dv, axis=0, keepdims=True)


