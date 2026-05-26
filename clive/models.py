import numpy as np 
from scipy.stats import t
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

# simple fitting functions
def plaw(x, p, s):
    """Power law function.

    Args:
        x (any): power law argument
        p (any): power law pivot
        s (any): power law slope

    Returns:
        float: Power law in x.
    """
    return p * x**s


def plawc(x, p, s, c):
    return p * x **s + c


def line(x, m, b):
    """Linear function.

    Args:
        x (any): argument
        m (any): slope
        b (any): intercept

    Returns:
        float: Linear function in x.
    """
    return m * x + b


class SkewTMeanInterpolator:
    """
    Pre-computes the relationship between (Delta_mean / sigma_v) and (omega, alpha)
    for a Skew-T distribution using the algebraic mean. 
    """
    def __init__(self, nu=6, alpha_bounds=(-15, 15), pts=1000):
        self.nu = nu
        
        # We can increase 'pts' since this is now an instantaneous vectorized calculation
        alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], pts)
        
        # 1. Delta coefficient for Azzalini Skew-T
        delta = alphas / np.sqrt(1 + alphas**2)
        
        # 2. Exact Mean of standard Skew-T (xi=0, omega=1)
        # mu_z = delta * sqrt(nu / pi) * [Gamma((nu - 1) / 2) / Gamma(nu / 2)]
        gamma_factor = gamma((nu - 1) / 2) / gamma(nu / 2)
        mu_z = delta * np.sqrt(nu / np.pi) * gamma_factor
        
        # 3. Exact Standard Deviation of standard Skew-T
        var_z = (nu / (nu - 2)) - mu_z**2
        sig_z = np.sqrt(var_z)
        
        # 4. Ratio R = Delta_mean / sigma_v for standard distribution
        # Assuming Delta_mean = xi - mean = 0 - mu_z = -mu_z
        R_vals = -mu_z / sig_z
        
        # interp1d requires the x-array (R_vals) to be strictly monotonically increasing.
        # As alpha increases, R_vals strictly decreases, so we must reverse the arrays.
        if R_vals[-1] < R_vals[0]:
            valid_R = R_vals[::-1]
            valid_alphas = alphas[::-1]
        else:
            valid_R = R_vals
            valid_alphas = alphas
            
        # Interpolator 1: Ratio -> alpha (Upgraded to cubic for smooth analytical data)
        self.R_to_alpha = interp1d(valid_R, valid_alphas, kind='cubic', 
                                   bounds_error=False, 
                                   fill_value=(valid_alphas[0], valid_alphas[-1]))
        
        # Interpolator 2: alpha -> standard deviation
        self.alpha_to_sigZ = interp1d(alphas, sig_z, kind='cubic', 
                                      bounds_error=False, 
                                      fill_value=(sig_z[0], sig_z[-1]))

    def get_skew_params(self, delta_mean, sig_v):
        """Maps physical parameters (Delta_mean, sigma_v) to Skew-T parameters (omega, alpha)"""
        ratio = delta_mean / sig_v
        alpha = self.R_to_alpha(ratio)
        sig_z = self.alpha_to_sigZ(alpha)
        omega = sig_v / sig_z
        return omega, alpha
    
# Instantiate this globally so it isn't rebuilt on every function call
skew_interp = SkewTMeanInterpolator(nu=6)

def Delta(x, M, pars):
    a_p, a_s, a_c = pars

    M_p = 1e14

    A = 170
    a = plawc(M/M_p, a_p, a_s, a_c)

    D = A*x/(a+x)

    return D

save_path = '/spiff/cosweeney/simulations/MDPL2/data/models'

data = np.load(save_path+'vr_mean_r_RBS_data.npz')

# Re-initialize the object (we use a dummy grid then overwrite)
RBinterp = RectBivariateSpline.__new__(RectBivariateSpline)
RBinterp.tck = (data['tx'], data['ty'], data['c'])
RBinterp.degrees = (data['kx'], data['ky'])

def vlos_mean(R, rlos, M):
    r    = np.sqrt(R**2 + rlos**2)
    sin = rlos / r

    mode = RBinterp(np.log10(M), r).squeeze()*sin 

    return mode

def rho_mod(x, M, pars):
    am, ab, s = pars
    a = line(M/1e14, am, ab)
    return a*np.exp(-x/s)

def sig_vm(x, M, pars):
    rho = rho_mod(x, M, pars)
    s_0 = 355

    return np.sqrt(2*s_0**2 * (1 - rho))


delta_mpars = [ 1.58565530e-01,  3.20425113e-01,  4.35742317e-05] 
rho_pars    = [-4.53981362e-02,  3.74929938e-01,  1.07841238e+01]

def Pv_R_rlos(v, R, rlos, M, pars_delta=delta_mpars, pars_sig=rho_pars):
    """
    Evaluates P(v | R, rlos, M) modeled as a Skew-T distribution.
    """
    # 1. Coordinate transformations
    r = np.sqrt(R**2 + rlos**2)
    x = r - R
    
    # 2. Evaluate physical quantities via your models
    mu = vlos_mean(R, rlos, M)
    sig_v = sig_vm(x, M, pars_sig)
    
    # Enforce symmetry: Delta flips sign when rlos is negative
    sign_rlos = np.sign(rlos)
    delta_val = Delta(x, M, pars_delta) * (sign_rlos if sign_rlos != 0 else 1.0)
    
    # 3. Interpolate standard Skew-T parameters
    omega, alpha = skew_interp.get_skew_params(delta_val, sig_v)
    
    # 4. Location parameter is purely algebraic
    xi = mu + delta_val
    
    # 5. Evaluate Skew-T PDF
    nu = skew_interp.nu
    z = (v - xi) / omega
    
    # Azzalini's Skew-T formulation: 2 * t.pdf(z) * t.cdf(alpha * z * sqrt...)
    arg = alpha * z * np.sqrt((nu + 1) / (nu + z**2))
    pdf = (2 / omega) * t.pdf(z, nu) * t.cdf(arg, nu + 1)
    
    return pdf


def vlos_mean_vec(R, rlos, M, RBinterp=RBinterp):
    """
    Vectorized version of vlos_mean that works with N-dimensional broadcasted arrays.
    """
    r = np.sqrt(R**2 + rlos**2)
    
    # 1. Safe sine calculation to prevent division by zero at R=0, rlos=0
    # np.where evaluates across the array safely without throwing warnings
    sin = np.where(r == 0, 0.0, rlos / r)
    
    # 2. Broadcast inputs to their shared target shape (e.g., 1, NM, NR, Nrlos)
    logM = np.log10(M)
    logM_b, r_b = np.broadcast_arrays(logM, r)
    
    # 3. Flatten the arrays to 1D for safe interpolation
    flat_logM = logM_b.ravel()
    flat_r = r_b.ravel()
    
    # 4. Evaluate the interpolator point-to-point.
    # Note: Because your original syntax was RBinterp(x, y), I am assuming this is 
    # a scipy.interpolate.RectBivariateSpline (or similar). 
    # Using `grid=False` ensures it evaluates paired coordinates (x[i], y[i]) 
    # rather than building a grid of all possible combinations.
    flat_interp = RBinterp(flat_logM, flat_r, grid=False)
    
    # 5. Reshape back to the broadcasted 4D shape and apply the geometric factor
    mode_interp = flat_interp.reshape(logM_b.shape)
    mode = mode_interp * sin
    
    return mode


def Pv_R_rlos_vec(v, R, rlos, M, pars_delta=delta_mpars, pars_sig=rho_pars):
    """
    Vectorized evaluation of P(v | R, rlos, M).
    Expects 1D arrays for v, R, rlos, M.
    Returns an array of shape (Nv, NM, NR, Nrlos).
    """
    # 1. Expand dimensions to create a 4D broadcastable grid
    # v shape:    (Nv, 1, 1, 1)
    # M shape:    (1, NM, 1, 1)
    # R shape:    (1, 1, NR, 1)
    # rlos shape: (1, 1, 1, Nrlos)
    v_b    = np.asarray(v)[:, None, None, None]
    M_b    = np.asarray(M)[None, :, None, None]
    R_b    = np.asarray(R)[None, None, :, None]
    rlos_b = np.asarray(rlos)[None, None, None, :]
    
    # 2. Coordinate transformations (Shape: 1, 1, NR, Nrlos)
    r = np.sqrt(R_b**2 + rlos_b**2)
    x = r - R_b
    
    # 3. Evaluate physical quantities
    # Note: Your underlying vlos_mean, sig_vm, and Delta functions 
    # will naturally inherit these broadcasted (1, NM, NR, Nrlos) shapes
    mu    = vlos_mean_vec(R_b, rlos_b, M_b)
    sig_v = sig_vm(x, M_b, pars_sig)
    
    # Enforce symmetry using np.where for array-safe conditionals
    sign_rlos = np.sign(rlos_b)
    sign_rlos = np.where(sign_rlos == 0, 1.0, sign_rlos)
    delta_val = Delta(x, M_b, pars_delta) * sign_rlos
    
    # 4. Interpolate standard Skew-T parameters
    # scipy.interpolate.interp1d natively supports N-dimensional arrays
    omega, alpha = skew_interp.get_skew_params(delta_val, sig_v)
    
    # 5. Location parameter
    xi = mu + delta_val
    
    # 6. Evaluate Skew-T PDF
    # z shape becomes (Nv, NM, NR, Nrlos) because v_b broadcasts against xi/omega
    nu = skew_interp.nu
    z = (v_b - xi) / omega
    
    arg = alpha * z * np.sqrt((nu + 1) / (nu + z**2))
    pdf = (2 / omega) * t.pdf(z, nu) * t.cdf(arg, nu + 1)
    
    return pdf