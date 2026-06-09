import numpy as np

from halotools.mock_observables import tpcf, s_mu_tpcf
from halotools.mock_observables import apply_zspace_distortion


from functools import partial
import multiprocessing

from halotools.mock_observables.pairwise_velocities.mean_radial_velocity_vs_r import (
    _process_args,
)
from halotools.mock_observables.pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes,
    _cell1_parallelization_indices,
)
from halotools.mock_observables.pair_counters.rectangular_mesh import (
    RectangularDoubleMesh,
)


# functions defined by https://github.com/florpi/GaussianStreamingModel

def compute_real_tpcf(r, pos, boxsize, num_threads=1, pos_cross=None):
    """
        Computes the real space two point correlation function using halotools
        Args:
                r: np.array
                         binning in pair distances.
                pos: np.ndarray
                         3-D array with the position of the tracers.
                boxsize: float
                        size of the simulation's box.
                num_threads: int
                        number of threads to use.
                pos_cross: np.ndarray
                         3-D array with the position of the tracers
                         (for cross_correlations).

        Returns:
                real_tpcf: np.array
                        1-D array with the real space tpcf.
        """
    if pos_cross is not None:
        do_auto = False
    else:
        do_auto = True
    real_tpcf = tpcf(
        pos,
        r,
        period=boxsize,
        num_threads=num_threads,
        sample2=pos_cross,
        do_auto=do_auto,
    )
    return real_tpcf


def move_to_redshift_space(pos, vel, cosmology, redshift, los_direction, boxsize):
    s_pos = pos.copy()
    z_pos = apply_zspace_distortion(
        true_pos=pos[:, los_direction],
        peculiar_velocity=vel[:, los_direction],
        redshift=redshift,
        cosmology=cosmology,
        Lbox=boxsize,
    )
    # Move tracers to redshift space
    s_pos[:, los_direction] = z_pos
    # Halotools tpcf_s_mu assumes the line of sight is always the z direction
    if los_direction != 2:
        s_pos_old = s_pos.copy()
        s_pos[:, 2] = s_pos_old[:, los_direction]
        s_pos[:, los_direction] = s_pos_old[:, 2]
    return s_pos

def compute_tpcf_s_mu(
    s,
    mu,
    pos,
    vel,
    los_direction,
    cosmology,
    boxsize,
    redshift,
    num_threads=1,
    pos_cross=None,
    vel_cross=None,
):
    """
        Computes the redshift space two point correlation function
        Args:
                s: np.array
                        binning in redshift space pair distances.
                mu: np.array
                         binning in the cosine of the angle respect to the line of sight.
                pos: np.ndarray
                        3-D array with the position of the tracers, in Mpc/h.
                vel: np.ndarray
                         3-D array with the velocities of the tracers, in km/s.
                los_direction: int
                        line of sight direction either 0(=x), 1(=y), 2(=z)
                cosmology: dict
                        dictionary containing the simulatoin's cosmological parameters.
                boxsize:  float
                        size of the simulation's box.
                num_threads: int 
                        number of threads to use.
        Returns:
                tpcf_s_mu: np.ndarray
                        2-D array with the redshift space tpcf.
        """
    if pos_cross is not None:
        do_auto = False
    else:
        do_auto = True

    s_pos = move_to_redshift_space(
        pos, vel, cosmology, redshift, los_direction, boxsize
    )
    if pos_cross is not None and vel_cross is not None:
        s_pos_cross = move_to_redshift_space(
            pos_cross, vel_cross, cosmology, redshift, los_direction, boxsize
        )
    else:
        s_pos_cross = None
    tpcf_s_mu = s_mu_tpcf(
        s_pos,
        s,
        mu,
        period=boxsize,
        estimator="Natural",#u"Landy-Szalay",
        num_threads=num_threads,
        sample2=s_pos_cross,
        do_auto=False#do_auto,
    )
    print(type(tpcf_s_mu))
    print(len(tpcf_s_mu) if isinstance(tpcf_s_mu, tuple) else tpcf_s_mu.shape)

    return tpcf_s_mu

    import numpy as np
from typing import NamedTuple, Callable
from scipy.special import binom
from collections import namedtuple
from scipy.stats import norm


def get_moment(
    moments: NamedTuple, r: np.array, r_order: int, t_order: int, mode: str
) -> np.array:
    """
    Given a named tuple containing the radial and transverse moments, returns the ```r_order```
    radial moment and the ```t_order``` transverse moment.

    Args:
        moments: Named tuple containing the radial and transverse moments. 
        r:  pair separation.
        r_order: order of the radial moment
        t_order: order of the transverse moments
        mode: either ```c``` for central moments or ```m``` for moments.

    Returns:
        moment

    Example naming moments Tuple: ('m_10': Radial mean, 'c_20': Second order radial central moment)
    """
    if t_order % 2 != 0:
        # Due to isotropy all momens with t_order odd vanish
        return np.zeros_like(r)
    elif (r_order == 0) and (t_order == 0):
        # The PDF is normalised
        return np.ones_like(r)
    elif (mode == "c") and (r_order + t_order == 1):
        # The first order central moments are zero
        return np.zeros_like(r)
    else:
        return getattr(moments, f"{mode}_{r_order}{t_order}")(r)


def project_to_los(moments: NamedTuple, n: int, mode: str = "c") -> Callable:
    """ 
    Project the moments of the radial and tangential velocity field onto the line of sight moments.

    Args:
        moments: Named tuple containing the radial and transverse moments.
        n: order of the moment.
        mode: Type of moment. If central moments use c, if moments about the origin use m.
    Returns:
        2D function of r_parallel and r_perpendicular that returns the 
        n-th moment of the line of sight velocity PDF 
    """

    def los_moment(r_perpendicular, r_parallel):
        r_perpedicular = np.atleast_2d(r_perpendicular)
        r_parallel = np.atleast_2d(r_parallel)

        r = np.sqrt(r_parallel ** 2 + r_perpendicular ** 2)
        mu = r_parallel / r

        return np.sum(
            [
                binom(n, k)
                * mu ** k
                * np.sqrt(1 - mu ** 2) ** (n - k)
                * get_moment(moments, r, r_order=k, t_order=n - k, mode=mode)
                for k in range(n + 1)
            ],
            axis=0,
        )

    return los_moment

def losmoments2gaussian(mean: Callable, scale: Callable)->Callable:
    """
    Args:
        mean: function that takes r_parallel and r_perp as inputs and returns the mean 
        line of sight pairwise velocity
        std:  function that takes r_parallel and r_perp as inputs and returns the standard 
        deviation of the line of sight pairwise velocity


    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    def pdf_los(vlos: np.array, r_perp: np.array, r_parallel: np.array):
        return norm.pdf(
            vlos, loc=mean(r_perp, r_parallel), scale=scale(r_perp, r_parallel)
        )

    return pdf_los

def project_moments(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    """
    Args:
        m_10: function that takes pair separation (r) as input and returns the mean 
        radial pairwise velocity
        c_20:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity
        c_02:  function that takes pair separation (r) as input and returns the standard 
        deviation of the radial pairwise velocity

    Returns:
        pdf_los: line of sight pairwise velocity PDF 
    """
    Moments = namedtuple('Moments', ['m_10', 'c_20', 'c_02'])
    moments = Moments(m_10, c_20, c_02)
    mean = project_to_los(moments, 1, mode='m')
    c_2 = project_to_los(moments, 2, mode='c')
    std = lambda r_perp, r_parallel: np.sqrt(c_2(r_perp, r_parallel))
    return mean, std

def moments2gaussian(m_10: Callable, c_20: Callable, c_02: Callable)->Callable:
    mean, std = project_moments(m_10, c_20, c_02)
    return losmoments2gaussian(mean, std)


from typing import Callable
from scipy.integrate import simpson#, quadrature, quad


def integrand_s_mu(
    s_c: float, mu_c: float, twopcf_function: Callable, los_pdf_function: Callable
):
    """
    Computes the streaming model integrand ( https://arxiv.org/abs/1710.09379, Eq 22 ) at s, mu
    Args:
        s_c: bin centers for the pair distance bins.
        mu_c: bin centers for the cosine of the angle rescpect to the line of sight bins.
        twopcf_function: function that given pair distance as an argument returns the real space two point 
                correlation function.
        los_pdf_function: given the line of sight velocity, perpendicular and parallel distances to the line
                of sight, returns the value of the line of sight pairwise velocity distribution.
    Returns:
        integrand: np.ndarray
            2-D array with the value of the integrand evaluated at the given s_c and mu_c.			
	"""

    def integrand(y):
        S = s_c.reshape(-1, 1)
        MU = mu_c.reshape(1, -1)
        s_parallel = S * MU
        s_perp = S * np.sqrt(1 - MU ** 2)
        # Use reshape to vectorize all possible combinations
        s_perp = s_perp.reshape(-1, 1)
        s_parallel = s_parallel.reshape(-1, 1)
        y = y.reshape(1, -1)
        vlos = (s_parallel - y) * np.sign(y)
        r = np.sqrt(s_perp ** 2 + y ** 2)
        los_pdf = np.nan_to_num(los_pdf_function(vlos, s_perp, y), #np.abs(y)
                copy=False)
        return los_pdf * (1 + twopcf_function(r.flatten()).reshape(r.shape))

    return integrand


def simps_integrate(
    s_c: np.array,
    mu_c: np.array,
    twopcf_function: Callable,
    los_pdf_function: Callable,
    limit: float = 120.0,
    epsilon: float = 0.0001,
    n: int = 300,
):
    """
    Computes the streaming model integral ( https://arxiv.org/abs/1710.09379, Eq 22 ) 
    Args:
        s_c: pair distance bins.
        mu_c: cosine of the angle rescpect to the line of sight bins.
        twopcf_function: function that given pair distance as an argument returns the real space two point 
                correlation function.
        los_pdf_function: given the line of sight velocity, perpendicular and parallel distances to the line
                of sight, returns the value of the line of sight pairwise velocity distribution.
        limit: r_parallel limits of the integral.
        epsilon: due to discontinuity at zero, add small offset +-epsilon to estimate integral.
        n: number of points to evaluate the integrand.
    Returns:
        twopcf_s: np.ndarray
            2-D array with the resulting redshift space two point correlation function
	"""

    streaming_integrand = integrand_s_mu(s_c, mu_c, twopcf_function, los_pdf_function)
    # split integrand in two due to discontinuity at 0
    r_integrand = np.linspace(-limit, -epsilon, n)
    integral_left = simpson(
        streaming_integrand(r_integrand), r_integrand, axis=-1
    ).reshape((s_c.shape[0], mu_c.shape[0]))

    r_integrand = np.linspace(epsilon, limit, n)
    integral_right = simpson(
        streaming_integrand(r_integrand), r_integrand, axis=-1
    ).reshape((s_c.shape[0], mu_c.shape[0]))

    twopcf_s = integral_left + integral_right - 1.0
    return twopcf_s


import sys
sys.path.insert(0, '/home/cosweeney/code/clive_bias/notebooks/streaming_model/')

from tangential_pvd_vs_r_engine import tangential_pvd_vs_r_engine

def tangential_pvd_vs_r(
    sample1, velocities1,
    rbins_absolute=None, rbins_normalized=None, normalize_rbins_by=None,
    sample2=None, velocities2=None,
    period=None, num_threads=1,
    approx_cell1_size=None, approx_cell2_size=None,
):
    result = _process_args(
        sample1, velocities1, sample2, velocities2,
        rbins_absolute, rbins_normalized, normalize_rbins_by,
        period, num_threads, approx_cell1_size, approx_cell2_size,
    )
    (
        sample1, velocities1, sample2, velocities2,
        max_rbins_absolute, period, num_threads,
        _sample1_is_sample2, PBCs,
        approx_cell1_size, approx_cell2_size,
        rbins_normalized, normalize_rbins_by,
    ) = result

    x1in, y1in, z1in = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2in, y2in, z2in = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    vx1in, vy1in, vz1in = velocities1[:, 0], velocities1[:, 1], velocities1[:, 2]
    vx2in, vy2in, vz2in = velocities2[:, 0], velocities2[:, 1], velocities2[:, 2]
    xperiod, yperiod, zperiod = period
    squared_normalize_rbins_by = normalize_rbins_by * normalize_rbins_by

    approx_cell1_size, approx_cell2_size = _set_approximate_cell_sizes(
        approx_cell1_size, approx_cell2_size, period)

    double_mesh = RectangularDoubleMesh(
        x1in, y1in, z1in, x2in, y2in, z2in,
        *approx_cell1_size, *approx_cell2_size,
        max_rbins_absolute, max_rbins_absolute, max_rbins_absolute,
        xperiod, yperiod, zperiod, PBCs,
    )

    engine = partial(
        tangential_pvd_vs_r_engine,
        double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in,
        vx1in, vy1in, vz1in, vx2in, vy2in, vz2in,
        squared_normalize_rbins_by, rbins_normalized,
    )

    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        counts, vtan_sum, vtansq_sum = result[:, 0], result[:, 1], result[:, 2]
        counts    = np.sum(counts, axis=0)
        vtan_sum  = np.sum(vtan_sum, axis=0)
        vtansq_sum = np.sum(vtansq_sum, axis=0)
        pool.close()
    else:
        counts, vtan_sum, vtansq_sum = np.array(engine(cell1_tuples[0]))

    counts    = np.diff(counts).astype('f4')
    vtan      = np.diff(vtan_sum)
    vtansq    = np.diff(vtansq_sum)

    vtan_dispersion_squared = np.zeros(len(vtan))
    has_pairs = counts > 0
    term1 = vtansq[has_pairs] / counts[has_pairs]
    term2 = (vtan[has_pairs] / counts[has_pairs])**2
    vtan_dispersion_squared[has_pairs] = term1 - term2
    return np.sqrt(vtan_dispersion_squared)