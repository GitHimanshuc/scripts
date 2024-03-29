import sxs
import scri
import numpy as np

import quaternion
import spherical_functions as sf
from spherical_functions import LM_index

from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

from quaternion.calculus import indefinite_integral as integrate

from functools import partial

def WM_to_MT(h_wm):
    h_mts = scri.ModesTimeSeries(
        sf.SWSH_modes.Modes(
            h_wm.data,
            spin_weight=h_wm.spin_weight,
            ell_min=h_wm.ell_min,
            ell_max=h_wm.ell_max,
            multiplication_truncator=max,
        ),
        time=h_wm.t,
    )
    return h_mts

def MT_to_WM(h_mts, sxs_version=False, dataType=scri.h):
    if not sxs_version:
        h = scri.WaveformModes(t=h_mts.t,\
                           data=np.array(h_mts, dtype=complex)[:,sf.LM_index(abs(h_mts.s),-abs(h_mts.s),0):],\
                           ell_min=abs(h_mts.s),\
                           ell_max=h_mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
        h.r_is_scaled_out = True
        h.m_is_scaled_out = True
        return h
    else:
        h = sxs.WaveformModes(input_array=np.array(h_mts, dtype=complex)[:,sf.LM_index(abs(h_mts.s),-abs(h_mts.s),0):],\
                              time=h_mts.t,\
                              time_axis=0,\
                              modes_axis=1,\
                              ell_min=abs(h_mts.s),\
                              ell_max=h_mts.ell_max,\
                              spin_weight=h_mts.s)
        return h

def compute_number_of_orbits_past_time(h, Norbits, t1):
    # return the time at which Norbits have been achieved after t1                                                                                                                                          
    h_news = MT_to_WM(WM_to_MT(h).dot, dataType=scri.hdot)
    omega = np.linalg.norm(h_news.angular_velocity(), axis=1)
    phase = integrate(omega, h_news.t)
    return h.t[np.argmin(abs(phase - (phase[np.argmin(abs(h.t - t1))] + Norbits * (2 * np.pi))))]

def cost(δt_δϕ, args):
    modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization = args
    
    # Take the sqrt because least_squares squares the inputs...
    diff = trapezoid(np.sum(abs(modes_A(t_reference + δt_δϕ[0]) * np.exp(1j * δt_δϕ[1]) ** δϕ_factor * δΨ_factor -\
                                modes_B)**2, axis=1), t_reference)
    return np.sqrt(diff / normalization)

def cost_function(h_A, h_B, t1, t2, δt_δϕ, δΨ_factor=1, include_modes=None):
    h_A = h_A.copy()
    h_B = h_B.copy()

    t_reference = h_A.t[np.argmin(abs(h_A.t - t1)):np.argmin(abs(h_A.t - t2)) + 1]
            
    # Remove certain modes, if requested
    ell_max = min(h_A.ell_max, h_B.ell_max)
    if include_modes != None:
        for L in range(2, ell_max + 1):
            for M in range(-L, L + 1):
                if not (L, M) in include_modes: 
                    h_A.data[:,LM_index(L, M, h_A.ell_min)] *= 0
                    h_B.data[:,LM_index(L, M, h_B.ell_min)] *= 0
                    
    # Define the cost function
    modes_A = CubicSpline(h_A.t, h_A[:,2:ell_max + 1].data)
    modes_B = CubicSpline(h_B.t, h_B[:,2:ell_max + 1].data)(t_reference)
    
    normalization = trapezoid(CubicSpline(h_B.t, h_B[:,2:ell_max + 1].norm())(t_reference), t_reference)
    
    δϕ_factor = np.array([M for L in range(h_A.ell_min, h_A.ell_max + 1) for M in range(-L, L + 1)])
    
    # Optimize by brute force with multiprocessing
    cost_wrapper = partial(cost, args = [modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization])
    
    return 0.5 * cost_wrapper(δt_δϕ) ** 2

def align2d(h_A, h_B, t1, t2, n_brute_force_δt=None, n_brute_force_δϕ=None, include_modes=None, nprocs=None):
    """Align waveforms by shifting in time and phase

    This function determines the optimal time and phase offset to apply to `h_A` by minimizing
    the averaged (over time) L² norm (over the sphere) of the difference of the h_Aveforms.

    The integral is taken from time `t1` to `t2`.

    Note that the input waveforms are assumed to be initially aligned at least well
    enough that:
    
      1) the time span from `t1` to `t2` in the two waveforms will overlap at
         least slightly after the second waveform is shifted in time; and
      2) waveform `h_B` contains all the times corresponding to `t1` to `t2`
         in waveform `h_A`.

    The first of these can usually be assured by simply aligning the peaks prior to
    calling this function:

        h_A.t -= h_A.max_norm_time() - h_B.max_norm_time()

    The second assumption will be satisfied as long as `t1` is not too close to the
    beginning of `h_B` and `t2` is not too close to the end.

    Parameters
    ----------
    h_A : WaveformModes
    h_B : WaveformModes
        Waveforms to be aligned
    t1 : float
    t2 : float
        Beginning and end of integration interval
    n_brute_force_δt : int, optional
        Number of evenly spaced δt values between (t1-t2) and (t2-t1) to sample
        for the initial guess.  By default, this is just the maximum number of
        time steps in the range (t1, t2) in the input waveforms.  If this is
        too small, an incorrect local minimum may be found.
    n_brute_force_δϕ : int, optional
        Number of evenly spaced δϕ values between 0 and 2π to sample
        for the initial guess.  By default, this is 2 * ell_max + 1.
    include_modes: list, optional
        A list containing the (ell, m) modes to be included in the L² norm.
    nprocs: int, optional
        Number of cpus to use. Default is maximum number.

    Returns
    -------
    optimum: OptimizeResult
        Result of scipy.optimize.least_squares
    h_A_prime: WaveformModes
        Resulting waveform after transforming `h_A` using `optimum`

    Notes
    -----
    Choosing the time interval is usually the most difficult choice to make when
    aligning waveforms.  Assuming you want to align during inspiral, the times
    must span sufficiently long that the waveforms' norm (equivalently, orbital
    frequency changes) significantly from `t1` to `t2`.  This means that you
    cannot always rely on a specific number of orbits, for example.  Also note
    that neither number should be too close to the beginning or end of either
    waveform, to provide some "wiggle room".

    Precession generally causes no problems for this function.  In principle,
    eccentricity, center-of-mass offsets, boosts, or other supertranslations could
    cause problems, but this function begins with a brute-force method of finding
    the optimal time offset that will avoid local minima in all but truly
    outrageous situations.  In particular, as long as `t1` and `t2` are separated
    by enough, there should never be a problem.

    """
    from scipy.optimize import least_squares

    import multiprocessing as mp

    h_A_copy = h_A.copy()
    h_B_copy = h_B.copy()

    # Check that (t1, t2) makes sense and is actually contained in both waveforms
    if t2 <= t1:
        raise ValueError(f"(t1,t2)=({t1}, {t2}) is out of order")
    if h_A_copy.t[0] > t1 or h_A_copy.t[-1] < t2:
        raise ValueError(f"(t1,t2)=({t1}, {t2}) not contained in h_A_copy.t, which spans ({h_A_copy.t[0]}, {h_A_copy.t[-1]})")
    if h_B_copy.t[0] > t1 or h_B_copy.t[-1] < t2:
        raise ValueError(f"(t1,t2)=({t1}, {t2}) not contained in h_B_copy.t, which spans ({h_B_copy.t[0]}, {h_B_copy.t[-1]})")

    # Figure out time offsets to try
    δt_lower = max(t1 - t2, h_A_copy.t[0] - t1)
    δt_upper = min(t2 - t1, h_A_copy.t[-1] - t2)

    # We'll start by brute forcing, sampling time offsets evenly at as many
    # points as there are time steps in (t1,t2) in the input waveforms
    if n_brute_force_δt is None:
        n_brute_force_δt = max(sum((h_A_copy.t>=t1)&(h_A_copy.t<=t2)), sum((h_B_copy.t>=t1)&(h_B_copy.t<=t2)))
    δt_brute_force = np.linspace(δt_lower, δt_upper, num=n_brute_force_δt)
    
    if n_brute_force_δϕ == None:
        n_brute_force_δϕ = 2*h_A_copy.ell_max + 1
    δϕ_brute_force = np.linspace(0, 2*np.pi, n_brute_force_δϕ, endpoint=False)
    
    δt_δϕ_brute_force = np.array(np.meshgrid(δt_brute_force, δϕ_brute_force)).T.reshape(-1,2)

    t_reference = h_B_copy.t[np.argmin(abs(h_B_copy.t - t1)):np.argmin(abs(h_B_copy.t - t2)) + 1]
            
    # Remove certain modes, if requested
    ell_max = min(h_A_copy.ell_max, h_B_copy.ell_max)
    if include_modes != None:
        for L in range(2, ell_max + 1):
            for M in range(-L, L + 1):
                if not (L, M) in include_modes: 
                    h_A_copy.data[:,LM_index(L, M, h_A_copy.ell_min)] *= 0
                    h_B_copy.data[:,LM_index(L, M, h_B_copy.ell_min)] *= 0
                    
    # Define the cost function
    modes_A = CubicSpline(h_A_copy.t, h_A_copy[:,2:ell_max + 1].data)
    modes_B = CubicSpline(h_B_copy.t, h_B_copy[:,2:ell_max + 1].data)(t_reference)
    
    normalization = trapezoid(CubicSpline(h_B_copy.t, h_B_copy[:,2:ell_max + 1].norm())(t_reference), t_reference)
    
    δϕ_factor = np.array([M for L in range(h_A_copy.ell_min, ell_max + 1) for M in range(-L, L + 1)])

    optimums = []
    h_A_primes = []
    for δΨ_factor in [-1, +1]:
        # Optimize by brute force with multiprocessing
        cost_wrapper = partial(cost, args = [modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization])

        if nprocs == None:
            nprocs = mp.cpu_count()
        nprocs = 4

        pool = mp.Pool(processes=nprocs)
        cost_brute_force = pool.map(cost_wrapper, δt_δϕ_brute_force)
        pool.close()
        pool.join()
    
        δt_δϕ = δt_δϕ_brute_force[np.argmin(cost_brute_force)]

        # Optimize explicitly
        optimum = least_squares(cost_wrapper, δt_δϕ, bounds=[(δt_lower, 0), (δt_upper, 2*np.pi)], max_nfev=50000)
        optimums.append(optimum)

        h_A_prime = h_A.copy()
        h_A_prime.t = h_A.t - optimum.x[0]
        h_A_prime.data = h_A[:,2:ell_max + 1].data * np.exp(1j * optimum.x[1]) ** δϕ_factor * δΨ_factor
        h_A_prime.ell_min = 2
        h_A_prime.ell_max = ell_max
        h_A_primes.append(h_A_prime)

    idx = np.argmin(abs(np.array([optimum.cost for optimum in optimums])))
        
    return optimums[idx].cost, h_A_primes[idx], optimums[idx]

def hybridize(h_NR, h_PN, t1, t2):
    from scipy.interpolate import CubicSpline

    h_NR_spline = CubicSpline(h_NR.t, h_NR.data)
    
    t_temp = np.array(np.append(h_PN.t[h_PN.t < t2], h_NR.t[h_NR.t > t2]))
    t_matching = h_PN.t[(h_PN.t >= t1) & (h_PN.t <= t2)]
    data_temp = np.empty((len(t_temp), len(h_NR.LM)), dtype=complex)

    smoothing_function = np.resize(scri.utilities.transition_function(t_matching, t1, t2), (len(h_PN.LM), len(t_matching))).T

    # hybridize data
    data_matching = (1 - smoothing_function) * h_PN.data[(h_PN.t >= t1) & (h_PN.t <= t2),:] + smoothing_function * h_NR_spline(t_matching)
    data_temp[t_temp < t1,:] = h_PN.data[h_PN.t < t1,:]
    data_temp[(t_temp >= t1) & (t_temp <= t2),:] = data_matching
    data_temp[t_temp > t2,:] = h_NR.data[h_NR.t > t2,:]

    # remove indices that cause tiny time step  
    minstep = min(min(np.diff(h_NR.t[(h_NR.t > t1) & (h_NR.t < t2)])), min(np.diff(h_PN.t[(h_PN.t > t1) & (h_PN.t < t2)])))
    bad_indices = np.nonzero(np.append(np.diff(t_temp) < minstep, 0) & (t_temp > t1) & (t_temp < t2))
    while len(bad_indices[0]) > 0:
        t_temp = np.delete(t_temp, bad_indices)
        data_temp = np.delete(data_temp, bad_indices, axis=0)
        bad_indices = np.nonzero(np.append(np.diff(t_temp) < minstep, 0) & (t_temp > t1) & (t_temp < t2))

    # construct hybrid waveform
    h_hybrid = scri.WaveformModes()
    h_hybrid.t = t_temp
    h_hybrid.data = data_temp
    h_hybrid.dataType = scri.h
    h_hybrid.frameType = scri.Inertial
    ell_min, ell_max = min(h_NR.LM[:, 0]), max(h_NR.LM[:, 0])
    h_hybrid.ells = ell_min, ell_max
    h_hybrid.r_is_scaled_out = True
    h_hybrid.m_is_scaled_out = True

    return h_hybrid

def time_translation(abd, t_0=0):
    """Time translate an abd object. This is necessary because creating a copy
    of an abd object and then changing it's time variable does not change
    the time variable of the waveform variables.
    """
    abd_prime = scri.asymptotic_bondi_data.AsymptoticBondiData(abd.t, abd.ell_max)

    abd_prime.sigma = abd.sigma
    abd_prime.psi4 = abd.psi4
    abd_prime.psi3 = abd.psi3
    abd_prime.psi2 = abd.psi2
    abd_prime.psi1 = abd.psi1
    abd_prime.psi0 = abd.psi0

    abd_prime.t -= t_0

    return abd_prime

def rotation(abd, phi=0):
    q = quaternion.from_rotation_vector(-phi*np.array([0,0,1]))

    h = MT_to_WM(2.0*abd.sigma.bar, False, scri.h)
    Psi4 = MT_to_WM(0.5*(-np.sqrt(2))**4*abd.psi4, False, scri.psi4)
    Psi3 = MT_to_WM(0.5*(-np.sqrt(2))**3*abd.psi3, False, scri.psi3)
    Psi2 = MT_to_WM(0.5*(-np.sqrt(2))**2*abd.psi2, False, scri.psi2)
    Psi1 = MT_to_WM(0.5*(-np.sqrt(2))**1*abd.psi1, False, scri.psi1)
    Psi0 = MT_to_WM(0.5*(-np.sqrt(2))**0*abd.psi0, False, scri.psi0)

    h.rotate_physical_system(q)
    Psi4.rotate_physical_system(q)
    Psi3.rotate_physical_system(q)
    Psi2.rotate_physical_system(q)
    Psi1.rotate_physical_system(q)
    Psi0.rotate_physical_system(q)

    abd_rot = abd.copy()
    abd_rot.sigma = 0.5*WM_to_MT(h).bar
    abd_rot.psi4 = 2*(-1./np.sqrt(2))**4*WM_to_MT(Psi4)
    abd_rot.psi3 = 2*(-1./np.sqrt(2))**3*WM_to_MT(Psi3)
    abd_rot.psi2 = 2*(-1./np.sqrt(2))**2*WM_to_MT(Psi2)
    abd_rot.psi1 = 2*(-1./np.sqrt(2))**1*WM_to_MT(Psi1)
    abd_rot.psi0 = 2*(-1./np.sqrt(2))**0*WM_to_MT(Psi0)

    return abd_rot

def PN_BMS_w_time_phase(abd, h_PN, PsiM_PN, t1, t2, include_modes, N=4):
    abd_prime = abd.copy()

    errors = []

    abds = []
    h_primes = []
    trans_and_convs = []
    for itr in range(N):
        abd_prime, trans_and_conv = abd_prime.map_to_superrest_frame(t_0=t1 + (t2 - t1)/2,
                                                                     target_h_input=h_PN,
                                                                     target_PsiM_input=PsiM_PN,
                                                                     padding_time=(t2 - t1)/2)
        
        h_CCE_PN_BMS = MT_to_WM(2.0*abd_prime.sigma.bar, False, scri.h)

        error, h_CCE_PN_BMS_prime, res = align2d(h_CCE_PN_BMS, h_PN, t1, t2, n_brute_force_δt=None, n_brute_force_δϕ=None, include_modes=include_modes, nprocs=None)
        
        abd_prime = time_translation(abd_prime, res.x[0])

        abd_prime = rotation(abd_prime, res.x[1])

        errors.append(error)
        abds.append(abd_prime)
        h_primes.append(h_CCE_PN_BMS_prime)
        trans_and_convs.append(trans_and_conv)

    idx = np.argmin(abs(np.array(errors)))

    if N == 1:
        return errors[idx], h_primes[idx], trans_and_convs[idx], idx, abds[idx]
    else:
        return errors[idx], h_primes[idx], None, idx, abds[idx]
