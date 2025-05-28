import scri
import numpy as np
import spherical_functions as sf
import sxs
from spherical_functions import LM_index

from functools import partial

import scipy as sp
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


class SplineArray:
    def __init__(self, x, y):
        self.complex = np.iscomplexobj(y)
        if self.complex:
            y = y.view(dtype=float)
        self.splines = [Spline(x, y[:, i]) for i in range(y.shape[1])]

    def __call__(self, xprime):
        yprime = np.concatenate(
            [spl(xprime)[:, np.newaxis] for spl in self.splines], axis=1
        )
        if self.complex:
            yprime = yprime.view(dtype=complex)
        return yprime


def SquaredError(W1, W2, t1, t2, modes=None, return_h1h2_h1h1=False):
    """
    Calculate the residue of W1 and W2 between t1 and t2.
    """
    W2_spline = SplineArray(W2.t, W2.data)
    t_filter = (W1.t >= t1) & (W1.t <= t2)
    filtered_time = W1.t[t_filter]

    if modes is None:
        h1h2 = np.sum(
            sp.integrate.simpson(
                abs(W2_spline(filtered_time) - W1.data[t_filter, :]) ** 2.0,
                filtered_time,
                axis=0,
            )
        )
        h1h1 = np.sum(
            sp.integrate.simpson(
                abs(W1.data[t_filter, :]) ** 2.0, filtered_time, axis=0
            )
        )
    else:
        h1h2 = np.sum(
            sp.integrate.simpson(
                abs(W2_spline(filtered_time) - W1.data[t_filter, :])[:, modes] ** 2.0,
                filtered_time,
                axis=0,
            )
        )
        h1h1 = np.sum(
            sp.integrate.simpson(
                abs(W1.data[t_filter, :][:, modes]) ** 2.0, filtered_time, axis=0
            )
        )
    if return_h1h2_h1h1:
        return h1h2 , h1h1
    else:
        return 0.5 * h1h2 / h1h1


def abd_to_WM(abd, lmin=2):
    W = scri.WaveformModes()
    W.t = abd.t
    W.data = 2 * abd.sigma.bar
    if len(W.data[0]) > 77 and lmin == 2:
        W.data = np.copy(W.data[:, 4:])
        W.ells = 2, 8
    else:
        W.ells = 0, 8
    W.frameType = scri.Inertial
    W.dataType = scri.h
    return W


def MT_to_WM(h_mts, sxs_version=False, dataType=scri.h):
    if not sxs_version:
        h = scri.WaveformModes(
            t=h_mts.t,
            data=np.array(h_mts, dtype=complex)[
                :, sf.LM_index(abs(h_mts.s), -abs(h_mts.s), 0) :
            ],
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            frameType=scri.Inertial,
            dataType=dataType,
        )
        h.r_is_scaled_out = True
        h.m_is_scaled_out = True
        return h
    else:
        h = sxs.WaveformModes(
            input_array=np.array(h_mts, dtype=complex)[
                :, sf.LM_index(abs(h_mts.s), -abs(h_mts.s), 0) :
            ],
            time=h_mts.t,
            time_axis=0,
            modes_axis=1,
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            spin_weight=h_mts.s,
        )
        return h


def align2d(
    h_A,
    h_B,
    t1,
    t2,
    n_brute_force_δt=None,
    n_brute_force_δϕ=None,
    include_modes=None,
    nprocs=None,
):
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
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_A_copy.t, which spans ({h_A_copy.t[0]}, {h_A_copy.t[-1]})"
        )
    if h_B_copy.t[0] > t1 or h_B_copy.t[-1] < t2:
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_B_copy.t, which spans ({h_B_copy.t[0]}, {h_B_copy.t[-1]})"
        )

    # Figure out time offsets to try
    δt_lower = max(t1 - t2, h_A_copy.t[0] - t1)
    δt_upper = min(t2 - t1, h_A_copy.t[-1] - t2)

    # We'll start by brute forcing, sampling time offsets evenly at as many
    # points as there are time steps in (t1,t2) in the input waveforms
    if n_brute_force_δt is None:
        n_brute_force_δt = max(
            sum((h_A_copy.t >= t1) & (h_A_copy.t <= t2)),
            sum((h_B_copy.t >= t1) & (h_B_copy.t <= t2)),
        )
    δt_brute_force = np.linspace(δt_lower, δt_upper, num=n_brute_force_δt)

    if n_brute_force_δϕ == None:
        n_brute_force_δϕ = 2 * h_A_copy.ell_max + 1
    δϕ_brute_force = np.linspace(0, 2 * np.pi, n_brute_force_δϕ, endpoint=False)

    δt_δϕ_brute_force = np.array(np.meshgrid(δt_brute_force, δϕ_brute_force)).T.reshape(
        -1, 2
    )

    t_reference = h_A_copy.t[
        np.argmin(abs(h_A_copy.t - t1)) : np.argmin(abs(h_A_copy.t - t2)) + 1
    ]

    # Remove certain modes, if requested
    ell_max = min(h_A_copy.ell_max, h_B_copy.ell_max)
    if include_modes != None:
        for L in range(2, ell_max + 1):
            for M in range(-L, L + 1):
                if not (L, M) in include_modes:
                    h_A_copy.data[:, LM_index(L, M, h_A_copy.ell_min)] *= 0
                    h_B_copy.data[:, LM_index(L, M, h_B_copy.ell_min)] *= 0

    # Define the cost function
    modes_A = sp.interpolate.CubicSpline(h_A_copy.t, h_A_copy[:, 2 : ell_max + 1].data)
    modes_B = sp.interpolate.CubicSpline(h_B_copy.t, h_B_copy[:, 2 : ell_max + 1].data)(
        t_reference
    )

    normalization = sp.integrate.trapezoid(
        sp.interpolate.CubicSpline(h_B_copy.t, h_B_copy[:, 2 : ell_max + 1].norm())(
            t_reference
        ),
        t_reference,
    )

    δϕ_factor = np.array(
        [M for L in range(h_A_copy.ell_min, ell_max + 1) for M in range(-L, L + 1)]
    )

    optimums = []
    h_A_primes = []
    for δΨ_factor in [-1, +1]:
        # Optimize by brute force with multiprocessing
        cost_wrapper = partial(
            cost,
            args=[modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization],
        )

        if nprocs == None:
            nprocs = mp.cpu_count()

        pool = mp.Pool(processes=nprocs)
        cost_brute_force = pool.map(cost_wrapper, δt_δϕ_brute_force)
        pool.close()
        pool.join()

        δt_δϕ = δt_δϕ_brute_force[np.argmin(cost_brute_force)]

        # Optimize explicitly
        optimum = least_squares(
            cost_wrapper,
            δt_δϕ,
            bounds=[(δt_lower, 0), (δt_upper, 2 * np.pi)],
            max_nfev=50000,
        )
        optimums.append(optimum)

        h_A_prime = h_A.copy()
        h_A_prime.t = h_A.t - optimum.x[0]
        h_A_prime.data = (
            h_A[:, 2 : ell_max + 1].data
            * np.exp(1j * optimum.x[1]) ** δϕ_factor
            * δΨ_factor
        )
        h_A_prime.ell_min = 2
        h_A_prime.ell_max = ell_max
        h_A_primes.append(h_A_prime)

    idx = np.argmin(abs(np.array([optimum.cost for optimum in optimums])))

    return optimums[idx].cost, h_A_primes[idx], optimums[idx]


def cost(δt_δϕ, args):
    modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization = args

    # Take the sqrt because least_squares squares the inputs...
    diff = sp.integrate.trapezoid(
        np.sum(
            abs(
                modes_A(t_reference + δt_δϕ[0])
                * np.exp(1j * δt_δϕ[1]) ** δϕ_factor
                * δΨ_factor
                - modes_B
            )
            ** 2,
            axis=1,
        ),
        t_reference,
    )
    return np.sqrt(diff / normalization)


def PN_BMS_w_time_phase(abd, h_PN, PsiM_PN, t1, t2, include_modes, N=4, write_dir=""):
    abd_prime = abd.copy()
    W_NR = scri.WaveformModes()
    W_NR.data = 2 * abd.sigma.bar
    W_NR.t = abd.t
    W_NR.ells = 0, 8
    W_NR.frameType = scri.Inertial
    W_NR.dataType = scri.h

    errors = []
    h_primes = []
    # abd_primes = []
    trans_and_convs = []
    for itr in range(N):
        print(f"Iteration {itr}")
        abd_prime, trans, abd_err = abd_prime.map_to_superrest_frame(
            t_0=t1 + (t2 - t1) / 2,
            target_strain_input=h_PN,
            target_PsiM_input=PsiM_PN,
            padding_time=(t2 - t1) / 2,
        )

        W_NR = MT_to_WM(2.0 * abd_prime.sigma.bar, False, scri.h)

        error, h_CCE_PN_BMS_prime, res = align2d(
            W_NR,
            h_PN,
            t1,
            t2,
            n_brute_force_δt=1000,
            n_brute_force_δϕ=20,
            include_modes=include_modes,
            nprocs=5,
        )
        print(f"{itr} error: {error}")

        # abd_prime = time_translation(abd_prime, res.x[0])

        # abd_prime = rotation(abd_prime, res.x[1])

        errors.append(error)
        h_primes.append(h_CCE_PN_BMS_prime)
        # abd_primes.append(abd_prime)
        trans_and_convs.append(trans)

    idx = np.argmin(abs(np.array(errors)))

    # write_abd_to_file(abd_primes[idx], write_dir)

    if N == 1:
        return errors[idx], h_primes[idx], trans_and_convs[idx], idx
    else:
        return errors[idx], h_primes[idx], trans_and_convs[idx], idx


def fix_BMS_NRNR(abd, abd2, hyb, PN):
    abd_prime, trans, abd_err = abd.map_to_superrest_frame(
        t_0=hyb.t_start + hyb.length / 2
    )
    W_NR = abd_to_WM(abd_prime)

    Psi_M = MT_to_WM(abd_prime.supermomentum("Moreschi"), False, dataType=scri.psi2)
    tp1, W_NR2, tp2, idx = PN_BMS_w_time_phase(
        abd2, W_NR, Psi_M, hyb.t_start, hyb.t_start + hyb.length, None
    )

    return W_NR, W_NR2


def fix_BMS_NRNR_t12(abd, abd2, t1, t2):
    abd_prime, trans, abd_err = abd.map_to_superrest_frame(t_0=(t1 + t2) / 2)
    W_NR = abd_to_WM(abd_prime)

    Psi_M = MT_to_WM(abd_prime.supermomentum("Moreschi"), False, dataType=scri.psi2)
    tp1, W_NR2, tp2, idx = PN_BMS_w_time_phase(abd2, W_NR, Psi_M, t1, t2, None)

    return W_NR, W_NR2
