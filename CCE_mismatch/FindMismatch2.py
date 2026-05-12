import scri
import numpy as np
import spherical_functions as sf
import sxs
from spherical_functions import LM_index

from functools import partial

import scipy as sp


class SplineArray:
    def __init__(self, x, y):
        y = np.asarray(y)
        self.complex = np.iscomplexobj(y)
        if self.complex:
            self._real_spline = sp.interpolate.CubicSpline(x, y.real, axis=0)
            self._imag_spline = sp.interpolate.CubicSpline(x, y.imag, axis=0)
        else:
            self._spline = sp.interpolate.CubicSpline(x, y, axis=0)

    def __call__(self, xprime):
        if self.complex:
            return self._real_spline(xprime) + 1j * self._imag_spline(xprime)
        return self._spline(xprime)


def _time_slice(t, t1, t2):
    start = np.searchsorted(t, t1, side="left")
    stop = np.searchsorted(t, t2, side="right")
    return slice(start, stop)


def _reference_times(t, t1, t2):
    return t[_time_slice(t, t1, t2)]


def _mode_view(data, modes):
    if modes is None:
        return data
    return data[:, modes]


def _all_m_factors(ell_min, ell_max):
    return np.array(
        [M for L in range(ell_min, ell_max + 1) for M in range(-L, L + 1)],
        dtype=int,
    )


def _selected_mode_indices(ell_min, ell_max, include_modes):
    m_factors = _all_m_factors(ell_min, ell_max)
    if include_modes is None:
        return np.arange(m_factors.size, dtype=int), m_factors

    include_modes = set(include_modes)
    indices = [
        LM_index(L, M, ell_min)
        for L in range(ell_min, ell_max + 1)
        for M in range(-L, L + 1)
        if (L, M) in include_modes
    ]
    indices = np.asarray(indices, dtype=int)
    return indices, m_factors[indices]


def _normalization_from_modes(modes, t_reference):
    return sp.integrate.trapezoid(np.sum(np.abs(modes) ** 2, axis=1), t_reference)


def _cross_cost_components(modes_A, modes_B, t_reference):
    aa = sp.integrate.trapezoid(np.sum(np.abs(modes_A) ** 2, axis=1), t_reference)
    cross = sp.integrate.trapezoid(
        modes_A * np.conjugate(modes_B), t_reference, axis=0
    )
    return aa, cross


def _single_mode_overlap_components(mode_A, mode_B, t_reference):
    aa = sp.integrate.trapezoid(np.abs(mode_A) ** 2, t_reference)
    cross = sp.integrate.trapezoid(mode_A * np.conjugate(mode_B), t_reference)
    return aa, cross


def _brute_force_guess(modes_A, modes_B, t_reference, δt_grid, δϕ_grid, δϕ_factor, normalization):
    phase_grid = np.exp(1j * np.outer(δϕ_grid, δϕ_factor))
    best = {}
    for δΨ_factor in (-1, +1):
        best_value = np.inf
        best_params = None
        for δt in δt_grid:
            shifted_A = modes_A(t_reference + δt)
            aa, cross = _cross_cost_components(shifted_A, modes_B, t_reference)
            overlap = phase_grid @ cross
            diff = aa + normalization - 2.0 * np.real(δΨ_factor * overlap)
            diff = np.maximum(diff, 0.0)
            cost_values = np.sqrt(diff / normalization)
            idx = int(np.argmin(cost_values))
            if cost_values[idx] < best_value:
                best_value = float(cost_values[idx])
                best_params = np.array([δt, δϕ_grid[idx]], dtype=float)
        best[δΨ_factor] = (best_value, best_params)
    return best


def _brute_force_guess_22(mode_A, mode_B, t_reference, δt_grid, normalization):
    best_value = np.inf
    best_params = None
    for δt in δt_grid:
        shifted_A = mode_A(t_reference + δt)
        aa, cross = _single_mode_overlap_components(shifted_A, mode_B, t_reference)
        diff = max(aa + normalization - 2.0 * np.abs(cross), 0.0)
        value = np.sqrt(diff / normalization)
        if value < best_value:
            best_value = float(value)
            best_params = np.array([δt, _phase_from_overlap_22(cross)], dtype=float)
    return best_value, best_params


def _phase_from_overlap_22(cross):
    # The 22-only objective is invariant under δϕ -> δϕ + π; choose [0, π).
    return np.mod(-0.5 * np.angle(cross), np.pi)


def _validate_alignment_inputs(h_A, h_B, t1, t2):
    if t2 <= t1:
        raise ValueError(f"(t1,t2)=({t1}, {t2}) is out of order")
    if h_A.t[0] > t1 or h_A.t[-1] < t2:
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_A.t, which spans ({h_A.t[0]}, {h_A.t[-1]})"
        )
    if h_B.t[0] > t1 or h_B.t[-1] < t2:
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_B.t, which spans ({h_B.t[0]}, {h_B.t[-1]})"
        )


def _δt_bounds(h_A, t1, t2):
    return max(t1 - t2, h_A.t[0] - t1), min(t2 - t1, h_A.t[-1] - t2)


def _default_n_brute_force_δt(h_A, h_B, t1, t2):
    return max(
        np.count_nonzero((h_A.t >= t1) & (h_A.t <= t2)),
        np.count_nonzero((h_B.t >= t1) & (h_B.t <= t2)),
    )


def _aligned_waveform(h_A, ell_max, δt, δϕ):
    h_A_prime = h_A.copy()
    h_A_prime.t = h_A.t - δt
    all_δϕ_factor = _all_m_factors(2, ell_max)
    h_A_prime.data = (
        h_A[:, 2 : ell_max + 1].data * np.exp(1j * δϕ * all_δϕ_factor)
    )
    h_A_prime.ell_min = 2
    h_A_prime.ell_max = ell_max
    return h_A_prime


def SquaredError(W1, W2, t1, t2, modes=None, return_h1h2_h1h1=False, return_min_max_time=False):
    """
    Calculate the residue of W1 and W2 between t1 and t2.
    """
    W2_spline = SplineArray(W2.t, W2.data)
    t_slice = _time_slice(W1.t, t1, t2)
    filtered_time = W1.t[t_slice]
    W1_data = _mode_view(W1.data[t_slice, :], modes)
    W2_data = _mode_view(W2_spline(filtered_time), modes)

    diff_sq = np.abs(W2_data - W1_data) ** 2.0
    ref_sq = np.abs(W1_data) ** 2.0
    h1h2 = np.sum(sp.integrate.simpson(diff_sq, filtered_time, axis=0))
    h1h1 = np.sum(sp.integrate.simpson(ref_sq, filtered_time, axis=0))

    if return_h1h2_h1h1:
        if return_min_max_time:
            return h1h2, h1h1, filtered_time[0], filtered_time[-1]
        return h1h2, h1h1
    else:
        mismatch = 0.5 * h1h2 / h1h1
        if return_min_max_time:
            return mismatch, filtered_time[0], filtered_time[-1]
        return mismatch


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
    """Align waveforms by shifting in time and phase."""
    from scipy.optimize import least_squares

    _validate_alignment_inputs(h_A, h_B, t1, t2)

    δt_lower, δt_upper = _δt_bounds(h_A, t1, t2)

    if n_brute_force_δt is None:
        n_brute_force_δt = _default_n_brute_force_δt(h_A, h_B, t1, t2)
    δt_brute_force = np.linspace(δt_lower, δt_upper, num=n_brute_force_δt)

    if n_brute_force_δϕ is None:
        n_brute_force_δϕ = 2 * h_A.ell_max + 1
    δϕ_brute_force = np.linspace(0, 2 * np.pi, n_brute_force_δϕ, endpoint=False)

    t_reference = _reference_times(h_A.t, t1, t2)

    ell_max = min(h_A.ell_max, h_B.ell_max)
    all_δϕ_factor = _all_m_factors(2, ell_max)
    mode_indices, δϕ_factor = _selected_mode_indices(2, ell_max, include_modes)

    data_A = h_A[:, 2 : ell_max + 1].data[:, mode_indices]
    data_B = h_B[:, 2 : ell_max + 1].data[:, mode_indices]
    modes_A = sp.interpolate.CubicSpline(h_A.t, data_A, axis=0)
    modes_B = sp.interpolate.CubicSpline(h_B.t, data_B, axis=0)(t_reference)
    normalization = _normalization_from_modes(modes_B, t_reference)

    brute_force = _brute_force_guess(
        modes_A, modes_B, t_reference, δt_brute_force, δϕ_brute_force, δϕ_factor, normalization
    )

    optimums = []
    h_A_primes = []
    for δΨ_factor in (-1, +1):
        cost_wrapper = partial(
            cost,
            args=[modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization],
        )
        optimum = least_squares(
            cost_wrapper,
            brute_force[δΨ_factor][1],
            bounds=[(δt_lower, 0), (δt_upper, 2 * np.pi)],
            max_nfev=50000,
        )
        optimum.δΨ_factor = δΨ_factor
        optimums.append(optimum)

        h_A_prime = _aligned_waveform(h_A, ell_max, optimum.x[0], optimum.x[1])
        h_A_prime.data *= δΨ_factor
        h_A_prime.ell_min = 2
        h_A_prime.ell_max = ell_max
        h_A_primes.append(h_A_prime)

    idx = int(np.argmin(np.abs(np.array([optimum.cost for optimum in optimums]))))
    return optimums[idx].cost, h_A_primes[idx], optimums[idx]


def align2d_22(h_A, h_B, t1, t2, n_brute_force_δt=None, nprocs=None):
    """Align waveforms using only the (2,2) mode, then transform all returned modes."""
    del nprocs

    _validate_alignment_inputs(h_A, h_B, t1, t2)

    ell_max = min(h_A.ell_max, h_B.ell_max)
    if ell_max < 2:
        raise ValueError("The (2,2) mode is not available in both waveforms")

    δt_lower, δt_upper = _δt_bounds(h_A, t1, t2)
    if n_brute_force_δt is None:
        n_brute_force_δt = _default_n_brute_force_δt(h_A, h_B, t1, t2)
    δt_brute_force = np.linspace(δt_lower, δt_upper, num=n_brute_force_δt)

    t_reference = _reference_times(h_A.t, t1, t2)
    mode_22_index = LM_index(2, 2, 2)
    mode_A = sp.interpolate.CubicSpline(
        h_A.t, h_A[:, 2 : ell_max + 1].data[:, mode_22_index]
    )
    mode_B = sp.interpolate.CubicSpline(
        h_B.t, h_B[:, 2 : ell_max + 1].data[:, mode_22_index]
    )(t_reference)
    normalization = sp.integrate.trapezoid(np.abs(mode_B) ** 2, t_reference)

    if normalization <= 0.0:
        raise ValueError("The (2,2) reference mode has zero norm on the alignment window")

    _, brute_force_guess = _brute_force_guess_22(
        mode_A, mode_B, t_reference, δt_brute_force, normalization
    )

    def objective(δt):
        shifted_A = mode_A(t_reference + δt)
        aa, cross = _single_mode_overlap_components(shifted_A, mode_B, t_reference)
        diff = max(aa + normalization - 2.0 * np.abs(cross), 0.0)
        return np.sqrt(diff / normalization)

    optimum_dt = sp.optimize.minimize_scalar(
        objective,
        bounds=(δt_lower, δt_upper),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )

    δt_opt = float(optimum_dt.x)
    shifted_A = mode_A(t_reference + δt_opt)
    _, cross = _single_mode_overlap_components(shifted_A, mode_B, t_reference)
    δϕ_opt = _phase_from_overlap_22(cross)
    cost_opt = objective(δt_opt)

    if brute_force_guess is not None and objective(brute_force_guess[0]) < cost_opt:
        δt_opt = float(brute_force_guess[0])
        shifted_A = mode_A(t_reference + δt_opt)
        _, cross = _single_mode_overlap_components(shifted_A, mode_B, t_reference)
        δϕ_opt = _phase_from_overlap_22(cross)
        cost_opt = objective(δt_opt)

    optimum = sp.optimize.OptimizeResult(
        x=np.array([δt_opt, δϕ_opt], dtype=float),
        cost=float(cost_opt),
        fun=float(cost_opt),
        success=bool(optimum_dt.success),
        status=int(optimum_dt.status),
        message=optimum_dt.message,
        nfev=int(optimum_dt.nfev),
        nit=int(optimum_dt.nit),
    )
    h_A_prime = _aligned_waveform(h_A, ell_max, δt_opt, δϕ_opt)
    return optimum.cost, h_A_prime, optimum


def cost(δt_δϕ, args):
    modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization = args
    shifted_A = modes_A(t_reference + δt_δϕ[0])
    aa, cross = _cross_cost_components(shifted_A, modes_B, t_reference)
    phase = np.exp(1j * δt_δϕ[1] * δϕ_factor)
    diff = aa + normalization - 2.0 * np.real(δΨ_factor * np.sum(cross * phase))

    return np.sqrt(max(diff, 0.0) / normalization)


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

        errors.append(error)
        h_primes.append(h_CCE_PN_BMS_prime)
        trans_and_convs.append(trans)

    idx = np.argmin(np.abs(np.array(errors)))

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
