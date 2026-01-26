"""
MIET-pFCS Analysis Pipeline with Membrane Fluctuation Model Fitting

This module provides comprehensive analysis of PTU files for MIET (Metal-Induced Energy Transfer)
combined with pseudo-FCS (fluorescence correlation spectroscopy) measurements.

Features:
- PTU file reading and TCSPC data processing
- MIET calibration curve generation
- Lifetime-based height determination
- Photobleaching correction
- Auto-correlation analysis for membrane fluctuation studies
- Multiple lifetime estimation methods (MLE, variance-based, intensity-weighted)
- Passive membrane bending model fitting using functions from Membrane_fluctuations.py

The pipeline processes time-tagged time-resolved (TTTR) photon data from PicoQuant PTU files
and correlates fluorescence lifetime changes with molecular height variations using MIET theory.

Enhanced functionality includes fitting height correlation functions with passive membrane
fluctuation models to extract biophysical parameters such as bending modulus (κ), membrane
tension (σ), and viscosity (η).

Key Functions:
- run_miet_ptu_pipeline(): Basic MIET-pFCS analysis
- enhanced_miet_ptu_pipeline(): Analysis with membrane model fitting
- fit_membrane_fluctuation_model(): Fit correlation data to membrane models
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Local MIET and PTU imports
# -----------------------------------------------------------------------------

# Import local MIET functions
from MIET_main import (
    MetalsDB, miet_calc, brightness_dipole
)

# Import PTU processing functions
from PTU_utils import (
    PTU_Read_Head, PTU_Read, mhist, harp_tcspc, tttr2xfcs,
    tttr2bin, tttr2bintcspc, mHist2_indices,
    _append_trace_with_merge, _append_tmptau_with_merge
)

# Import membrane fluctuation analysis functions
from Membrane_fluctuations import (
    fluctuation_coeff, active_profile, corr_model_unscaled,
    corr_model_normalized, fit_active_params, autocorr_fft_raw
)

# Load metals database
try:
    from pathlib import Path
    metals_path = Path(__file__).parent / "metals.mat"
    metals_db = MetalsDB(str(metals_path))
    
    # Get wavelength range and metal properties
    wavelengths_nm = np.arange(400, 800, 1)  # 1nm resolution from 400-800nm
    gold_indices = []
    titan_indices = []
    
    for wl in wavelengths_nm:
        gold_indices.append(metals_db.get_index(20, wl))  # 20 = gold code
        titan_indices.append(metals_db.get_index(22, wl))  # 22 = titanium code
    
    wavelength = wavelengths_nm
    gold = np.array(gold_indices)
    titan = np.array(titan_indices)
    
except Exception as e:
    print(f"Warning: Could not load metals database: {e}")
    print("Using fallback metal values...")
    # Fallback values at 690nm
    wavelength = np.array([690.0])
    gold = np.array([0.13 + 3.8j])  # Typical gold at 690nm
    titan = np.array([2.8 + 3.2j])  # Typical titanium at 690nm

# Set function aliases for compatibility
MIET_calc_fn = miet_calc
BrightnessDipole_fn = brightness_dipole


# -----------------------------------------------------------------------------
# Interpolation and binning utilities
# -----------------------------------------------------------------------------
def interp1_nan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation with NaN extrapolation outside bounds."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)

    # Ensure increasing x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.interp(xq, x, y, left=np.nan, right=np.nan)
    return out



def photobleach_fit_exp(normtrace: np.ndarray) -> np.ndarray:
    """
    Exponential photobleaching correction curve fitting.

    Fits normtrace to model: ztrace ~ A*exp(-t/tau) + C using least squares.
    Uses grid search over baseline parameter C for robust fitting.
    
    Parameters:
    normtrace: Normalized intensity trace
    
    Returns:
    ztrace: Photobleaching correction curve (same length as normtrace), normalized to mean ~1
    """
    y = np.asarray(normtrace, dtype=float)
    n = y.size
    x = np.arange(n, dtype=float)

    # Safety
    y = np.where(np.isfinite(y), y, np.nan)
    m = np.nanmedian(y)
    y = np.where(np.isnan(y), m, y)

    # Grid over baseline C (small fraction)
    c_grid = np.linspace(0.0, np.percentile(y, 30) * 0.9, 60)
    best = None
    best_sse = np.inf

    for c in c_grid:
        yy = y - c
        if np.any(yy <= 0):
            continue
        logy = np.log(yy)

        # Fit logy ~ a + b*x  => yy ~ exp(a)*exp(b*x)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
        a, b = coef
        yhat = np.exp(a + b * x) + c
        sse = float(np.mean((y - yhat) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (a, b, c)

    if best is None:
        # fallback: no correction
        return np.ones_like(y)

    a, b, c = best
    z = np.exp(a + b * x) + c
    # normalize curve so dividing doesn’t change mean too much
    z /= np.mean(z[z > 0])
    return z


def mseb_like(x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, label: str):
    """Plot mean curve with standard deviation shading."""
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    plt.semilogx(x, y_mean, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

def normalize_corr(y: np.ndarray, mode: str = "g0") -> np.ndarray:
    """
    Normalize a correlation curve.
    mode:
      - "g0": y / y[0]   (common for ACF overlays)
      - "end": y / y[-1] - 1  (your earlier style)
      - "none": no normalization
    """
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.full_like(y, np.nan)

    if mode == "g0":
        # use first finite value (more robust than y[0] if y[0] is NaN)
        i0 = int(np.argmax(finite))
        denom = y[i0]
        if denom == 0 or not np.isfinite(denom):
            return np.full_like(y, np.nan)
        return y / denom

    if mode == "end":
        # use last finite value
        i1 = int(np.where(finite)[0][-1])
        denom = y[i1]
        if denom == 0 or not np.isfinite(denom):
            return np.full_like(y, np.nan)
        return y / denom - 1.0

    return y


def plot_normalized_acf_with_fits(results: dict, *, norm_mode: str = "g0", title: str = "Normalized height ACFs + fits"):
    """
    Plot normalized correlation curves (int/var/MLE) and overlay normalized fit curves.
    Expects results from enhanced_miet_ptu_pipeline (i.e., includes 'membrane_fits').
    """
    tau_s = np.asarray(results["tau_s"], dtype=float)

    auto  = results.get("auto", None)
    auto2 = results.get("auto2", None)
    auto3 = results.get("auto3", None)

    auto_mean  = np.mean(auto,  axis=1) if auto  is not None else None
    auto2_mean = np.mean(auto2, axis=1) if auto2 is not None else None
    auto3_mean = np.mean(auto3, axis=1) if auto3 is not None else None

    plt.figure(figsize=(10, 6))

    # --- data curves (normalized) ---
    if auto3_mean is not None:
        plt.semilogx(tau_s, normalize_corr(auto3_mean, norm_mode), label="Data: MLE")
    if auto2_mean is not None:
        plt.semilogx(tau_s, normalize_corr(auto2_mean, norm_mode), label="Data: Variance")
    if auto_mean is not None:
        plt.semilogx(tau_s, normalize_corr(auto_mean, norm_mode), label="Data: Intensity")

    # --- overlay fits (normalized) ---
    fits = results.get("membrane_fits", {}) or {}
    # Map keys -> legend labels
    fit_label = {
        "mle_based": "Fit: MLE",
        "variance_based": "Fit: Variance",
        "intensity_based": "Fit: Intensity",
    }

    for key, fd in fits.items():
        if not isinstance(fd, dict):
            continue
        if not fd.get("success", False):
            continue
        tau_fit = np.asarray(fd.get("tau_fit", []), dtype=float)
        fit_curve = np.asarray(fd.get("fitted_curve", []), dtype=float)
        if tau_fit.size == 0 or fit_curve.size == 0:
            continue

        plt.semilogx(
            tau_fit,
            normalize_corr(fit_curve, norm_mode),
            "--",
            linewidth=2,
            label=f"{fit_label.get(key, 'Fit')}"
        )

    plt.xlabel("Time lag (s)")
    plt.ylabel(f"Normalized g(t) [{norm_mode}]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.title(title)
    plt.tight_layout()



# -----------------------------------------------------------------------------
# Main MIET-pFCS analysis pipeline
# -----------------------------------------------------------------------------
def run_miet_ptu_pipeline(
    ptu_path: Union[str, Path],
    *,
    # Dye properties
    tau0: float = 2.9,
    qy0: float = 0.6,
    tau1: float = 2.2,
    # Wavelengths in um
    lamex_um: float = 0.640,
    lamem_um: float = 0.690,
    # Optics
    NA: float = 1.49,
    # Geometry
    glass_n: float = 1.52,
    n1: float = 1.33,
    n: float = 1.33,
    top_n: float = 1.46,
    # layer thicknesses in um
    d0_um: Tuple[float, float, float, float] = (2e-3, 10e-3, 1e-3, 10e-3),
    d_um: float = 3e-1,
    d1_um: Tuple[float, ...] = (),
    # MIET calc
    curveType: int = 2,
    al_res: int = 100,
    # TTTR binning
    tbin_s: float = 1e-4,
    photons_per_chunk: int = int(5e6),
    cutoff_ns: float = 10.0,
    shift_ns: float = 0.3,
    micro_rebin: int = 8,
    # Correlation
    Ncasc: int = 10,
    Nsub: int = 6,
    nbunches: int = 10,
    # Debug / sanity checks
    sanity_checks: bool = True,
    sanity_max_chunks: Optional[int] = None,  # e.g. 20 to stop after 20 chunks
    sanity_print_every: int = 1,              # print every N chunks
    sanity_assert: bool = False,              # raise on suspicious conditions
):
    """
    End-to-end MIET-pFCS pipeline with robust chunk-boundary handling.

    Key fixes vs common "never stops" / huge arrays issues:
    - Uses PTU_Read's `loc` to avoid skipping overflow markers at chunk end:
        cnts += (num - loc)
    - Uses block-local binning and merges boundary bins.
    - Adds sanity checks that detect:
        * cnts not increasing
        * unexpected large macro bins
        * weird sync monotonicity
        * runaway tmptau size
    """
    ptu_path = Path(ptu_path)

    # ---- helpers: merge boundary macro-bin across chunks ----
    # These functions are now imported from PTU_utils

    def _dbg(msg: str) -> None:
        if sanity_checks:
            print(msg)

    def _maybe_assert(cond: bool, msg: str) -> None:
        if cond:
            return
        if sanity_assert:
            raise RuntimeError(msg)
        _dbg(f"[SANITY WARNING] {msg}")
            
    def mle_grid_argmin_blockwise(tmptau_block, logpmf, grid_block=512, dtype=np.float32):
        """
        tmptau_block: (T, M) int/float
        logpmf:       (T, G) float
        returns:
          best_idx: (M,) int
        """
        # Work in float32 for speed/memory (usually fine for argmin)
        X = tmptau_block.T.astype(dtype, copy=False)      # (M, T)
        L = logpmf.astype(dtype, copy=False)              # (T, G)
    
        M = X.shape[0]
        G = L.shape[1]
    
        best_val = np.full(M, np.inf, dtype=dtype)
        best_idx = np.zeros(M, dtype=np.int32)
    
        for j0 in range(0, G, grid_block):
            j1 = min(j0 + grid_block, G)
    
            # (M, T) @ (T, gb) -> (M, gb)  but gb is small now
            s = -(X @ L[:, j0:j1])
    
            # per-row min inside this block
            local_idx = np.argmin(s, axis=1)
            local_val = s[np.arange(M), local_idx]
    
            # update global best
            better = local_val < best_val
            best_val[better] = local_val[better]
            best_idx[better] = (j0 + local_idx[better]).astype(np.int32)
    
        return best_idx
    
         

    # ---- quantum yield update ----
    qy = qy0 * tau1 / tau0
    tau_free = tau1

    # ---- MIET calibration curves ----
    lamem_nm = lamem_um * 1e3
    idx = int(np.argmin(np.abs(wavelength - lamem_nm)))
    n_ti = titan[idx]
    n_au = gold[idx]

    n0 = np.array([glass_n, n_ti, n_au, n_ti, top_n], dtype=complex)
    d0_nm = np.array(d0_um, dtype=float) * 1e3
    d1_nm = np.array(d1_um, dtype=float) * 1e3
    d_nm = float(d_um * 1e3)

    z_nm, lifecurve = MIET_calc_fn(
        al_res,
        lamem_nm,
        n0,
        n,
        n1,
        d0_nm,
        d_nm,
        d1_nm,
        qy,
        tau_free,
        1,
        curveType,
    )
    z_nm = np.asarray(z_nm, dtype=float).ravel()
    lifecurve = np.asarray(lifecurve, dtype=float).ravel()

    ok = np.isfinite(z_nm) & np.isfinite(lifecurve)
    z_nm = z_nm[ok]
    lifecurve = lifecurve[ok]
    if z_nm.size == 0:
        raise RuntimeError("MIET calibration curve is empty after NaN filtering.")
    maxz_life = int(np.nanargmax(lifecurve))

    # ---- brightness curve (dipole) ----
    fac = 2 * np.pi / lamem_um
    zfac = (z_nm * 1e-3) * fac
    d0fac = (np.array(d0_um, dtype=float) * fac)
    dfac = float(d_um * fac)
    d1fac = (np.array(d1_um, dtype=float) * fac) if len(d1_um) else np.array([])

    br_out = BrightnessDipole_fn(zfac, n0, n, n1, d0fac, dfac, d1fac, NA, qy, "false")
    br = np.asarray(br_out[-1] if isinstance(br_out, tuple) else br_out, dtype=float).ravel()
    br = interp1_nan(z_nm[np.isfinite(br)], br[np.isfinite(br)], z_nm)

    # ---- read PTU: get TCSPC peak position ----
    harp = harp_tcspc(ptu_path)
    tcspc_full = harp.tcspcdata
    if tcspc_full is None or tcspc_full.size == 0:
        raise RuntimeError("harp_tcspc returned empty TCSPC data.")
    pos = int(np.argmax(np.sum(tcspc_full, axis=1)))

    # ---- timing from PTU header ----
    head = PTU_Read_Head(str(ptu_path))
    SyncRate = float(head.TTResult_SyncRate)
    Resolution_s = float(head.MeasDesc_Resolution)

    Ngate_full = int(np.ceil((1.0 / SyncRate) / Resolution_s))
    bin_tcspc = np.arange(Ngate_full + 1, dtype=np.int64)

    macrobin_ticks = float(tbin_s * SyncRate)

    shift_bins = int(np.round((shift_ns * 1e-9) / Resolution_s))
    ind1 = pos + shift_bins
    cutoff_bins_native = int(np.ceil((cutoff_ns * 1e-9) / Resolution_s))
    ind2 = ind1 + cutoff_bins_native

    n_micro_bins = int(np.round(cutoff_bins_native / float(micro_rebin)))
    if n_micro_bins <= 0:
        raise ValueError("n_micro_bins <= 0; check cutoff_ns, Resolution_s, micro_rebin.")

    # ---- streaming TTTR processing ----
    cnts = 0                      # record index, advanced by (num - loc)
    flag = True
    k = 0
    last_cnts = -1

    ttrace_parts: list[np.ndarray] = []
    tmptau_parts: list[np.ndarray] = []
    tmptcspc = np.zeros(Ngate_full, dtype=np.int64)

    # sanity trackers
    total_photons_kept = 0
    total_raw_records = 0
    max_tau_macro_seen = 0
    max_trace_bins_seen = 0

    while flag:
        k += 1
        if sanity_max_chunks is not None and k > sanity_max_chunks:
            _dbg(f"[SANITY STOP] Reached sanity_max_chunks={sanity_max_chunks}. Breaking.")
            break

        sync, tcspc, chan, special, num, loc, _ = PTU_Read(
            str(ptu_path),
            [cnts + 1, photons_per_chunk],  # 1-based start
            head,
        )

        num = int(num)
        loc = int(loc)
        total_raw_records += num

        # --- EOF / termination sanity ---
        flag = (num > 0)
        if not flag:
            _dbg(f"[EOF] num=0 at chunk {k}. Done.")
            break

        # IMPORTANT: advance by (num - loc), not num
        cnts_next = cnts + (num - loc)

        if sanity_checks and (k % sanity_print_every == 0):
            _dbg(
                f"[CHUNK {k}] cnts={cnts} -> {cnts_next} | num={num} loc={loc} "
                f"| raw_len={len(sync)}"
            )

        _maybe_assert(cnts_next > cnts, f"cnts did not advance (cnts={cnts}, next={cnts_next}).")
        if cnts_next <= cnts:
            # avoid infinite loop
            _dbg("[SANITY STOP] cnts did not advance; breaking to avoid infinite loop.")
            break
        cnts = cnts_next

        # Convert
        sync = np.asarray(sync, dtype=np.int64)
        tcspc = np.asarray(tcspc, dtype=np.int64)
        special = np.asarray(special, dtype=np.int64)

        if sync.size == 0:
            _dbg(f"[CHUNK {k}] sync empty after read; continuing.")
            continue

        # Keep photons only (special==0)
        keep_ph = (special == 0)
        n_before = int(sync.size)
        sync = sync[keep_ph]
        tcspc = tcspc[keep_ph]
        n_after = int(sync.size)
        total_photons_kept += n_after

        if sanity_checks and (k % sanity_print_every == 0):
            _dbg(f"          photons kept: {n_after}/{n_before}")

        if n_after == 0:
            continue

        # sync monotonic sanity (should be nondecreasing typically)
        if sanity_checks and n_after > 1:
            n_neg = int(np.sum(np.diff(sync) < 0))
            if n_neg > 0:
                _maybe_assert(False, f"sync decreased within chunk ({n_neg} negative diffs). Overflow handling may be wrong.")

        # Accumulate global TCSPC histogram
        tmptcspc += mhist(tcspc, np.arange(Ngate_full, dtype=np.int64))

        # Gate microtimes
        gate = (tcspc > ind1) & (tcspc <= ind2)
        tcspc_shifted = (tcspc - ind1) * gate.astype(np.int64)

        # Intensity trace
        t_block = tttr2bin(sync, macrobin_ticks, rebase=True)
        max_trace_bins_seen = max(max_trace_bins_seen, int(t_block.size))
        _append_trace_with_merge(ttrace_parts, t_block)

        # Lifetime 2D trace (micro, macro)
        tau_block = tttr2bintcspc(
            sync,
            (macrobin_ticks, micro_rebin),
            tcspc_shifted,
            Ngate=cutoff_bins_native,
        )
        max_tau_macro_seen = max(max_tau_macro_seen, int(tau_block.shape[1]))
        _append_tmptau_with_merge(tmptau_parts, tau_block)

        # runaway-size warnings (these catch “MLE matrix multiply never ends”)
        if sanity_checks and (k % sanity_print_every == 0):
            _dbg(
                f"          t_block bins={t_block.size}, tau_block shape={tau_block.shape}, "
                f"max_tau_macro_seen={max_tau_macro_seen}"
            )

        if sanity_checks:
            # If a single block produces a ridiculous number of macro bins, it’s almost always a sync/overflow problem.
            if tau_block.shape[1] > 5_000_000:
                _maybe_assert(False, f"tau_block has {tau_block.shape[1]} macro bins (runaway). Likely overflow/cnts/loc bug.")
                _dbg("[SANITY STOP] runaway tau_block macro bins; breaking.")
                break

    # ---- assemble traces ----
    ttrace = np.concatenate(ttrace_parts, axis=0) if ttrace_parts else np.zeros((0,), dtype=np.int64)
    tmptau = np.concatenate(tmptau_parts, axis=1) if tmptau_parts else np.zeros((n_micro_bins, 0), dtype=np.int64)

    if sanity_checks:
        _dbg(
            f"[SUMMARY] chunks={k}, total_raw_records={total_raw_records}, "
            f"total_photons_kept={total_photons_kept}, "
            f"ttrace_len={ttrace.size}, tmptau_shape={tmptau.shape}"
        )

    # Ensure tmptau has expected micro dimension (avoid off-by-1 due to rounding)
    if tmptau.shape[0] != n_micro_bins:
        if tmptau.shape[0] > n_micro_bins:
            tmptau = tmptau[:n_micro_bins, :]
        else:
            pad = np.zeros((n_micro_bins - tmptau.shape[0], tmptau.shape[1]), dtype=tmptau.dtype)
            tmptau = np.vstack([tmptau, pad])

    # ---- photobleaching correction ----
    if ttrace.size == 0:
        raise RuntimeError("ttrace is empty; check PTU_Read and special filtering.")
    meantrace = float(np.mean(ttrace[ttrace > 0])) if np.any(ttrace > 0) else float(np.mean(ttrace))
    normtrace = ttrace / meantrace
    ztrace = photobleach_fit_exp(normtrace)
    normtrace = normtrace / ztrace

    # ---- lifetime trace -> height trace ----
    delay = np.arange(tmptau.shape[0], dtype=float) * Resolution_s * 1e9 * micro_rebin
    
    grid_lts = np.linspace(0.1, 3.0, 200)
    grid_bs  = np.linspace(0.0, 0.2, 60)
    
    ltmat, bmat = np.meshgrid(grid_lts, grid_bs, indexing="xy")
    ltvec = ltmat.reshape(-1, order="F")
    bvec  = bmat.reshape(-1, order="F")
    
    delay_col = delay[:, None]
    exp_part = np.exp(-delay_col / ltvec[None, :])
    exp_part /= np.sum(exp_part, axis=0, keepdims=True)
    pmf = (bvec[None, :] / delay.size) + (1.0 - bvec[None, :]) * exp_part
    logpmf = np.log(np.clip(pmf, 1e-300, None))


    n_macro = tmptau.shape[1]
    grid_ind = np.zeros(n_macro, dtype=int)
    nbunch = 100
    edges = np.linspace(0, n_macro, nbunch + 1).astype(int)

    if sanity_checks:
        _dbg(f"[MLE] tmptau micro={tmptau.shape[0]} macro={tmptau.shape[1]} grid={logpmf.shape[1]}")

    for kk in range(nbunch):
        a, b = edges[kk], edges[kk + 1]
        if b <= a:
            continue
    
        grid_ind[a:b] = mle_grid_argmin_blockwise(
            tmptau[:, a:b],
            logpmf,
            grid_block=512,   # try 256/512/1024
            dtype=np.float32
        )
        print(kk)

    gridMLE_tau = ltvec[grid_ind]
    htrace3 = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], gridMLE_tau)

    counts = tmptau.astype(float)
    den = np.sum(counts, axis=0)
    den = np.where(den == 0, np.nan, den)
    E1 = np.sum(counts * delay[:, None], axis=0) / den
    E2 = np.sum(counts * (delay[:, None]**2), axis=0) / den
    meantau = np.sqrt(np.maximum(E2 - E1**2, 0.0))

    htrace2 = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], meantau)

    # ---- height trace from intensity fluctuations ----
    pos2 = int(np.argmax(tmptcspc))
    ind1b = max(int(pos2 + shift_bins), 0)

    t_ns = bin_tcspc[ind1b:Ngate_full] * Resolution_s * 1e9
    y_tail = tmptcspc[ind1b:Ngate_full].astype(float)

    def _fit_biexp(t, y):
        t1_grid = np.linspace(0.2, 5.0, 60)
        t2_grid = np.linspace(0.2, 5.0, 60)
        best = None
        best_sse = np.inf
        for t1 in t1_grid:
            e1 = np.exp(-t / t1)
            for t2 in t2_grid:
                e2 = np.exp(-t / t2)
                A = np.vstack([e1, e2, np.ones_like(t)]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                yhat = A @ coef
                sse = float(np.mean((y - yhat) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best = (t1, t2, coef)
        if best is None:
            return 1.0, 2.0, 1.0, 0.0, float(np.median(y))
        t1, t2, coef = best
        a1, a2, c = coef
        return float(t1), float(t2), float(a1), float(a2), float(c)

    t1, t2, a1, a2, c_bg = _fit_biexp(t_ns, y_tail)
    amps = np.clip(np.array([a1, a2], dtype=float), 0.0, None)
    taus = np.clip(np.array([t1, t2], dtype=float), 1e-12, None)
    tau_avg = float(np.sum(amps) / np.sum(amps / taus)) if np.sum(amps) > 0 else float(np.mean(taus))

    z_avg = float(
        interp1_nan(
            lifecurve[np.isfinite(lifecurve)],
            z_nm[np.isfinite(lifecurve)],
            np.array([tau_avg]),
        )[0]
    )

    iz = int(np.argmin(np.abs(z_nm - z_avg)))
    br_fac = float(br[iz]) if np.isfinite(br[iz]) else float(np.nanmean(br))
    norm_br = br / br_fac

    maxz_br = int(np.nanargmax(norm_br))
    htrace = interp1_nan(norm_br[: maxz_br + 1], z_nm[: maxz_br + 1], normtrace)
    bhmean = meantrace * br_fac

    # ---- correlate height trajectory ----
    mask1 = np.isfinite(htrace)
    tpoints = np.where(mask1)[0]
    h1 = htrace[mask1]
    i1 = ttrace[mask1]

    mask2 = np.isfinite(htrace2)
    tpoints2 = np.where(mask2)[0]
    h2 = htrace2[mask2]

    mask3 = np.isfinite(htrace3)
    tpoints3 = np.where(mask3)[0]
    h3 = htrace3[mask3]

    def _bunch_edges(L, nb):
        return np.round(np.linspace(0, L, nb + 1)).astype(int)

    e1 = _bunch_edges(len(tpoints), nbunches)
    e2 = _bunch_edges(len(tpoints2), nbunches)
    e3 = _bunch_edges(len(tpoints3), nbunches)

    auto = auto2 = auto3 = autoi = None
    autotime = None

    for kk in range(nbunches):
        a, b = e1[kk], e1[kk + 1]
        a2, b2 = e2[kk], e2[kk + 1]
        a3, b3 = e3[kk], e3[kk + 1]

        tmpauto, autotime = tttr2xfcs(tpoints[a:b], h1[a:b], Ncasc, Nsub)
        tmpauto = tmpauto.squeeze()
        if auto is None:
            auto = np.zeros((tmpauto.size, nbunches), dtype=float)
        auto[:, kk] = tmpauto

        tmpauto2, _ = tttr2xfcs(tpoints2[a2:b2], h2[a2:b2], Ncasc, Nsub)
        tmpauto2 = tmpauto2.squeeze()
        if auto2 is None:
            auto2 = np.zeros((tmpauto2.size, nbunches), dtype=float)
        auto2[:, kk] = tmpauto2

        tmpauto3, _ = tttr2xfcs(tpoints3[a3:b3], h3[a3:b3], Ncasc, Nsub)
        tmpauto3 = tmpauto3.squeeze()
        if auto3 is None:
            auto3 = np.zeros((tmpauto3.size, nbunches), dtype=float)
        auto3[:, kk] = tmpauto3

        tmpautoi, _ = tttr2xfcs(tpoints[a:b], i1[a:b], Ncasc, Nsub)
        tmpautoi = tmpautoi.squeeze()
        if autoi is None:
            autoi = np.zeros((tmpautoi.size, nbunches), dtype=float)
        autoi[:, kk] = tmpautoi

    autotime = autotime.squeeze().astype(float)
    tau_s = autotime * tbin_s

    def _norm_end(x):
        return x / x[-1:, :] - 1.0

    auto_n = _norm_end(auto)
    auto2_n = _norm_end(auto2)
    auto3_n = _norm_end(auto3)
    autoi_n = _norm_end(autoi)

    # ---- plots ----
    x = tau_s[:-1]
    plt.figure()
    mseb_like(x, np.mean(auto3_n[:-1, :], axis=1), np.std(auto3_n[:-1, :], axis=1), label="dACF_MLE")
    mseb_like(x, np.mean(auto2_n[:-1, :], axis=1), np.std(auto2_n[:-1, :], axis=1), label="dACF_var")
    mseb_like(x, np.mean(auto_n[:-1, :], axis=1), np.std(auto_n[:-1, :], axis=1), label="dACF_b(h)")
    mseb_like(
        x,
        np.mean(autoi_n[:-1, :], axis=1) * bhmean,
        np.std(autoi_n[:-1, :], axis=1) * bhmean,
        label="ACF (scaled)",
    )
    plt.xlabel("t / (s)")
    plt.ylabel("g(t)")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.figure()
    plt.semilogx(tau_s, np.mean(auto3 / auto3[0:1, :], axis=1), label="dACF_MLE")
    plt.semilogx(tau_s, np.mean(auto2 / auto2[0:1, :], axis=1), label="dACF_var")
    plt.semilogx(tau_s, np.mean(auto / auto[0:1, :], axis=1), label="dACF_b(h)")
    plt.semilogx(tau_s, np.mean(autoi / autoi[0:1, :], axis=1), label="ACF")
    plt.xlabel("t / (s)")
    plt.ylabel("g(t) / g(0)")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()

    return {
        "z_nm": z_nm,
        "lifecurve": lifecurve,
        "br": br,
        "ttrace": ttrace,
        "normtrace": normtrace,
        "tmptcspc": tmptcspc,
        "tmptau": tmptau,
        "htrace_int": htrace,
        "htrace_var": htrace2,
        "htrace_mle": htrace3,
        "auto": auto,
        "auto2": auto2,
        "auto3": auto3,
        "autoi": autoi,
        "autotime": autotime,
        "tau_s": tau_s,
        "z_avg_nm": z_avg,
        "tau_avg_ns": tau_avg,
        "sanity": {
            "chunks": k,
            "total_raw_records": total_raw_records,
            "total_photons_kept": total_photons_kept,
            "max_t_block_bins": max_trace_bins_seen,
            "max_tau_block_macro_bins": max_tau_macro_seen,
            "final_cnts": cnts,
        },
    }



def fit_membrane_fluctuation_model(
    tau_s: np.ndarray,
    height_acf: np.ndarray,
    *,
    initial_params: Optional[dict] = None,
    fit_method: str = "leastsq"
) -> dict:
    """
    Fit height correlation function with passive membrane bending model.
    
    This function uses the membrane fluctuation analysis from Membrane_fluctuations.py
    to fit the height auto-correlation data obtained from MIET-pFCS measurements.
    
    Parameters:
    -----------
    tau_s : np.ndarray
        Time lag array in seconds
    height_acf : np.ndarray
        Height auto-correlation function values
    initial_params : dict, optional
        Initial parameter guess dictionary with keys:
        - 'kappa': bending modulus (default: 10.0 kT)
        - 'sigma': membrane tension (default: 1e-4 N/m)
        - 'eta': viscosity (default: 1e-3 Pa·s)
        - 'amplitude': correlation amplitude (default: 1.0)
    fit_method : str
        Fitting method - 'leastsq' or 'minimize'
        
    Returns:
    --------
    dict : Fitted parameters and goodness of fit metrics
        - 'params': fitted parameter values
        - 'fitted_curve': fitted correlation function
        - 'residuals': fitting residuals
        - 'r_squared': coefficient of determination
        - 'chi_squared': chi-squared statistic
    """
    
    # Set default initial parameters if not provided
    if initial_params is None:
        initial_params = {
            'kappa': 20.0,      # bending modulus in kT
            'sigma': 1e-4,      # membrane tension in N/m
            'eta': 1.2e-3,        # viscosity in Pa·s
            'amplitude': 1.0    # correlation amplitude
        }
    
    # Ensure finite values only
    finite_mask = np.isfinite(tau_s) & np.isfinite(height_acf)
    tau_fit = tau_s[finite_mask]
    acf_fit = height_acf[finite_mask]
    
    if len(tau_fit) < 4:
        raise ValueError("Insufficient finite data points for fitting")
    
    # Use the normalized correlation model from Membrane_fluctuations.py
    def model_func(params):
        """Model function for fitting"""
        try:
            # Get the correlation model (already normalized)
            model_acf = corr_model_normalized(
                tau_fit,
                params['kappa'],
                params['sigma'],
                params['eta']
            )
            return params['amplitude'] * model_acf
        except Exception:
            # Return large residuals if model fails
            return np.full_like(tau_fit, np.inf)
    
    def residuals_func(param_array):
        """Calculate residuals for optimization"""
        params = {
            'kappa': param_array[0],
            'sigma': param_array[1],
            'eta': param_array[2],
            'amplitude': param_array[3]
        }
        model_values = model_func(params)
        return acf_fit - model_values
    
    # Initial parameter array
    p0 = [
        initial_params['kappa'],
        initial_params['sigma'],
        initial_params['eta'],
        initial_params['amplitude']
    ]
    
    # Parameter bounds (physical constraints)
    bounds = [
        (0.1, 100.0),     # kappa: 0.1 to 1000 kT
        (1e-6, 1e-2),      # sigma: 1 μN/m to 10 mN/m
        (1e-5, 1.0),       # eta: 0.01 to 1000 mPa·s
        (0.01, 100.0)      # amplitude: positive scaling
    ]
    
    try:
        if fit_method == "leastsq":
            from scipy.optimize import least_squares
            
            # Use least_squares with bounds
            result = least_squares(
                residuals_func,
                p0,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                method='trf'
            )
            fitted_params = result.x
            success = result.success
            
        else:  # minimize method
            from scipy.optimize import minimize
            
            def objective(param_array):
                residuals = residuals_func(param_array)
                return np.sum(residuals**2)
            
            result = minimize(
                objective,
                p0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            fitted_params = result.x
            success = result.success
            
    except ImportError:
        # Fallback to simple grid search if scipy not available
        print("Warning: scipy not available, using simple grid search")
        success = False
        fitted_params = p0
    
    if not success:
        print("Warning: Fitting did not converge, using initial parameters")
        fitted_params = p0
    
    # Calculate fitted curve and metrics
    final_params = {
        'kappa': fitted_params[0],
        'sigma': fitted_params[1],
        'eta': fitted_params[2],
        'amplitude': fitted_params[3]
    }
    
    fitted_curve = model_func(final_params)
    residuals = acf_fit - fitted_curve
    
    # Calculate goodness of fit metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((acf_fit - np.mean(acf_fit))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Chi-squared (assuming Poisson statistics)
    chi_squared = np.sum(residuals**2 / np.abs(fitted_curve + 1e-12))
    
    return {
        'params': final_params,
        'fitted_curve': fitted_curve,
        'residuals': residuals,
        'r_squared': r_squared,
        'chi_squared': chi_squared,
        'tau_fit': tau_fit,
        'acf_fit': acf_fit,
        'success': success
    }


def fit_membrane_minimal_model(
    tau_s: np.ndarray,
    height_acf: np.ndarray,
    z_avg: float,
    *,
    fixed_kappa: float = 20.0,  # Fixed bending modulus in kT
    fixed_eta: float = 1.2e-3,  # Fixed viscosity in Pa·s
    beam_waist: float = 280e-9,  # Beam waist radius in m
    initial_params: Optional[dict] = None,
    show_plot: bool = True
) -> dict:
    """
    Minimal membrane fluctuation fitting with fixed bending modulus and viscosity.
    
    This function follows the approach from the original HeightCorrFit where:
    - Bending modulus κ is fixed at 20 kT
    - Viscosity η is fixed at 1.2 mPa·s
    - Only membrane tension σ and curvature interaction γ are fitted
    
    Parameters:
    -----------
    tau_s : np.ndarray
        Time lag array in seconds
    height_acf : np.ndarray
        Height auto-correlation function values
    z_avg : float
        Average height in meters
    fixed_kappa : float
        Fixed bending modulus in kT (default: 20.0)
    fixed_eta : float
        Fixed viscosity in Pa·s (default: 1.2e-3)
    beam_waist : float
        Beam waist radius in meters (default: 280e-9)
    initial_params : dict, optional
        Initial parameter guess dictionary with keys:
        - 'sigma': membrane tension (default: 1e-3 N/m)
        - 'gamma': curvature interaction potential (default: 1e-6 J/m⁴)
    show_plot : bool
        Whether to display fitting plots
        
    Returns:
    --------
    dict : Fitted parameters and goodness of fit metrics
    """
    
    # Convert constants to SI units
    kBT = 4.11e-21  # Boltzmann constant × temperature in J
    kappa_J = fixed_kappa * kBT  # Bending modulus in J
    eta = fixed_eta  # Viscosity in Pa·s
    w = beam_waist  # Beam waist radius in m
    
    # Set default initial parameters if not provided
    if initial_params is None:
        initial_params = {
            'sigma': 1e-3,    # membrane tension in N/m
            'gamma': 1e-6     # curvature interaction in J/m⁴
        }
    
    # Ensure finite values only
    finite_mask = np.isfinite(tau_s) & np.isfinite(height_acf)
    tau_fit = tau_s[finite_mask]
    acf_fit = height_acf[finite_mask]
    
    if len(tau_fit) < 3:
        raise ValueError("Insufficient finite data points for fitting")
    
    def model_func(sigma, gamma):
        """
        Calculate theoretical correlation function following the original implementation
        
        Model: sum over q-space with exponential decay and fluctuation coefficient
        """
        try:
            # Calculate q limits
            q0 = (gamma / kappa_J)**(1/4)  # Lower q limit
            q_max = 1.0 / z_avg  # Upper q limit based on average height
            
            # Create q vector (1000 points between q0 and q_max)
            q = np.linspace(max(q0, 1e5), min(q_max, 1e7), 1000)
            
            # Calculate fluctuation coefficient T(q)
            # Note: Using fluctuation_coeff from Membrane_fluctuations.py
            T = fluctuation_coeff(q, z_avg, eta, sigma, gamma)
            
            # Calculate model correlation at each time point
            model_corr = np.zeros_like(tau_fit)
            
            for i, t in enumerate(tau_fit):
                # Exponential decay with beam waist correction
                f1 = np.exp(-t * (T - 0.25 * w**2 * q**2))
                
                # Denominator: energy terms
                f2 = kappa_J * q**4 + sigma * q**2 + gamma
                
                # Integrate over q-space (trapezoidal rule)
                integrand = f1 / f2
                model_corr[i] = np.trapz(integrand, q)
            
            return model_corr
            
        except Exception as e:
            print(f"Warning in model calculation: {e}")
            return np.full_like(tau_fit, np.nan)
    
    def objective_func(params):
        """Objective function for optimization"""
        sigma, gamma = params
        
        # Apply physical bounds
        if sigma <= 0 or sigma > 1e-2 or gamma <= 0 or gamma > 1e-3:
            return np.inf
            
        try:
            model_vals = model_func(sigma, gamma)
            if np.any(np.isnan(model_vals)):
                return np.inf
                
            # Use non-negative least squares to find amplitude
            try:
                from scipy.optimize import nnls
                A = model_vals.reshape(-1, 1)
                coeffs, residual = nnls(A, acf_fit)
                amplitude = coeffs[0] if len(coeffs) > 0 else 1.0
            except ImportError:
                # Fallback: simple least squares amplitude
                amplitude = np.sum(acf_fit * model_vals) / np.sum(model_vals**2)
                amplitude = max(0, amplitude)  # Ensure non-negative
            
            fitted_vals = amplitude * model_vals
            error = np.sum((acf_fit - fitted_vals)**2)
            return error
            
        except Exception:
            return np.inf
    
    # Perform optimization
    try:
        from scipy.optimize import minimize
        
        # Initial guess
        p0 = [initial_params['sigma'], initial_params['gamma']]
        
        # Bounds for parameters
        bounds = [(1e-6, 1e-2), (1e-9, 1e-3)]  # sigma, gamma bounds
        
        result = minimize(
            objective_func,
            p0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        fitted_sigma, fitted_gamma = result.x
        success = result.success and result.fun < np.inf
        
    except ImportError:
        print("Warning: scipy not available, using grid search")
        # Simple grid search fallback
        sigma_grid = np.logspace(-6, -3, 50)
        gamma_grid = np.logspace(-9, -4, 50)
        
        best_error = np.inf
        fitted_sigma, fitted_gamma = initial_params['sigma'], initial_params['gamma']
        
        for s in sigma_grid:
            for g in gamma_grid:
                error = objective_func([s, g])
                if error < best_error:
                    best_error = error
                    fitted_sigma, fitted_gamma = s, g
        
        success = best_error < np.inf
    
    # Calculate final fitted curve
    final_model = model_func(fitted_sigma, fitted_gamma)
    
    # Find optimal amplitude using NNLS
    try:
        from scipy.optimize import nnls
        A = final_model.reshape(-1, 1)
        coeffs, residual = nnls(A, acf_fit)
        amplitude = coeffs[0] if len(coeffs) > 0 else 1.0
    except ImportError:
        # Fallback: simple least squares amplitude
        amplitude = np.sum(acf_fit * final_model) / np.sum(final_model**2)
        amplitude = max(0, amplitude)  # Ensure non-negative
    
    fitted_curve = amplitude * final_model
    residuals = acf_fit - fitted_curve
    
    # Calculate goodness of fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((acf_fit - np.mean(acf_fit))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Create results dictionary
    results = {
        'params': {
            'kappa': fixed_kappa,  # Fixed value in kT
            'sigma': fitted_sigma,  # Fitted membrane tension in N/m
            'gamma': fitted_gamma,  # Fitted curvature interaction in J/m⁴
            'eta': fixed_eta,      # Fixed viscosity in Pa·s
            'amplitude': amplitude
        },
        'fitted_curve': fitted_curve,
        'residuals': residuals,
        'r_squared': r_squared,
        'tau_fit': tau_fit,
        'acf_fit': acf_fit,
        'success': success
    }
    
    # Generate plot similar to original version
    if show_plot:
        plt.figure(figsize=(10, 8))
        
        # Main plot (top 3/4)
        plt.subplot(4, 1, (1, 3))
        plt.semilogx(tau_fit, acf_fit, '--', linewidth=2, label='data', markersize=2.5)
        plt.semilogx(tau_fit, fitted_curve, '-', linewidth=2, label='fit')
        
        plt.legend(frameon=False)
        plt.ylabel('g(t)')
        plt.grid(True, alpha=0.3)
        
        # Add parameter text
        ax = plt.axis()
        x_text = np.exp(np.log(ax[0]) + 0.5 * np.log(ax[1]/ax[0]))
        y_text = ax[2] + 0.8 * (ax[3] - ax[2])
        
        param_text = [
            f'η = {fixed_eta*1e3:.1f} mPa·s',
            f'σ = {fitted_sigma:.1e} N/m',
            f'γ = {fitted_gamma:.1e} J/m⁴',
            f'κ = {fixed_kappa:.0f} kT (fixed)'
        ]
        plt.text(x_text, y_text, '\n'.join(param_text),
                fontsize=10, verticalalignment='top')
        
        # Residuals plot (bottom 1/4)
        plt.subplot(4, 1, 4)
        plt.semilogx(tau_fit, residuals, 'r-', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.xlabel('t / s')
        plt.ylabel('residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


def enhanced_miet_ptu_pipeline(
    ptu_path: Union[str, Path],
    fit_membrane_model: bool = True,
    use_minimal_model: bool = False,
    **kwargs
) -> dict:
    """
    Enhanced MIET-pFCS pipeline with membrane fluctuation model fitting.
    
    This function extends the basic run_miet_ptu_pipeline by automatically
    fitting the height correlation functions with passive membrane bending models.
    
    Parameters:
    -----------
    ptu_path : str or Path
        Path to PTU file
    fit_membrane_model : bool
        Whether to fit membrane fluctuation models to correlation data
    use_minimal_model : bool
        If True, use minimal model with fixed κ and η (like original version)
        If False, use full model fitting all parameters
    **kwargs : dict
        Additional parameters passed to run_miet_ptu_pipeline
        
    Returns:
    --------
    dict : Results including fitted membrane parameters
    """
    
    # Run the basic MIET-pFCS pipeline
    results = run_miet_ptu_pipeline(ptu_path, **kwargs)
    
    if not fit_membrane_model:
        return results
    
    # Extract correlation data for fitting
    tau_s = results['tau_s']
    auto = results['auto']
    auto2 = results['auto2']
    auto3 = results['auto3']
    
    # Calculate mean correlation functions
    auto_mean = np.mean(auto, axis=1) if auto is not None else None
    auto2_mean = np.mean(auto2, axis=1) if auto2 is not None else None
    auto3_mean = np.mean(auto3, axis=1) if auto3 is not None else None
    
    # Fit membrane models to each height correlation method
    membrane_fits = {}
    z_avg_m = results['z_avg_nm'] * 1e-9  # Convert nm to meters for minimal model
    
    # Choose fitting function based on model type
    if use_minimal_model:
        fit_func = lambda t, h: fit_membrane_minimal_model(t, h, z_avg_m, show_plot=False)
        model_description = "minimal (fixed κ,η)"
    else:
        fit_func = fit_membrane_fluctuation_model
        model_description = "full parameter"
    
    if auto_mean is not None:
        try:
            fit_result = fit_func(tau_s, auto_mean)
            membrane_fits['intensity_based'] = fit_result
            if use_minimal_model:
                print(f"Intensity-based fit ({model_description}): σ={fit_result['params']['sigma']:.2e} N/m, "
                      f"γ={fit_result['params']['gamma']:.2e} J/m⁴, R²={fit_result['r_squared']:.3f}")
            else:
                print(f"Intensity-based fit ({model_description}): κ={fit_result['params']['kappa']:.2f} kT, "
                      f"σ={fit_result['params']['sigma']:.2e} N/m, R²={fit_result['r_squared']:.3f}")
        except Exception as e:
            print(f"Warning: Could not fit intensity-based correlation: {e}")
    
    if auto2_mean is not None:
        try:
            fit_result = fit_func(tau_s, auto2_mean)
            membrane_fits['variance_based'] = fit_result
            if use_minimal_model:
                print(f"Variance-based fit ({model_description}): σ={fit_result['params']['sigma']:.2e} N/m, "
                      f"γ={fit_result['params']['gamma']:.2e} J/m⁴, R²={fit_result['r_squared']:.3f}")
            else:
                print(f"Variance-based fit ({model_description}): κ={fit_result['params']['kappa']:.2f} kT, "
                      f"σ={fit_result['params']['sigma']:.2e} N/m, R²={fit_result['r_squared']:.3f}")
        except Exception as e:
            print(f"Warning: Could not fit variance-based correlation: {e}")
    
    if auto3_mean is not None:
        try:
            fit_result = fit_func(tau_s, auto3_mean)
            membrane_fits['mle_based'] = fit_result
            if use_minimal_model:
                print(f"MLE-based fit ({model_description}): σ={fit_result['params']['sigma']:.2e} N/m, "
                      f"γ={fit_result['params']['gamma']:.2e} J/m⁴, R²={fit_result['r_squared']:.3f}")
            else:
                print(f"MLE-based fit ({model_description}): κ={fit_result['params']['kappa']:.2f} kT, "
                      f"σ={fit_result['params']['sigma']:.2e} N/m, R²={fit_result['r_squared']:.3f}")
        except Exception as e:
            print(f"Warning: Could not fit MLE-based correlation: {e}")
    
    # Add membrane fitting results to output
    results['membrane_fits'] = membrane_fits
    
    # Create enhanced correlation plot with fits
    # Create normalized correlation plot with normalized fits
    if membrane_fits:
          # norm_mode can be "g0" (divide by g(0)) or "end" 
          plot_normalized_acf_with_fits(
              results,
              norm_mode="end",
              title=f"Normalized height ACFs + membrane fits ({model_description})"
          )
    
    return results


# -----------------------------------------------------------------------------
# Example usage (edit path)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: Basic MIET-pFCS pipeline
    print("Running basic MIET-pFCS analysis...")
    out = run_miet_ptu_pipeline(
        r"D:\MIET\fromTao\data for share\BOTTOM-22.ptu"
    )
    
    # Example 2: Enhanced pipeline with full membrane fluctuation fitting
    print("\nRunning enhanced MIET-pFCS analysis with full membrane model fitting...")
    enhanced_out = enhanced_miet_ptu_pipeline(
        r"D:\MIET\fromTao\data for share\BOTTOM-22.ptu",
        fit_membrane_model=True,
        use_minimal_model=False
    )
    
    # Example 3: Enhanced pipeline with minimal model 
    print("\nRunning enhanced MIET-pFCS analysis with minimal membrane model...")
    minimal_out = enhanced_miet_ptu_pipeline(
        r"D:\MIET\fromTao\data for share\BOTTOM-22.ptu",
        fit_membrane_model=True,
        use_minimal_model=True
    )
    
    # Display fitted membrane parameters if available
    print("\n=== Membrane Fluctuation Analysis Results ===")
    
    # Full model results
    if 'membrane_fits' in enhanced_out:
        fits = enhanced_out['membrane_fits']
        print("\nFull Model Results:")
        for method, fit_data in fits.items():
            if fit_data['success']:
                params = fit_data['params']
                print(f"  {method.replace('_', ' ').title()}:")
                print(f"    κ: {params['kappa']:.2f} kT")
                print(f"    σ: {params['sigma']:.2e} N/m")
                print(f"    η: {params['eta']:.2e} Pa·s")
                print(f"    R²: {fit_data['r_squared']:.3f}")
    
    # Minimal model results
    if 'membrane_fits' in minimal_out:
        fits = minimal_out['membrane_fits']
        print("\nMinimal Model Results (fixed κ=20kT, η=1.2mPa·s):")
        for method, fit_data in fits.items():
            if fit_data['success']:
                params = fit_data['params']
                print(f"  {method.replace('_', ' ').title()}:")
                print(f"    σ: {params['sigma']:.2e} N/m")
                print(f"    γ: {params['gamma']:.2e} J/m⁴")
                print(f"    R²: {fit_data['r_squared']:.3f}")
    
    plt.show()
