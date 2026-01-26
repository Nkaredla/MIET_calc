"""
MIET-pFCS Analysis Pipeline

This module provides comprehensive analysis of PTU files for MIET (Metal-Induced Energy Transfer)
combined with pseudo-FCS (fluorescence correlation spectroscopy) measurements.

Features:
- PTU file reading and TCSPC data processing
- MIET calibration curve generation
- Lifetime-based height determination
- Photobleaching correction
- Auto-correlation analysis for membrane fluctuation studies
- Multiple lifetime estimation methods (MLE, variance-based, intensity-weighted)

The pipeline processes time-tagged time-resolved (TTTR) photon data from PicoQuant PTU files
and correlates fluorescence lifetime changes with molecular height variations using MIET theory.
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
    PTU_Read_Head, PTU_Read, mhist, harp_tcspc, tttr2xfcs
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


def tttr2bin(sync_ticks: np.ndarray, macrobin_ticks: float) -> np.ndarray:
    """
    Bin photon arrivals by macro time bins.
    Returns counts per bin.

    Parameters:
    sync_ticks: Array of photon arrival times in sync ticks
    macrobin_ticks: Width of each macro time bin in sync ticks
    """
    s = np.asarray(sync_ticks, dtype=np.int64)
    if s.size == 0:
        return np.zeros((0,), dtype=np.int64)

    w = int(np.round(macrobin_ticks))
    if w <= 0:
        raise ValueError("macrobin_ticks must be >= 1 tick")

    # Bin index from start of this block
    idx = (s // w).astype(np.int64)
    counts = np.bincount(idx, minlength=idx.max() + 1)
    return counts.astype(np.int64)


def tttr2bintcspc(
    sync_ticks: np.ndarray,
    params: Tuple[float, int],
    tcspc_shifted: np.ndarray,
    n_micro_bins: int,
) -> np.ndarray:
    """
    Bin TCSPC data into macro and micro time bins simultaneously.

    Parameters:
    sync_ticks: Array of photon arrival times in sync ticks
    params: Tuple of (macrobin_width_ticks, microtime_rebin_factor)
    tcspc_shifted: Shifted microtime channel indices for gated photons
    n_micro_bins: Number of microtime bins in output

    Returns:
    2D array with shape (n_micro_bins, n_macro_bins)
    """
    s = np.asarray(sync_ticks, dtype=np.int64)
    t = np.asarray(tcspc_shifted, dtype=np.int64)
    if s.size == 0:
        return np.zeros((n_micro_bins, 0), dtype=np.int64)

    macro_w = int(np.round(params[0]))
    micro_rebin = int(params[1])
    if macro_w <= 0:
        raise ValueError("macrobin width must be >=1")
    if micro_rebin <= 0:
        raise ValueError("micro rebin must be >=1")

    # Gate photons: valid photons have t>0 after shifting
    valid = t > 0
    if not np.any(valid):
        # number of macro bins still determined by time span
        n_macro = int((s.max() // macro_w) + 1)
        return np.zeros((n_micro_bins, n_macro), dtype=np.int64)

    s = s[valid]
    t = t[valid]

    macro_idx = (s // macro_w).astype(np.int64)
    micro_idx = (t // micro_rebin).astype(np.int64)

    # Bound micro indices to requested range
    keep = (micro_idx >= 0) & (micro_idx < n_micro_bins)
    macro_idx = macro_idx[keep]
    micro_idx = micro_idx[keep]

    n_macro = int(macro_idx.max() + 1) if macro_idx.size else int((s.max() // macro_w) + 1)
    out = np.zeros((n_micro_bins, n_macro), dtype=np.int64)
    np.add.at(out, (micro_idx, macro_idx), 1)
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
    photons_per_chunk: int = int(1e6),
    cutoff_ns: float = 10.0,
    shift_ns: float = 0.3,
    micro_rebin: int = 8,
    # Correlation
    Ncasc: int = 13,
    Nsub: int = 6,
    nbunches: int = 10,
):
    ptu_path = Path(ptu_path)

    # --- quantum yield update ---
    qy1 = qy0 * tau1 / tau0
    qy = qy1
    tau_free = tau1

    # --- MIET calibration curves ---
    lamem_nm = lamem_um * 1e3

    # metal refractive indices at emission wavelength
    # Use nearest wavelength match for robust lookup
    idx = int(np.argmin(np.abs(wavelength - lamem_nm)))
    n_ti = titan[idx]
    n_au = gold[idx]

    n0 = np.array([glass_n, n_ti, n_au, n_ti, top_n], dtype=complex)
    d0 = np.array(d0_um, dtype=float) * 1e3  # convert to nm
    d1 = np.array(d1_um, dtype=float) * 1e3  # convert to nm
    d = float(d_um * 1e3)  # convert to nm

    # Calculate MIET calibration curve
    z_nm, lifecurve = MIET_calc_fn(
        al_res,
        lamem_nm,
        n0,
        n,
        n1,
        d0,
        d,
        d1,
        qy,
        tau_free,
        1,
        curveType,
    )
    z_nm = np.asarray(z_nm, dtype=float)
    lifecurve = np.asarray(lifecurve, dtype=float)

    # Remove NaN values from calibration curve
    ind = np.isfinite(lifecurve) & np.isfinite(z_nm)
    z_nm = z_nm[ind]
    lifecurve = lifecurve[ind]

    # BrightnessDipole uses dimensionless scaling factors
    fac = 2 * np.pi / lamem_um  # wave vector factor (1/um)
    zfac = (z_nm * 1e-3) * fac  # convert nm to dimensionless units

    d0fac = (np.array(d0_um) * fac)  # layer thicknesses in dimensionless units
    dfac = (d_um * fac)
    d1fac = (np.array(d1_um) * fac) if len(d1_um) else np.array([])

    # Calculate brightness enhancement factors
    br_out = BrightnessDipole_fn(zfac, n0, n, n1, d0fac, dfac, d1fac, NA, qy, "false")
    if isinstance(br_out, tuple):
        br = np.asarray(br_out[-1], dtype=float)
    else:
        br = np.asarray(br_out, dtype=float)

    # Interpolate brightness onto z grid
    br = interp1_nan(z_nm[np.isfinite(br)], br[np.isfinite(br)], z_nm)

    # Find maximum in lifetime curve for height calibration
    maxz_life = int(np.nanargmax(lifecurve))

    # --- Read PTU data and calculate intensity trace ---
    # Get overall TCSPC histogram and peak position
    harp = harp_tcspc(ptu_path)
    tcspc_full = harp.tcspcdata
    pos = int(np.argmax(np.sum(tcspc_full, axis=1)))

    # Extract timing parameters from PTU header
    head = PTU_Read_Head(str(ptu_path))
    SyncRate = float(head.TTResult_SyncRate)
    Resolution_s = float(head.MeasDesc_Resolution)
    Ngate = int(np.ceil((1.0 / SyncRate) / Resolution_s))
    bin_tcspc = np.arange(Ngate + 1, dtype=np.int64)

    # Initialize streaming TTTR processing
    cnts = 0
    flag = True
    ttrace_list = []
    tmptcspc = np.zeros(Ngate, dtype=np.int64)
    tmptau_list = []
    sync_offset = 0  # Ensure sync timing continuity across chunks

    macrobin_ticks = tbin_s * SyncRate

    shift = int(np.round((shift_ns * 1e-9) / Resolution_s))
    ind1 = pos + shift
    cutoff_bins_native = int(np.ceil((cutoff_ns * 1e-9) / Resolution_s))
    ind2 = ind1 + cutoff_bins_native

    n_micro_bins = int(np.ceil((cutoff_ns * 1e-9) / Resolution_s))

    while flag:
        sync, tcspc, chan, special, num, loc, _ = PTU_Read(str(ptu_path), [cnts + 1, photons_per_chunk], head)

        sync = np.asarray(sync, dtype=np.int64)
        tcspc = np.asarray(tcspc, dtype=np.int64)
        special = np.asarray(special, dtype=np.int64)

        # Remove marker events (keep only photon events)
        keep = special == 0
        sync = sync[keep]
        tcspc = tcspc[keep]

        cnts += int(num)
        flag = num > 0

        if sync.size == 0:
            continue

        # Ensure continuous sync timing across chunks
        sync_cont = sync + sync_offset
        sync_offset += int(sync[-1])

        # Accumulate TCSPC histogram
        tmptcspc += mhist(tcspc, np.arange(Ngate, dtype=np.int64))

        # Gate microtimes around prompt peak
        gate = (tcspc > ind1) & (tcspc <= ind2)
        tcspc_shifted = (tcspc - ind1) * gate.astype(np.int64)

        # Lifetime trace binned in macro time bins
        tau_block = tttr2bintcspc(
            sync_cont,
            (macrobin_ticks, micro_rebin),
            tcspc_shifted,
            n_micro_bins=n_micro_bins,
        )
        tmptau_list.append(tau_block)

        # Intensity trace (counts per macrobin)
        ttrace_list.append(tttr2bin(sync_cont, macrobin_ticks))

    # Concatenate blocks; pad to equal macro length
    # (tttr2bin and tttr2bintcspc return variable lengths per chunk)
    ttrace = np.concatenate(ttrace_list, axis=0)

    # tmptau: concatenate along macro time axis (columns)
    # Each block tau is (n_micro_bins, n_macro_bins_block)
    tmptau = np.concatenate(tmptau_list, axis=1) if tmptau_list else np.zeros((n_micro_bins, 0), dtype=np.int64)

    # --- photobleaching correction ---
    meantrace = float(np.mean(ttrace[ttrace > 0])) if np.any(ttrace > 0) else float(np.mean(ttrace))
    normtrace = ttrace / meantrace

    ztrace = photobleach_fit_exp(normtrace)
    normtrace = normtrace / ztrace

    # --- lifetime trace -> height trace (MIET lifetime calibration) ---
    delay_ns = (np.arange(tmptau.shape[0], dtype=float) * Resolution_s * 1e9 * micro_rebin)

    # Grid MLE (monoexp + uniform bg)
    grid_lts = np.linspace(0.1, 3.0, 200)  # ns
    grid_bs = np.linspace(0.0, 0.2, 60)

    ltmat, bmat = np.meshgrid(grid_lts, grid_bs, indexing="xy")
    ltvec = ltmat.ravel()
    bvec = bmat.ravel()

    # Build normalized monoexp(+bg) PMF per tau,b
    delay_col = delay_ns[:, None]  # (T,1)
    exp_part = np.exp(-delay_col / ltvec[None, :])
    exp_part /= np.sum(exp_part, axis=0, keepdims=True)
    pmf = (bvec[None, :] / delay_ns.size) + (1.0 - bvec[None, :]) * exp_part
    logpmf = np.log(np.clip(pmf, 1e-300, None))  # avoid -inf

    # Evaluate in bunches for memory
    nbunch = 100
    n_macro = tmptau.shape[1]
    edges = np.linspace(0, n_macro, nbunch + 1).astype(int)

    grid_ind = np.zeros(n_macro, dtype=int)
    for k in range(nbunch):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        # score = - counts^T logpmf  (equivalent to negative log-likelihood up to constants)
        scores = -(tmptau[:, a:b].T @ logpmf)  # (macro_block, grid)
        grid_ind[a:b] = np.argmin(scores, axis=1)

    gridMLE_tau = ltvec[grid_ind]  # ns
    # gridMLE_b = bvec[grid_ind]   # if you need it

    htrace3 = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], gridMLE_tau)

    # Variance-based lifetime estimator
    # Calculate standard deviation of decay: sqrt(E[t^2] - E[t]^2)
    counts = tmptau.astype(float)
    denom = np.sum(counts, axis=0)
    denom = np.where(denom == 0, np.nan, denom)

    Et = np.sum(counts * delay_ns[:, None], axis=0) / denom
    Et2 = np.sum(counts * (delay_ns[:, None] ** 2), axis=0) / denom
    meantau = np.sqrt(np.maximum(Et2 - Et**2, 0.0))

    htrace2 = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], meantau)

    # --- height trace from intensity fluctuations ---
    pos2 = int(np.argmax(tmptcspc))
    ind1b = pos2 + shift

    # Simple biexponential tail fit (replacing Simplex/ExpFun)
    # Model: y = a1 exp(-t/t1) + a2 exp(-t/t2) + c
    t_ns = bin_tcspc[ind1b : Ngate] * Resolution_s * 1e9
    y_tail = tmptcspc[ind1b : Ngate].astype(float)

    def _fit_biexp(t, y):
        # crude grid for t1,t2, linear solve for a1,a2,c
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
        t1, t2, coef = best
        a1, a2, c = coef
        return float(t1), float(t2), float(a1), float(a2), float(c)

    t1, t2, a1, a2, c_bg = _fit_biexp(t_ns, y_tail)
    # Calculate amplitude-weighted average lifetime
    amps = np.array([a1, a2], dtype=float)
    taus = np.array([t1, t2], dtype=float)
    amps = np.clip(amps, 0.0, None)
    if np.sum(amps) > 0:
        tau_avg = float(np.sum(amps) / np.sum(amps / np.clip(taus, 1e-12, None)))
    else:
        tau_avg = float(np.mean(taus))

    z_avg = float(interp1_nan(lifecurve[np.isfinite(lifecurve)], z_nm[np.isfinite(lifecurve)], np.array([tau_avg]))[0])

    # Brightness normalization using nearest z value
    iz = int(np.argmin(np.abs(z_nm - z_avg)))
    br_fac = float(br[iz]) if np.isfinite(br[iz]) else float(np.nanmean(br))
    norm_br = br / br_fac

    maxz_br = int(np.nanargmax(norm_br))
    htrace = interp1_nan(norm_br[: maxz_br + 1], z_nm[: maxz_br + 1], normtrace)
    bhmean = meantrace * br_fac

    # --- correlate height trajectory ---
    def _finite_mask(x):
        return np.isfinite(x)

    mask1 = _finite_mask(htrace)
    tpoints = np.where(mask1)[0]
    h1 = htrace[mask1]
    i1 = ttrace[mask1]

    mask2 = _finite_mask(htrace2)
    tpoints2 = np.where(mask2)[0]
    h2 = htrace2[mask2]

    mask3 = _finite_mask(htrace3)
    tpoints3 = np.where(mask3)[0]
    h3 = htrace3[mask3]

    # Divide data into bunches for correlation analysis
    def _bunch_edges(L, nb):
        return np.round(np.linspace(0, L, nb + 1)).astype(int)

    e1 = _bunch_edges(len(tpoints), nbunches)
    e2 = _bunch_edges(len(tpoints2), nbunches)
    e3 = _bunch_edges(len(tpoints3), nbunches)

    # auto arrays: store as (tau, bunch)
    auto = None
    auto2 = None
    auto3 = None
    autoi = None
    autotime = None

    for k in range(nbunches):
        a, b = e1[k], e1[k + 1]
        a2, b2 = e2[k], e2[k + 1]
        a3, b3 = e3[k], e3[k + 1]

        # Height (intensity-derived)
        tmpauto, autotime = tttr2xfcs(tpoints[a:b], h1[a:b], Ncasc, Nsub)
        tmpauto = tmpauto.squeeze()  # (M,1,1) -> (M,)
        if auto is None:
            auto = np.zeros((tmpauto.size, nbunches), dtype=float)
        auto[:, k] = tmpauto

        # Height (variance lifetime)
        tmpauto2, _ = tttr2xfcs(tpoints2[a2:b2], h2[a2:b2], Ncasc, Nsub)
        tmpauto2 = tmpauto2.squeeze()
        if auto2 is None:
            auto2 = np.zeros((tmpauto2.size, nbunches), dtype=float)
        auto2[:, k] = tmpauto2

        # Height (MLE lifetime)
        tmpauto3, _ = tttr2xfcs(tpoints3[a3:b3], h3[a3:b3], Ncasc, Nsub)
        tmpauto3 = tmpauto3.squeeze()
        if auto3 is None:
            auto3 = np.zeros((tmpauto3.size, nbunches), dtype=float)
        auto3[:, k] = tmpauto3

        # Intensity ACF
        tmpautoi, _ = tttr2xfcs(tpoints[a:b], i1[a:b], Ncasc, Nsub)
        tmpautoi = tmpautoi.squeeze()
        if autoi is None:
            autoi = np.zeros((tmpautoi.size, nbunches), dtype=float)
        autoi[:, k] = tmpautoi

    autotime = autotime.squeeze().astype(float)  # in time bin units

    # Normalize correlation functions: auto = auto/auto(end,:) - 1
    def _norm_end(x):
        return x / x[-1:, :] - 1.0

    auto_n = _norm_end(auto)
    auto2_n = _norm_end(auto2)
    auto3_n = _norm_end(auto3)
    autoi_n = _norm_end(autoi)

    # --- Generate plots ---
    tau_s = autotime * tbin_s
    x = tau_s[:-1]  # exclude final time point

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

    # Second figure: normalized by first value
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
    }


# -----------------------------------------------------------------------------
# Example usage (edit path)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # You must have your PTU parsing + harp_tcspc + correlator in scope (same file or imports).
    out = run_miet_ptu_pipeline(
        r"C:\Users\narai\OneDrive\Documents\MIET\fromTao\data for share\BOTTOM-22.ptu"
    )
    plt.show()
