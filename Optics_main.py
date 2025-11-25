# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:58:37 2025

@author: narai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import os, sys
sys.path.insert(0, os.getcwd())   # ensure current working dir is on the path
from scipy.special import jv  # Bessel J_n
from scipy.ndimage import map_coordinates
from typing import Optional, Dict, Any, Tuple
from scipy.interpolate import PchipInterpolator
from utilities import mim, Disk, mpcolor, FocusImage2D, FocusImage3D, reconstruct_intensity_over_phi
from MIET_main import fresnel, dipoleL, lifetimeL  # Fresnel()
try:
    from scipy.interpolate import interp1d
    def _interp1_cubic_complex(x, y, xnew):
        f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0.0, assume_sorted=True)
        return f(xnew)
except Exception:
    def _interp1_cubic_complex(x, y, xnew):
        # numpy.interp can't do complex; do real/imag separately
        yr = np.interp(xnew, x, np.real(y), left=0.0, right=0.0)
        yi = np.interp(xnew, x, np.imag(y), left=0.0, right=0.0)
        return yr + 1j*yi


#%%


def gauss_exc(
    rhofield,
    zfield,
    NA,
    fd,
    n0,
    n,
    n1,
    d0,
    d,
    d1,
    lamex,
    over,
    focpos,
    atf=None,
    resolution=(20, 20),
    ring=None,
    maxm=2,
):
    """
    Vectorial Gaussian excitation (Richards–Wolf, layered media).

    Returns
    -------
    fxc, fxs, fyc, fys, fzc, fzs, rho, z
      - fxc, fyc, fzc : (Nrho, Nz, maxm+1)  cosine-like harmonics j=0..maxm
      - fxs, fys, fzs : (Nrho, Nz, maxm)    sine-like   harmonics j=1..maxm
      - rho, z        : (Nrho, Nz) midpoint grids
    """
    # --------------------------
    # Normalize inputs / dtypes
    # --------------------------
    maxnum = int(1e3)

    # resolution can be int or (Nrho_per_lambda, Nz_samples)
    if resolution is None:
        resolution = (20, 20)
    if np.isscalar(resolution):
        resolution = (int(resolution), int(resolution))
    Nrho_per_lambda = int(resolution[0])
    Nz_samples      = int(resolution[1])

    n0 = np.atleast_1d(np.asarray(n0, dtype=complex)).ravel()
    n1 = np.atleast_1d(np.asarray(n1, dtype=complex)).ravel()
    d0 = np.atleast_1d(np.asarray(d0, dtype=float)).ravel()
    d1 = np.atleast_1d(np.asarray(d1, dtype=float)).ravel()

    n  = complex(n)

    # d may be [], scalar, or array-like → take first element or 0.0
    d_arr = np.atleast_1d(np.asarray(d, dtype=float)).ravel()
    d = 0.0 if d_arr.size == 0 else float(d_arr[0])

    lamex = float(lamex)
    fd    = float(fd)

    # focpos → [z, x, y, tilt_x, tilt_y]
    foc_arr = np.atleast_1d(np.asarray(focpos, dtype=float)).ravel()
    if foc_arr.size == 1:
        foc_arr = np.array([foc_arr[0], 0.0, 0.0, 0.0, 0.0], dtype=float)
    elif foc_arr.size < 5:
        foc_arr = np.pad(foc_arr, (0, 5 - foc_arr.size), constant_values=0.0)
    foc_z, foc_x, foc_y, tilt_x, tilt_y = foc_arr

    # Ensure dipole layer thickness spans z-range if needed
    zfield_arr = np.asarray(zfield, dtype=float).ravel() if np.ndim(zfield) else np.array([float(zfield)])
    if d == 0.0 or (np.ndim(zfield) > 0 and d < np.max(zfield_arr)):
        d = float(np.max(zfield_arr))

    # Wavenumbers
    k0_vec = 2.0 * np.pi / lamex * n0
    k0_1   = k0_vec[0]
    k      = 2.0 * np.pi / lamex * n

    d0_k = 2.0 * np.pi * d0 / lamex
    d1_k = 2.0 * np.pi * d1 / lamex

    # --------------------------------
    # Build rho, z grids (mid-point)
    # --------------------------------
    if np.size(rhofield) == 1:
        rhov = np.array([float(rhofield)], float)
        if np.size(zfield) > 1:
            z1, z2 = float(zfield[0]), float(zfield[1])
            Nz = max(Nz_samples, 1)
            zv = z1 + (np.arange(Nz) + 0.5) * (z2 - z1) / Nz
        else:
            zv = zfield_arr
        rho = rhov[:, None] * np.ones_like(zv)[None, :]
        z   = np.ones_like(rhov)[:, None] * zv[None, :]

    elif np.size(rhofield) == 2:
        drho = lamex / max(Nrho_per_lambda, 1)
        r1, r2 = float(rhofield[0]), float(rhofield[1])
        if r2 <= r1:
            rhov = np.array([r1], float)
        else:
            Nrho = int(np.floor((r2 - r1) / drho))
            Nrho = max(Nrho, 1)
            rhov = r1 + (np.arange(Nrho) + 0.5) * drho

        if np.size(zfield) > 1:
            z1, z2 = float(zfield[0]), float(zfield[1])
            Nz = max(Nz_samples, 1)
            zv = z1 + (np.arange(Nz) + 0.5) * (z2 - z1) / Nz
        else:
            zv = zfield_arr

        rho, z = np.meshgrid(rhov, zv, indexing="ij")

    else:
        rho = np.asarray(rhofield, dtype=float)
        z   = np.asarray(zfield, dtype=float)
        rhov = rho[:, 0]
        zv   = z[0, :]

    rhov = rho[:, 0]
    zv   = z[0, :]

    # ------------------------------------------
    # Pupil coordinates, refraction mapping
    # ------------------------------------------
    chimax = np.arcsin(np.clip(np.real(NA / n0[0]), 0.0, 1.0))
    chimin = 0.0
    chi = np.linspace(chimin, chimax, maxnum + 1)[:, None]  # (L,1)
    c0  = np.cos(chi)
    s0  = np.sin(chi)

    # Snell into dipole layer
    ss = (n0[0] / n) * s0
    cc = np.sqrt(1.0 - ss**2 + 0j)  # complex dtype safe for TIR

    # ------------------------------------------
    # Beam envelope (over) → fac(chi), om_x/y
    # ------------------------------------------
    def _beam_over_params(over_param):
        over_arr = np.asarray(over_param, dtype=float).ravel()
        fac_local = (s0 * np.sqrt(c0)).ravel().astype(complex)  # aplanatic apodization

        if over_arr.size == 1:
            w0 = float(over_arr[0])
            if w0 == 0.0:
                fac_local = np.zeros_like(fac_local); fac_local[0] = 1.0 + 0j
            om = np.array([1.0, 1.0], dtype=complex) / (w0**2 if w0 != 0 else 1.0)

        elif over_arr.size == 2:
            z0, w0 = over_arr
            if w0 == 0.0:
                fac_local = np.zeros_like(fac_local); fac_local[0] = 1.0 + 0j
                om = np.array([1.0, 1.0], dtype=complex)
            else:
                zeta  = (z0 * lamex) / (np.pi * w0**2) if np.isfinite(z0) else 0.0
                w_eff = w0 * np.sqrt(1.0 + zeta**2)
                om    = np.array([1.0, 1.0], dtype=complex) * (1 - 1j * zeta) / (w_eff**2)

        elif over_arr.size == 3:
            ast, w0_in, _ = over_arr
            zeta = 2 * ast * np.pi * (w0_in**2) / (NA * fd) ** 2
            w0   = w0_in / np.sqrt(1.0 + zeta**2)
            zetas = np.array([-zeta, zeta], dtype=float)
            om    = (1 - 1j * zetas) / (1 + zetas**2) / (w0**2) / 2.0

        else:
            z1, z2, w1, w2 = over_arr[:4]
            zeta = np.array([z1, z2]) * lamex / (np.pi * np.array([w1, w2]) ** 2)
            om   = (1 - 1j * zeta) / (1 + zeta**2) / (np.array([w1, w2]) ** 2) / 2.0

        return fac_local, np.asarray(om, dtype=complex)

    fac, om = _beam_over_params(over)

    # ------------------------------------------
    # Fresnel coefficients on excitation path
    # ------------------------------------------
    # reflections seen from the dipole layer into bottom & top stacks
    rp0, rs0, _, _ = fresnel((n * cc).ravel(), np.concatenate(([n], n0[::-1])), d0_k[::-1])
    rp1, rs1, _, _ = fresnel((n * cc).ravel(), np.concatenate(([n], n1)), d1_k)

    # Optional coverslip (kept for parity; not applied to fields here)
    if atf is not None and np.size(atf) > 0:
        atf_arr = np.atleast_1d(np.asarray(atf, dtype=float)).ravel()
        w_bottom = (n0[0] * c0).ravel()
        if atf_arr.size == 1:
            n_cs = atf_arr[0]
            _, _, tmp_p, tms_s = fresnel(w_bottom, n0[0], n_cs)
            _, _, mp_p,  ms_s  = fresnel(np.sqrt(n_cs**2 - (n0[0]**2) * (s0**2)).ravel(), n_cs, n0[0])
        else:
            n_cs, dz_mm = atf_arr[0], atf_arr[1]
            _, _, tmp_p, tms_s = fresnel(w_bottom, np.array([n0[0], n0[0], n_cs]), -2 * np.pi * dz_mm / lamex)
            _, _, mp_p,  ms_s  = fresnel(np.sqrt(n_cs**2 - (n0[0]**2) * (s0**2)).ravel(),
                                         np.array([n_cs, n_cs, n]), 2 * np.pi * dz_mm / lamex)
        # (If you want, you can multiply tp/ts equivalents with these; left neutral here.)

    # Effective coefficients incl. multiple reflections in n-layer
    denom_p = (1 - rp1 * rp0 * np.exp(2j * k * (cc.ravel()) * d))
    denom_s = (1 - rs1 * rs0 * np.exp(2j * k * (cc.ravel()) * d))
    tp_eff  = (1.0 / denom_p)[:, None]   # (L,1) up to a common factor
    ts_eff  = (1.0 / denom_s)[:, None]
    rp_eff  = tp_eff * rp1[:, None]
    rs_eff  = ts_eff * rs1[:, None]

    # ------------------------------------------
    # Phase to z-planes (direct + reflected)
    # ------------------------------------------
    cc_col = cc.ravel()[:, None]  # (L,1)
    c0_col = c0.ravel()[:, None]  # (L,1)
    row    = np.ones_like(zv)[None, :]    # (1,Nz)

    phase1 = k * cc_col @ zv[None, :]           - (k0_1 * c0_col) @ (row * foc_z)
    phase2 = k * cc_col @ (2 * d - zv)[None, :] - (k0_1 * c0_col) @ (row * foc_z)
    ez1 = np.exp(1j * phase1)  # (L,Nz)
    ez2 = np.exp(1j * phase2)  # (L,Nz)

    # ------------------------------------------
    # Pupil azimuth sampling and Fourier series
    # ------------------------------------------
    L = ss.size
    ss_vec = ss.ravel()
    cc_vec = cc.ravel()
    rad    = (s0 / np.max(s0)).ravel()  # (L,)
    n_ = n

    K = 2 * maxm + 1
    tmpxt = np.zeros((L, K), dtype=complex)
    tmpxr = np.zeros((L, K), dtype=complex)
    tmpyt = np.zeros((L, K), dtype=complex)
    tmpyr = np.zeros((L, K), dtype=complex)
    tmpzt = np.zeros((L, K), dtype=complex)
    tmpzr = np.zeros((L, K), dtype=complex)

    rho0 = np.hypot(foc_x, foc_y)
    psi0 = np.arctan2(foc_y, foc_x)

    for j in range(K):
        psi = j / (2 * maxm + 1) * 2 * np.pi
        cp, sp = np.cos(psi), np.sin(psi)

        # Gaussian pupil envelope with lateral focus/tilt
        argx = (fd * n_ * ss_vec * cp - tilt_x)
        argy = (fd * n_ * ss_vec * sp - tilt_y)
        ef = fac * np.exp(-om[0] * (argx**2) - om[1] * (argy**2) + 1j * k * ss_vec * rho0 * np.cos(psi - psi0))

        # Optional pupil mask "ring"
        if ring is not None:
            if callable(ring):
                mask = np.asarray(ring(rad, psi), dtype=complex)
                if mask.shape != ef.shape:
                    raise ValueError("ring(rad, psi) must return (L,), matching pupil samples.")
                ef *= mask
            elif isinstance(ring, str):
                # evaluate a numpy expression with variables rad (L,) and psi (scalar)
                _locals = {"rad": rad, "psi": psi, "np": np}
                mask = eval(ring, {"__builtins__": {}}, _locals)
                mask = np.asarray(mask, dtype=complex)
                ef *= (mask if mask.shape == ef.shape else mask * np.ones_like(ef))
            else:
                ef *= complex(ring)

        # Field factors (transmitted/reflected)
        tx = (cp**2 * (tp_eff.ravel() * cc_vec) + sp**2 * ts_eff.ravel()) * ef
        ty = (cp * sp * (tp_eff.ravel() * cc_vec - ts_eff.ravel())) * ef
        tz = (-cp * ss_vec * tp_eff.ravel()) * ef

        rx = (-cp**2 * (rp_eff.ravel() * cc_vec) + sp**2 * rs_eff.ravel()) * ef
        ry = (cp * sp * (-rp_eff.ravel() * cc_vec - rs_eff.ravel())) * ef
        rz = (-cp * ss_vec * rp_eff.ravel()) * ef

        tmpxt[:, j] = tx; tmpxr[:, j] = rx
        tmpyt[:, j] = ty; tmpyr[:, j] = ry
        tmpzt[:, j] = tz; tmpzr[:, j] = rz

    # Fourier series coefficients over psi
    scale = (maxm + 0.5)
    tmpxt = fft(tmpxt, axis=1) / scale
    tmpxr = fft(tmpxr, axis=1) / scale
    tmpyt = fft(tmpyt, axis=1) / scale
    tmpyr = fft(tmpyr, axis=1) / scale
    tmpzt = fft(tmpzt, axis=1) / scale
    tmpzr = fft(tmpzr, axis=1) / scale

    # ------------------------------------------
    # Hankel-like reconstruction via Bessel J
    # ------------------------------------------
    Nrho, Nz = rho.shape
    barg = (k * rhov[:, None]) * (ss_vec[None, :])  # (Nrho,L)

    fxc = np.zeros((Nrho, Nz, maxm + 1), dtype=complex)
    fyc = np.zeros((Nrho, Nz, maxm + 1), dtype=complex)
    fzc = np.zeros((Nrho, Nz, maxm + 1), dtype=complex)
    fxs = np.zeros((Nrho, Nz, maxm    ), dtype=complex)
    fys = np.zeros((Nrho, Nz, maxm    ), dtype=complex)
    fzs = np.zeros((Nrho, Nz, maxm    ), dtype=complex)

    row = np.ones((1, Nz), complex)

    def _term(col_t, col_r):
        term_t = col_t[:, None] * row
        term_r = col_r[:, None] * row
        return term_t * ez1 + term_r * ez2  # (L,Nz)

    # j = 0
    J0 = jv(0, barg)
    Tx0 = _term(tmpxt[:, 0], tmpxr[:, 0])
    Ty0 = _term(tmpyt[:, 0], tmpyr[:, 0])
    Tz0 = _term(tmpzt[:, 0], tmpzr[:, 0])
    fxc[:, :, 0] = J0 @ Tx0
    fyc[:, :, 0] = J0 @ Ty0
    fzc[:, :, 0] = J0 @ Tz0

    # j >= 1
    Klen = tmpxt.shape[1]  # 2*maxm+1
    for j in range(1, maxm + 1):
        Jj = jv(j, barg)
        a = j
        b = Klen - j

        Tx_c = _term(tmpxt[:, a] + tmpxt[:, b], tmpxr[:, a] + tmpxr[:, b])
        Ty_c = _term(tmpyt[:, a] + tmpyt[:, b], tmpyr[:, a] + tmpyr[:, b])
        Tz_c = _term(tmpzt[:, a] + tmpzt[:, b], tmpzr[:, a] + tmpzr[:, b])

        Tx_s = _term(tmpxt[:, a] - tmpxt[:, b], tmpxr[:, a] - tmpxr[:, b])
        Ty_s = _term(tmpyt[:, a] - tmpyt[:, b], tmpyr[:, a] - tmpyr[:, b])
        Tz_s = _term(tmpzt[:, a] - tmpzt[:, b], tmpzr[:, a] - tmpzr[:, b])

        phase_c = (1j) ** (-j)
        phase_s = (1j) ** (-j + 1)

        fxc[:, :, j]     = phase_c * (Jj @ Tx_c)
        fyc[:, :, j]     = phase_c * (Jj @ Ty_c)
        fzc[:, :, j]     = phase_c * (Jj @ Tz_c)
        fxs[:, :, j - 1] = phase_s * (Jj @ Tx_s)
        fys[:, :, j - 1] = phase_s * (Jj @ Ty_s)
        fzs[:, :, j - 1] = phase_s * (Jj @ Tz_s)

    return fxc, fxs, fyc, fys, fzc, fzs, rho, z

def _pack_exc_struct(fxc, fxs, fyc, fys, fzc, fzs, rho, z, d0, d, d1, maxm):
    return {
        "maxm": int(maxm),
        "rho": np.asarray(rho, float),
        "z":   np.asarray(z, float),
        "fxc": np.asarray(fxc, complex),
        "fxs": np.asarray(fxs, complex),
        "fyc": np.asarray(fyc, complex),
        "fys": np.asarray(fys, complex),
        "fzc": np.asarray(fzc, complex),
        "fzs": np.asarray(fzs, complex),
        "d0":  np.atleast_1d(np.asarray(d0, float)),
        "d":   float(d),
        "d1":  np.atleast_1d(np.asarray(d1, float)),
    }


def PulsedExcitation(x, k10, Tpulse, Trep, kisc=0, extinction=None, wavelength=640, w0=250, compute_derivatives=0):
    """

    Parameters
    ----------
    x : float or array-like
        Excitation chance per pulse / photon rate (photons/ns), OR laser power in µW
        if `extinction` is provided (not None).
    k10 : float
        Inverse lifetime [1/ns].
    Tpulse : float
        Pulse duration [ns].
    Trep : float
        Excitation period [ns].
    kisc : float, optional
        Ratio of intersystem crossing rate to phosphorescence rate. Default 0.
    extinction : float or None, optional
        Molar extinction coefficient [L/mol/cm]. If not None, `x` is interpreted as power in µW
        and converted to photons/ns using `wavelength` and `w0`. Default None.
    wavelength : float, optional
        Wavelength [nm]. Used only when `extinction` is not None. Default 640.
    w0 : float, optional
        Focus diameter [nm]. Used only when `extinction` is not None. Default 250.
    compute_derivatives : {0,1,2}
        0 -> return y
        1 -> return (y, y1)
        2 -> return (y, y1, y2)

    Returns
    -------
    y or (y, y1) or (y, y1, y2)
        y in 1/ns (or per pulse), and its first/second derivatives (matching MATLAB outputs y, y1, y2).
    """
    # Constants (NIST values as in the MATLAB code)
    AvogadroConstant = 6.0221419947e23
    PlanckConstant = 6.6260687652e-34
    SpeedOfLight = 299792458.0

    x = np.asarray(x, dtype=float)
    k10 = float(k10)
    Tpulse = float(Tpulse)
    Trep = float(Trep)
    kisc = 0.0 if kisc is None else float(kisc)

    # If extinction is provided, interpret x as power in µW and convert to photons/ns
    if extinction is not None:
        extinction = 100e3 if (extinction is None or (isinstance(extinction, float) and np.isnan(extinction))) else float(extinction)
        if wavelength is None:
            wavelength = 640.0
        if w0 is None:
            w0 = 250.0
        wavelength = float(wavelength)
        w0 = float(w0)
        # photons/ns (faithful to MATLAB conversion)
        x = (
            2.0 * extinction * np.log(10.0) / AvogadroConstant
            * x * 1e-6
            / (PlanckConstant * SpeedOfLight / (wavelength * 1e-9))
            / (np.pi * (w0 ** 2) * 1e-18)
            / 1e10
        )

    k01 = x * Trep / Tpulse

    # chance to be in the excited state S1:
    denom1 = (k01 + k10)
    term1 = (k01 / denom1) * (Tpulse / Trep)
    term2 = (
        (k01 ** 2) / (denom1 ** 2) / Trep / k10
        * (1.0 - np.exp(-(k01 + k10) * Tpulse))
        * (1.0 - np.exp(-k10 * (Trep - Tpulse)))
        / (1.0 - np.exp(-k10 * Trep - k01 * Tpulse))
    )
    kappa = term1 + term2
    y = k10 * kappa / (1.0 + kisc * kappa)

    if compute_derivatives == 0:
        return y

    E = np.exp
    kappa1 = (
        (
            2.0 * E(k10 * Trep) * k01 * k10
            - 2.0 * E(2.0 * (k01 + k10) * Tpulse + k10 * Trep) * k01 * k10
            - 2.0 * E(k01 * Tpulse + k10 * (Tpulse + Trep)) * (k01 + k10) * (k01 ** 2 + k10 ** 2) * Tpulse
            + E(k01 * Tpulse + 2.0 * k10 * Trep) * k01 * (-2.0 * k10 + k01 * (k01 + k10) * Tpulse)
            + E((k01 + 2.0 * k10) * Tpulse) * k01 * (2.0 * k10 + k01 * (k01 + k10) * Tpulse)
            + E(k10 * Tpulse) * k10 * (-2.0 * k01 + k10 * (k01 + k10) * Tpulse)
            + E(2.0 * k01 * Tpulse + k10 * Tpulse + 2.0 * k10 * Trep) * k10 * (2.0 * k01 + k10 * (k01 + k10) * Tpulse)
        )
        / (
            E(k10 * Tpulse)
            * ( -1.0 + E(k01 * Tpulse + k10 * Trep) ) ** 2
            * k10 * (k01 + k10) ** 3 * Trep
        )
    )
    y1 = k10 / (1.0 + kisc * kappa) ** 2 * kappa1 * Trep / Tpulse

    if compute_derivatives == 1:
        return y, y1

    kappa2 = -(
        (
            (
                -6.0 * ( -1.0 + E((k01 + k10) * Tpulse) )
                * ( -1.0 + E(k10 * (-Tpulse + Trep)) ) * (k01 ** 2)
            )
            / ( -1.0 + E(k01 * Tpulse + k10 * Trep) )
            + 2.0 * (k10 ** 2) * (k01 + k10) * Tpulse
            - (
                ( -1.0 + E(k10 * (-Tpulse + Trep)) )
                * (k01 + k10) ** 2
                * (
                    -2.0
                    + 2.0 * E(3.0 * k01 * Tpulse + k10 * Tpulse + 2.0 * k10 * Trep)
                    + E(2.0 * k01 * Tpulse + k10 * (Tpulse + Trep)) * ( -4.0 + k01 * Tpulse * ( -4.0 + k01 * Tpulse ) )
                    - E(2.0 * k01 * Tpulse + 2.0 * k10 * Trep) * ( 2.0 + k01 * Tpulse * ( -4.0 + k01 * Tpulse ) )
                    + E(k01 * Tpulse + k10 * Trep) * ( 4.0 - k01 * Tpulse * ( 4.0 + k01 * Tpulse ) )
                    + E((k01 + k10) * Tpulse) * ( 2.0 + k01 * Tpulse * ( 4.0 + k01 * Tpulse ) )
                )
            )
            / ( -1.0 + E(k01 * Tpulse + k10 * Trep) ) ** 3
            + (
                4.0 * ( -1.0 + E(k10 * (-Tpulse + Trep)) )
                * k01 * (k01 + k10)
                * (
                    2.0
                    + E(k01 * Tpulse)
                    * (
                        -( E(k10 * Tpulse) * (2.0 + k01 * Tpulse) )
                        + E(k10 * Trep) * ( -2.0 + 2.0 * E((k01 + k10) * Tpulse) + k01 * Tpulse )
                    )
                )
            )
            / ( -1.0 + E(k01 * Tpulse + k10 * Trep) ) ** 2
        )
        / ( k10 * (k01 + k10) ** 4 * Trep )
    )
    y2 = (
        ( -k10 * kisc / (1.0 + kisc * kappa) ** 3 * (kappa1 ** 2) )
        + ( k10 / (1.0 + kisc * kappa) ** 2 * kappa2 )
    ) * (Trep / Tpulse) ** 2

    return y, y1, y2

from scipy.special import jv as besselj

def gauss_exc_to_mdf(
    exc, NA, n0, n, n1, focpos, lamem, mag, av,
    zpin=0, atf=None, kappa=0, lt=0, pulse=None, sat=0, triplet=0
):
    """
    Faithful translation of MATLAB GaussExc2MDF.
    Returns dict with fields: rho, z, volx, voly, etc.
    """

    # -------- constants / inputs --------
    maxnum = int(1e3)
    ni = 1.0  # imaging medium (air)
    ki = 2*np.pi/lamem * ni
    n0_vec = np.atleast_1d(np.asarray(n0, dtype=float)).ravel()
    k0 = 2*np.pi/lamem * n0_vec   # use k0[0] below, like MATLAB

    # pulse default: [1,1]
    if pulse is None:
        pulse = [1, 1]
    pulse = np.atleast_1d(pulse).astype(float).ravel()
    if pulse.size == 1:
        pulse = np.array([pulse[0], pulse[0]], float)

    # unpack excitation "struct"
    maxm = int(np.asarray(exc["maxm"]))
    rho = np.asarray(exc["rho"], dtype=float)
    z   = np.asarray(exc["z"],   dtype=float)
    fxc = np.asarray(exc["fxc"], dtype=complex)
    fxs = np.asarray(exc["fxs"], dtype=complex)
    fyc = np.asarray(exc["fyc"], dtype=complex)
    fys = np.asarray(exc["fys"], dtype=complex)
    fzc = np.asarray(exc["fzc"], dtype=complex)
    fzs = np.asarray(exc["fzs"], dtype=complex)

    # radial coords for image-plane integration
    if rho.size > 1:  # 2D grid
        drho = float(rho[1, 0] - rho[0, 0])
        rhov = mag * np.arange(rho[0, 0], rho[-1, 0] + av/mag + 1e-12, drho)
        uv   = mag * rho[:, 0]
    else:             # scalar
        drho = float(z[1] - z[0])
        rhov = mag * np.arange(rho - 0.0, rho + av/mag + 1e-12, drho)
        uv   = mag * np.atleast_1d(rho)

    rhov = np.abs(rhov)
    uv   = np.abs(uv)
    zv   = z[0, :].copy()
    Nz   = zv.size

    # pupil angles in image space (exclude above critical angle)
    chimin = 0.0
    chimax = float(np.arcsin(np.clip(NA/(ni*mag), 0.0, 1.0)))
    dchi   = chimax/maxnum
    chi    = np.arange(chimin, chimax + 0.5*dchi, dchi)[:, None]  # (L,1)
    ci     = np.cos(chi)  # (L,1)
    si     = np.sin(chi)  # (L,1)

    # Abbe sine condition: image → sample pupil
    # s0 must be real and <=1 for the rem/phase to stay real like MATLAB
    n0_1 = float(n0_vec[0])
    s0   = (ni/n0_1) * mag * si
    s0   = np.clip(s0, 0.0, 1.0)          # (L,1), real
    c0   = np.sqrt(1.0 - s0**2)           # (L,1), real

    # layer distances in "k units" (MATLAB: 2*pi/lamem * exc.d*)
    d0k = 2*np.pi/lamem * np.asarray(exc["d0"], dtype=float)
    dk  = 2*np.pi/lamem * float(exc["d"])
    d1k = 2*np.pi/lamem * np.asarray(exc["d1"], dtype=float)

    # dipole emission inside stack for those sample angles
    theta0 = np.arcsin(s0.ravel())  # (L,)
    # v, pc, ps should come back as (L, Nz)
    v, pc, ps, *_ = dipoleL(theta0, 2*np.pi*zv/lamem, n0, n, n1, d0k, dk, d1k)
    v  = np.asarray(v,  dtype=complex)
    pc = np.asarray(pc, dtype=complex)
    ps = np.asarray(ps, dtype=complex)

    # optional coverslip correction (exactly like MATLAB)
    atf_arr = np.atleast_1d(np.asarray([] if atf is None else atf, dtype=float)).ravel()
    if atf_arr.size > 0:
        if atf_arr.size == 1:
            ncs = float(atf_arr[0])
            # [tmp, tms, tmp, tms] = Fresnel(n0(1)*c0, n0(1), atf)
            tmp_p, tms_s, _, _ = fresnel((n0_1*c0).ravel(), n0_1, ncs)
            # [mp, ms, mp, ms] = Fresnel(sqrt(atf^2-n0(1)^2*s0.^2), atf, n0(1))
            mp_p, ms_s, _, _  = fresnel(np.sqrt(ncs**2 - (n0_1**2)*(s0**2)).ravel(), ncs, n0_1)
        else:
            ncs  = float(atf_arr[0])
            dzmm = float(atf_arr[1])
            # Fresnel(n0(1)*c0, [n0(1) atf(1) atf(1)], 2*pi*atf(2)/lamem)
            tmp_p, tms_s, _, _ = fresnel((n0_1*c0).ravel(), np.array([n0_1, ncs, ncs], float),  2*np.pi*dzmm/lamem)
            # Fresnel(sqrt(atf(1)^2-n0(1)^2*s0.^2), [atf(1) n0(1) n0(1)], -2*pi*atf(2)/lamem)
            mp_p, ms_s, _, _  = fresnel(np.sqrt(ncs**2 - (n0_1**2)*(s0**2)).ravel(), np.array([ncs, n0_1, n0_1], float), -2*np.pi*dzmm/lamem)

        # scale factors, broadcast to (L,Nz)
        row = np.ones((1, Nz), float)
        scale_p = ((tmp_p * mp_p)[:, None]) @ row
        scale_s = ((tms_s * ms_s)[:, None]) @ row
        v  = scale_p * v
        pc = scale_p * pc
        ps = scale_s * ps
    else:
        atf = None  # store as [] in result below

    # defocus/pinhole phase (use real phase like MATLAB's rem(...,2*pi))
    foc_arr = np.atleast_1d(np.asarray(focpos, dtype=float)).ravel()
    foc_z = float(foc_arr[0]) if foc_arr.size > 0 else 0.0
    phase = (-k0[0] * c0 * foc_z) + (ki * float(zpin) * ci)  # (L,1), real
    phase = np.mod(phase, 2*np.pi)
    ez = np.exp(1j * phase)

    # Bessel/Hankel-like radial integrals
    fac = dchi * si * np.sqrt(ci / c0)          # (L,1)
    barg = (ki * rhov[:, None]) * si.T          # (Nrhov, L)
    j0   = besselj(0, barg)
    j1   = besselj(1, barg)
    j2   = besselj(2, barg)

    row = np.ones((1, Nz), float)
    ezi = (fac * ci * ez) @ row                 # (L,Nz)
    ezr = (fac * ez)     @ row                  # (L,Nz)

    # f*, g* components (Nrhov x Nz)
    f0 = j0 @ (ezi * pc + ezr * ps)             # cos(0φ)
    f2 = -j2 @ (ezi * pc - ezr * ps)            # cos(2φ)
    f1 = -2j * j1 @ (ezi * v)                   # cos(1φ)

    g0 = j0 @ (ezr * pc + ezi * ps)             # cos(0φ)
    g2 = -j2 @ (ezr * pc - ezi * ps)            # cos(2φ)
    g1 = -2j * j1 @ (ezr * v)                   # cos(1φ)

    f00 = np.real(f0 * np.conj(g0))
    f11 = np.real(f1 * np.conj(g1))
    f22 = np.real(f2 * np.conj(g2))
    f01 = np.real(f0 * np.conj(g1) + f1 * np.conj(g0))
    f02 = np.real(f0 * np.conj(g2) + f2 * np.conj(g0))
    f12 = np.real(f1 * np.conj(g2) + f2 * np.conj(g1))

    # pinhole weights s0..s4 (Nu x Nrhov)
    uv_vec   = uv
    rhov_vec = rhov
    Nu       = uv_vec.size
    Nrhov    = rhov_vec.size

    # phi = acos((av^2 - u^2 - r^2)/(2 u r)) with safe division/clipping
    num = (av**2) - (uv_vec[:, None]**2) - (rhov_vec[None, :]**2)
    den = 2.0 * uv_vec[:, None] * rhov_vec[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    ratio = np.clip(ratio, -1.0, 1.0)
    phi = np.arccos(ratio)

    tmp_mat = (np.ones((Nu, 1)) @ rhov_vec[None, :]) * drho
    s0w = 2.0 * (np.pi - phi) * tmp_mat
    s1w = 2.0 * np.sin(phi)   * tmp_mat
    s2w = np.sin(2.0 * phi)   * tmp_mat
    s3w = (2.0/3.0) * np.sin(3.0 * phi) * tmp_mat
    s4w = 0.5 * np.sin(4.0 * phi)       * tmp_mat

    # lifetime weighting
    qv = []
    qp = []

    if (lt is None) or (lt == 0):
        lam1 = 3.0/(2.0*n)
        lam2 = 1.0/(2.0*n)
        lam3 = 3.0/(10.0*n)
        lam1_vec = np.full((Nz,), lam1, float)
        lam2_vec = np.full((Nz,), lam2, float)
        lam3_vec = np.full((Nz,), lam3, float)
        do_tau = False

    elif lt < 3:
        lvd = np.zeros((Nz,), float); lvu = np.zeros_like(lvd)
        lpd = np.zeros_like(lvd);     lpu = np.zeros_like(lvd)
        qvd = np.zeros_like(lvd);     qvu = np.zeros_like(lvd)
        qpd = np.zeros_like(lvd);     qpu = np.zeros_like(lvd)
        for j in range(Nz):
            (lvd[j], lvu[j], lpd[j], lpu[j],
             qvd[j], qvu[j], qpd[j], qpu[j], *_rest) = lifetimeL(2*np.pi*zv[j]/lamem, n0, n, n1, d0k, dk, d1k)

        qv = qvd + qvu
        qp = qpd + qpu
        with np.errstate(divide='ignore', invalid='ignore'):
            qd  = np.sqrt(np.divide(qv - qp, qp, out=np.zeros_like(qv), where=(qp != 0)))
        aqd = np.arctan(qd)

        lam1_vec = np.empty_like(zv)
        lam2_vec = np.empty_like(zv)
        lam3_vec = np.empty_like(zv)
        for j in range(Nz):
            if np.isfinite(qd[j]) and (abs(qd[j]) > 1e-5):
                lam1_vec[j] = 2*np.real(aqd[j]/qd[j]) / qp[j]
                lam2_vec[j] = 2*np.real((1 - aqd[j]/qd[j])/(qd[j]**2)) / qp[j]
                lam3_vec[j] = 2*np.real(((qd[j]**2 - 3)*qd[j] + 3*aqd[j])/(qd[j]**5)) / 3 / qp[j]
            elif not np.isfinite(qd[j]):
                lam1_vec[j] = 0.0; lam2_vec[j] = 0.0; lam3_vec[j] = 0.0
            else:
                lam1_vec[j] = 3.0/(2.0*n)
                lam2_vec[j] = 1.0/(2.0*n)
                lam3_vec[j] = 3.0/(10.0*n)

        do_tau = (lt == 2)
        if do_tau:
            tau1_vec = np.empty_like(zv)
            tau2_vec = np.empty_like(zv)
            tau3_vec = np.empty_like(zv)
            for j in range(Nz):
                if np.isfinite(qd[j]) and (abs(qd[j]) > 1e-5):
                    tau1_vec[j] = (1/qv[j] + np.real(aqd[j]/qd[j]) / qp[j]) / qp[j]
                    tau2_vec[j] = np.real(aqd[j]/(qd[j]**3) / qp[j] - 1/(qd[j]**2)/qv[j]) / qp[j]
                    tau3_vec[j] = np.real(((qp[j] + 2*qv[j])*qd[j] - 3*qv[j]*aqd[j])/(qd[j]**5)) / (qp[j]**2) / qv[j]
                elif not np.isfinite(qd[j]):
                    tau1_vec[j] = 0.0; tau2_vec[j] = 0.0; tau3_vec[j] = 0.0
                else:
                    tau1_vec[j] = 9.0/(8.0*n**2)
                    tau2_vec[j] = 3.0/(8.0*n**2)
                    tau3_vec[j] = 9.0/(40.0*n**2)

    else:  # lt == 3
        lam1_vec = lam2_vec = lam3_vec = None
        do_tau = False

    # saturation prepass (approximate), identical to MATLAB
    if sat and (sat > 0):
        fx = rho[:, 0].conj().T @ (np.abs(fxc[:, :, 0])**2 + np.abs(fyc[:, :, 0])**2 + np.abs(fzc[:, :, 0])**2)
        for k in range(fxs.shape[2]):
            fx += rho[:, 0].conj().T @ (np.abs(fxc[:, :, k+1])**2 + np.abs(fyc[:, :, k+1])**2 + np.abs(fzc[:, :, k+1])**2)/2.0
            fx += rho[:, 0].conj().T @ (np.abs(fxs[:, :, k])**2 + np.abs(fys[:, :, k])**2 + np.abs(fzs[:, :, k])**2)/2.0
        tmp_flux = np.max(2*np.pi*fx*drho)
        sat = (float(sat)/tmp_flux) if (tmp_flux > 0) else 0.0
    else:
        sat = 0.0

    # allocate volumes
    Nrho, Nz = rho.shape
    if (lt is None) or (lt < 2) or (lt == 0):
        volx = np.zeros((Nrho, Nz, 2*maxm + 1), dtype=float)
        voly = np.zeros_like(volx)
    else:
        K = 1 if (lt == 2) else 3
        volx = np.zeros((Nrho, Nz, 2*maxm + 1, K), dtype=float)
        voly = np.zeros_like(volx)

    # precompute S*F** matrices (Nu x Nz)
    S0F00 = s0w @ f00
    S0F11 = s0w @ f11
    S0F22 = s0w @ f22
    S2F02 = s2w @ f02
    S2F11 = s2w @ f11
    S4F22 = s4w @ f22
    S1F01 = s1w @ f01
    S1F12 = s1w @ f12
    S3F12 = s3w @ f12

    # main azimuth loop
    denom_base = (4*maxm + 1)
    for j in range(0, 4*maxm + 1):
        psi = j/denom_base * 2*np.pi

        # reconstruct focal fields at this azimuth
        fx = fxc[:, :, 0].copy()
        fy = fyc[:, :, 0].copy()
        fz = fzc[:, :, 0].copy()
        for k in range(1, maxm + 1):
            c = np.cos(k*psi); s = np.sin(k*psi)
            fx += fxc[:, :, k]*c + fxs[:, :, k-1]*s
            fy += fyc[:, :, k]*c + fys[:, :, k-1]*s
            fz += fzc[:, :, k]*c + fzs[:, :, k-1]*s

        if sat > 0:
            inten = sat*(np.abs(fx)**2 + np.abs(fy)**2 + np.abs(fz)**2)
            pe = PulsedExcitation(inten, 1, pulse[0], pulse[1])[0]
            tmp0 = 1.0 + triplet*pe
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.sqrt(np.divide(sat*pe, inten, out=np.zeros_like(inten), where=(inten != 0)))
            fx *= scale; fy *= scale; fz *= scale
        else:
            tmp0 = 1.0 + triplet*(np.abs(fx)**2 + np.abs(fy)**2 + np.abs(fz)**2)

        # shorthand
        Ax2 = np.abs(fx)**2
        Ay2 = np.abs(fy)**2
        Az2 = np.abs(fz)**2
        Rxy = np.real(fx*np.conj(fy))
        Rxz = np.real(fx*np.conj(fz))
        Ryz = np.real(fy*np.conj(fz))

        c1 = np.cos(psi);   s1 = np.sin(psi)
        c2 = np.cos(2*psi); s2 = np.sin(2*psi)
        c3 = np.cos(3*psi); s3 = np.sin(3*psi)
        c4 = np.cos(4*psi); s4 = np.sin(4*psi)

        # ux1, ux2, ux3  (Nu x Nz)
        ux1 = (
            (1/4)*(S0F00*Ax2) + (5/16)*kappa*(S0F00*Ax2) +
            (1/4)*(S0F22*Ax2) + (1/8)*kappa*(S0F22*Ax2) +
            (1/4)*(S0F00*Ay2) - (1/16)*kappa*(S0F00*Ay2) +
            (1/4)*(S0F22*Ay2) + (1/8)*kappa*(S0F22*Ay2) +
            (1/4)*(S0F00*Az2) - (1/4)*kappa*(S0F00*Az2) +
            (1/4)*(S0F22*Az2) - (1/4)*kappa*(S0F22*Az2)
        ) + (1/16)*(
            -4*c2*(S2F02*Ax2) - 5*kappa*c2*(S2F02*Ax2) + 3*kappa*c4*(S4F22*Ax2)
            - 6*kappa*s2*(S2F02*Rxy) + 6*kappa*s4*(S4F22*Rxy)
            - 4*c2*(S2F02*Ay2) + kappa*c2*(S2F02*Ay2) - 3*kappa*c4*(S4F22*Ay2)
            - 4*c2*(S2F02*Az2) + 4*kappa*c2*(S2F02*Az2)
        )

        ux2 = (
            (1/4)*c2*(S2F02*Ax2) + (7/8)*kappa*c2*(S2F02*Ax2) +
            (1/4)*c2*(S2F11*Ax2) + (1/8)*kappa*c2*(S2F11*Ax2) -
            (3/8)*kappa*c4*(S4F22*Ax2) +
            (3/4)*kappa*s2*(S2F02*Rxy) - (3/4)*kappa*s4*(S4F22*Rxy) +
            (1/4)*c2*(S2F02*Ay2) + (1/8)*kappa*c2*(S2F02*Ay2) +
            (1/4)*c2*(S2F11*Ay2) + (1/8)*kappa*c2*(S2F11*Ay2) +
            (3/8)*kappa*c4*(S4F22*Ay2) +
            (3/2)*kappa*c1*(S1F01*Rxz) - (3/4)*kappa*c1*(S1F12*Rxz) - (3/4)*kappa*c3*(S3F12*Rxz) -
            (3/4)*kappa*s1*(S1F12*Ryz) - (3/4)*kappa*s3*(S3F12*Ryz) +
            (1/4)*c2*(S2F02*Az2) - kappa*c2*(S2F02*Az2) +
            (1/4)*c2*(S2F11*Az2) - (1/4)*kappa*c2*(S2F11*Az2)
        ) + (
            -(1/4)*(S0F00*Ax2) - (7/8)*kappa*(S0F00*Ax2) +
            (1/4)*(S0F11*Ax2) + (1/8)*kappa*(S0F11*Ax2) -
            (1/4)*(S0F22*Ax2) - (1/2)*kappa*(S0F22*Ax2) -
            (1/4)*(S0F00*Ay2) - (1/8)*kappa*(S0F00*Ay2) +
            (1/4)*(S0F11*Ay2) + (1/8)*kappa*(S0F11*Ay2) -
            (1/4)*(S0F22*Ay2) - (1/2)*kappa*(S0F22*Ay2) -
            (1/4)*(S0F00*Az2) + kappa*(S0F00*Az2) +
            (1/4)*(S0F11*Az2) - (1/4)*kappa*(S0F11*Az2) -
            (1/4)*(S0F22*Az2) + kappa*(S0F22*Az2)
        )

        ux3 = (
            (9/16)*kappa*(S0F00*Ax2) - (3/8)*kappa*(S0F11*Ax2) + (3/8)*kappa*(S0F22*Ax2) +
            (3/16)*kappa*(S0F00*Ay2) - (3/8)*kappa*(S0F11*Ay2) + (3/8)*kappa*(S0F22*Ay2) -
            (3/4)*kappa*(S0F00*Az2) + (3/4)*kappa*(S0F11*Az2) - (3/4)*kappa*(S0F22*Az2)
        ) - (3/16)*(
            3*kappa*c2*(S2F02*Ax2) + 2*kappa*c2*(S2F11*Ax2) - kappa*c4*(S4F22*Ax2) +
            2*kappa*s2*(S2F02*Rxy) - 2*kappa*s4*(S4F22*Rxy) +
            kappa*c2*(S2F02*Ay2) + 2*kappa*c2*(S2F11*Ay2) + kappa*c4*(S4F22*Ay2) +
            8*kappa*c1*(S1F01*Rxz) - 4*kappa*c1*(S1F12*Rxz) - 4*kappa*c3*(S3F12*Rxz) -
            4*kappa*s1*(S1F01*Ryz) + 4*kappa*s1*(S1F12*Ryz) - 4*kappa*s3*(S3F12*Ryz) -
            4*kappa*c2*(S2F02*Az2) - 4*kappa*c2*(S2F11*Az2)
        )

        # uy1, uy2, uy3
        uy1 = (
            (1/4)*(S0F00*Ax2) - (1/16)*kappa*(S0F00*Ax2) + (1/4)*(S0F22*Ax2) + (1/8)*kappa*(S0F22*Ax2) +
            (1/4)*(S0F00*Ay2) + (5/16)*kappa*(S0F00*Ay2) + (1/4)*(S0F22*Ay2) + (1/8)*kappa*(S0F22*Ay2) +
            (1/4)*(S0F00*Az2) - (1/4)*kappa*(S0F00*Az2) + (1/4)*(S0F22*Az2) - (1/4)*kappa*(S0F22*Az2)
        ) + (1/16)*(
            4*c2*(S2F02*Ax2) - kappa*c2*(S2F02*Ax2) - 3*kappa*c4*(S4F22*Ax2) -
            6*kappa*s2*(S2F02*Rxy) - 6*kappa*s4*(S4F22*Rxy) +
            4*c2*(S2F02*Ay2) + 5*kappa*c2*(S2F02*Ay2) + 3*kappa*c4*(S4F22*Ay2) +
            4*c2*(S2F02*Az2) - 4*kappa*c2*(S2F02*Az2)
        )

        uy2 = (
            -(1/4)*c2*(S2F02*Ax2) - (1/8)*kappa*c2*(S2F02*Ax2) -
            (1/4)*c2*(S2F11*Ax2) - (1/8)*kappa*c2*(S2F11*Ax2) +
            (3/8)*kappa*c4*(S4F22*Ax2) +
            (3/4)*kappa*s2*(S2F02*Rxy) + (3/4)*kappa*s4*(S4F22*Rxy) -
            (1/4)*c2*(S2F02*Ay2) - (7/8)*kappa*c2*(S2F02*Ay2) -
            (1/4)*c2*(S2F11*Ay2) - (1/8)*kappa*c2*(S2F11*Ay2) -
            (3/8)*kappa*c4*(S4F22*Ay2) -
            (3/4)*kappa*c1*(S1F12*Rxz) + (3/4)*kappa*c3*(S3F12*Rxz) +
            (3/2)*kappa*s1*(S1F01*Ryz) - (3/4)*kappa*s1*(S1F12*Ryz) + (3/4)*kappa*s3*(S3F12*Ryz) -
            (1/4)*c2*(S2F02*Az2) + kappa*c2*(S2F02*Az2) -
            (1/4)*c2*(S2F11*Az2) + (1/4)*kappa*c2*(S2F11*Az2)
        ) + (
            -(1/4)*(S0F00*Ax2) - (1/8)*kappa*(S0F00*Ax2) +
            (1/4)*(S0F11*Ax2) + (1/8)*kappa*(S0F11*Ax2) -
            (1/4)*(S0F22*Ax2) - (1/2)*kappa*(S0F22*Ax2) -
            (1/4)*(S0F00*Ay2) - (7/8)*kappa*(S0F00*Ay2) +
            (1/4)*(S0F11*Ay2) + (1/8)*kappa*(S0F11*Ay2) -
            (1/4)*(S0F22*Ay2) - (1/2)*kappa*(S0F22*Ay2) -
            (1/4)*(S0F00*Az2) + kappa*(S0F00*Az2) +
            (1/4)*(S0F11*Az2) - (1/4)*kappa*(S0F11*Az2) -
            (1/4)*(S0F22*Az2) + kappa*(S0F22*Az2)
        )

        uy3 = (
            (3/16)*kappa*(S0F00*Ax2) - (3/8)*kappa*(S0F11*Ax2) + (3/8)*kappa*(S0F22*Ax2) +
            (9/16)*kappa*(S0F00*Ay2) - (3/8)*kappa*(S0F11*Ay2) + (3/8)*kappa*(S0F22*Ay2) -
            (3/4)*kappa*(S0F00*Az2) + (3/4)*kappa*(S0F11*Az2) - (3/4)*kappa*(S0F22*Az2)
        ) + (3/16)*(
            kappa*c2*(S2F02*Ax2) + 2*kappa*c2*(S2F11*Ax2) - kappa*c4*(S4F22*Ax2) -
            2*kappa*s2*(S2F02*Rxy) - 2*kappa*s4*(S4F22*Rxy) +
            3*kappa*c2*(S2F02*Ay2) + 2*kappa*c2*(S2F11*Ay2) + kappa*c4*(S4F22*Ay2) +
            4*kappa*c1*(S1F12*Rxz) - 4*kappa*c3*(S3F12*Rxz) -
            8*kappa*s1*(S1F01*Ryz) + 4*kappa*s1*(S1F12*Ryz) - 4*kappa*s3*(S3F12*Ryz) -
            4*kappa*c2*(S2F02*Az2) - 4*kappa*c2*(S2F11*Az2)
        )

        denom = tmp0 * denom_base

        if (lt is None) or (lt < 2) or (lt == 0):
            tmpx = (ux1*lam1_vec + ux2*lam2_vec + ux3*lam3_vec) / denom
            tmpy = (uy1*lam1_vec + uy2*lam2_vec + uy3*lam3_vec) / denom

            volx[:, :, 0] += tmpx
            voly[:, :, 0] += tmpy
            for k in range(1, maxm + 1):
                c = np.cos(k*psi); s = np.sin(k*psi)
                volx[:, :, k]         += 2*tmpx*c
                volx[:, :, maxm + k]  += 2*tmpx*s
                voly[:, :, k]         += 2*tmpy*c
                voly[:, :, maxm + k]  += 2*tmpy*s

        elif lt == 2:
            tmpx = (ux1*lam1_vec + ux2*lam2_vec + ux3*lam3_vec) / denom
            tmpy = (uy1*lam1_vec + uy2*lam2_vec + uy3*lam3_vec) / denom
            volx[:, :, 0, 0] += tmpx
            voly[:, :, 0, 0] += tmpy
            for k in range(1, maxm + 1):
                c = np.cos(k*psi); s = np.sin(k*psi)
                volx[:, :, k, 0]        += 2*tmpx*c
                volx[:, :, maxm + k, 0] += 2*tmpx*s
                voly[:, :, k, 0]        += 2*tmpy*c
                voly[:, :, maxm + k, 0] += 2*tmpy*s

            tmpx_tau = (ux1*tau1_vec + ux2*tau2_vec + ux3*tau3_vec) / denom
            tmpy_tau = (uy1*tau1_vec + uy2*tau2_vec + uy3*tau3_vec) / denom
            volx[:, :, 0, 1] += tmpx_tau
            voly[:, :, 0, 1] += tmpy_tau
            for k in range(1, maxm + 1):
                c = np.cos(k*psi); s = np.sin(k*psi)
                volx[:, :, k, 1]        += 2*tmpx_tau*c
                volx[:, :, maxm + k, 1] += 2*tmpx_tau*s
                voly[:, :, k, 1]        += 2*tmpy_tau*c
                voly[:, :, maxm + k, 1] += 2*tmpy_tau*s

        elif lt == 3:
            ux1n = ux1/denom; ux2n = ux2/denom; ux3n = ux3/denom
            volx[:, :, 0, 0] += ux1n
            volx[:, :, 0, 1] += ux2n
            volx[:, :, 0, 2] += ux3n
            for k in range(1, maxm + 1):
                c = np.cos(k*psi); s = np.sin(k*psi)
                volx[:, :, k, 0]        += 2*ux1n*c
                volx[:, :, maxm + k, 0] += 2*ux1n*s
                volx[:, :, k, 1]        += 2*ux2n*c
                volx[:, :, maxm + k, 1] += 2*ux2n*s
                volx[:, :, k, 2]        += 2*ux3n*c
                volx[:, :, maxm + k, 2] += 2*ux3n*s

            uy1n = uy1/denom; uy2n = uy2/denom; uy3n = uy3/denom
            voly[:, :, 0, 0] += uy1n
            voly[:, :, 0, 1] += uy2n
            voly[:, :, 0, 2] += uy3n
            for k in range(1, maxm + 1):
                c = np.cos(k*psi); s = np.sin(k*psi)
                voly[:, :, k, 0]        += 2*uy1n*c
                voly[:, :, maxm + k, 0] += 2*uy1n*s
                voly[:, :, k, 1]        += 2*uy2n*c
                voly[:, :, maxm + k, 1] += 2*uy2n*s
                voly[:, :, k, 2]        += 2*uy3n*c
                voly[:, :, maxm + k, 2] += 2*uy3n*s

    return {
        "NA": NA, "n0": n0, "n": n, "n1": n1, "focpos": focpos, "lambda": lamem,
        "mag": mag, "av": av, "zpin": zpin, "atf": ([] if atf is None else atf),
        "kappa": kappa, "lt": lt, "pulse": pulse, "sat": sat, "triplet": triplet,
        "rho": exc["rho"], "z": exc["z"], "volx": volx, "voly": voly,
        "qv": qv, "qp": qp,
    }



def SEPDipole(rho, z, NA, n1, n, n2, d1, d, d2, lamb, mag, focpos,
              atf=None, ring=None, block=None, orient=None, maxnum=2000):
    """

    Parameters
    ----------
    rho : array-like or (2,)
        Radial camera coordinates. If len(rho)==2, treat as [rho_min, rho_max] and
        generate a grid with step lamb/50 (as in MATLAB).
        Units: same as in your optical train (typically μm at the image plane).
    z : float
        Molecule distance from the bottom of its layer (same length units as d,* and lamb).
    NA : float
        Numerical aperture.
    n1 : array-like (bottom -> up)   below the dipole layer. n1[0] is the medium
        just below the dipole layer (this is the one used in aplanatic prefactor).
    n : complex or float
        Refractive index of dipole (emitter) layer.
    n2 : array-like (bottom -> up)   above the dipole layer.
    d1 : array-like
        Thicknesses for layers below the dipole layer. len(d1) == len(n1)-1.
    d : float
        Thickness of the dipole layer.
    d2 : array-like
        Thicknesses for layers above the dipole layer. len(d2) == len(n2)-1.
    lamb : float
        Wavelength (same length units as z, d*, rho-step choice).
    mag : float
        Magnification.
    focpos : float
        Focus position along the optical axis (same length units).
    atf : None, scalar, or length-2 iterable, optional
        Coverslip/immersion correction.
        - None: no correction.
        - scalar -> index of coverslip; only reflection losses.
        - (n_cs, dz): with aberration (thickness mismatch dz).
    ring : None, callable, or string, optional
        Phase modulation as a function of sin(theta)/sin(theta_max).
        If callable, called as ring(rad) and added to phase.
        If string, evaluated as Python expression with `rad` (NumPy array) in scope.
    block : None, scalar, or (bmin,bmax), optional
        Dark-field beam block. When scalar b in [0,1]: chi_min = b*chi_max.
        When (bmin,bmax): chi_min = bmin*chi_max, chi_max = bmax*chi_max.
    orient : None or (theta, phi), optional
        If provided, compute oriented-molecule image intensities (intx only is used
        in that branch in MATLAB). Units: radians.
    maxnum : int
        Angular sampling (default 2000 like MATLAB).

    Returns
    -------
    intx, inty, intz : (P, Nrho) float arrays
        Intensity distributions for x-, y-, z- dipoles (or oriented case).
    rho_out : (Nrho,) float
        The radial sample used (image plane, after `mag*rho`).
    phi_out : (P,) float
        Azimuth grid (0 .. 2π, step π/100).
    fxx0, fxx2, fxz, byx0, byx2, byz : 1D complex arrays, length = Nrho
        Field component Hankel transforms (cosine/sine harmonics) as in MATLAB.
    """
    # --- constants & inputs ---
    ni = 1.0  # imaging medium (air) in original code
    k_i = 2*np.pi / lamb * ni

    n1 = np.atleast_1d(np.asarray(n1, dtype=complex)).ravel()
    n2 = np.atleast_1d(np.asarray(n2, dtype=complex)).ravel()
    d1 = np.atleast_1d(np.asarray(d1, dtype=float)).ravel()
    d2 = np.atleast_1d(np.asarray(d2, dtype=float)).ravel()
    n  = complex(n)
    d  = float(d)

    # rho grid
    rho = np.asarray(rho, dtype=float).ravel()
    if rho.size == 2:
        drho = lamb / 50.0
        rho = np.arange(rho[0], rho[1] + 1e-12, drho)
    # camera plane scaling
    rho = mag * rho
    rho = rho.reshape(-1)      # (Nrho,)
    Nrho = rho.size

    # --- angular ranges (objective) ---
    # clamp NA/(mag*ni) to <= 1 to avoid invalid asin
    s = NA / (mag * ni)
    s = float(min(max(s, 0.0), 1.0))
    chi_max = np.arcsin(s)

    if block is None:
        chi_min = 0.0
    else:
        b = np.atleast_1d(block).astype(float)
        if b.size == 1:
            chi_min = float(b[0]) * chi_max
        else:
            chi_min = float(b[0]) * chi_max
            chi_max = float(b[1]) * chi_max

    dchi = (chi_max - chi_min) / maxnum
    chi  = chi_min + (np.arange(maxnum) + 0.5) * dchi  # (M,)
    M    = chi.size

    ci = np.cos(chi)                         # (M,)
    si = np.sin(chi)                         # (M,)
    s0 = (ni / n1[0]) * mag * si             # (M,)
    # allow complex TIR region
    c0 = np.sqrt(1.0 - s0**2 + 0j)           # (M,)
    psi = np.arcsin(s0)                      # (M,)

    # --- Dipole fields in layered medium (downward half-space) ---
    # DipoleL expects angles in the dipole layer (psi), and z in phase units 2π/λ
    v, pc, ps, *_ = dipoleL(psi, 2*np.pi*z/lamb,
                            n1, n, n2,
                            2*np.pi*d1/lamb, 2*np.pi*d/lamb, 2*np.pi*d2/lamb)
    # v,pc,ps : (M, Nz). z is scalar here => Nz=1
    v  = v[:, 0]
    pc = pc[:, 0]
    ps = ps[:, 0]

    # --- coverslip / immersion correction (atf) ---
    if atf is not None:
        atf = np.atleast_1d(atf)
        if atf.size == 1:
            ncs = complex(atf[0])
            # Fresnel: from n1[0] to coverslip (p/s)
            # our fresnel(w1, n1, n2) returns rp, rs, tp, ts
            _, _, t_p1, t_s1 = fresnel(n1[0]*c0, n1[0], ncs)
            # back to n1[0] (opposite direction)
            w_cs = np.sqrt(ncs**2 - (n1[0]**2) * (s0**2) + 0j)
            _, _, t_p2, t_s2 = fresnel(w_cs, ncs, n1[0])
            tmp, tms = t_p1, t_s1
            mp,  ms  = t_p2, t_s2
        else:
            ncs, dz = complex(atf[0]), float(atf[1])
            # upward: [n1(1) -> ncs -> ncs] with internal thickness +dz
            _, _, t_p1, t_s1 = fresnel(n1[0]*c0, np.array([n1[0], ncs, ncs], dtype=complex),
                                       np.array([2*np.pi*dz/lamb], dtype=float))
            # downward: [ncs -> n1(1) -> n1(1)] with internal thickness -dz
            w_cs = np.sqrt(ncs**2 - (n1[0]**2) * (s0**2) + 0j)
            _, _, t_p2, t_s2 = fresnel(w_cs, np.array([ncs, n1[0], n1[0]], dtype=complex),
                                       np.array([-2*np.pi*dz/lamb], dtype=float))
            tmp, tms = t_p1, t_s1
            mp,  ms  = t_p2, t_s2

        # apply to fields
        v  = tmp * mp * v
        pc = tmp * mp * pc
        ps = tms * ms * ps

    # --- phase through the imaging system & optional ring modulation ---
    # Default phase from focus position: phase = -n1(1)*c0*focpos
    phase = -n1[0] * c0 * focpos
    if ring is not None:
        # rad = sin(theta)/sin(theta_max) in MATLAB, here s0/max(s0)
        rad = s0 / (np.max(np.abs(s0)) + 1e-30)
        if callable(ring):
            phase = phase + ring(rad)
        elif isinstance(ring, str):
            # Evaluate with 'rad' and numpy in scope (safe-ish)
            phase = phase + eval(ring, {"np": np}, {"rad": rad})
        else:
            raise ValueError("`ring` must be None, callable(rad)->phase, or a string expression using `rad`.")

    ez = np.exp(1j * 2*np.pi/lamb * phase)   # (M,)

    # aplanatic objective prefactor
    fac = dchi * si * np.sqrt(ci / c0)       # (M,)

    # --- Bessel transforms (Hankel-like integrals) ---
    # argument matrix: (M, Nrho) = (k_i * si)[:,None] * rho[None,:]
    barg = (k_i * si)[:, None] * rho[None, :]
    j0 = jv(0, barg)
    j1 = jv(1, barg)
    j2 = jv(2, barg)

    ezi = fac * ci * ez      # (M,)
    ezr = fac * ez           # (M,)

    # NOTE: In MATLAB these are row vectors (1 x Nrho) after the angular integration.
    # We compute them as (Nrho,) 1D arrays via dot-products over angle M.
    # (ezi*pc + ezr*ps) has shape (M,), j0 is (M,Nrho) => rowvec @ matrix -> (Nrho,)
    fxx0 = (ezi * pc + ezr * ps) @ j0
    fxx2 = -(ezi * pc - ezr * ps) @ j2
    fxz  = -2j * (ezi * v)        @ j1

    byx0 = (ezr * pc + ezi * ps) @ j0
    byx2 = -(ezr * pc - ezi * ps) @ j2
    byz  = -2j * (ezr * v)        @ j1

    # --- build phi grid and intensities ---
    phi = np.arange(0, 2*np.pi + 1e-12, np.pi/100.0)  # (P,)
    P   = phi.size
    phi_col = phi[:, None]                             # (P,1)

    # expand the 1D "radial" arrays across phi rows
    # shapes (P, Nrho)
    F0 = np.broadcast_to(fxx0[None, :], (P, Nrho))
    F2 = np.broadcast_to(fxx2[None, :], (P, Nrho))
    V1 = np.broadcast_to(fxz[None, :],  (P, Nrho))
    B0 = np.broadcast_to(byx0[None, :], (P, Nrho))
    B2 = np.broadcast_to(byx2[None, :], (P, Nrho))
    BZ = np.broadcast_to(byz[None, :],  (P, Nrho))

    if orient is not None and len(np.atleast_1d(orient)) >= 2:
        th0 = float(orient[0])
        ph0 = float(orient[1])
        phi_shift = phi_col - ph0

        s_th, c_th = np.sin(th0), np.cos(th0)
        c2 = np.cos(2*phi_shift)
        c1 = np.cos(phi_shift)
        s2 = np.sin(2*phi_shift)
        s1 = np.sin(phi_shift)

        # x-polarized camera channel for oriented dipole
        termE = s_th * (F0 + c2 * F2) + c_th * (c1 * V1)
        termB = s_th * (B0 + c2 * B2) + c_th * (c1 * BZ)
        inty  = np.real(termE * np.conj(termB))

        # z-polarized camera component
        termE_z = s_th * (s2 * F2) + c_th * (s1 * V1)
        termB_z = s_th * (s2 * B2) + c_th * (s1 * BZ)
        intz    = np.real(termE_z * np.conj(termB_z))

        # MATLAB returns intx = inty + intz in the oriented branch
        intx = inty + intz
    else:
        # Standard (unoriented) x/y/z dipole channels
        c2 = np.cos(2*phi_col)
        s2 = np.sin(2*phi_col)

        intx = np.real((F0 + c2 * F2) * np.conj(B0 + c2 * B2))
        inty = np.real((s2 * F2)      * np.conj(s2 * B2))
        # intz independent of phi in this formulation
        intz = np.real(fxz * np.conj(byz))[None, :]
        intz = np.broadcast_to(intz, (P, Nrho))

    return (intx, inty, intz,
            rho, phi,
            fxx0, fxx2, fxz, byx0, byx2, byz)


def SEPImage(al, be, nn, pixel, rho, fxx0, fxx2, fxz, byx0, byx2, byz):
    """
      [int, x, y, ex, ey, bx, by] = SEPImage(al,be,nn,pixel,rho,fxx0,fxx2,fxz,byx0,byx2,byz)

    Parameters
    ----------
    al, be : float
        Dipole polar (alpha) and azimuth (beta) angles [radians].
    nn : int or (ny, nx)
        Half-extent in pixels (MATLAB-style). If scalar n, use (n, n).
        The output grid spans y = -ny:ny and x = -nx:nx (inclusive).
    pixel : float
        Pixel size (same units as rho), used to scale rho → pixel units.
    rho : (Nr,) array
        Radial sample positions corresponding to the 1D field terms below (same units as pixel).
    fxx0, fxx2, fxz, byx0, byx2, byz : (Nr,) complex arrays
        Radial harmonics returned from SEPDipole.

    Returns
    -------
    int : (Ny, Nx) float
        Image intensity.
    x, y : (Ny, Nx) float
        Pixel coordinate grids (as in MATLAB meshgrid(-nx:nx, -ny:ny)).
    ex, ey, bx, by : (Ny, Nx) complex
        Field components after rotations (for diagnostics).
    """
    # --- parse nn like MATLAB ---
    if np.isscalar(nn):
        ny = nx = int(nn)
    else:
        nn = np.asarray(nn).ravel()
        if nn.size != 2:
            raise ValueError("nn must be scalar or length-2 (ny, nx).")
        ny, nx = int(nn[0]), int(nn[1])

    # --- coordinate grids (MATLAB: [x,y] = meshgrid(-nx:nx, -ny:ny)) ---
    xv = np.arange(-nx, nx + 1, dtype=float)
    yv = np.arange(-ny, ny + 1, dtype=float)
    x, y = np.meshgrid(xv, yv, indexing='xy')  # shapes (Ny, Nx)

    # polar coords
    p = np.angle(x + 1j*y)
    r = np.hypot(x, y)  # radius in pixels

    # --- radial interpolation points (rho in pixel units) ---
    rho_pix = np.asarray(rho, dtype=float) / float(pixel)

    # PCHIP interpolators (handle complex by splitting)
    def pchip_complex(x, y):
        xr = PchipInterpolator(x, np.real(y), extrapolate=False)
        xi = PchipInterpolator(x, np.imag(y), extrapolate=False)
        def eval_at(z):
            zr = xr(z)
            zi = xi(z)
            out = zr + 1j*zi
            # outside domain -> NaN; make them 0 like typical optical PSF tails
            return np.nan_to_num(out, nan=0.0)
        return eval_at

    I_fxx0 = pchip_complex(rho_pix, fxx0)
    I_fxx2 = pchip_complex(rho_pix, fxx2)
    I_fxz  = pchip_complex(rho_pix, fxz)
    I_byx0 = pchip_complex(rho_pix, byx0)
    I_byx2 = pchip_complex(rho_pix, byx2)
    I_byz  = pchip_complex(rho_pix, byz)

    # evaluate on the (Ny, Nx) radius grid
    F0  = I_fxx0(r)
    F2  = I_fxx2(r)
    FXZ = I_fxz(r)
    B0  = I_byx0(r)
    B2  = I_byx2(r)
    BYZ = I_byz(r)

    # shorthand trig
    sa, ca = np.sin(al), np.cos(al)
    c1 = np.cos(p - be)
    s1 = np.sin(p - be)
    c2 = np.cos(2*(p - be))
    s2 = np.sin(2*(p - be))

    # fields (exactly mirroring MATLAB expressions)
    ex = sa * (F0 + c2 * F2) + ca * c1 * FXZ
    by = sa * (B0 + c2 * B2) + ca * c1 * BYZ
    ey = sa * (s2 * F2)      + ca * s1 * FXZ
    bx = -(sa * (s2 * B2)    + ca * s1 * BYZ)

    # rotate camera axes by beta (MATLAB post-rotation)
    tmp = ex * np.cos(be) - ey * np.sin(be)
    ey  = ey * np.cos(be) + ex * np.sin(be)
    ex  = tmp

    tmp = by * np.cos(be) + bx * np.sin(be)
    bx  = bx * np.cos(be) - by * np.sin(be)
    by  = tmp

    # image intensity
    inten = np.real(ex * np.conj(by) - ey * np.conj(bx))

    return inten, x, y, ex, ey, bx, by


def _pchip_complex(x, y):
    """PCHIP for complex y over x; returns callable that zero-fills out-of-range."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, complex).ravel()
    xr = PchipInterpolator(x, y.real, extrapolate=False)
    xi = PchipInterpolator(x, y.imag, extrapolate=False)
    def eval_at(z):
        zr = xr(z); zi = xi(z)
        out = zr + 1j*zi
        return np.nan_to_num(out, nan=0.0)
    return eval_at

def PatternGeneratePos_SEP(
    xcent, ycent, z,
    NA, n0, n, n1, d0, d, d1,
    lamem, mag, focus,
    atf=None, ring=None,
    pixel=24, nn=30, field=100,
    be=0.0, al=np.pi/2,
    pic=0
):
    """
    Python port of PatternGeneratePos_SEP.m
    - Computes SEPdata internally via SEPDipole.
    - Performs subpixel shift using bilinear interpolation (like interp2(...,'linear',0)).

    Returns
    -------
    int_img : (2*field_y+1, 2*field_x+1) ndarray
        Intensity image after placing and subpixel shifting the PSF.
    Sx, Sy : coordinate grids of the output canvas (same shapes as int_img)
    """
    # --- defaults & argument normalization (keep MATLAB parity) ---
    if atf is None: atf = None
    if ring is None: ring = None
    if pixel is None: pixel = 24
    if nn is None: nn = 30
    if field is None: field = 100
    if be is None: be = 0.0
    if al is None: al = np.pi/2
    if focus is None: focus = 0.9
    if mag is None: mag = 400
    if lamem is None: lamem = 0.690
    if d1 is None: d1 = []
    if d  is None: d  = 0.001
    if d0 is None: d0 = []
    if n1 is None: n1 = 1.0
    if n  is None: n  = 1.0
    if n0 is None: n0 = 1.52

    # field can be scalar or (fy, fx)
    if np.isscalar(field):
        fy = fx = int(field)
    else:
        field = np.asarray(field).ravel()
        fy, fx = int(field[0]), int(field[1])

    # ensure numeric types
    al = float(al); be = float(be)
    xcent = float(0 if xcent is None else xcent)
    ycent = float(0 if ycent is None else ycent)

    # --- precompute radial harmonics (SEPdata) on-demand ---
    # MATLAB: [0, 1.5*max(nn)*pixel/mag]
    rho_max = 1.5 * max(nn, nn) * pixel / mag
    # SEPDipole signature: (rho_range, z, NA, n1, n, n2, d1, d, d2, lambda, mag, focpos, atf, ring, block, orient)
    # Note: MATLAB uses (n0 below, n1 above); our SEPDipole uses (n1 below, n2 above). Map: n1<-n0, n2<-n1 here.
    intx, inty, intz, rho, _, fxx0, fxx2, fxz, byx0, byx2, byz = SEPDipole(
        np.array([0.0, rho_max]),
        z, NA,
        np.atleast_1d(n0),  # below
        n,
        np.atleast_1d(n1),  # above
        np.atleast_1d(d0), d, np.atleast_1d(d1),
        lamem, mag, focus,
        atf=atf, ring=ring, block=None, orient=None
    )

    # --- create output canvas coordinates (for optional diagnostics) ---
    Sx, Sy = np.meshgrid(
        np.arange(-fx - xcent, fx - xcent + 1, dtype=float),
        np.arange(-fy + ycent, fy + ycent + 1, dtype=float),
        indexing='xy'
    )

    # --- local PSF patch on (-nn:nn) ---
    grid = np.arange(-nn, nn+1, dtype=float)
    x, y = np.meshgrid(grid, grid, indexing='xy')
    p = np.angle(x + 1j*y)
    r = np.hypot(x, y)

    # interpolate radial harmonics at r (convert rho to pixel units)
    rho_pix = np.asarray(rho, float) / float(pixel)
    F0  = _pchip_complex(rho_pix, fxx0)(r)
    F2  = _pchip_complex(rho_pix, fxx2)(r)
    FXZ = _pchip_complex(rho_pix, fxz )(r)
    B0  = _pchip_complex(rho_pix, byx0)(r)
    B2  = _pchip_complex(rho_pix, byx2)(r)
    BYZ = _pchip_complex(rho_pix, byz )(r)

    sa, ca = np.sin(al), np.cos(al)
    c1 = np.cos(p - be); s1 = np.sin(p - be)
    c2 = np.cos(2*(p - be)); s2 = np.sin(2*(p - be))

    # fields (match MATLAB)
    ex = sa * (F0 + c2 * F2) + ca * c1 * FXZ
    by = sa * (B0 + c2 * B2) + ca * c1 * BYZ
    ey = sa * (s2 * F2)      + ca * s1 * FXZ
    bx = -(sa * (s2 * B2)    + ca * s1 * BYZ)

    # rotate camera axes by be (post-rotation like MATLAB)
    tmp = ex * np.cos(be) - ey * np.sin(be)
    ey  = ey * np.cos(be) + ex * np.sin(be)
    ex  = tmp
    tmp = by * np.cos(be) + bx * np.sin(be)
    bx  = bx * np.cos(be) - by * np.sin(be)
    by  = tmp

    # intensity (Ny, Nx) for the PSF patch
    int_patch = np.real(ex * np.conj(by) - ey * np.conj(bx))

    # --- paste into a larger zero canvas, top-left, then subpixel-shift like MATLAB ---
    H, W = 2*fy + 1, 2*fx + 1
    tmpint = np.zeros((H, W), dtype=float)
    h0, w0 = int_patch.shape
    tmpint[0:h0, 0:w0] = int_patch

    # subpixel shift (MATLAB's interp2 on index grids with origin 1)
    # xshift = field(2)-floor(size(int,2)/2)+xcent;
    # yshift = field(1)-floor(size(int,1)/2)-ycent;
    xshift = fx - np.floor(w0/2.0) + xcent
    yshift = fy - np.floor(h0/2.0) - ycent

    # MATLAB xi,yi are 1-based; for map_coordinates we switch to 0-based:
    # xi_matlab = 1..W - xshift  -> xi_python = xi_matlab - 1
    xi_mat = np.arange(1, W+1, dtype=float)[None, :] - xshift  # (1,W)
    yi_mat = np.arange(1, H+1, dtype=float)[:, None] - yshift  # (H,1)
    XI = np.broadcast_to(xi_mat, (H, W)) - 1.0
    YI = np.broadcast_to(yi_mat, (H, W)) - 1.0

    # sample with bilinear interpolation, constant outside = 0
    int_img = map_coordinates(tmpint, [YI, XI], order=1, mode='constant', cval=0.0)

    # Optional: display like mim if pic==1
    if pic == 1:
        import matplotlib.pyplot as plt
        plt.imshow(int_img, cmap='hot', origin='upper', aspect='equal')
        plt.axis('off'); plt.show()

    return int_img, Sx, Sy


def PatternGeneration(z, NA, n0, n, n1, d0, d, d1,
                      lamem, mag, focus,
                      atf=None, ring=None,
                      pixel=None, nn=None,
                      be_res=None, al_res=None, pic=None):
    """
    Python port of PatternGeneratePos_SEP / PatternGeneration (MATLAB).

    Parameters
    ----------
    z : float
        Molecule distance from bottom of dipole layer (same units as d,* and lamem).
    NA : float
    n0 : array-like
        Stack below dipole layer (bottom -> up). In MATLAB this is `n0`.
    n : complex/float
        Index of dipole (emitter) layer.
    n1 : array-like
        Stack above dipole layer (bottom -> up). In MATLAB this is `n1`.
    d0 : array-like
        Thicknesses for layers below (len == len(n0)-1).
    d : float
        Thickness of dipole layer.
    d1 : array-like
        Thicknesses for layers above (len == len(n1)-1).
    lamem : float
        Wavelength.
    mag : float
        Magnification.
    focus : float
        Focus position along optical axis (same units).
    atf : optional
    ring : optional
    pixel : float, default 24
        Pixel size (same units as rho).
    nn : int or (ny, nx), default [10,10] (MATLAB style)
        Half-size of the PSF stamp.
    be_res : float (deg), default 10
        Minimum resolution for in-plane angle.
    al_res : float (deg), default 10
        Minimum resolution for out-of-plane angle.
    pic : int {0,1}, default 0
        If 1, shows a tiled preview of the masks.

    Returns
    -------
    model : dict
        Keys: rho, theta, phi, mask, fxx0, fxx2, fxz, byx0, byx2, byz
    """
    # ---- defaults like MATLAB ----
    if atf is None: atf = None
    if ring is None: ring = None
    if pic is None: pic = 0
    if be_res is None or be_res == 0: be_res = 10.0
    if al_res is None or al_res == 0: al_res = 10.0
    if nn is None: nn = [10, 10]
    if pixel is None: pixel = 24.0

    # Ensure shapes
    n0 = np.atleast_1d(n0)
    n1 = np.atleast_1d(n1)
    d0 = np.atleast_1d(d0)
    d1 = np.atleast_1d(d1)

    # Disk apodization (normalized)
    bck = Disk(nn)

    # ---- Sample orientations like MATLAB ----
    thetas = []
    phis   = []
    # k from 90 down to 0 in steps of al_res (degrees)
    for k in np.arange(90, -1, -al_res):
        al = np.deg2rad(k)
        if k == 90:
            jj  = int(np.round(180.0 / be_res))
            dbe = np.pi / max(jj, 1)
        elif k == 0:
            jj  = 1
            dbe = 0.0
        else:
            jj  = int(np.round(np.sin(al) * 360.0 / be_res))
            jj  = max(jj, 1)
            dbe = 2*np.pi / jj
        for j in range(jj):
            thetas.append(al)
            phis.append(dbe * j)
    thetas = np.array(thetas, float)
    phis   = np.array(phis, float)

    # ---- Radial harmonics from SEPDipole (SEPdata) ----
    # [0, 1.5*max(nn)*pixel/mag]
    nn_arr = np.atleast_1d(nn).astype(int).ravel()
    if nn_arr.size == 1:
        nn_arr = np.array([nn_arr[0], nn_arr[0]], dtype=int)
    rho_max = 1.5 * max(nn_arr) * pixel / mag

    intx, inty, intz, rho, _, fxx0, fxx2, fxz, byx0, byx2, byz = SEPDipole(
        np.array([0.0, rho_max]),
        z, NA,
        n0, n, n1,   # below, dip, above (MATLAB order preserved)
        d0, d, d1,
        lamem, mag, focus,
        atf=atf, ring=ring
    )

    # ---- Build masks over orientations (PSF stamp size nn) ----
    # SEPImage returns: (int, x, y, ex, ey, bx, by)
    masks = []
    for al, phi in zip(thetas, phis):
        be = -phi
        img, *_ = SEPImage(al, be, nn, pixel, rho, fxx0, fxx2, fxz, byx0, byx2, byz)
        masks.append(img)
    mask = np.stack(masks, axis=2)  # (Ny, Nx, Norient)

    # ---- Apodize by Disk, normalize per-slice by max ----
    for j in range(mask.shape[2]):
        sl = mask[:, :, j]
        sl = sl * bck
        m  = np.max(sl)
        if m > 0:
            sl = sl / m
        mask[:, :, j] = sl

    # ---- Optional tiled preview ----
    if pic == 1:
        Nor = mask.shape[2]
        col = int(np.ceil(np.sqrt(Nor)))
        h, w = mask.shape[:2]
        rows = int(np.ceil(Nor / col))
        im = np.zeros((rows*h, col*w), dtype=float)
        for j in range(Nor):
            r = j // col
            c = j % col
            im[r*h:(r+1)*h, c*w:(c+1)*w] = mask[:, :, j]
        plt.figure()
        plt.imshow(im, cmap='hot', origin='upper', aspect='equal')
        plt.axis('off')
        plt.title('Orientation masks')
        plt.show()

    # ---- Pack model ----
    model = {
        "rho":  rho,
        "theta": thetas,
        "phi":   phis,
        "mask":  mask,
        "fxx0":  fxx0,
        "fxx2":  fxx2,
        "fxz":   fxz,
        "byx0":  byx0,
        "byx2":  byx2,
        "byz":   byz,
    }
    return model



def RotateEMField(fxc, fxs=None, fyc=None, fys=None, fzc=None, fzs=None, phi=None):
    """
    Python port of:
      [fxcr, fxsr, fycr, fysr, fzcr, fzsr] = RotateEMField(fxc, fxs, fyc, fys, fzc, fzs, phi)

    Two calling modes:

    1) Struct mode (MATLAB nargin==2):
       RotateEMField(F, phi)
       where F is a dict-like with keys: 'fxc','fyc','fzc','fxs','fys','fzs'.
       Returns (F_rot, None, None, None, None, None) — the rotated dict is the first element.

    2) Array mode:
       RotateEMField(fxc, fxs, fyc, fys, fzc, fzs, phi)
       Arrays may be complex. The harmonic convention follows MATLAB:
         - order-0 terms at index [:,:,0]
         - order-j cosine terms at [:,:,j] paired with sine terms at [:,:,j-1]

    Notes
    -----
    - This port **exactly** follows your MATLAB equations (even lines like `fxsr = fys; fysr = fys;`).
    - The loop uses zero-based indexing in Python: `j` in 0..(K-1) maps to MATLAB's `j=1..K`,
      and we access `[..., j+1]` vs `[..., j]` to match your `j+1` / `j` pattern.
    """

    # ---- Struct mode: RotateEMField(F, phi) ----
    if fyc is None and isinstance(fxc, (dict,)):
        phi = float(fxs)  # here 'fxs' carries phi in MATLAB's nargin==2
        F = fxc

        # Pull components (allow missing keys to be None)
        Ffxc = np.array(F.get('fxc', None))
        Ffyc = np.array(F.get('fyc', None))
        Ffzc = np.array(F.get('fzc', None))
        Ffxs = F.get('fxs', None)
        Ffys = F.get('fys', None)
        Ffzs = F.get('fzs', None)

        c = np.cos(phi); s = np.sin(phi)

        # Base (order-0) rotation
        out = {}
        out['fxc'] = c * Ffxc - s * Ffyc
        out['fyc'] = s * Ffxc + c * Ffyc
        out['fzc'] = Ffzc
        out['fxs'] = Ffys  # NOTE: this mirrors your MATLAB exactly
        out['fys'] = Ffys
        out['fzs'] = Ffzs

        # Higher harmonics (if present)
        if Ffxs is not None and np.size(Ffxs) > 0:
            Ffxs = np.array(Ffxs)
            Ffys = np.array(Ffys)
            Ffzs = np.array(Ffzs)

            K = Ffxs.shape[2]  # number of sine-harmonic slices
            # ensure we can assign into [:,:,j+1]
            # We assume Ffxc/Ffyc/Ffzc have at least K+1 slices in axis=2.
            for j in range(K):  # MATLAB j=1..K
                jj = j + 1
                cjp = np.cos(jj * phi)
                sjp = np.sin(jj * phi)

                # tmp.fxc(:,:,j+1)
                out['fxc'][..., jj] = (
                    -s * (Ffyc[..., jj] * cjp - Ffys[..., j] * sjp)
                    + c * (Ffxc[..., jj] * cjp - Ffxs[..., j] * sjp)
                )
                # tmp.fxs(:,:,j)
                out['fxs'][..., j] = (
                    -s * (Ffyc[..., jj] * sjp + Ffys[..., j] * cjp)
                    + c * (Ffxc[..., jj] * sjp + Ffxs[..., j] * cjp)
                )
                # tmp.fyc(:,:,j+1)
                out['fyc'][..., jj] = (
                    s * (Ffxc[..., jj] * cjp - Ffxs[..., j] * sjp)
                    + c * (Ffyc[..., jj] * cjp - Ffys[..., j] * sjp)
                )
                # tmp.fys(:,:,j)
                out['fys'][..., j] = (
                    s * (Ffxc[..., jj] * sjp + Ffxs[..., j] * cjp)
                    + c * (Ffyc[..., jj] * sjp + Ffys[..., j] * cjp)
                )
                # tmp.fzc(:,:,j+1), tmp.fzs(:,:,j)
                out['fzc'][..., jj] = (Ffzc[..., jj] * cjp - Ffzs[..., j] * sjp)
                out['fzs'][..., j]  = (Ffzc[..., jj] * sjp + Ffzs[..., j] * cjp)

        # Return like MATLAB's first output; pad with Nones to keep signature uniform
        return out, None, None, None, None, None

    # ---- Array mode: RotateEMField(fxc, fxs, fyc, fys, fzc, fzs, phi) ----
    if phi is None:
        raise ValueError("Array mode requires phi (7th argument).")

    phi = float(phi)
    c = np.cos(phi); s = np.sin(phi)

    fxc = np.array(fxc); fyc = np.array(fyc); fzc = np.array(fzc)
    fxcr = c * fxc - s * fyc
    fycr = s * fxc + c * fyc
    fzcr = fzc

    # Copy base for sine parts (as per your MATLAB lines)
    fxsr = np.array(fys)
    fysr = np.array(fys)
    fzsr = np.array(fzs)

    if fxs is not None and np.size(fxs) > 0:
        fxs = np.array(fxs); fys = np.array(fys); fzs = np.array(fzs)
        K = fxs.shape[2]
        for j in range(K):  # MATLAB j=1..K → Python 0..K-1 (use j+1 for the cosine slices)
            jj = j + 1
            cjp = np.cos(jj * phi)
            sjp = np.sin(jj * phi)

            # fxcr(:,:,j+1)
            fxcr[..., jj] = (
                -s * (fyc[..., jj] * cjp - fys[..., j] * sjp)
                + c * (fxc[..., jj] * cjp - fxs[..., j] * sjp)
            )
            # fxsr(:,:,j)
            fxsr[..., j] = (
                -s * (fyc[..., jj] * sjp + fys[..., j] * cjp)
                + c * (fxc[..., jj] * sjp + fxs[..., j] * cjp)
            )
            # fycr(:,:,j+1)
            fycr[..., jj] = (
                s * (fxc[..., jj] * cjp - fxs[..., j] * sjp)
                + c * (fyc[..., jj] * cjp - fys[..., j] * sjp)
            )
            # fysr(:,:,j)
            fysr[..., j] = (
                s * (fxc[..., jj] * sjp + fxs[..., j] * cjp)
                + c * (fyc[..., jj] * sjp + fys[..., j] * cjp)
            )
            # fzcr(:,:,j+1), fzsr(:,:,j)
            fzcr[..., jj] = (fzc[..., jj] * cjp - fzs[..., j] * sjp)
            fzsr[..., j]  = (fzc[..., jj] * sjp + fzs[..., j] * cjp)

    return fxcr, fxsr, fycr, fysr, fzcr, fzsr


def radialpattern(pattern,
                  pixel,
                  NA,
                  n=None,
                  *,
                  # ---- defaults from your previous MATLAB code ----
                  nn=None,
                  lamex=0.564,
                  resolution=None,
                  rhofield=None,
                  zfield=(0.0, 0.01),
                  fd=3e3,
                  n0=1.52,
                  d0=None,
                  d=0.01,
                  d1=None,
                  over=np.inf,
                  focpos=0.0,
                  atf=None,
                  ring=None,
                  maxm=3,
                  theta_min=None,
                  be_res=10.0,
                  al_res=10.0,
                  pic=0):
    """
    Extended radialpattern:
      - builds excitation harmonics via gauss_exc(...),
      - rotates fields by π/2 and sums (like your MATLAB),
      - samples orientations (θ, φ),
      - synthesizes per-orientation PSFs by radial-harmonic interpolation (cubic),
      - tiles them into 'imtheo' (optionally shown if pic=1).

    Returns
    -------
    fxc, fxs, fyc, fys, fzc, fzs : complex arrays (Nrho, Nz, ...)
    rho : (Nrho,) radial samples used for interpolation
    rr, psi : 2D grids on (-nn..nn)^2 (in pixel units and angles)
    mask : (H, W, K) stack of per-orientation PSFs (|…|^2)
    imtheo : tiled montage image (float)
    theta, phi : (K,) orientation lists (radians)
    """
    # ---- indices (n, n1) handling ----
    if n is None:
        n = 1.0
        n1 = 1.0
    else:
        n_arr = np.atleast_1d(n).astype(float)
        if n_arr.size == 1:
            n = float(n_arr[0]); n1 = n
        else:
            n  = float(n_arr[0]); n1 = float(n_arr[1])

    # ---- defaults for grid/fields ----
    if NA is None or NA == 0:
        NA = 1.49
    if nn is None:
        nn = int(np.ceil(0.6 / float(pixel)))
    if resolution is None:
        resolution = np.array([lamex/0.02, lamex/0.001], dtype=float)
    if rhofield is None:
        rhofield = np.array([-lamex/resolution[0]/2.0, nn*pixel*1.1], float)
    if d0 is None: d0 = []
    if d1 is None: d1 = []
    if theta_min is None:
        theta_min = np.arcsin(0.4 / NA)
    if ring is None:
        # 'cos(psi).*(rad>sin(theta_min))' as string (gauss_exc should evaluate it)
        ring = f"cos(psi).*(rad>sin({theta_min}))"

    # ---- build rr, psi on (-nn..nn) ----
    grid = np.arange(-nn, nn + 1, dtype=float)
    xx, yy = np.meshgrid(grid, grid, indexing='xy')
    rr  = pixel * np.hypot(xx, yy)
    psi = np.angle(xx + 1j*yy)

    # ---- excitation harmonics from your GaussExc port ----
    fxc, fxs, fyc, fys, fzc, fzs, rho, _z = gauss_exc(
        np.asarray(rhofield, float), np.asarray(zfield, float), NA, fd,
        n0, n, n1, np.asarray(d0, float), float(d), np.asarray(d1, float),
        float(lamex), over, float(focpos), atf,
        np.asarray(resolution, float), ring, int(maxm)
    )

    # Rotate by π/2 and sum (exactly your MATLAB sequence)
    fxc2, fxs2, fyc2, fys2, fzc2, fzs2 = RotateEMField(fxc, fxs, fyc, fys, fzc, fzs, np.pi/2.0)
    fxc = fxc + fxc2
    fxs = fxs + fxs2
    fyc = fyc + fyc2
    fys = fys + fys2
    fzc = fzc + fzc2
    fzs = fzs + fzs2

    # pattern='azimuthal' swap/zero (matches your MATLAB)
    if isinstance(pattern, str) and pattern.lower() == 'azimuthal':
        fzc = np.zeros_like(fzc); fzs = np.zeros_like(fzs)
        fxc, fyc = fyc, fxc
        fxs, fys = fys, fxs

    # ---- orientation sampling (θ, φ) as in your snippet ----
    thetas = []
    phis   = []
    for k in np.arange(90, -1, -al_res):
        al = np.deg2rad(k)
        if k == 90:
            jj  = int(np.round(180.0 / be_res))
            jj  = max(jj, 1)
            dbe = np.pi / jj
        elif k == 0:
            jj  = 1
            dbe = 0.0
        else:
            jj  = int(np.round(np.sin(al) * 360.0 / be_res))
            jj  = max(jj, 1)
            dbe = 2*np.pi / jj
        for j in range(jj):
            thetas.append(al)
            phis.append(dbe * j)
    theta = np.array(thetas, float)
    phi   = np.array(phis, float)

    # ---- build masks via radial-harmonic interpolation (cubic) ----
    # MATLAB uses rho(:,1) and ( :,1, … ) → here: [:, 0] and [:, 0, …]
    rho_vec = np.ravel(rho[:, 0]) if np.ndim(rho) == 2 else np.ravel(rho)
    # ensure strictly increasing x for interp
    if np.any(np.diff(rho_vec) <= 0):
        order = np.argsort(rho_vec)
        rho_vec = rho_vec[order]
        def _ord(a):
            return a[order, ...]
        fxc = _ord(fxc); fxs = _ord(fxs)
        fyc = _ord(fyc); fys = _ord(fys)
        fzc = _ord(fzc); fzs = _ord(fzs)

    H, W = rr.shape
    K    = theta.size
    mask = np.zeros((H, W, K), dtype=float)

    # precompute sin/cos(theta/phi)
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi);   sp = np.sin(phi)

    # zero-order combinations as 1D profiles over rho
    A0 = fxc[:, 0, 0]; B0 = fyc[:, 0, 0]; C0 = fzc[:, 0, 0]
    # iterate orientations
    for k in range(K):
        # base profile at this orientation
        prof0 = (A0 * st[k] * cp[k] +
                 B0 * st[k] * sp[k] +
                 C0 * ct[k])

        img = _interp1_cubic_complex(rho_vec, prof0, rr)

        # higher orders j=1..maxm
        for j in range(1, int(maxm) + 1):
            Ac = fxc[:, 0, j];  Bc = fyc[:, 0, j];  Cc = fzc[:, 0, j]
            As = fxs[:, 0, j-1]; Bs = fys[:, 0, j-1]; Cs = fzs[:, 0, j-1]

            prof_c = (Ac * st[k] * cp[k] +
                      Bc * st[k] * sp[k] +
                      Cc * ct[k])
            prof_s = (As * st[k] * cp[k] +
                      Bs * st[k] * sp[k] +
                      Cs * ct[k])

            img += (_interp1_cubic_complex(rho_vec, prof_c, rr) * np.cos(j * psi) +
                    _interp1_cubic_complex(rho_vec, prof_s, rr) * np.sin(j * psi))

        mask[..., k] = np.abs(img) ** 2

    # ---- tile montage (imtheo) ----
    col  = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / col))
    imtheo = np.zeros((rows * H, col * W), dtype=float)
    for j in range(K):
        r = j // col; c = j % col
        imtheo[r*H:(r+1)*H, c*W:(c+1)*W] = mask[..., j]

    # ---- optional display ----
    if pic == 1:
       mim(imtheo)
       

    return fxc, fxs, fyc, fys, fzc, fzs, rho_vec, rr, psi, mask, imtheo, theta, phi


def _coerce_float(x, default=0.0) -> float:
    """Turn scalars/empty/listlike into a single float, with sane default."""
    if x is None:
        return float(default)
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _coerce_array(x) -> np.ndarray:
    """Always return a 1D float array (possibly empty)."""
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()



def _pack_intensity_harmonics(fxc, fxs, fyc, fys, fzc, fzs):
    Nr, Nz, M1 = fxc.shape
    M = M1 - 1
    vol = np.zeros((Nr, Nz, 2*M+1), dtype=float)
    # DC
    vol[..., 0] = (np.abs(fxc[...,0])**2 + np.abs(fyc[...,0])**2 + np.abs(fzc[...,0])**2)
    # j>=1 (cos, sin)
    for j in range(1, M+1):
        # cross terms vanish on average; store per-harmonic energy split into cos/sin “buckets”
        vol[..., j]       = 0.5*(np.abs(fxc[...,j])**2 + np.abs(fyc[...,j])**2 + np.abs(fzc[...,j])**2)
        vol[..., M + j]   = 0.5*(np.abs(fxs[...,j-1])**2 + np.abs(fys[...,j-1])**2 + np.abs(fzs[...,j-1])**2)
    return vol
    
    

# def mdf_confocal_microscopy_py(
#     rhofield: Optional[np.ndarray] = None,
#     zfield: Optional[np.ndarray] = None,
#     NA: float = 1.2,
#     fd: float = 3e3,
#     n0: float = 1.35,
#     n: float = 1.35,
#     n1: float = 1.333,
#     d0: Optional[np.ndarray] = None,
#     d: Optional[float] = None,
#     d1: Optional[np.ndarray] = None,
#     lamex: float = 0.645,
#     over: float = 5e3,
#     focpos: float = 0.0,
#     defoc: float = 0.0,
#     av: float = 75.0,      # unused in this proxy
#     lamem: float = 0.650,  # unused in this proxy
#     mag: float = 60.0,     # unused in this proxy
#     zpin: float = 0.0,     # unused in this proxy
#     atf: Optional[np.ndarray] = None,
#     resolution: int = 50,
#     ring: Optional[Any] = None,
#     maxm: int = 2,
#     kappa: float = 1.0,    # unused in this proxy
#     lt: Optional[Any] = None,     # unused in this proxy
#     pulse: Optional[Any] = None,  # unused in this proxy
#     sat: Optional[Any] = None,    # unused in this proxy
#     triplet: Optional[Any] = None,# unused in this proxy
#     bild: bool = True,
# ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
#     """
#     Python version of the MATLAB MDFConfocalMicroscopy example.
#     Returns four dicts: exc_lin, exc_rad, mdf_lin, mdf_rad.
#     Each dict has keys: 'rho', 'z', 'wave', 'vol' (where 'vol' is wave[...,None]
#     so it can be plotted by FocusImage2D).
#     """
#     # Defaults matching MATLAB example
#     if rhofield is None: rhofield = [0, 2]
#     if zfield is None:   zfield   = [-4, 4]
#     if atf is None:      atf      = []

#     # Coerce layer inputs
#     d0 = _coerce_array(d0 if d0 is not None else [])
#     d1 = _coerce_array(d1 if d1 is not None else [])
#     d  = _coerce_float(0.0 if d is None else d)

#     # ---- LINEAR excitation ----
#     fxc, fxs, fyc, fys, fzc, fzs, rho, z = gauss_exc(
#         rhofield, zfield, NA, fd, n0, n, n1, d0, d, d1,
#         lamex, over, focpos + defoc, atf, resolution, ring, maxm
#     )
#     wave_lin = (np.abs(fxc[...,0])**2 + np.abs(fyc[...,0])**2 + np.abs(fzc[...,0])**2
#                 + 0.5*np.sum(np.abs(fxc[...,1:])**2 + np.abs(fyc[...,1:])**2 + np.abs(fzc[...,1:])**2, axis=2))
    
   
#     # ---- RADIAL “pupil-ring” excitation (as in MATLAB comments) ----
#     # Build from two orthogonal cos(psi) illuminations and add fields
#     fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r, _, _ = gauss_exc(
#         rhofield, zfield, NA, fd, n0, n, n1, d0, d, d1,
#         lamex, over, focpos + defoc, atf, resolution, ring='np.cos(psi)'
#         , maxm=maxm
#     )
#     # rotate by π/2 and add (radial = Ex(ψ)+Ex(ψ+π/2))
#     fxc_rot, fxs_rot, fyc_rot, fys_rot, fzc_rot, fzs_rot = RotateEMField(
#         fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r, np.pi/2
#     )
#     fxc_r += fxc_rot;  fxs_r += fxs_rot
#     fyc_r += fyc_rot;  fys_r += fys_rot
#     fzc_r += fzc_rot;  fzs_r += fzs_rot
    
    
#     wave_rad = (np.abs(fxc_r[...,0])**2 + np.abs(fyc_r[...,0])**2 + np.abs(fzc_r[...,0])**2
#                 + 0.5*np.sum(np.abs(fxc_r[...,1:])**2 + np.abs(fyc_r[...,1:])**2 + np.abs(fzc_r[...,1:])**2, axis=2))
    
#     vol_lin = _pack_intensity_harmonics(fxc, fxs, fyc, fys, fzc, fzs)
#     vol_rad = _pack_intensity_harmonics(fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r)

#     # Proxy MDF: use excitation intensity (reciprocity, no pinhole) as stand-in
#     mdf_lin_2d = wave_lin.copy()
#     mdf_rad_2d = wave_rad.copy()

#     # Pack results; 'vol' is wave[...,None] so FocusImage2D can plot it
#     exc_lin = {"rho": rho, "z": z, "wave": wave_lin, "vol": wave_lin[..., None]}
#     exc_rad = {"rho": rho, "z": z, "wave": wave_rad, "vol": wave_rad[..., None]}
#     mdf_lin = {"rho": rho, "z": z, "wave": mdf_lin_2d, "vol": mdf_lin_2d[..., None]}
#     mdf_rad = {"rho": rho, "z": z, "wave": mdf_rad_2d, "vol": mdf_rad_2d[..., None]}

#     vol_mdf_lin = mdf_lin_2d[..., None]   # DC-only stack so FocusImage3D can reconstruct
#     vol_mdf_rad = mdf_rad_2d[..., None]


#     # ---- Plots (use FocusImage2D, with 'vertical' mirror like common MATLAB layout) ----
#     if bild:
#         plt.figure(figsize=(11, 4.6))
#         ax1 = plt.subplot(1, 2, 1)
#         FocusImage2D(exc_lin["rho"], exc_lin["z"], exc_lin["vol"], phi=0.0, flag="vertical", ax=ax1, cmap="hot")
#         ax1.set_xlabel(r'$\rho$ [μm]'); ax1.set_ylabel('z [μm]')
#         ax1.set_title('Excitation (linear)')

#         ax2 = plt.subplot(1, 2, 2)
#         FocusImage2D(exc_rad["rho"], exc_rad["z"], exc_rad["vol"], phi=0.0, flag="vertical", ax=ax2, cmap="hot")
#         ax2.set_xlabel(r'$\rho$ [μm]'); ax2.set_ylabel('z [μm]')
#         ax2.set_title('Excitation (radial)')

#         plt.figure(figsize=(11, 4.6))
#         ax3 = plt.subplot(1, 2, 1)
#         FocusImage2D(mdf_lin["rho"], mdf_lin["z"], mdf_lin["vol"], phi=0.0, flag="vertical", ax=ax3, cmap="hot")
#         ax3.set_xlabel(r'$\rho$ [μm]'); ax3.set_ylabel('z [μm]')
#         ax3.set_title('MDF (linear, proxy)')

#         ax4 = plt.subplot(1, 2, 2)
#         FocusImage2D(mdf_rad["rho"], mdf_rad["z"], mdf_rad["vol"], phi=0.0, flag="vertical", ax=ax4, cmap="hot")
#         ax4.set_xlabel(r'$\rho$ [μm]'); ax4.set_ylabel('z [μm]')
#         ax4.set_title('MDF (radial, proxy)')

#         plt.tight_layout()
        
#         tsh = 1.0 / np.exp(np.array([1, 2, 3], float))
        
#         L = len(tsh)
#         colors = [(1.0, j / L, 0.0) for j in range(L-1, -1, -1)].reverse()
        
#         plt.figure(figsize=(12, 5))
#         ax = plt.subplot(1, 2, 1, projection='3d')
      
#         FocusImage3D(rho, z, vol_lin, flag=None, tsh=tsh, maxangle=2*np.pi,
#                      ax=ax, plot_isosurfaces=True,
#                      colors=colors,
#                      alphas=[0.50, 0.35, 0.25], aggregate_for_plot=True)
#         ax.set_title('Excitation (linear) — isosurfaces')

#         ax = plt.subplot(1, 2, 2, projection='3d')
#         FocusImage3D(rho, z, vol_rad, flag=None, tsh=tsh, maxangle=2*np.pi,
#                      ax=ax, plot_isosurfaces=True,
#                      colors=colors,
#                      alphas=[0.50, 0.35, 0.25],aggregate_for_plot=True,)
#         ax.set_title('Excitation (radial) — isosurfaces')
        
#         # MDF 3D (DC-only, but still useful shape)
#         plt.figure(figsize=(12, 5))
#         ax = plt.subplot(1, 2, 1, projection='3d')
#         FocusImage3D(rho, z, vol_mdf_lin, flag=None, tsh=tsh, maxangle=2*np.pi,
#                      ax=ax, plot_isosurfaces=True,
#                      colors=colors,
#                      alphas=[0.50, 0.35, 0.25],aggregate_for_plot=True,)
#         ax.set_title('MDF (linear, proxy) — isosurfaces')

#         ax = plt.subplot(1, 2, 2, projection='3d')
#         FocusImage3D(rho, z, vol_mdf_rad, flag=None, tsh=tsh, maxangle=2*np.pi,
#                      ax=ax, plot_isosurfaces=True,
#                      colors=colors,
#                      alphas=[0.50, 0.35, 0.25],aggregate_for_plot=True,)
#         ax.set_title('MDF (radial, proxy) — isosurfaces')
        

#     return exc_lin, exc_rad, mdf_lin, mdf_rad




def mdf_confocal_microscopy_py(
    rhofield=None, zfield=None,
    NA=1.2, fd=3e3,
    n0=1.35, n=1.35, n1=1.333,
    d0=None, d=None, d1=None,
    lamex=0.645, over=5e3,
    focpos=0.0, defoc=0.0,
    av=75.0, lamem=0.650, mag=60.0, zpin=0.0,
    atf=None, resolution=50, ring=None, maxm=2,
    kappa=1.0, lt=None, pulse=None, sat=None, triplet=None,
    bild=True
):
    import numpy as np
    import matplotlib.pyplot as plt

    if rhofield is None: rhofield = [0, 1]
    if zfield   is None: zfield   = [0, 1]
    if atf      is None: atf = []
    d0 = _coerce_array(d0 if d0 is not None else [])
    d1 = _coerce_array(d1 if d1 is not None else [])
    d  = _coerce_float(0.0 if d is None else d)

    # --- Excitation: linear ---
    fxc, fxs, fyc, fys, fzc, fzs, rho, z = gauss_exc(
        rhofield, zfield, NA, fd, n0, n, n1, np.asarray(d0,float), float(d), np.asarray(d1,float),
        lamex, over, float(focpos)+float(defoc), atf, resolution, ring, int(maxm)
    )
    wave_lin = (np.abs(fxc[...,0])**2 + np.abs(fyc[...,0])**2 + np.abs(fzc[...,0])**2
                + 0.5*np.sum(np.abs(fxc[...,1:])**2 + np.abs(fyc[...,1:])**2 + np.abs(fzc[...,1:])**2, axis=2))

    # --- Excitation: radial (cos ψ + rotated by π/2) ---
    fxc_cos, fxs_cos, fyc_cos, fys_cos, fzc_cos, fzs_cos, _, _ = gauss_exc(
        rhofield, zfield, NA, fd, n0, n, n1, np.asarray(d0,float), float(d), np.asarray(d1,float),
        lamex, over, float(focpos)+float(defoc), atf, resolution, 'np.cos(psi)', int(maxm)
    )
    fxc_rot, fxs_rot, fyc_rot, fys_rot, fzc_rot, fzs_rot = RotateEMField(
        fxc_cos, fxs_cos, fyc_cos, fys_cos, fzc_cos, fzs_cos, np.pi/2
    )

    # coherent sum (do not overwrite originals!)
    fxc_rad = fxc_cos + fxc_rot
    fxs_rad = fxs_cos + fxs_rot
    fyc_rad = fyc_cos + fyc_rot
    fys_rad = fys_cos + fys_rot
    fzc_rad = fzc_cos + fzc_rot
    fzs_rad = fzs_cos + fzs_rot

    wave_rad = (np.abs(fxc_rad[...,0])**2 + np.abs(fyc_rad[...,0])**2 + np.abs(fzc_rad[...,0])**2
                + 0.5*np.sum(np.abs(fxc_rad[...,1:])**2 + np.abs(fyc_rad[...,1:])**2 + np.abs(fzc_rad[...,1:])**2, axis=2))

    # --- Build exc “structs” for MDF step ---
    exc_lin_struct = _pack_exc_struct(fxc, fxs, fyc, fys, fzc, fzs, rho, z, d0, d, d1, maxm)
    exc_rad_struct = _pack_exc_struct(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, rho, z, d0, d, d1, maxm)

    # --- MDF from excitation (full model) ---
    mdf_lin = gauss_exc_to_mdf(exc_lin_struct, NA, n0, n, n1, focpos,
                               lamem, mag, av, zpin, atf, kappa or 0, lt or 0,
                               pulse, sat or 0, triplet or 0)
    mdf_rad = gauss_exc_to_mdf(exc_rad_struct, NA, n0, n, n1, focpos,
                               lamem, mag, av, zpin, atf, kappa or 0, lt or 0,
                               pulse, sat or 0, triplet or 0)

    # Gather simple dicts (for plotting with FocusImage2D/3D)
    exc_lin = {"rho": rho, "z": z, "wave": wave_lin, "vol": wave_lin[...,None]}
    exc_rad = {"rho": rho, "z": z, "wave": wave_rad, "vol": wave_rad[...,None]}

    def _sum_vol(mdf):
        Vx, Vy = mdf["volx"], mdf["voly"]
        if Vx.ndim == 4:
            return np.sum(Vx, axis=3) + np.sum(Vy, axis=3)
        else:
            return Vx + Vy

    vol_mdf_lin = _sum_vol(mdf_lin)
    vol_mdf_rad = _sum_vol(mdf_rad)

    # --- Optional plotting (2D + 3D isosurfaces) ---
    if bild:
        tsh = 1.0 / np.exp(np.array([1,2,3],float))
        L = len(tsh)
        colors = [(1.0, j/L, 0.0) for j in range(L-1,-1,-1)]
        alphas = [0.50, 0.35, 0.25]

        # 2D plots
        plt.figure(figsize=(11, 4.6))
        ax1 = plt.subplot(1, 2, 1)
        FocusImage2D(exc_lin["rho"], exc_lin["z"], exc_lin["vol"], phi=0.0, flag="vertical", ax=ax1, cmap="hot")
        ax1.set_title("Excitation (linear)")
        ax2 = plt.subplot(1, 2, 2)
        FocusImage2D(exc_rad["rho"], exc_rad["z"], exc_rad["vol"], phi=0.0, flag="vertical", ax=ax2, cmap="hot")
        ax2.set_title("Excitation (radial)")

        plt.figure(figsize=(11, 4.6))
        ax3 = plt.subplot(1, 2, 1)
        FocusImage2D(rho, z, vol_mdf_lin, phi=0.0, flag="vertical", ax=ax3, cmap="hot")
        ax3.set_title("MDF (linear)")
        ax4 = plt.subplot(1, 2, 2)
        FocusImage2D(rho, z, vol_mdf_rad, phi=0.0, flag="vertical", ax=ax4, cmap="hot")
        ax4.set_title("MDF (radial)")

        # 3D plots
        plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 2, 1, projection="3d")
        FocusImage3D(rho, z, vol_mdf_lin, tsh=tsh, maxangle=2*np.pi,
                     ax=ax, plot_isosurfaces=True, colors=colors, alphas=alphas)
        ax.set_title("MDF (linear) — isosurfaces")

        ax = plt.subplot(1, 2, 2, projection="3d")
        FocusImage3D(rho, z, vol_mdf_rad, tsh=tsh, maxangle=2*np.pi,
                     ax=ax, plot_isosurfaces=True, colors=colors, alphas=alphas)
        ax.set_title("MDF (radial) — isosurfaces")

    return exc_lin, exc_rad, mdf_lin, mdf_rad