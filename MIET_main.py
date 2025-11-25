# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:35:43 2025

@author: narai
"""

import numpy as np
from math import isfinite  # just in case; numpy handles complex fine
import os, sys
sys.path.insert(0, os.getcwd())   # ensure current working dir is on the path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed to register 3D projection)
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

#%%


def fresnel(w1, n1, n2):
    """
    [rp, rs, tp, ts] = fresnel(w1, n1, n2)

    Fresnel reflection (r) and transmission (t) coefficients for p- and s-polarized waves.

    Two input modes:

    CASE 1 (single interface):
        n1, n2 are scalars: refractive indices of medium 1 and 2.

    CASE 2 (stack of media):
        n1, n2 are 1D arrays with len(n1) == len(n2) + 2
        - n1: refractive indices of all layers (top ... bottom)
        - n2: physical thicknesses of the INTERNAL layers (i.e., all except top & bottom)

    Parameters
    ----------
    w1 : array_like or scalar (complex/real)
        Normal component of the wavevector in the first medium (row vector in MATLAB).
        Shape will be flattened to 1D.
    n1 : scalar or array_like
    n2 : scalar or array_like

    Returns
    -------
    rp, rs, tp, ts : np.ndarray (complex), shape = (len(w1),)
        Reflection / transmission coefficients for p- and s-polarization.
    """
    # Make row-like 1D arrays of complex numbers
    w1 = np.atleast_1d(w1).astype(complex).ravel()
    n1_arr = np.atleast_1d(n1).astype(complex).ravel()
    n2_arr = np.atleast_1d(n2).astype(complex).ravel()

    # ---- CASE 1: single interface ----
    if n1_arr.size == 1 and n2_arr.size == 1:
        n1s = n1_arr[0]
        n2s = n2_arr[0]

        # w2 = sqrt(n2^2 - n1^2 + w1^2)
        w2 = np.sqrt(n2s**2 - n1s**2 + w1**2)

        # enforce imag(w2) >= 0
        neg_im = (np.imag(w2) < 0)
        if np.any(neg_im):
            w2[neg_im] = np.conj(w2[neg_im])

        # p-pol
        rp = (w1 * n2s**2 - w2 * n1s**2) / (w1 * n2s**2 + w2 * n1s**2)
        tp = 2 * n1s * n2s * w1 / (w1 * n2s**2 + w2 * n1s**2)

        # s-pol
        rs = (w1 - w2) / (w1 + w2)
        ts = 2 * w1 / (w1 + w2)

        # special case: w1 == 0 & w2 == 0 (only if n1 == n2); avoid NaNs
        ind = (np.isclose(w1, 0) & np.isclose(w2, 0))
        if np.any(ind):
            rp[ind] = 0
            rs[ind] = 0
            tp[ind] = 1
            ts[ind] = 1

        return rp, rs, tp, ts

    # ---- CASE 2: stack of different materials ----
    elif n1_arr.size == n2_arr.size + 2:
        if n1_arr.size == 2:
            # reduce to single interface case
            return fresnel(w1, n1_arr[0], n1_arr[1])

        n = n1_arr  # refractive indices per layer
        d = np.concatenate(([0], n2_arr, [0]))  # thicknesses including 0 for first & last

        L = w1.size
        NL = n.size
        w = np.zeros((NL, L), dtype=complex)
        w[0, :] = w1
        # q^2 = n(1)^2 - w1^2 = n(j)^2 - wj^2  => wj = sqrt(n(j)^2 - n(1)^2 + w1^2)
        for j in range(1, NL):
            wj = np.sqrt(n[j]**2 - n[0]**2 + w1**2)
            # enforce imag(wj) >= 0
            neg_im = (np.imag(wj) < 0)
            if np.any(neg_im):
                wj[neg_im] = np.conj(wj[neg_im])
            w[j, :] = wj

        # Numerical guard threshold (matches MATLAB comments ~ log limits)
        THRESH = 708.0

        # ---------- p-polarization ----------
        j = NL - 1
        M11 = ( (w[j, :] / w[j-1, :]) * (n[j-1] / n[j]) + (n[j] / n[j-1]) ) / 2
        M21 = (-(w[j, :] / w[j-1, :]) * (n[j-1] / n[j]) + (n[j] / n[j-1]) ) / 2

        inftyalert = np.full(L, False, dtype=bool)

        for j in range(NL - 2, 0, -1):
            # exponential phase through layer j
            # guard: imag(w)*d > THRESH or imag(w)*d + log(M11) > THRESH
            # use abs(M11) inside log for safety (MATLAB uses log(M11); can be complex)
            guard_left  = (np.imag(w[j, :]) * d[j] > THRESH)
            with np.errstate(divide='ignore', invalid='ignore'):
                guard_right = (np.imag(w[j, :]) * d[j] + np.log(np.abs(M11) + np.finfo(float).eps) > THRESH)
            inftyIndex = np.where(guard_left | guard_right)[0]

            keep = np.ones(L, dtype=bool)
            keep[inftyIndex] = False

            # apply phases only to safe indices
            M11[keep] = np.exp(-1j * w[j, keep] * d[j]) * M11[keep]
            M21[keep] = np.exp( 1j * w[j, keep] * d[j]) * M21[keep]

            # when overflow would happen, set M21 -> 0 and mark alert
            if inftyIndex.size:
                M21[inftyIndex] = 0
                inftyalert[inftyIndex] = True

            # interface transfer from layer j to j-1
            N11 = ( (w[j, :] / w[j-1, :]) * (n[j-1] / n[j]) + (n[j] / n[j-1]) ) / 2
            N21 = (-(w[j, :] / w[j-1, :]) * (n[j-1] / n[j]) + (n[j] / n[j-1]) ) / 2

            tmp11 = N11 * M11 + N21 * M21
            tmp21 = N21 * M11 + N11 * M21
            M11, M21 = tmp11, tmp21

        rp = M21 / M11
        tp = 1.0 / M11
        tp[inftyalert] = 0.0

        # ---------- s-polarization ----------
        j = NL - 1
        M11 = (w[j, :] / w[j-1, :] + 1.0) / 2.0
        M21 = (-(w[j, :] / w[j-1, :]) + 1.0) / 2.0

        inftyalert = np.full(L, False, dtype=bool)

        for j in range(NL - 2, 0, -1):
            guard_left  = (np.imag(w[j, :]) * d[j] > THRESH)
            with np.errstate(divide='ignore', invalid='ignore'):
                guard_right = (np.imag(w[j, :]) * d[j] + np.log(np.abs(M11) + np.finfo(float).eps) > THRESH)
            inftyIndex = np.where(guard_left | guard_right)[0]

            keep = np.ones(L, dtype=bool)
            keep[inftyIndex] = False

            M11[keep] = np.exp(-1j * w[j, keep] * d[j]) * M11[keep]
            M21[keep] = np.exp( 1j * w[j, keep] * d[j]) * M21[keep]

            if inftyIndex.size:
                M21[inftyIndex] = 0.0
                inftyalert[inftyIndex] = True

            N11 = (w[j, :] / w[j-1, :] + 1.0) / 2.0  # N22 = N11
            N21 = (-(w[j, :] / w[j-1, :]) + 1.0) / 2.0  # N12 = N21

            tmp11 = N11 * M11 + N21 * M21
            tmp21 = N21 * M11 + N11 * M21
            M11, M21 = tmp11, tmp21

        rs = M21 / M11
        ts = 1.0 / M11
        ts[inftyalert] = 0.0

        return rp, rs, tp, ts

    else:
        raise ValueError("Wrong input: use scalars (single interface) or len(n1) == len(n2) + 2 for stacks.")

def dipoleL(theta, z, n0, n, n1, d0, d, d1):
    """

    Parameters
    ----------
    theta : array_like
        Emission angles (radians). Direction is downward; only 0 <= theta <= pi/2 contribute.
    z : array_like or scalar
        Molecule's distance(s) from the bottom of its layer (same units as d).
    n0 : array_like (complex or real)
        Refractive indices of the stack below the molecule's layer (length >= 1).
        n0[0] is the index of the medium immediately below the molecule's layer.
    n : scalar (complex or real)
        Refractive index of the molecule's layer.
    n1 : array_like (complex or real)
        Refractive indices of the stack above the molecule's layer.
        n1[-1] is the index of the topmost semi-infinite medium.
    d0 : array_like (real)
        Thicknesses of layers below the molecule's layer. Must satisfy len(d0) == len(n0) - 1.
    d : scalar (real)
        Thickness of the molecule's layer.
    d1 : array_like (real)
        Thicknesses of layers above the molecule's layer. Must satisfy len(d1) == len(n1) - 1.

    Returns
    -------
    v, pc, ps : complex ndarray, shape (len(theta), len(z))
        Field amplitudes (components) as in the MATLAB code:
        - v  : component in a coordinate system whose z-axis points away from the surface (longitudinal)
        - pc : p-polarized (cosine-like) component
        - ps : s-polarized component
    tp, ts, tp1, ts1 : complex ndarray, shape (len(theta),)
        Effective transmission coefficients into the lower half-space (direct and once-reflected paths).
    fac : complex ndarray, shape (len(theta),)
        Angular prefactor sqrt(n0(1)) * n0(1) * cos(theta) / w (with w the normal component in the dipole layer).
    """
    # Ensure 1D arrays; keep complex dtype for safety
    theta = np.abs(np.atleast_1d(theta).astype(complex).ravel())
    z = np.atleast_1d(z).astype(complex).ravel()   # row-like
    n0 = np.atleast_1d(np.asarray(n0, dtype=complex)).ravel()
    n1 = np.atleast_1d(np.asarray(n1, dtype=complex)).ravel()
    d0 = np.atleast_1d(np.asarray(d0, dtype=float)).ravel()
    d1 = np.atleast_1d(np.asarray(d1, dtype=float)).ravel()
    n = complex(n)
    d = float(d)

    # masks and output shapes
    ind = (theta.real <= np.pi/2)  # only those contribute
    n_theta = theta.size
    n_z = z.size

    # Initialize outputs
    v  = np.zeros((n_theta, n_z), dtype=complex)
    pc = np.zeros_like(v)
    ps = np.zeros_like(v)

    # Also return these per-theta coefficients; initialize with zeros
    tp  = np.zeros(n_theta, dtype=complex)
    ts  = np.zeros(n_theta, dtype=complex)
    tp1 = np.zeros(n_theta, dtype=complex)
    ts1 = np.zeros(n_theta, dtype=complex)
    fac = np.zeros(n_theta, dtype=complex)

    if np.any(ind):
        tmp = theta[ind]  # contributing angles

        # Normal component in dipole layer (w in MATLAB)
        # w = sqrt(n^2 - n0(1)^2 * sin(tmp)^2)
        n0_1 = n0[0]
        w = np.sqrt(n**2 - (n0_1**2) * (np.sin(tmp) ** 2))

        # Fresnel stacks:
        # Upwards: layers [n, n1...] with internal thicknesses d1
        # Downwards: layers [n, n0[::-1]...] with internal thicknesses d0[::-1]
      

        # Ensure we have the fresnel function from the previous step
        # rp/rs: reflections seen looking upward/downward from the dipole layer
        # tp/ts: transmissions into the downward side (we'll assemble effective ones below)
        rpu, rsu, tpu, tsu = fresnel(w, np.concatenate(([n], n1)), d1)
        rpd, rsd, tpd, tsd = fresnel(w, np.concatenate(([n], n0[::-1])), d0[::-1])

        # Effective transmission coefficients including multiple reflections within the dipole layer
        # Note: use broadcasting-safe shapes; everything is 1D over angles here
        denom_p = (1 - rpu * rpd * np.exp(2j * w * d))
        denom_s = (1 - rsu * rsd * np.exp(2j * w * d))
        tp_eff = tpd / denom_p
        ts_eff = tsd / denom_s
        tp1_eff = tp_eff * (rpu * np.exp(2j * w * d))  # once-reflected
        ts1_eff = ts_eff * (rsu * np.exp(2j * w * d))

        # Angular prefactor (per-angle)
        fac_ang = np.sqrt(n0_1) * n0_1 * np.cos(tmp) / w

        # Longitudinal phase factor vs z; shape (n_angles_kept, n_z)
        ez = np.exp(1j * w[:, None] * z[None, :])

        # Build the base per-angle coefficients (column vectors) and broadcast over z via ez
        base_v  = (fac_ang * (n0_1 / n) * np.sin(tmp) * tp_eff)[:, None]   # (M,1)
        base_v1 = (fac_ang * (n0_1 / n) * np.sin(tmp) * tp1_eff)[:, None]

        base_pc  = (fac_ang * (w / n) * tp_eff)[:, None]
        base_pc1 = (fac_ang * (w / n) * tp1_eff)[:, None]

        base_ps  = (fac_ang * ts_eff)[:, None]
        base_ps1 = (fac_ang * ts1_eff)[:, None]

        # Compose fields: note "+ ... * ez - ... / ez" etc. (elementwise)
        v_block  =  base_v * ez +  base_v1 / ez
        pc_block =  base_pc * ez -  base_pc1 / ez
        ps_block =  base_ps * ez +  base_ps1 / ez

        # Insert into full arrays at the rows corresponding to contributing angles
        v[ind, :]  = v_block
        pc[ind, :] = pc_block
        ps[ind, :] = ps_block

        # Store per-theta coefficients back into full-length arrays
        tp[ind]  = tp_eff
        ts[ind]  = ts_eff
        tp1[ind] = tp1_eff
        ts1[ind] = ts1_eff
        fac[ind] = fac_ang

    return v, pc, ps, tp, ts, tp1, ts1, fac
 

def hash_waveguide_mode(n, d, dq=1e-6, prefer_cubic=True):
    """
 
    Parameters
    ----------
    n : array_like (complex)
        Refractive indices per layer, ordered from bottom cladding to top cladding.
        (Same order as MATLAB code: n = n(:).')
    d : array_like (real)
        Physical thicknesses of INTERNAL layers (len(d) == len(n) - 2).
    dq : float
        Resolution of the in-plane wavevector scan.
    prefer_cubic : bool
        If True and SciPy is available, use cubic interpolation for root refinement.
        Otherwise falls back to linear bracketing.

    Returns
    -------
    qp, qs : np.ndarray
        In-plane wavevector components (guided modes) for p and s polarization.
    fp, fs : np.ndarray
        Characteristic functions sampled on the grid qq (same length).
        Zeros of fp/fs correspond to guided modes.
    """
    n = np.atleast_1d(np.asarray(n, dtype=complex)).ravel()
    d = np.atleast_1d(np.asarray(d, dtype=float)).ravel()

    L = n.size
    if L < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # MATLAB: d = [0 d 0]
    d_ext = np.concatenate(([0.0], d, [0.0]))

    # Scan range in q: from max(|n(1)|, |n(end)|) to max(|n|)
    n_edge_max = max(np.abs(n[0]), np.abs(n[-1]))
    n_all_max  = np.max(np.abs(n))
    if not (n_all_max > n_edge_max):
        return np.array([]), np.array([]), np.array([]), np.array([])

    qmin = n_edge_max + dq/2
    qmax = n_all_max  - dq/2
    # ensure inclusive upper range
    Q = int(np.floor((qmax - qmin) / dq)) + 1
    qq = qmin + dq*np.arange(Q)          # (Q,)

    def _enforce_pos_imag(x):
        x = x.astype(complex, copy=True)
        m = (np.imag(x) < 0)
        if np.any(m):
            x[m] = np.conj(x[m])
        return x

    # ---------- p-polarization ----------
    j = L - 1  # last index (0-based)
    w1 = _enforce_pos_imag(np.sqrt(n[j]**2 - qq**2))
    w0 = _enforce_pos_imag(np.sqrt(n[j-1]**2 - qq**2))

    M11 = ( (w1/w0)*(n[j-1]/n[j]) + (n[j]/n[j-1]) ) / 2
    M12 = (-(w1/w0)*(n[j-1]/n[j]) + (n[j]/n[j-1]) ) / 2
    M21 = M12.copy()
    M22 = M11.copy()

    for j in range(L-2, 0, -1):  # j = L-2,...,1   (equiv. MATLAB 2..L-1)
        w1 = _enforce_pos_imag(np.sqrt(n[j]**2   - qq**2))
        w0 = _enforce_pos_imag(np.sqrt(n[j-1]**2 - qq**2))

        phase = np.exp(-1j*w1*d_ext[j])
        M11 *= phase
        M12 *= phase
        phase = np.exp( 1j*w1*d_ext[j])
        M21 *= phase
        M22 *= phase

        N11 = ( (w1/w0)*(n[j-1]/n[j]) + (n[j]/n[j-1]) ) / 2
        N12 = (-(w1/w0)*(n[j-1]/n[j]) + (n[j]/n[j-1]) ) / 2
        N21 = N12
        N22 = N11

        tmp11 = N11*M11 + N12*M21
        tmp12 = N11*M12 + N12*M22
        M21   = N21*M11 + N22*M21
        M22   = N21*M12 + N22*M22
        M11   = tmp11
        M12   = tmp12

    fp = M11  # characteristic function for p

    # ---------- s-polarization ----------
    j = L - 1
    w1 = _enforce_pos_imag(np.sqrt(n[j]**2 - qq**2))
    w0 = _enforce_pos_imag(np.sqrt(n[j-1]**2 - qq**2))

    M11 = ( w1/w0 + 1 ) / 2
    M12 = (-w1/w0 + 1 ) / 2
    M21 = M12.copy()
    M22 = M11.copy()

    for j in range(L-2, 0, -1):
        w1 = _enforce_pos_imag(np.sqrt(n[j]**2   - qq**2))
        w0 = _enforce_pos_imag(np.sqrt(n[j-1]**2 - qq**2))

        phase = np.exp(-1j*w1*d_ext[j])
        M11 *= phase
        M12 *= phase
        phase = np.exp( 1j*w1*d_ext[j])
        M21 *= phase
        M22 *= phase

        N11 = ( w1/w0 + 1 ) / 2
        N12 = (-w1/w0 + 1 ) / 2
        N21 = N12
        N22 = N11

        tmp11 = N11*M11 + N12*M21
        tmp12 = N11*M12 + N12*M22
        M21   = N21*M11 + N22*M21
        M22   = N21*M12 + N22*M22
        M11   = tmp11
        M12   = tmp12

    fs = M11  # characteristic function for s

    # ---------- find zeros via sign changes and refine ----------
    def _refine_zeros(fvals, qq, prefer_cubic=True):
        f = np.real(fvals)  # guided-mode characteristic is real on real qq
        sign = f[:-1] * f[1:] < 0
        idx = np.where(sign)[0]  # zero between i and i+1
        roots = []

        # optional cubic interpolation q = g(f)
        cs_ok = False
        if prefer_cubic:
            try:
                from scipy.interpolate import CubicSpline
                cs_ok = True
            except Exception:
                cs_ok = False

        for i in idx:
            lo = max(0, i-2)
            hi = min(len(qq)-1, i+3)
            x = f[lo:hi+1]
            y = qq[lo:hi+1]
            if x.size >= 3 and cs_ok:
                # build cubic spline y(x) and evaluate at x=0
                # guard: if x has duplicates (rare), fall back to linear
                if np.all(np.diff(x) != 0):
                    cs = CubicSpline(x, y, bc_type='not-a-knot')
                    roots.append(cs(0.0))
                    continue
            # linear bracketing on (i, i+1)
            f0, f1 = f[i], f[i+1]
            q0, q1 = qq[i], qq[i+1]
            roots.append(q0 - f0 * (q1 - q0) / (f1 - f0))
        return np.asarray(roots, dtype=complex)

    qp = _refine_zeros(fp, qq, prefer_cubic=prefer_cubic)
    qs = _refine_zeros(fs, qq, prefer_cubic=prefer_cubic)

    return qp, qs, fp, fs
        

def lifetimeL(z, n0, n, n1, d0, d, d1, hash_waveguide_mode=None):
    """
  
    Parameters
    ----------
    z : array_like
        Molecule distances from the bottom of its layer (same units as d).
    n0 : array_like (complex/real)
        Refractive indices below the molecule layer (closest first).
    n : scalar complex/real
        Refractive index of the molecule layer.
    n1 : array_like (complex/real)
        Refractive indices above the molecule layer (closest first).
    d0 : array_like (real)
        Thicknesses of n0 stack (len = len(n0) - 1).
    d : float
        Thickness of molecule layer.
    d1 : array_like (real)
        Thicknesses of n1 stack (len = len(n1) - 1).
    hash_waveguide_mode : callable or None
        Optional function returning (ppos, spos) for guided modes:
        ppos,spos are arrays of in-plane propagation constants.

    Returns
    -------
    lvd, lvu, lpd, lpu, qvd, qvu, qpd, qpu, qv, qp
        Each is a 1D complex ndarray of length len(z) (row-wise per z).
    """
    # ---- helpers / casting ----
    z = np.atleast_1d(z).astype(complex).ravel()            # (M,)
    M = z.size
    n0 = np.atleast_1d(np.asarray(n0, dtype=complex)).ravel()
    n1 = np.atleast_1d(np.asarray(n1, dtype=complex)).ravel()
    d0 = np.atleast_1d(np.asarray(d0, dtype=float)).ravel()
    d1 = np.atleast_1d(np.asarray(d1, dtype=float)).ravel()
    n = complex(n)
    d = float(d)

    # outputs per z
    lvd = np.zeros(M, dtype=complex)
    lvu = np.zeros(M, dtype=complex)
    lpd = np.zeros(M, dtype=complex)
    lpu = np.zeros(M, dtype=complex)
    qvd = np.zeros(M, dtype=complex)
    qvu = np.zeros(M, dtype=complex)
    qpd = np.zeros(M, dtype=complex)
    qpu = np.zeros(M, dtype=complex)
    qv  = np.zeros(M, dtype=complex)
    qp  = np.zeros(M, dtype=complex)

    # ---------- propagating part ----------
    dw0 = 1e-3
    all_real_env = (np.allclose(np.imag(n0), 0.0) and np.allclose(np.imag(n1), 0.0))
    max_env = np.max(np.real(np.concatenate([n0, n1]))) if (n0.size + n1.size) else 0.0

    if all_real_env and (max_env < np.real(n)):  # real surroundings and smaller than dipole layer index
        w_start = np.sqrt(n**2 - max_env**2)
        # (0.5:1/dw0)' * dw0 gives 1000 samples; scale to span [w_start, n]
        u = (np.arange(0.5, 1.0/dw0, 1.0)) * dw0   # length ~1000
        w = w_start + u * (n - w_start)
    else:
        u = (np.arange(0.5, 1.0/dw0, 1.0)) * dw0
        w = u * n  # 0..n

    w = np.atleast_1d(w).astype(complex).ravel()            # (L,)
    if w.size < 2:
        return lvd, lvu, lpd, lpu, qvd, qvu, qpd, qpu, qv, qp
    dw = (w[1] - w[0]).real

    st = np.sqrt(1 - (w / n)**2)                            # (L,)

    # Fresnel (propagating)
    rpu, rsu, tpu, tsu = fresnel(w, np.concatenate(([n], n1)), d1)                  # (L,)
    rpd, rsd, tpd, tsd = fresnel(w, np.concatenate(([n], n0[::-1])), d0[::-1])      # (L,)

    # Multiple-reflection denominators
    ed = np.exp(2j * w * d)                                # (L,)
    tp = 1.0 / (1.0 - rpu * rpd * ed)
    ts = 1.0 / (1.0 - rsu * rsd * ed)

    # z-grid broadcasting
    ez0 = np.exp(2j * w[:, None] * z[None, :])             # (L,M)
    ezd = np.exp(2j * w[:, None] * (d - z)[None, :])       # (L,M)

    # Downward quenching (propagating)
    v1 = tp[:, None] + (rpu * tp)[:, None] * ezd
    v2 = (rpd * tp)[:, None] + (rpd * rpu * tp)[:, None] * ezd
    pc1 = tp[:, None] - (rpu * tp)[:, None] * ezd
    pc2 = (rpd * tp)[:, None] - (rpd * rpu * tp)[:, None] * ezd
    ps1 = ts[:, None] + (rsu * ts)[:, None] * ezd
    ps2 = (rsd * ts)[:, None] + (rsu * rsd * ts)[:, None] * ezd

    qvd += dw * np.sum((st**2)[:, None] * (np.abs(v1)**2 - np.abs(v2)**2), axis=0)
    qpd += (dw * np.sum(((w**2 / n**2)[:, None] * (np.abs(pc1)**2 - np.abs(pc2)**2)
                         + (np.abs(ps1)**2 - np.abs(ps2)**2)), axis=0)) / 2.0

    # Upward quenching (propagating) – swap rpu<->rpd, rsu<->rsd and use ez0
    v1 = tp[:, None] + (rpd * tp)[:, None] * ez0
    v2 = (rpu * tp)[:, None] + (rpd * rpu * tp)[:, None] * ez0
    pc1 = tp[:, None] - (rpd * tp)[:, None] * ez0
    pc2 = (rpu * tp)[:, None] - (rpd * rpu * tp)[:, None] * ez0
    ps1 = ts[:, None] + (rsd * ts)[:, None] * ez0
    ps2 = (rsu * ts)[:, None] + (rsu * rsd * ts)[:, None] * ez0

    qvu += dw * np.sum((st**2)[:, None] * (np.abs(v1)**2 - np.abs(v2)**2), axis=0)
    qpu += (dw * np.sum(((w**2 / n**2)[:, None] * (np.abs(pc1)**2 - np.abs(pc2)**2)
                         + (np.abs(ps1)**2 - np.abs(ps2)**2)), axis=0)) / 2.0

    # Far-field leakage into real half-spaces, if exit medium is lossless
    if np.isclose(np.imag(n0[0]), 0.0):
        tp_d = (tpd / (1.0 - rpu * rpd * ed))     # (L,)
        ts_d = (tsd / (1.0 - rsu * rsd * ed))
        tp1_d = tp_d * (rpu * ed)
        ts1_d = ts_d * (rsu * ed)

        ez = np.exp(1j * w[:, None] * z[None, :])
        ww = np.real(np.sqrt((n0[0]**2 - n**2) / (w**2) + 1.0))[:, None]  # (L,1)

        v  = (st * tp_d)[:, None] * ez + (st * tp1_d)[:, None] / ez
        pc = ((w / n) * tp_d)[:, None] * ez - ((w / n) * tp1_d)[:, None] / ez
        ps = ts_d[:, None] * ez + ts1_d[:, None] / ez

        lvd += dw * np.sum(ww * np.abs(v)**2, axis=0)
        lpd += (dw * np.sum(ww * (np.abs(pc)**2 + np.abs(ps)**2), axis=0)) / 2.0

    if np.isclose(np.imag(n1[-1]), 0.0):
        tp_u = (tpu / (1.0 - rpu * rpd * ed))
        ts_u = (tsu / (1.0 - rsu * rsd * ed))
        tp1_u = tp_u * (rpd * ed)
        ts1_u = ts_u * (rsd * ed)

        ez = np.exp(1j * w[:, None] * (d - z)[None, :])
        ww = np.real(np.sqrt((n1[-1]**2 - n**2) / (w**2) + 1.0))[:, None]

        v  = (st * tp_u)[:, None] * ez + (st * tp1_u)[:, None] / ez
        pc = ((w / n) * tp_u)[:, None] * ez - ((w / n) * tp1_u)[:, None] / ez
        ps = ts_u[:, None] * ez + ts1_u[:, None] / ez

        lvu += dw * np.sum(ww * np.abs(v)**2, axis=0)
        lpu += (dw * np.sum(ww * (np.abs(pc)**2 + np.abs(ps)**2), axis=0)) / 2.0

    # ---------- guided modes (optional) ----------
    if (hash_waveguide_mode is not None and all_real_env and
        (np.max(np.real(np.concatenate([n0[1:], [n], n1[:-1]]))) > np.max(np.real([n0[0], n1[-1]])))):

        dwg = 1j * 1e-7  # tiny complex step for two-point derivative

        ppos, spos = hash_waveguide_mode(np.concatenate([n0, [n], n1]),
                                         np.concatenate([d0, [d], d1]))
        # p-polarized guided modes
        if ppos is not None and len(ppos) > 0:
            for pp in np.atleast_1d(ppos):
                wp = np.sqrt(n**2 - pp**2)
                ww = wp + np.array([-0.5, 0.5]) * dwg  # two-point stencil (2,)
                rpd = fresnel(ww, np.concatenate(([n], n0[::-1])), d0[::-1])[0]  # rp
                rpu = fresnel(ww, np.concatenate(([n], n1)),       d1)[0]       # rp

                # Build F(ww, z) and finite-difference derivative wrt ww
                num = 1 - rpd * rpu * np.exp(2j * ww[:, None] * d)
                den = (rpd * np.exp(2j * ww[:, None] * z[None, :]) +
                       rpu * np.exp(2j * ww[:, None] * (d - z)[None, :]) +
                       2 * rpu * rpd * np.exp(2j * ww[:, None] * d))
                F = num / den                        # (2, M)
                fp = (F[1, :] - F[0, :]) / dwg       # (M,)
                qv += 4 * np.pi * (pp**2) * np.imag(1.0 / fp) / (n**2)
                qp += 2 * np.pi * np.imag(np.abs(wp)**2 / fp) / (n**2)

        # s-polarized guided modes
        if spos is not None and len(spos) > 0:
            for sp in np.atleast_1d(spos):
                ws = np.sqrt(n**2 - sp**2)
                ww = ws + np.array([-0.5, 0.5]) * dwg
                rsd = fresnel(ww, np.concatenate(([n], n0[::-1])), d0[::-1])[1]  # rs
                rsu = fresnel(ww, np.concatenate(([n], n1)),       d1)[1]        # rs

                num = 1 - rsd * rsu * np.exp(2j * ww[:, None] * d)
                den = (rsd * np.exp(2j * ww[:, None] * z[None, :]) +
                       rsu * np.exp(2j * ww[:, None] * (d - z)[None, :]) +
                       2 * rsu * rsd * np.exp(2j * ww[:, None] * d))
                F = num / den
                fs = (F[1, :] - F[0, :]) / dwg
                qp += 2 * np.pi * np.imag(1.0 / fs)

    # ---------- evanescent (near-field) part ----------
    if all_real_env:
        max_all = np.max(np.real(np.concatenate([n0, [n], n1])))
        # keep positive float for wmax (slight 1.01 margin like MATLAB)
        arg = 1.01 * (max_all**2) - (np.real(n)**2)
        wmax = np.sqrt(max(arg, 0.0))
        N = 1000
    else:
        max_all = np.max(np.real(np.concatenate([n0, [n], n1])))
        arg = 1e3 * (max_all**2) - (np.real(n)**2)
        wmax = np.sqrt(max(arg, 0.0))
        N = 10000

    if wmax > 0:
        t = (np.arange(0.5, N + 0.5) / N)   # (N,) in (0,1)
        w = np.exp(-6.0 + (6.0 + np.log(wmax)) * t)  # (N,)
        dw_log = (6.0 + np.log(wmax)) / N            # "dw" in transformed variable

        st = np.sqrt(1.0 + (w / n)**2)               # evanescent branch

        # Fresnel with i*w
        wi = 1j * w
        rpu, rsu, tpu, tsu = fresnel(wi, np.concatenate(([n], n1)), d1)
        rpd, rsd, tpd, tsd = fresnel(wi, np.concatenate(([n], n0[::-1])), d0[::-1])

        any_finite = (np.isfinite(rpu).any())
        if any_finite:
            ed = np.exp(-2.0 * w * d)

            tp = 1.0 / (1.0 - rpu * rpd * ed)
            ts = 1.0 / (1.0 - rsu * rsd * ed)

            ez0 = np.exp(-2.0 * w[:, None] * z[None, :])          # (N,M)
            ezd = np.exp(-2.0 * w[:, None] * (d - z)[None, :])    # (N,M)

            # down (evanescent)
            v1 = tp[:, None] + (rpu * tp)[:, None] * ezd
            v2 = (rpd * tp)[:, None] + (rpd * rpu * tp)[:, None] * ezd
            pc1 = tp[:, None] - (rpu * tp)[:, None] * ezd
            pc2 = (rpd * tp)[:, None] - (rpd * rpu * tp)[:, None] * ezd
            ps1 = ts[:, None] + (rsu * ts)[:, None] * ezd
            ps2 = (rsd * ts)[:, None] + (rsu * rsd * ts)[:, None] * ezd

            # boundary terms at the last row (w[-1]) for convergence improvement
            v1e, v2e = v1[-1, :], v2[-1, :]
            pc1e, pc2e = pc1[-1, :], pc2[-1, :]
            ps1e, ps2e = ps1[-1, :], ps2[-1, :]

            qvd += (1.0 / z + (1.0 / (2.0 * n**2)) / (z**3)) * np.imag(np.conj(v1e) * v2e)
            qvd += 2.0 * dw_log * np.sum((w * (st**2))[:, None] * ez0 *
                                          (np.imag(np.conj(v1) * v2) - np.imag(np.conj(v1e) * v2e)[None, :]),
                                          axis=0)

            qpd += ( (1.0 / (4.0 * n**2)) / (z**3) ) * np.imag(np.conj(pc1e) * pc2e) \
                   + (0.5 / z) * np.imag(np.conj(ps1e) * ps2e)
            qpd += dw_log * np.sum(ez0 * (
                        (w**3 / n**2)[:, None] * (np.imag(np.conj(pc1) * pc2) - np.imag(np.conj(pc1e) * pc2e)[None, :]) +
                        (w[:, None])            * (np.imag(np.conj(ps1) * ps2) - np.imag(np.conj(ps1e) * ps2e)[None, :])
                    ), axis=0)

            # up (swap roles)
            v1 = tp[:, None] + (rpd * tp)[:, None] * ez0
            v2 = (rpu * tp)[:, None] + (rpd * rpu * tp)[:, None] * ez0
            pc1 = tp[:, None] - (rpd * tp)[:, None] * ez0
            pc2 = (rpu * tp)[:, None] - (rpd * rpu * tp)[:, None] * ez0
            ps1 = ts[:, None] + (rsd * ts)[:, None] * ez0
            ps2 = (rsu * ts)[:, None] + (rsu * rsd * ts)[:, None] * ez0

            v1e, v2e = v1[-1, :], v2[-1, :]
            pc1e, pc2e = pc1[-1, :], pc2[-1, :]
            ps1e, ps2e = ps1[-1, :], ps2[-1, :]

            qvu += (1.0 / (d - z) + (1.0 / (2.0 * n**2)) / (d - z)**3) * np.imag(np.conj(v1e) * v2e)
            qvu += 2.0 * dw_log * np.sum((w * (st**2))[:, None] * ezd *
                                          (np.imag(np.conj(v1) * v2) - np.imag(np.conj(v1e) * v2e)[None, :]),
                                          axis=0)

            qpu += ( (1.0 / (4.0 * n**2)) / (d - z)**3 ) * np.imag(np.conj(pc1e) * pc2e) \
                   + (0.5 / (d - z)) * np.imag(np.conj(ps1e) * ps2e)
            qpu += dw_log * np.sum(ezd * (
                        (w**3 / n**2)[:, None] * (np.imag(np.conj(pc1) * pc2) - np.imag(np.conj(pc1e) * pc2e)[None, :]) +
                        (w[:, None])            * (np.imag(np.conj(ps1) * ps2) - np.imag(np.conj(ps1e) * ps2e)[None, :])
                    ), axis=0)

            # evanescent leakage into real media if lossless; else equate to quenching
            if np.isclose(np.imag(n0[0]), 0.0):
                tp_d = (tpd / (1.0 - rpu * rpd * ed))
                ts_d = (tsd / (1.0 - rsu * rsd * ed))
                tp1_d = tp_d * (rpu * ed)
                ts1_d = ts_d * (rsu * ed)

                ez = np.exp(-w[:, None] * z[None, :])
                # clip huge exponents as in MATLAB
                ez[(w[:, None] * z[None, :]) > 100] = 1e-20
                v  = (st * tp_d)[:, None] * ez + (st * tp1_d)[:, None] / ez
                pc = ((w / n) * tp_d)[:, None] * ez - ((w / n) * tp1_d)[:, None] / ez
                ps = ts_d[:, None] * ez + ts1_d[:, None] / ez
                ww = np.real(np.sqrt((n0[0]**2 - n**2) / (w**2) - 1.0))[:, None]

                lvd += dw_log * np.sum((w[:, None]) * ww * np.abs(v)**2, axis=0)
                lpd += (dw_log * np.sum((w[:, None]) * ww * (np.abs(pc)**2 + np.abs(ps)**2), axis=0)) / 2.0
            else:
                lvd = qvd
                lpd = qpd

            if np.isclose(np.imag(n1[-1]), 0.0):
                tp_u = (tpu / (1.0 - rpu * rpd * ed))
                ts_u = (tsu / (1.0 - rsu * rsd * ed))
                tp1_u = tp_u * (rpd * ed)
                ts1_u = ts_u * (rsd * ed)

                ez = np.exp(-w[:, None] * (d - z)[None, :])
                ez[(w[:, None] * (d - z)[None, :]) > 100] = 1e-20
                v  = (st * tp_u)[:, None] * ez + (st * tp1_u)[:, None] / ez
                pc = ((w / n) * tp_u)[:, None] * ez - ((w / n) * tp1_u)[:, None] / ez
                ps = ts_u[:, None] * ez + ts1_u[:, None] / ez
                ww = np.real(np.sqrt((n1[-1]**2 - n**2) / (w**2) - 1.0))[:, None]

                lvu += dw_log * np.sum((w[:, None]) * ww * np.abs(v)**2, axis=0)
                lpu += (dw_log * np.sum((w[:, None]) * ww * (np.abs(pc)**2 + np.abs(ps)**2), axis=0)) / 2.0
            else:
                lvu = qvu
                lpu = qpu
        else:
            # No valid Fresnel returned → set to inf as in MATLAB fallback
            qvu[:] = np.inf
            qpu[:] = np.inf
            qvd[:] = np.inf
            qpd[:] = np.inf
            lvu[:] = np.inf
            lpu[:] = np.inf
            lvd[:] = np.inf
            lpd[:] = np.inf

    return lvd, lvu, lpd, lpu, qvd, qvu, qpd, qpu, qv, qp

class MetalsDB:
    """
    Load a MATLAB .mat with fields like:
      wavelength (nm), silver, gold, platinum, palladium, copper, aluminum,
      chromium, titan, tungsten, nickel, beryllium, ito
    Each metal array should be complex-valued n+ik at those wavelengths.
    """
    def __init__(self, mat_path):
        # Try scipy first; if v7.3 (HDF5), fall back to h5py
        try:
            from scipy.io import loadmat
            raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            self._raw = {k: v for k, v in raw.items() if not k.startswith('__')}
        except Exception:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                self._raw = {k: np.array(f[k]).squeeze() for k in f.keys()}

        # Normalize keys to lowercase
        self._raw = {k.lower(): v for k, v in self._raw.items()}
        # Required
        self.wavelength = np.asarray(self._raw['wavelength']).astype(float).ravel()

        # Map the material code to field name (match your MATLAB file)
        self.code_to_field = {
            10: 'silver',
            20: 'gold',
            30: 'platinum',
            40: 'palladium',
            50: 'copper',
            60: 'aluminum',
            70: 'chromium',
            80: 'titan',      # same as in your MATLAB: "titan"
            90: 'tungsten',
            100: 'nickel',
            110: 'beryllium',
            120: 'ito',
        }

    def get_index(self, material_code, lam_nm):
        """Return complex refractive index n+ik at lam_nm (nm), interpolated."""
        field = self.code_to_field.get(int(material_code))
        if field is None or field.lower() not in self._raw:
            raise ValueError(f'Unknown material code {material_code} or missing field in .mat')

        # Stored values might be complex; interpolate real & imag separately
        vals = np.asarray(self._raw[field.lower()]).astype(complex).ravel()
        wl   = self.wavelength

        # Ensure lam_nm is array-like
        lam_nm = np.atleast_1d(lam_nm).astype(float)

        # Linear interpolation separately on Re and Im
        re = np.interp(lam_nm, wl, np.real(vals))
        im = np.interp(lam_nm, wl, np.imag(vals))
        out = re + 1j*im
        return out if out.size > 1 else complex(out.item())


def _load_spectrum_any(path):
    """
    Load a two-column spectrum file [wavelength, intensity].
    Accepts .txt/.csv; if it's a .mat with variables 'wavelength' and 'intensity',
    you can expand this as needed.
    Returns array shape (N,2) with wavelength in nm and normalized intensity.
    """
    if path.lower().endswith('.mat'):
        from scipy.io import loadmat
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = {k.lower(): k for k in raw.keys()}
        wl = np.asarray(raw[keys.get('wavelength')]).ravel().astype(float)
        I  = np.asarray(raw[keys.get('intensity')]).ravel().astype(float)
    else:
        arr = np.loadtxt(path, delimiter=None)
        wl = arr[:, 0].astype(float)
        I  = arr[:, 1].astype(float)

    # If given in micrometers (like MATLAB check), convert to nm
    if wl.max() < 2.0:
        wl = wl * 1000.0

    I = np.clip(I, 0, None)
    if I.sum() > 0:
        I = I / I.sum()
    else:
        I = np.ones_like(I) / len(I)
    return np.column_stack([wl, I])


def _group_into_5nm_bins(spectrum_nm_I):
    """
      spectrum(:,1)=round(spectrum(:,1)/5);
      group by 5 nm bins and normalize.
    """
    wl = spectrum_nm_I[:, 0]
    I  = spectrum_nm_I[:, 1]
    bins = np.round(wl / 5.0).astype(int)
    # compact
    u, inv = np.unique(bins, return_inverse=True)
    sums = np.bincount(inv, weights=I)
    wl5 = u * 5.0
    I5  = sums / sums.sum()
    return np.column_stack([wl5, I5])


def _replace_material_placeholders(n_vec, n_vec_backup, lam_nm, metals_db: MetalsDB | None):
    """
    Replace entries >= 10 with refractive indices from metals_db (if provided).
    Operates in-place on a copy of n_vec.
    """
    n_vec = np.array(n_vec, dtype=complex).ravel()
    if metals_db is None:
        return n_vec

    for idx, code in enumerate(np.asarray(n_vec_backup).ravel()):
        try:
            if float(code) >= 10:
                n_vec[idx] = metals_db.get_index(code, lam_nm)
        except Exception:
            # leave as-is if code isn't numeric/material-coded
            pass
    return n_vec


"""
Dipole Radiation Angular Distribution (ADR) demo / utility.

Dependencies:
- numpy
- matplotlib
- (from earlier steps) fresnel(), dipoleL()

This module exposes:
- plot_dipole_adr(flag=1, ...)  # main entry with scenario flags
- plot_adr_frame(...)           # low-level single-frame helper
"""


def draw_guides(ax, strt, Lx=3.0, Ly=1.5, Lz=1.5, color='k', lw=1):
    """Replicates the three MATLAB line() calls."""
    x0, y0, z0 = np.asarray(strt, float)

    # line([x0 x0-3],[y0 y0],[z0 z0], 'color','k','linewidth',1)
    ax.plot([x0, x0 - Lx], [y0, y0],       [z0, z0],       color=color, linewidth=lw)

    # line([x0 x0],[y0 y0-1.5],[z0 z0], 'color','k','linewidth',1)
    ax.plot([x0, x0],       [y0, y0 - Ly], [z0, z0],       color=color, linewidth=lw)

    # line([x0 x0],[y0 y0],[z0 z0+1.5], 'color','k','linewidth',1)
    ax.plot([x0, x0],       [y0, y0],       [z0, z0 + Lz], color=color, linewidth=lw)


def draw_surface(X, Y, Z, C, cmap='viridis', norm = None, alpha = None, ax = None, 
                 linewidth = 1.5, antialiased=True, shade = False):
    # --- make everything safely 2D and aligned ---
    X = np.atleast_2d(np.asarray(X))
    Y = np.atleast_2d(np.asarray(Y))
    Z = np.atleast_2d(np.asarray(Z))
    if X.shape != Y.shape or X.shape != Z.shape:
        # broadcast if needed
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
    C = np.asarray(C)
    # match C to (M,N)
    if C.ndim == 1:
        # expand along the missing axis
        if C.size == X.shape[1]:
            C = np.tile(C[np.newaxis, :], (X.shape[0], 1))
        elif C.size == X.shape[0]:
            C = np.tile(C[:, np.newaxis], (1, X.shape[1]))
        else:
            C = np.tile(C.ravel()[None, :], (X.shape[0], int(np.ceil(X.shape[1]*1.0/C.size))))[:, :X.shape[1]]
    elif C.shape != X.shape:
        # try broadcasting or fall back to tiling
        try:
            C = np.broadcast_to(C, X.shape)
        except ValueError:
            C = np.tile(C, (int(np.ceil(X.shape[0]/C.shape[0])), int(np.ceil(X.shape[1]/C.shape[1]))))[:X.shape[0], :X.shape[1]]

    cmap_obj = plt.colormaps.get_cmap(cmap)  # modern API
    FC = cmap_obj(norm(C))
    FC[..., -1] = alpha  # set alpha into RGBA per-face
    return ax.plot_surface(X, Y, Z,
                           facecolors=FC,
                           rstride=1, cstride=1,
                           linewidth=linewidth,
                           antialiased=antialiased,
                           shade=shade)


def Pfeil(an, en, base=1.0, pos=None, cax=None, al=None, arrowlen=None, rad=None,
          ax=None, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9, shade=False):
    """
    Draw a 3D arrow (shaft + head) from 'an' to 'en' .
    """

    # defaults
    an = np.asarray(an, dtype=float).ravel()
    en = np.asarray(en, dtype=float).ravel()
    if pos is None: pos = np.zeros(3, dtype=float)
    else: pos = np.asarray(pos, dtype=float).ravel()
    if cax is None: cax = np.array([0.0, 1.0], dtype=float)
    else: cax = np.asarray(cax, dtype=float).ravel()
    if al is None: al = np.pi/5
    head_fac = 0.3*(1.0 if arrowlen is None else arrowlen)
    rad_fac  = 0.05*(1.0 if rad is None else rad)

    # geometry
    v   = en - an
    L   = np.linalg.norm(v)
    if L == 0:
        raise ValueError("Start and end points are identical; arrow length is zero.")
    dir = v / L

    # orthonormal-ish side directions
    k = np.sqrt(dir[0]**2 + dir[1]**2)
    if k == 0:
        nx = np.array([1.0, 0.0, 0.0])
        ny = np.array([0.0, 1.0, 0.0])
    else:
        nx = np.array([-(dir[2]*dir[0])/k, -(dir[2]*dir[1])/k, k])
        ny = np.array([nx[1]*dir[2] - nx[2]*dir[1],
                       nx[2]*dir[0] - nx[0]*dir[2],
                       nx[0]*dir[1] - nx[1]*dir[0]])

    nx *= (rad_fac * L)
    ny *= (rad_fac * L)

    hvec   = dir * (head_fac * L)
    en_eff = an + base * (en - an)

    # parameter grids
    u = np.linspace(0, 2*np.pi, 101)                 # around circumference
    t = np.linspace(0, 1, 11)[:, None]               # along shaft (11,1)
    ones_u = np.ones_like(u)[None, :]                # (1,101)
    col    = np.ones_like(t)                         # (11,1)

    # helper rings (1,101)
    cu, su = np.cos(u)[None, :], np.sin(u)[None, :]
    ring_x = nx[0]*cu + ny[0]*su
    ring_y = nx[1]*cu + ny[1]*su
    ring_z = nx[2]*cu + ny[2]*su

    # translations
    base_x = pos[0] + an[0]
    base_y = pos[1] + an[1]
    base_z = pos[2] + an[2]
    dv = (en_eff - an) - hvec  # vector from an to (en-headbase)

    # ===== Shaft (11x101) =====
    X1 = base_x + col @ ring_x + t @ (dv[None, 0] * ones_u)
    Y1 = base_y + col @ ring_y + t @ (dv[None, 1] * ones_u)
    Z1 = base_z + col @ ring_z + t @ (dv[None, 2] * ones_u)
    C1 = cax[0] + (cax[1] - cax[0])*(1 - head_fac) * (t @ ones_u)  # (11,101)

    # ===== Outer sleeve near head (2x101) =====
    scale2 = np.array([[1.0], [1.0 + np.tan(al)]])   # (2,1)
    X2 = base_x + scale2 @ ring_x + np.array([[1.0],[1.0]]) @ (dv[None,0]*ones_u)
    Y2 = base_y + scale2 @ ring_y + np.array([[1.0],[1.0]]) @ (dv[None,1]*ones_u)
    Z2 = base_z + scale2 @ ring_z + np.array([[1.0],[1.0]]) @ (dv[None,2]*ones_u)
    C2 = cax[0] + (cax[1] - cax[0]) * ((1 - head_fac) * np.ones((2, u.size)))

    # ===== Base cap near 'an' (2x101) =====
    X3 = base_x + np.array([[0.0],[1.0]]) @ ring_x
    Y3 = base_y + np.array([[0.0],[1.0]]) @ ring_y
    Z3 = base_z + np.array([[0.0],[1.0]]) @ ring_z
    C3 = cax[0] * np.ones((2, u.size))

    # ===== Arrow head (11x101) =====
    head_base = en_eff - hvec
    X4 = (pos[0] + head_base[0]
          + (1 + np.tan(al)) * (t @ ring_x)
          + (1 - t) @ (hvec[0] * ones_u))
    Y4 = (pos[1] + head_base[1]
          + (1 + np.tan(al)) * (t @ ring_y)
          + (1 - t) @ (hvec[1] * ones_u))
    Z4 = (pos[2] + head_base[2]
          + (1 + np.tan(al)) * (t @ ring_z)
          + (1 - t) @ (hvec[2] * ones_u))
    C4 = cax[0] + (cax[1] - cax[0]) * ((1 - head_fac + head_fac*(1 - t)) @ ones_u)

    # plotting
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=cax[0], vmax=cax[1])
   
    cmap_obj = plt.colormaps.get_cmap(cmap)  # modern API
    

    s1 = draw_surface(X1, Y1, Z1, C1, cmap='viridis', norm = norm, alpha = alpha, ax = ax, 
                     linewidth = linewidth, antialiased=True, shade = False)
    s2 = draw_surface(X2, Y2, Z2, C2, cmap='viridis', norm = norm, alpha = alpha, ax = ax, 
                     linewidth = linewidth, antialiased=True, shade = False)
    s3 = draw_surface(X3, Y3, Z3, C3, cmap='viridis', norm = norm, alpha = alpha, ax = ax, 
                     linewidth = linewidth, antialiased=True, shade = False)
    s4 = draw_surface(X4, Y4, Z4, C4, cmap='viridis', norm = norm, alpha = alpha, ax = ax, 
                     linewidth = linewidth, antialiased=True, shade = False)

    # emulate caxis
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    ax.set_box_aspect([1,1,1])

    return ax, (s1, s2, s3, s4), mappable


def _intensity_surface(theta, phi, v, pc, ps, al, be,
                       scale_by_intensity=True, flip_z_sign=True):
    """
    Return X,Y,Z (P,T) and I (P,T) for the ADR surface.
    This version guarantees all outputs are 2D with shape (P,T).
    """
    # Shapes
    phi   = np.atleast_2d(phi).reshape(-1, 1)     # (P,1)
    theta = np.atleast_1d(theta).ravel()          # (T,)
    P, T = phi.shape[0], theta.size

    # Select z-column 0 (fields should be (T, Z) or (T,))
    v = np.asarray(v); pc = np.asarray(pc); ps = np.asarray(ps)
    if v.ndim == 1:        v0  = v
    else:                  v0  = v[:, 0]
    if pc.ndim == 1:       pc0 = pc
    else:                  pc0 = pc[:, 0]
    if ps.ndim == 1:       ps0 = ps
    else:                  ps0 = ps[:, 0]
    # Sanity
    if v0.shape[0] != T or pc0.shape[0] != T or ps0.shape[0] != T:
        raise ValueError(f"Field column length mismatch: T={T}, got v0={v0.shape}, pc0={pc0.shape}, ps0={ps0.shape}")

    # Orientation
    cos_al, sin_al = np.cos(al), np.sin(al)
    cphi = np.cos(phi - be)       # (P,1)
    sphi = np.sin(phi - be)       # (P,1)

    # Intensity (P,T)
    term_v  = (cos_al * np.ones_like(phi, dtype=complex)) * v0[np.newaxis, :]
    term_pc = (sin_al * cphi) * pc0[np.newaxis, :]
    term_ps = (sin_al * sphi) * ps0[np.newaxis, :]
    I = np.abs(term_v + term_pc)**2 + np.abs(term_ps)**2     # (P,T)
    I = I.real

    # Unit sphere (P,T)
    st = np.sin(theta)[np.newaxis, :]   # (1,T)
    ct = np.cos(theta)[np.newaxis, :]   # (1,T)
    cp = np.cos(phi)                    # (P,1)
    sp = np.sin(phi)
    X = cp @ st                         # (P,T)
    Y = sp @ st                         # (P,T)
    Z = np.tile(ct, (P, 1))             # (P,T)
    if flip_z_sign:
        Z = -Z

    if scale_by_intensity:
        X = X * I
        Y = Y * I
        Z = Z * I

    return X, Y, Z, I


def plot_adr_frame(theta, phi, al, be,
                   stack_down, n_dip, stack_up,
                   d_down, d_dip, d_up, z_in_layer=0.0,
                   ax=None, draw_upper_halfspace=True, lw=1.5,
                   title=None, annotate_angles=True,
                   cmap='viridis', caxis=(-0.1, 1.5), alpha=0.7):
    """
    Plot both hemispheres with MATLAB-like display (intensity-scaled coordinates, colormap, alpha).
    Guarantees facecolors match surface shapes.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=caxis[0], vmax=caxis[1])
    cmap_obj = plt.colormaps.get_cmap(cmap).reversed()   # or plt.colormaps.get_cmap(cmap + "_r")

     # --- Down (−Z) hemisphere ---
    v, pc, ps, *_ = dipoleL(theta, z_in_layer,
                            np.asarray(stack_down), n_dip, np.asarray(stack_up),
                            np.asarray(d_down), d_dip, np.asarray(d_up))
    # v,pc,ps may come out (T,1) already; _intensity_surface handles both
    Xd, Yd, Zd, Id = _intensity_surface(theta, phi, v, pc, ps, al, be,
                                        scale_by_intensity=True, flip_z_sign=True)
    # Color array aligned to (P,T)
    
    Cd = cmap_obj(norm(Id))
    Cd[..., -1] = alpha  # apply alpha into RGBA
    
    strt = np.array([0., 0., 0.]);
    an = strt+np.array([0., 0., 0])
    en = strt + 1.5*np.array([np.cos(be)*np.sin(al), np.sin(be)*np.sin(al), np.cos(al)])
    Pfeil(an,en, ax=ax)
    
    ax.plot([strt[0], strt[0]-3.0], [strt[1], strt[1]],     [strt[2], strt[2]],     color='k', linewidth=1)
    ax.plot([strt[0], strt[0]],     [strt[1], strt[1]-1.5], [strt[2], strt[2]],     color='k', linewidth=1)
    ax.plot([strt[0], strt[0]],     [strt[1], strt[1]],     [strt[2], strt[2]+1.5], color='k', linewidth=1)

    
    ax.plot_surface(Xd, Yd, Zd, rstride=1, cstride=1,
                    facecolors=Cd, linewidth=0, antialiased=False, shade=False, alpha = alpha)

    # red edge profiles at φ = 0 and π
    for phi_edge in (0.0, np.pi):
        Xe, Ye, Ze, Ie = _intensity_surface(theta, np.array([phi_edge]), v, pc, ps, al, be,
                                            scale_by_intensity=True, flip_z_sign=True)
        ax.plot(Xe.ravel(), Ye.ravel(), Ze.ravel(), 'r', linewidth=lw)

    # --- Up (+Z) hemisphere ---
    if draw_upper_halfspace:
        v1, pc1, ps1, *_ = dipoleL(theta,
                                   d_dip - z_in_layer,
                                   np.asarray(stack_up), n_dip, np.asarray(stack_down),
                                   np.asarray(d_up), d_dip, np.asarray(d_down))

        Xu, Yu, Zu, Iu = _intensity_surface(theta, phi, v1, pc1, ps1, al, be,
                                            scale_by_intensity=True, flip_z_sign=False)
        Cu = cmap_obj(norm(Iu))
        
        Cu[..., -1] = alpha
        ax.plot_surface(Xu, Yu, Zu, rstride=1, cstride=1,
                        facecolors=Cu, linewidth=0, antialiased=False, shade=False, alpha = alpha)

        for phi_edge in (0.0, np.pi):
            Xe, Ye, Ze, Ie = _intensity_surface(theta, np.array([phi_edge]), v1, pc1, ps1, al, be,
                                                scale_by_intensity=True, flip_z_sign=False)
            ax.plot(Xe.ravel(), Ye.ravel(), Ze.ravel(), 'r', linewidth=lw)

    # Cosmetics similar to MATLAB
    ax.axis("off")
    ax.view_init(elev=30, azim=-40)
    # fixed box from your code
    ax.set_xlim(-12.5*0.5, 12.5*0.5)
    ax.set_ylim(-7.0*0.5, 9.5*0.5)
    ax.set_zlim(-15.0*0.5, 10.0*0.5)
    ax.set_box_aspect([1, 1, 1])

    if annotate_angles:
        th_deg = int(round(al*180/np.pi))
        ph_deg = int(round(be*180/np.pi - 180))
        ax.text(1, 1, 2, rf'$\theta = {th_deg}^\circ, \ \phi = {ph_deg}^\circ$')

    if title:
        ax.set_title(title)

    return ax


def set_axes_equal(ax):
    """Equal scale for 3D axes."""
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def plot_dipole_adr(wavelength_nm=690.0,
        n_glass=1.52, n_top=1.33, n_polymer=1.46,
        stack_metal=None,    # e.g., [n_titan, n_gold, n_titan, n_polymer]
        d_metal_nm=None,     # e.g., [2, 10, 1, 10]  (thicknesses between indices where needed)
        dip_z_nm=60.0,
        al_list=None,
        be=np.pi,            # like bev = pi
        lw=1.5):
    """
    High-level entry with “scenarios” controlled by `flag`.

    Flags:
      1  Changing dipole orientation in both half spaces (polymer over glass)
      2  Polymer + metal stack both sides (gold paper style)
      3  Restricted rotation (uses a Gaussian angular spread -> shows weighting example)

    Parameters
    ----------
    wavelength_nm : float
        Emission wavelength (for scaling nm -> k-units via 2π/λ).
    n_glass, n_top, n_polymer : float/complex
        Typical indices used in the examples.
    metal_triplet : (complex, complex)
        Example metal+spacer indices.
    d_triplet_nm : (float, float)
        Metal+spacer thicknesses in nm.
    dip_z_nm : float
        Emitter height within dipole layer (nm).
    dip_angle_list : list of radians
        If provided, iterate over these al (β fixed = π).
    save_path : str or None
        If set, saves each frame as PNG with incremental numbering.
    """
    theta = np.linspace(0, np.pi/2, 301)       # (0:300)/300*pi/2
    phi   = np.linspace(0, np.pi, 201)[:, None]  # (0:200)'/200*pi

    # critical angle insertion (psi) like in MATLAB
    psi = np.arcsin(n_top / n_glass)           # asin(n1/n2), with n1=n_top, n2=n_glass
    theta = np.sort(np.concatenate(([psi], theta)))

    if al_list is None:
        al_list = np.arange(0, np.pi/2 + 1e-12, np.pi/6)

    # thickness unit converter (nm -> k0^-1)
    k0 = 2*np.pi / (wavelength_nm * 1e-3)      # if nm * 1e-3 gives μm
    to_k = lambda nm: (nm * 1e-3) * k0

    # default simple stacks (polymer on glass)
    if stack_metal is None:
        stack_down = np.array([n_glass])          # glass
        stack_up   = np.array([n_top])            # top half-space
        n_dip = n_polymer
        d_down = np.array([]); d_up = np.array([])
        d_dip  = to_k(100.0)                      # thick layer
    else:
        # interpret user-provided four-layer like MATLAB example
        # Example: stack_down = [glass, metal..., polymer], stack_up = [top]
        stack_down = np.array([n_glass, *stack_metal])    # glass + metals + polymer
        stack_up   = np.array([n_top])
        n_dip = n_polymer
        # You can tailor this to your DipoleL’s required thickness layout
        d_down = to_k(np.array(d_metal_nm, dtype=float))
        d_up   = np.array([])
        d_dip  = to_k(100.0)

    z_in = to_k(dip_z_nm)

    for al in al_list:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        plot_adr_frame(theta, phi, al, be,
                       stack_down, n_dip, stack_up,
                       d_down, d_dip, d_up, z_in_layer=z_in,
                       ax=ax, draw_upper_halfspace=True, lw=lw,
                       title=None, annotate_angles=True,
                       cmap='viridis', caxis=(-0.1, 1.5), alpha=0.7)

        # Optional axes guides like MATLAB lines:
        # ax.plot([-3, 0], [0, 0], [0, 0], 'k', lw=1)
        # ax.plot([0, 0], [-1.5, 0], [0, 0], 'k', lw=1)
        # ax.plot([0, 0], [0, 0], [0, 1.5], 'k', lw=1)

        plt.tight_layout()
        plt.show()
  
        
 

def miet_calc(
    al_res,
    lamem,        # float nm OR dict with keys: SpectrumFile, Wavel_Small, Wavel_Large
    n0, n, n1,    # stacks and dipole layer index (can include material codes >=10)
    d0, d, d1,    # thicknesses in nm (d0: len=n0-1, d: scalar, d1: len=n1-1)
    qyield,       # quantum yield (fraction or %, both accepted)
    tau_free,     # free-space lifetime (ns)
    fig=False,
    curveType=None,     # None or 'maximum' or bool/int (kept for parity)
    metals_db: MetalsDB | None = None,
    hash_wgm=None       # optional: callable(n_all, d_all)->(qp, qs)
):
    """

    Returns
    -------
    z_nm : (M,)
    life : (M,) if random orientation (al_res is NaN/None),
           or (M, K) for K polar angles (radians).
    """
    # figure flag like MATLAB
    fig = bool(fig)

    # qy: accept % or fraction
    qy = float(qyield)
    if qy > 1.0:
        qy = qy / 100.0

    # curve type
    if isinstance(curveType, str) and curveType.lower() == 'maximum':
        maxCurve = 1
    elif curveType is None:
        maxCurve = 0
    else:
        maxCurve = int(curveType)

    # spectrum handling
    polychrome = isinstance(lamem, dict)
    if polychrome:
        # load spectrum and crop to filter window
        spec = _load_spectrum_any(lamem['SpectrumFile'])
        wl_small = float(lamem['Wavel_Small'])
        wl_large = float(lamem['Wavel_Large'])
        spec = spec[(spec[:, 0] >= wl_small) & (spec[:, 0] <= wl_large)]
        spec = _group_into_5nm_bins(spec)  # shape (S, 2): [lambda_nm, weight]
    else:
        spec = np.array([[float(lamem), 1.0]])

    # build z-grid (nm), following MATLAB logic
    d_var = float(d)
    n1_arr = np.atleast_1d(n1)
    same_as_n = (np.size(n1_arr) == 1) and (complex(n1_arr[0]) == complex(n))

    if (np.size(n1_arr) > 1) or (not same_as_n):
        if d_var < 100.0:
            # 1 : (d-1)/100 : d-1
            step = max((d_var - 1.0) / 100.0, 1.0)  # keep sane step
            z_nm = np.arange(1.0, d_var, step, dtype=float)
        else:
            z_nm = np.arange(1.0, min(d_var, d_var), 1.0, dtype=float)  # effectively 1:1:d-1
            if z_nm.size == 0 and d_var > 1:
                z_nm = np.arange(1.0, d_var, 1.0, dtype=float)
    else:
        z_nm = np.arange(1.0, d_var, 1.0, dtype=float)

    if z_nm.size == 0:
        # Nothing to compute
        return z_nm, np.array([])

    # prepare outputs
    if al_res is None or (np.isscalar(al_res) and (np.isnan(al_res))):
        life_accum = np.zeros(z_nm.shape, dtype=float)
    else:
        if np.isscalar(al_res):
            theta = np.deg2rad(np.arange(90.0, -1.0, -float(al_res)))
        else:
            theta = np.array(al_res, dtype=float)
        life_accum = np.zeros((z_nm.size, theta.size), dtype=float)

    # store backups for material codes
    n0_backup = np.array(n0, dtype=complex).ravel()
    n1_backup = np.array(n1, dtype=complex).ravel()

    # loop wavelengths
    for lam_nm, weight in spec:
        fac = 2.0 * np.pi / lam_nm

        # replace placeholders >=10 by actual metal indices at lam_nm
        n0_use = _replace_material_placeholders(n0, n0_backup, lam_nm, metals_db)
        n1_use = _replace_material_placeholders(n1, n1_backup, lam_nm, metals_db)

        # call LifetimeL with scaled distances/thicknesses
        lvd, lvu, lpd, lpu, qvd, qvu, qpd, qpu, _, _ = lifetimeL(
            fac * z_nm,
            n0_use, complex(n),
            n1_use,
            fac * np.asarray(d0, dtype=float),
            fac * float(d),
            fac * np.asarray(d1, dtype=float),
            hash_waveguide_mode=hash_wgm,
        )

        Sv = (qvu + qvd)   # vertical
        Sp = (qpu + qpd)   # parallel

        if al_res is None or (np.isscalar(al_res) and (np.isnan(al_res))):
            # random orientation
            life = tau_free / ( (1.0 - qy) + qy * (Sv + 2.0 * Sp) / (4.0 * complex(n)) )
            life = np.real(life)
            # trim for uniqueness
            dlife = np.diff(life)
            first_peak = np.argmax(dlife < 0) if np.any(dlife < 0) else None
            if first_peak is not None and first_peak != 0:
                if maxCurve == 1:
                    life[first_peak+1:] = np.nan
                else:
                    limit_LT = np.nanmin(life[first_peak+1:])
                    idx = np.argmax(life > limit_LT)
                    if idx > 0:
                        life[idx:] = np.nan
        else:
            # fixed orientations
            life = np.zeros((z_nm.size, theta.size), dtype=float)
            for i, th in enumerate(theta):
                Sr = Sv * (np.cos(th)**2) + Sp * (np.sin(th)**2)
                life[:, i] = np.real(tau_free / ((1.0 - qy) + (Sr / (4.0/3.0 * complex(n))) * qy))

            # trim each curve
            for i in range(theta.size):
                dlife = np.diff(life[:, i])
                first_peak = np.argmax(dlife < 0) if np.any(dlife < 0) else None
                if first_peak is not None and first_peak != 0:
                    if maxCurve == 0:
                        life[first_peak+1:, i] = np.nan
                    else:
                        limit_LT = np.nanmin(life[first_peak+1:, i])
                        idx = np.argmax(life[:, i] > limit_LT)
                        if idx > 0:
                            life[idx:, i] = np.nan

        life_accum += float(weight) * life

    # final result
    life_out = life_accum

    # plotting (optional)
    if fig:
        plt.figure()
        if life_out.ndim == 1:
            plt.plot(z_nm, life_out, lw=2)
            plt.xlabel('distance from surface (nm)')
            plt.ylabel('lifetime (ns)')
            plt.title('MIET Calibration — Random Orientation')
            plt.tight_layout()
        else:
            cmap = plt.get_cmap('hsv', life_out.shape[1])
            for i in range(life_out.shape[1]):
                plt.plot(z_nm, life_out[:, i], color=cmap(i), lw=2, label=fr'polar angle = {np.rad2deg(theta[i]):.0f}°')
            plt.xlabel('distance from surface (nm)')
            plt.ylabel('lifetime (ns)')
            plt.legend(loc='lower right')
            plt.title('MIET Calibration — Various Orientations')
            plt.tight_layout()
        plt.show()

    return z_nm, life_out    
    

def brightness_dipole(z, n0, n, n1, d0, d, d1, NA, QY, curves=False):
    """

    Inputs
    ------
    z   : array-like, emitter height(s) within its layer (units with k0=1)
    n0  : array of refractive indices below (bottom->top)
    n   : scalar refractive index of the emitter layer
    n1  : array of refractive indices above (bottom->top)
    d0  : thicknesses below (len = len(n0)-1), same units as z
    d   : scalar thickness of emitter layer, same units as z
    d1  : thicknesses above (len = len(n1)-1), same units as z
    NA  : numerical aperture
    QY  : quantum yield in medium n, far from surfaces (0..1)
    curves : bool, if True plots collection efficiency, local QY, brightness

    Returns
    -------
    bv, bp, br, bf : arrays (len(z),)
        brightness for vertical, parallel (in-plane), fast-rotating, and
        fixed-randomly-oriented dipoles.
    """
    # z as 1D array
    z = np.atleast_1d(np.asarray(z, dtype=float)).ravel()

    # --- orientations for quickly rotating emitter ---
    alpha = (np.arange(0, 101) / 100.0 * np.pi)[:, None]  # (101,1)
    dalpha = float(alpha[1, 0] - alpha[0, 0])

    # --- angular distribution + collection cone ---
    # theta_max = asin(NA / n0(1))  (clip to [0,1] for safety)
    n0_1 = complex(np.atleast_1d(n0).ravel()[0])
    ratio = np.real(NA / n0_1)
    ratio = np.clip(ratio, 0.0, 1.0)
    theta_max = np.arcsin(ratio)
    theta = np.linspace(0.0, theta_max, 1001)  # (0:1000)/1000 * theta_max
    dtheta = float(theta[1] - theta[0])

    # Dipole radiation (θ rows, z cols)
    v, pc, ps, *_ = dipoleL(theta, z, n0, n, n1, d0, d, d1)

    # collection integrals over θ
    sin_th_col = np.sin(theta)[:, None]               # (T,1)
    collv = np.sum((np.abs(v)**2) * sin_th_col, axis=0) * dtheta
    collp = np.sum((np.abs(pc)**2 + np.abs(ps)**2) * sin_th_col, axis=0) * 0.5 * dtheta

    # total electromagnetic decay rates (down+up) at z
    _, _, _, _, qvd, qvu, qpd, qpu, _, _ = lifetimeL(z, n0, n, n1, d0, d, d1)

    # collection efficiency for each dipole model
    uv = collv / (qvd + qvu)
    up = collp / (qpd + qpu)
    ur = (collv + 2.0 * collp) / ((qvd + qvu) + 2.0 * (qpd + qpu))

    # ensemble of fixed, randomly oriented dipoles (integrate over alpha)
    # use outer products like MATLAB: (cos^2 * collv + sin^2 * collp) / (cos^2*(qv) + sin^2*(qp))
    cos2 = (np.cos(alpha) ** 2)                       # (A,1)
    sin2 = (np.sin(alpha) ** 2)                       # (A,1)
    sin_a = np.sin(alpha)                             # (A,1)

    collv_row = collv.reshape(1, -1)                  # (1,Nz)
    collp_row = collp.reshape(1, -1)
    qv_row = (qvd + qvu).reshape(1, -1)
    qp_row = (qpd + qpu).reshape(1, -1)

    num_coll = cos2 @ collv_row + sin2 @ collp_row    # (A,Nz)
    den_coll = cos2 @ qv_row  + sin2 @ qp_row         # (A,Nz)
    uf = np.sum(sin_a * (num_coll / den_coll), axis=0) * dalpha / 2.0

    if curves:
        plt.figure()
        plt.plot(z, np.real(uv), '-r', z, np.real(up), '-b',
                 z, np.real(ur), '-g', z, np.real(uf), '-k')
        plt.legend(['vertical', 'parallel', 'rotating', 'fixed'])
        plt.xlabel('z*k [-]')
        plt.ylabel('collection efficiency [-]')
        plt.tight_layout()

    # local quantum yields
    # QYv = (qv) / (qv + 4/3*n*(1/QY - 1)), etc.
    fac_nr = (4.0 / 3.0) * complex(n) * (1.0 / float(QY) - 1.0)
    QYv = (qvd + qvu) / ((qvd + qvu) + fac_nr)
    QYp = (qpd + qpu) / ((qpd + qpu) + fac_nr)
    QYr = ((qvd + qvu) + 2.0 * (qpd + qpu)) / 3.0
    QYr = QYr / (QYr + fac_nr)

    num_qy = cos2 @ qv_row + sin2 @ qp_row
    den_qy = num_qy + fac_nr
    QYf = np.sum(sin_a * (num_qy / den_qy), axis=0) * dalpha / 2.0

    if curves:
        plt.figure()
        plt.plot(z, np.real(QYv), '-r', z, np.real(QYp), '-b',
                 z, np.real(QYr), '-g', z, np.real(QYf), '-k')
        plt.legend(['vertical', 'parallel', 'rotating', 'fixed'])
        plt.xlabel('z*k [-]')
        plt.ylabel('local quantum yield [-]')
        plt.tight_layout()

    # brightness = collection efficiency * local quantum yield
    bv = uv * QYv
    bp = up * QYp
    br = ur * QYr

    den_bf = (cos2 @ qv_row + sin2 @ qp_row) + fac_nr
    bf = np.sum(sin_a * (num_coll / den_bf), axis=0) * dalpha / 2.0

    if curves:
        plt.figure()
        plt.plot(z, np.real(bv), '-r', z, np.real(bp), '-b',
                 z, np.real(br), '-g', z, np.real(bf), '-k')
        plt.legend(['vertical', 'parallel', 'rotating', 'fixed'])
        plt.xlabel('z*k [-]')
        plt.ylabel('brightness [-]')
        plt.tight_layout()
        plt.show()

    # return real parts (tiny imaginary numerical noise can appear)
    return (np.real_if_close(bv), np.real_if_close(bp),
            np.real_if_close(br), np.real_if_close(bf))

        


