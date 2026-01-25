# -*- coding: utf-8 -*-
"""
MIET-FLIM Analysis Utilities

This module contains utility functions for MIET-FLIM data processing and visualization,
including functions extracted from FlavMetaFLIM.py for FLIM analysis and image processing.

@author: MIET-FLIM Pipeline
"""

import numpy as np
import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tifffile import imwrite

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter

# Custom module imports for FLIM processing
from PTU_ScanRead import PTU_ScanRead, Process_Frame, mHist, cim
from FLIM_fitter import Calc_mIRF, FluoFit, PatternMatchIm

ArrayLike = Union[np.ndarray, list, tuple, float, int]

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#%%


def _get_ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax

def _normalize_01(a):
    a = np.asarray(a, float)
    m = np.nanmin(a)
    M = np.nanmax(a)
    if not np.isfinite(m) or not np.isfinite(M) or M == m:
        return np.zeros_like(a)
    return (a - m) / (M - m)

def _jet_black():
    jet = plt.colormaps.get_cmap('jet', 256)
    jet_arr = jet(np.linspace(0, 1, 256))
    jet_black = np.vstack([[0, 0, 0, 1], jet_arr])  # prepend black
    return ListedColormap(jet_black, name='jet_black')

def mim(x, p1=None, p2=None, p3=None, ax=None):
    """

    Behaviors:
      - 2D: imagesc with hot, axis image off
      - 3D: tile slices horizontally (normalize each)
      - 4D: tile [j] horizontally for each k, then stack rows for k
      - p1 == [vmin vmax] -> use those limits
      - p1 == 'h'/'v' -> show colorbar (horizontal/vertical)
      - p1 as string -> render text over image
      - p1 same size as x -> masked jet-on-black composite with auto scaling
      - nargin>3 branch: p1 mask same size as x and p2 == [vmin vmax]
                         or p1/p2 == 'h'/'v' for colorbar after a basic show
    Returns:
      handle (AxesImage) for the shown image.
    """
    x = np.asarray(x)
    fig, ax = _get_ax(ax)

    # --- no modifiers: behave by ndims ---
    if p1 is None and p2 is None and p3 is None:
        nd = x.ndim
        if nd == 2:
            im = ax.imshow(x, cmap='hot', aspect='equal', origin='upper')
            ax.set_axis_off()
            return im
        elif nd == 3:
            nj = x.shape[2]
            row = []
            for j in range(nj):
                sl = x[:, :, j]
                row.append(_normalize_01(sl))
            mosaic = np.concatenate(row, axis=1)
            return mim(mosaic, ax=ax)
        elif nd == 4:
            nj, nk = x.shape[2], x.shape[3]
            rows = []
            # first row (k=0)
            row = []
            for j in range(nj):
                sl = x[:, :, j, 0]
                row.append(_normalize_01(sl))
            rows.append(np.concatenate(row, axis=1))
            # remaining rows (k=1..nk-1)
            for k in range(1, nk):
                row = []
                for j in range(nj):
                    sl = x[:, :, j, k]
                    row.append(_normalize_01(sl))
                rows.append(np.concatenate(row, axis=1))
            mosaic = np.concatenate(rows, axis=0)
            return mim(mosaic, ax=ax)
        else:
            raise ValueError("mim: x must be 2D/3D/4D when no p1/p2/p3 provided.")

    # --- nargin == 2 behaviors ---
    if p2 is None and p3 is None:
        # p1 numeric [vmin vmax]
        if np.isscalar(p1) is False and np.size(p1) == 2 and np.all(np.isfinite(p1)):
            vmin, vmax = float(p1[0]), float(p1[1])
            im = ax.imshow(x, cmap='hot', aspect='equal', origin='upper',
                           vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            return im

        # p1 == 'h' / 'v' -> colorbar placement
        if isinstance(p1, str) and p1 in ('h', 'v'):
            im = mim(x, ax=ax)
            orientation = 'horizontal' if p1 == 'h' else 'vertical'
            fig.colorbar(im, ax=ax, orientation=orientation, fraction=0.046, pad=0.04)
            return im

        # p1 is arbitrary string -> overlay text
        if isinstance(p1, str):
            im = mim(x, ax=ax)
            a, b = x.shape[:2]
            # mimic MATLAB: text at near bottom-right; scale factor ~ 0.025 per char
            ax.text(b * (1 - 0.025 * len(p1)), 0.06 * a, p1,
                    fontname='Times New Roman', fontsize=16, color='w')
            return im

        # p1 is an array same size as x -> masked jet-on-black composite
        p1 = np.asarray(p1)
        if p1.shape == x.shape:
            mask = p1.astype(float)
            # normalize mask to [0,1] over finite entries
            finite_mask = np.isfinite(mask)
            if np.any(finite_mask):
                mm = np.nanmin(mask[finite_mask])
                span = np.nanmax(mask[finite_mask]) - mm
            else:
                mm, span = 0.0, 0.0
            if span > 0:
                mask01 = np.zeros_like(mask, dtype=float)
                mask01[finite_mask] = (mask[finite_mask] - mm) / span
            else:
                mask01 = np.zeros_like(mask, dtype=float)

            # scale x using only where mask is finite (like MATLAB)
            finite_x = finite_mask
            if np.any(finite_x):
                xm = np.nanmin(x[finite_x])
                xspan = np.nanmax(x[finite_x]) - xm
            else:
                xm, xspan = 0.0, 0.0

            if xspan > 0:
                # map to [1..64] index in "jet_black"
                Xidx = 1 + (x - xm) / xspan * 63.0
            else:
                Xidx = np.full_like(x, 64.0)

            Xidx = np.nan_to_num(Xidx, nan=1.0)
            Xidx = np.clip(Xidx, 1.0, 64.0)

            # build RGB using jet_black
            cmap = _jet_black()
            lut = cmap(np.linspace(0, 1, 65))  # 0..64
            # indices are float in [1,64] -> floor/ceil blend like MATLAB
            flo = np.floor(Xidx).astype(int)
            cei = np.ceil(Xidx).astype(int)
            # gather RGB (ignore alpha channel)
            c = np.zeros(x.shape + (3,), float)
            for ch in range(3):
                cflo = lut[flo, ch]
                ccei = lut[cei, ch]
                c[..., ch] = ((flo + 1 - Xidx) * cflo + (Xidx - flo) * ccei) * mask01

            im = ax.imshow(c, aspect='equal', origin='upper')
            ax.set_axis_off()

            # optional colorbar with relabeled ticks to data scale
            if xspan > 0:
                mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=xm, vmax=xm + xspan),
                                                 cmap=cmap)
                mappable.set_array([])
                # choose orientation based on aspect (like original heuristic)
                if x.shape[0] >= x.shape[1]:
                    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
                    ticks = cb.get_ticks()
                    # round formatting similar to MATLAB
                    if len(ticks):
                        scale = 10 ** (np.floor(np.log10(ticks[-1])) - 2) if ticks[-1] != 0 else 1
                        labels = (xm + ticks * xspan)
                        labels = np.round(labels / scale) * scale
                        cb.set_ticks(ticks)
                        cb.set_ticklabels([f"{v:g}" for v in labels])
                else:
                    cb = fig.colorbar(mappable, ax=ax, orientation='horizontal',
                                      fraction=0.046, pad=0.08)
                    ticks = cb.get_ticks()
                    if len(ticks):
                        scale = 10 ** (np.floor(np.log10(ticks[-1])) - 2) if ticks[-1] != 0 else 1
                        labels = (xm + ticks * xspan)
                        labels = np.round(labels / scale) * scale
                        cb.set_ticks(ticks)
                        cb.set_ticklabels([f"{v:g}" for v in labels])

            return im

        # otherwise, fall back to simple show
        im = ax.imshow(x, cmap='hot', aspect='equal', origin='upper')
        ax.set_axis_off()
        return im

    # --- nargin > 3 branch ---
    # Case: p1 same size as x AND p2 == [vmin vmax]  -> masked colored composite with fixed scaling
    if isinstance(p2, (list, tuple, np.ndarray)) and np.size(p2) == 2 and np.asarray(p1).shape == x.shape:
        vmin, vmax = float(p2[0]), float(p2[1])
        span = vmax - vmin
        mask = np.asarray(p1, float)

        # normalize mask to [0,1]
        finite_mask = np.isfinite(mask)
        if np.any(finite_mask):
            mm = np.nanmin(mask[finite_mask])
            mspan = np.nanmax(mask[finite_mask]) - mm
        else:
            mm, mspan = 0.0, 0.0
        if mspan > 0:
            mask01 = np.zeros_like(mask)
            mask01[finite_mask] = (mask[finite_mask] - mm) / mspan
        else:
            mask01 = np.zeros_like(mask)

        # map x to [1..64] using fixed [vmin,vmax]
        if span > 0:
            Xidx = 1 + (x - vmin) / span * 63.0
        else:
            Xidx = np.full_like(x, 64.0)
        Xidx = np.nan_to_num(Xidx, nan=1.0)
        Xidx = np.clip(Xidx, 1.0, 64.0)

        cmap = plt.colormaps.get_cmap('jet')
        lut = cmap(np.linspace(0, 1, 64))
        lut = np.vstack([[1, 1, 1, 1], lut])  # prepend white (like original added [1 1 1])
        c = np.zeros(x.shape + (3,), float)
        flo = np.floor(Xidx).astype(int)
        cei = np.ceil(Xidx).astype(int)
        for ch in range(3):
            cflo = lut[flo, ch]
            ccei = lut[cei, ch]
            c[..., ch] = ((flo + 1 - Xidx) * cflo + (Xidx - flo) * ccei) * mask01

        im = ax.imshow(c, aspect='equal', origin='upper')
        ax.set_axis_off()

        # Decide colorbar orientation: vertical if tall, or if p3=='v'
        orient = 'vertical'
        if (x.shape[0] <= x.shape[1]) and not (isinstance(p3, str) and p3 == 'v'):
            orient = 'horizontal'
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                                cmap='jet'),
                          ax=ax, orientation=orient, fraction=0.046, pad=0.04 if orient=='vertical' else 0.08)
        return im

    # Cases: p1 or p2 are [vmin vmax] plus the other is 'h'/'v' for colorbar
    if isinstance(p1, (list, tuple, np.ndarray)) and np.size(p1) == 2 and isinstance(p2, str) and p2 in ('h', 'v'):
        im = mim(x, p1, ax=ax)
        fig.colorbar(im, ax=ax,
                     orientation=('horizontal' if p2 == 'h' else 'vertical'),
                     fraction=0.046, pad=0.04)
        return im

    if isinstance(p2, (list, tuple, np.ndarray)) and np.size(p2) == 2 and isinstance(p1, str) and p1 in ('h', 'v'):
        im = mim(x, p2, ax=ax)
        fig.colorbar(im, ax=ax,
                     orientation=('horizontal' if p1 == 'h' else 'vertical'),
                     fraction=0.046, pad=0.04)
        return im

    # Fallback: simple image
    im = ax.imshow(x, cmap='hot', aspect='equal', origin='upper')
    ax.set_axis_off()
    return im


def Disk(m, opt=None):
    """

    Parameters
    ----------
    m : int or (m1, m2)
        If scalar, uses [m, m]. The output grid is
        y in [-m1..m1], x in [-m2..m2].
    opt : None, str, callable, or array-like (optional)
        - None: returns a normalized binary disk/ellipse mask (using m1 in both axes,
                matching the MATLAB code).
        - str:  a Python expression evaluated with variables:
                rad (sqrt(jx^2+jy^2)), jx, jy, m1, m2, and numpy as np.
                The result multiplies the base Disk mask.
        - callable: a function f(rad) returning weights; multiplied with base Disk.
        - array-like: weights of same shape as the mask; multiplied with base Disk.

    Returns
    -------
    y : 2D float ndarray
        Mask normalized so that y.sum() == 1 (unless it would be 0, then left as all-zeros).
    """
    m = np.atleast_1d(m).astype(int).ravel()
    if m.size == 1:
        m = np.array([m[0], m[0]], dtype=int)
    m1, m2 = int(m[0]), int(m[1])

    # coordinate grids: jy varies along rows (y), jx along columns (x)
    jx, jy = np.meshgrid(np.arange(-m2, m2 + 1),
                         np.arange(-m1, m1 + 1),
                         indexing='xy')

    if opt is None:
        # NOTE: matches MATLAB exactly (uses m1 for both axes):
        r = np.sqrt((jx.astype(float) ** 2) / (m1 ** 2) +
                    (jy.astype(float) ** 2) / (m1 ** 2))
        y = (np.ceil(r) < 2).astype(float)  # effectively r <= 1
    else:
        # Base disk (no opt), then weight by 'opt'
        y_base = Disk([m1, m2])  # recursion with opt=None

        rad = np.sqrt(jx.astype(float) ** 2 + jy.astype(float) ** 2)

        if callable(opt):
            w = np.asarray(opt(rad), dtype=float)
        elif isinstance(opt, str):
            # Evaluate with limited, explicit context
            ctx_globals = {"np": np}
            ctx_locals = {"rad": rad, "jx": jx, "jy": jy, "m1": m1, "m2": m2}
            w = np.asarray(eval(opt, ctx_globals, ctx_locals), dtype=float)
        else:
            w = np.asarray(opt, dtype=float)

        if w.shape != y_base.shape:
            raise ValueError(f"'opt' produced shape {w.shape}, expected {y_base.shape}.")

        y = y_base * w

    s = y.sum()
    if s > 0:
        y = y / s
    return y


def _as_array(a: ArrayLike) -> np.ndarray:
    return np.asarray(a)

def _expand_to_match(x: np.ndarray, g: ArrayLike) -> np.ndarray:
    """
    Expand grid-like input g (scalar, row, column, or grid) to the shape of x.
    Mimics the MATLAB logic in your function.
    """
    G = _as_array(g)
    m, n = x.shape
    if G.size == 1:
        return np.full_like(x, float(G), dtype=float)

    mr, nr = G.shape if G.ndim == 2 else (G.size, 1)

    # Already matching shape?
    if G.shape == x.shape:
        return G

    # Column vector length m
    if mr == m and (nr == 1 or G.ndim == 1):
        return np.tile(G.reshape(m, 1), (1, n))

    # Row vector length n
    if nr == n and (mr == 1 or G.ndim == 1):
        return np.tile(G.reshape(1, n), (m, 1))

    # Column vector length n (needs transpose then tile across rows)
    if mr == n and nr == 1:
        return np.tile(G.reshape(n, 1).T, (m, 1))

    # Row vector length m (needs transpose then tile across cols)
    if mr == 1 and nr == m:
        return np.tile(G.reshape(m, 1), (1, n))

    # Try transpose fallback (like the MATLAB code’s last resort)
    Gt = G.T if G.ndim == 2 else G
    if Gt.shape == x.shape:
        return Gt

    raise ValueError(f"Cannot expand grid of shape {G.shape} to match {x.shape}")

def mpcolor(
    r: ArrayLike,
    z: Optional[ArrayLike] = None,
    x: Optional[ArrayLike] = None,
    flag: Optional[Union[str, np.ndarray]] = None,
    ax: Optional[Axes] = None,
    cmap: str = "hot"
):
    """
    mpcolor(r)                        -> quick image of r (shading-like)
    mpcolor(r, z, x)                  -> pcolormesh over grids r,z with values x
    mpcolor(r, z, x, 'lr')            -> mirror left-right (negated flipped r on left)
    mpcolor(r, z, x, 'ud')            -> mirror up-down   (negated flipped z on bottom)
    mpcolor(r, z, x, 'both')          -> mirror both axes
    mpcolor(r, z, x, XRIGHT)          -> stitch XRIGHT to the right of mirrored left

    Returns the Matplotlib artist.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # --- Case A: only r provided (like MATLAB pcolor(r); shading interp)
    if z is None and x is None and flag is None:
        R = _as_array(r)
        im = ax.imshow(R, origin="lower", cmap=cmap, aspect="equal", interpolation="bilinear")
        ax.axis("off")  # MATLAB turns axis off in many helpers; keep visuals clean
        plt.set_cmap(cmap)
        return im

    # Ensure arrays
    r = _as_array(r)
    z = _as_array(z) if z is not None else None
    x = _as_array(x) if x is not None else None

    if z is None or x is None:
        raise ValueError("When providing more than one arg, call as mpcolor(r, z, x, [flag]).")

    # Expand r,z to shape of x if needed (mimic MATLAB broadcasting rules)
    r = _expand_to_match(x, r)
    z = _expand_to_match(x, z)

    # Helper to draw pcolormesh for grid-sized arrays (like pcolor)
    def _draw(R, Z, X):
        # pcolormesh expects corners; for pcolor-like behavior, we can
        # pass centers with shading='nearest'/'auto'. We'll use shading='auto'
        pm = ax.pcolormesh(R, Z, X, shading="auto", cmap=cmap)
        ax.set_aspect("equal", adjustable="box")
        return pm

    artist = None

    # --- Case B: r, z, x provided (no flag)
    if flag is None:
        artist = _draw(r, z, x)

    # --- Case C: string mirroring flags
    elif isinstance(flag, (str, np.str_)):
        f = str(flag).lower()

        if f == "lr":
            R = np.vstack([-np.flipud(r), r])
            Z = np.vstack([z, z])
            X = np.vstack([np.flipud(x), x])
            artist = _draw(R, Z, X)
            # xticks → absolute values
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])

        elif f == "ud":
            R = np.hstack([r, r])
            Z = np.hstack([-np.fliplr(z), z])
            X = np.hstack([np.fliplr(x), x])
            artist = _draw(R, Z, X)
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])

        elif f == "both":
            # Left-right block then duplicated horizontally with up-down flip
            R_lr = np.vstack([-np.flipud(r), r])
            Z_lr = np.vstack([z, z])
            X_lr = np.vstack([np.flipud(x), x])

            R = np.hstack([R_lr, R_lr])
            Z = np.hstack([-np.fliplr(Z_lr), Z_lr])
            X = np.hstack([np.fliplr(X_lr), X_lr])

            artist = _draw(R, Z, X)
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])

        else:
            raise ValueError(f"Unknown flag: {flag}")

    # --- Case D: matrix “flag” → stitch custom right-half
    else:
        flag_arr = _as_array(flag)
        if flag_arr.shape != x.shape:
            raise ValueError(f"When flag is an array, it must match x.shape. Got {flag_arr.shape} vs {x.shape}.")

        R = np.vstack([-np.flipud(r), r])
        Z = np.vstack([z, z])
        X = np.vstack([np.flipud(x), flag_arr])
        artist = _draw(R, Z, X)

    # Cosmetics to emulate MATLAB
    plt.set_cmap(cmap)
    # shading interp is visually similar to smooth colormapping; keep grid hidden
    ax.set_aspect("equal", adjustable="box")

    return artist



def FocusImage2D(
    rho: np.ndarray,
    z: np.ndarray,
    volx: np.ndarray,
    phi: Optional[Union[float, np.ndarray]] = None,
    flag: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "hot",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct two 2D images (fx, ex) from harmonic stacks over azimuth φ and (optionally) plot.

    Parameters
    ----------
    rho, z : 2D arrays (Nrho x Nz)
        Polar grids.
    volx : ndarray
        Harmonic stack. Shape (Nrho, Nz, 2*M+1[, K]) or a 2D image (Nrho, Nz).
        If 2D, it is promoted to (Nrho, Nz, 1, 1).
        The 3rd dim packs [DC, cos1..cosM, sin1..sinM].
    phi : float or 2D array, optional
        Scalar or grid (Nrho x Nz) angle.
    flag : {'log','surf','horizontal','vertical'}, optional
        If provided, also draws a plot.
    ax : matplotlib Axes, optional
        Axis to draw on if plotting.
    cmap : str
        Colormap for plotting.

    Returns
    -------
    fx, ex : 2D ndarrays
        Reconstructed images (same shape as rho,z).
    """
    if phi is None:
        phi = 0.0

    # Promote volx to the canonical 4D shape (Nrho, Nz, 2*M+1, K)
    V = np.asarray(volx)
    if V.ndim == 2:
        V = V[:, :, None]            # (Nrho,Nz,1)
    if V.ndim == 3:
        V = V[..., None]             # (Nrho,Nz,2M+1,1)
    if V.ndim != 4:
        raise ValueError("volx must be (Nrho,Nz,2M+1[,K]) or a 2D (Nrho,Nz) image.")

    Nrho, Nz, K3, K4 = V.shape
    if (K3 % 2) != 1:
        raise ValueError("3rd dim of volx must be odd (2*M+1).")
    M = (K3 - 1) // 2

    # Prepare phi with broadcasting
    phi_arr = np.asarray(phi)
    if phi_arr.ndim == 0:
        # scalar
        cos_m = lambda m: np.cos(m * phi_arr)
        sin_m = lambda m: np.sin(m * phi_arr)
        expand = lambda A: A
    else:
        if phi_arr.shape != (Nrho, Nz):
            raise ValueError("phi must be scalar or have shape (Nrho, Nz).")
        cos_m = lambda m: np.cos(m * phi_arr)
        sin_m = lambda m: np.sin(m * phi_arr)
        expand = lambda A: A[..., np.newaxis, np.newaxis]  # -> (Nrho,Nz,1,1)

    # Reconstruct fx, ex from harmonics
    fx = V[:, :, 0, :].astype(np.complex128)  # DC
    for m in range(1, M + 1):
        fx += expand(cos_m(m)) * V[:, :, m, :] + expand(sin_m(m)) * V[:, :, M + m, :]

    # ex uses (phi + pi) ⇒ cos, sin pick up (-1)^m
    ex = V[:, :, 0, :].astype(np.complex128)
    for m in range(1, M + 1):
        sgn = -1 if (m % 2 == 1) else 1
        ex += sgn * (expand(cos_m(m)) * V[:, :, m, :] + expand(sin_m(m)) * V[:, :, M + m, :])

    # --- Sum across *channel axis* (last axis), robust for both 3D (...,1) and 4D (...,K)
    ch_ax = -1
    if np.iscomplexobj(volx):
        fx = np.sum(np.abs(fx) ** 2, axis=ch_ax)
        ex = np.sum(np.abs(ex) ** 2, axis=ch_ax)
    else:
        fx = np.abs(np.sum(fx, axis=ch_ax))
        ex = np.abs(np.sum(ex, axis=ch_ax))

    # Optional plotting (mirror layouts like MATLAB)
    if flag is not None:
        if ax is None:
            _, ax = plt.subplots()

        def _flipud(A): return np.flipud(A)
        def _fliplr(A): return np.fliplr(A)

        if flag == "log":
            R = np.vstack([-_flipud(rho), rho])
            Z = np.vstack([z, z])
            X = np.vstack([_flipud(ex), fx])
            X = np.log10(np.maximum(X, np.finfo(float).tiny))
            mpcolor(R, Z, X, ax=ax, cmap=cmap)
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])
        elif flag == "surf":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = ax.figure if hasattr(ax, "plot_surface") else plt.figure()
            if not hasattr(ax, "plot_surface"):
                ax = fig.add_subplot(111, projection="3d")
            R = np.vstack([-_flipud(rho), rho])
            Z = np.vstack([z, z])
            X = np.vstack([_flipud(ex), fx])
            ax.plot_surface(R, Z, X, cmap=cmap, linewidth=0, antialiased=True)
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])
        elif flag == "horizontal":
            R = np.vstack([z, z])
            Z = np.vstack([-_flipud(rho), rho])
            X = np.vstack([_flipud(ex), fx])
            mpcolor(R, Z, X, ax=ax, cmap=cmap)
            yt = ax.get_yticks()
            ax.set_yticklabels([f"{abs(v):g}" for v in yt])
        elif flag == "vertical":
            zrow = z[0, :]
            dz = np.mean(np.diff(zrow)) if zrow.size > 1 else 0.0
            z0 = np.min(z) - dz
            R = np.hstack([np.vstack([-_flipud(rho), rho]),
                           np.vstack([-_flipud(rho), rho])])
            Z_left = np.fliplr(np.vstack([z, z]))
            Z = np.hstack([2 * z0 - Z_left, np.vstack([z, z])])
            X_left = np.fliplr(np.vstack([_flipud(ex), fx]))
            X = np.hstack([X_left, np.vstack([_flipud(ex), fx])])
            mpcolor(R, Z, X, ax=ax, cmap=cmap)
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{abs(v):g}" for v in xt])
        else:
            # default stacked view
            R = np.vstack([-_flipud(rho), rho])
            Z = np.vstack([z, z])
            X = np.vstack([_flipud(ex), fx])
            mpcolor(R, Z, X, ax=ax, cmap=cmap)

        plt.set_cmap(cmap)

    return fx, ex


def reconstruct_intensity_over_phi(fxc, fxs, fyc, fys, fzc, fzs, rho, z, nphi=128):
    """
    Build 3D intensity volume feld(r,z,phi) from harmonic fields.
    Returns rr, zz, phi_grid, feld.
    """
    Nr, Nz, Kc = fxc.shape
    M = Kc - 1
    phi_grid = np.linspace(0, 2*np.pi, nphi, endpoint=False)

    # prepare 3D cylindrical grids (r and z repeated along phi)
    r_axis = rho[:, 0]
    z_axis = z[0, :]
    rr, zz = np.meshgrid(r_axis, z_axis, indexing='ij')  # (Nr, Nz)
    rr = np.repeat(rr[..., None], nphi, axis=2)          # (Nr, Nz, nphi)
    zz = np.repeat(zz[..., None], nphi, axis=2)          # (Nr, Nz, nphi)

    # reconstruct E(r,z,phi) from harmonics
    Ex = np.zeros((Nr, Nz, nphi), dtype=complex)
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)

    # DC term
    Ex += fxc[..., 0][..., None]
    Ey += fyc[..., 0][..., None]
    Ez += fzc[..., 0][..., None]

    # j = 1..M terms
    for j in range(1, M+1):
        c = np.cos(j*phi_grid)[None, None, :]
        s = np.sin(j*phi_grid)[None, None, :]

        Ex += fxc[..., j][..., None]*c + fxs[..., j-1][..., None]*s
        Ey += fyc[..., j][..., None]*c + fys[..., j-1][..., None]*s
        Ez += fzc[..., j][..., None]*c + fzs[..., j-1][..., None]*s

    # intensity
    feld = (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2).astype(float)
    return rr, zz, phi_grid, feld



def FocusImage3D(
    rho: np.ndarray,
    z: np.ndarray,
    vol: np.ndarray,
    flag=None,
    tsh=None,
    maxangle: float = 2*np.pi,
    clfflag=None,
    *,
    maxphi: int = 2**7,
    aggregate_for_plot: bool = True,
    ax: plt.Axes = None,
    plot_isosurfaces: bool = True,
    colors=None,
    alphas=None,
):
    """
    Reconstruct feld(r, z, phi) from harmonic volume vol (DC + cos/sin slices)
    and optionally draw 3D isosurfaces on the provided Axes3D.

    Inputs
    ------
    rho, z : 2D grids, shape (Nrho, Nz)
    vol    : (Nrho, Nz, 2*M+1) or (Nrho, Nz, 2*M+1, K)
             3rd dim packs [DC, cos1..cosM, sin1..sinM].
    tsh    : iterable of threshold FRACTIONS (e.g. 1/exp(1:3)).
    ax     : 3D axes to draw on; if None, a new figure+axes is created.
    plot_isosurfaces : if True, compute marching cubes and plot surfaces.

    Returns
    -------
    feld, phi, rr, zz
      feld: (Nrho, Nz, Nphi) if aggregate_for_plot=True (default)
            else (Nrho, Nz, Nphi, K)
      phi, rr, zz: cylindrical 3D grids (same shapes as feld).
    """
    rho = np.asarray(rho)
    z   = np.asarray(z)
    V   = np.asarray(vol)

    # ensure channel axis
    if V.ndim == 3:
        V = V[..., np.newaxis]  # (Nr, Nz, 2M+1, 1)
    Nr, Nz, K3, K = V.shape
    if (K3 % 2) != 1:
        raise ValueError("vol 3rd dimension must be odd (2*M+1).")
    M = (K3 - 1) // 2

    # phi grid and cylindrical mesh
    phi_vals = np.linspace(0.0, float(maxangle), int(maxphi) + 1)
    rr, zz, phi = np.meshgrid(rho[:, 0], z[0, :], phi_vals, indexing="ij")  # (Nr,Nz,Nphi)

    # reconstruct per-channel harmonic series over phi
    # feld_full: (Nr, Nz, Nphi, K)
    feld_full = np.empty((Nr, Nz, phi_vals.size, K), dtype=V.dtype)
    feld_full[:] = V[:, :, 0, :][:, :, None, :]  # DC term
    for j in range(1, M + 1):
        c = np.cos(j * phi)[..., None]  # (Nr,Nz,Nphi,1)
        s = np.sin(j * phi)[..., None]
        feld_full += c * V[:, :, j, :][:, :, None, :] + s * V[:, :, M + j, :][:, :, None, :]

    # aggregate across channels if requested (like MATLAB "no output" case)
    if aggregate_for_plot:
        if np.iscomplexobj(V):
            feld = np.sum(np.abs(feld_full)**2, axis=3)
        else:
            feld = np.sum(feld_full, axis=3)
    else:
        feld = feld_full

    # plotting
    if plot_isosurfaces:
        try:
            from skimage.measure import marching_cubes
        except Exception as e:
            raise ImportError(
                "FocusImage3D isosurface plotting needs scikit-image "
                "(install with `pip install scikit-image`)."
            ) from e

        # choose thresholds if not provided
        if tsh is None:
            tsh = 1.0 / np.exp(np.array([1, 2, 3], float))
        tsh = np.asarray(tsh, float)

        # default colors/alphas
        if colors is None:
            colors = [(1.0, 0.8, 0.5), (1.0, 0.6, 0.3), (1.0, 0.4, 0.2)]
        if alphas is None:
            alphas = [0.5, 0.35, 0.25]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Normalize feld to [0,1] for levels
        f = np.asarray(feld, float)
        fmax = f.max() if f.size else 1.0
        if fmax <= 0:
            fmax = 1.0
        f_norm = f / fmax

        Nr, Nz, Np = f_norm.shape
        r_axis = rho[:, 0]
        z_axis = z[0, :]
        p_axis = phi_vals

        # Marching cubes works on a regular voxel grid. Our axes are (r,z,phi).
        # We'll pass data as (Nr, Nz, Np) and then map verts -> (x,y,z) via r,phi.
        for level, col, alpha in zip(tsh, colors, alphas):
            try:
                verts, faces, _, _ = marching_cubes(f_norm, level=level, allow_degenerate=False)
            except ValueError:
                # level outside data range — skip
                continue

            # verts columns are indices along (r, z, phi) axes, in voxel coordinates
            r_idx = verts[:, 0]
            z_idx = verts[:, 1]
            p_idx = verts[:, 2]

            # map index space -> actual coordinates
            r = np.interp(r_idx, np.arange(Nr), r_axis)
            zc = np.interp(z_idx, np.arange(Nz), z_axis)
            p = np.interp(p_idx, np.arange(Np), p_axis)

            x = r * np.cos(p)
            y = r * np.sin(p)

            ax.plot_trisurf(x, y, zc, triangles=faces, linewidth=0, antialiased=True, color=col, alpha=alpha)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('x [μm]'); ax.set_ylabel('y [μm]'); ax.set_zlabel('z [μm]')
        ax.view_init(elev=20.0, azim=35.0)

    return feld, phi, rr, zz


def _plot_slice_fallback(feld, rr, zz, phi):
    phi_idx = 0
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(zz[..., phi_idx], rr[..., phi_idx], np.asarray(feld[..., phi_idx]), shading='auto')
    plt.xlabel('z (μm)'); plt.ylabel('ρ (μm)')
    plt.title(f'φ = {float(phi[0,0,phi_idx]):.2f} rad (install scikit-image for 3D iso)')
    plt.colorbar(label='Intensity (a.u.)')
    plt.tight_layout(); plt.show()


def _set_axes_equal_3d(ax):
    """Equal aspect for 3D axes."""
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xmid = 0.5*(xlim[0]+xlim[1]); ymid = 0.5*(ylim[0]+ylim[1]); zmid = 0.5*(zlim[0]+zlim[1])
    radius = 0.5*max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    ax.set_xlim3d(xmid-radius, xmid+radius)
    ax.set_ylim3d(ymid-radius, ymid+radius)
    ax.set_zlim3d(zmid-radius, zmid+radius)


#%%
# ============================================================================
# FLIM ANALYSIS UTILITY FUNCTIONS (extracted from FlavMetaFLIM.py)
# ============================================================================

def vignette_correction(int_im, blur_factor=6, lower_percentile=2, upper_percentile=98):
    """
    Apply vignette correction to intensity image following MATLAB algorithm.
    
    This function corrects for illumination non-uniformity by:
    1. Creating a blurred background estimate using Gaussian filter
    2. Normalizing the original image by this background
    3. Applying percentile-based contrast enhancement
    4. Final normalization to [0,1] range
    
    MATLAB equivalent:
    bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = bblur ./ max(bblur(:));
    I2 = int_im ./ bblur;
    lb = prctile(I2(:), 2);
    ub = prctile(I2(:), 98);
    I2(I2 < lb) = lb;
    I2(I2 > ub) = ub;
    I2 = I2 - lb;
    int2 = I2 ./ max(I2(:));
    
    Parameters
    ----------
    int_im : ndarray
        Input intensity image
    blur_factor : float, optional (default=6)
        Gaussian blur factor = min(image_size) / blur_factor
    lower_percentile : float, optional (default=2)
        Lower percentile for contrast adjustment (removes dark outliers)
    upper_percentile : float, optional (default=98)
        Upper percentile for contrast adjustment (removes bright outliers)
        
    Returns
    -------
    int2 : ndarray
        Vignette-corrected and normalized intensity image [0,1]
    bblur : ndarray
        Gaussian-blurred background estimate
    """
    
    print("Applying vignette correction...")
    print(f"  - Image shape: {int_im.shape}")
    print(f"  - Blur factor: 1/{blur_factor} of min image dimension")
    
    # Calculate Gaussian blur sigma: min(image_size) / blur_factor
    sigma = min(int_im.shape) / blur_factor
    print(f"  - Gaussian sigma: {sigma:.2f} pixels")
    
    # Create blurred background (excitation floor estimate)
    # MATLAB: bblur = imgaussfilt(int_im, min(size(int_im))/6);
    bblur = gaussian_filter(int_im, sigma=sigma)
    
    # Normalize blurred background to [0,1] with max=1
    # MATLAB: bblur = bblur ./ max(bblur(:));
    bblur = bblur / np.max(bblur)
    
    # Vignette correction: divide by normalized background
    # MATLAB: I2 = int_im ./ bblur;
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    I2 = int_im / (bblur + epsilon)
    
    # Calculate percentile bounds for contrast enhancement
    # MATLAB: lb = prctile(I2(:), 2); ub = prctile(I2(:), 98);
    lb = np.percentile(I2, lower_percentile)  # Lower bound (2nd percentile)
    ub = np.percentile(I2, upper_percentile)  # Upper bound (98th percentile)
    
    print(f"  - Intensity range before clipping: [{np.min(I2):.3f}, {np.max(I2):.3f}]")
    print(f"  - Percentile bounds: [{lb:.3f}, {ub:.3f}]")
    
    # Clip values to percentile bounds
    # MATLAB: I2(I2 < lb) = lb; I2(I2 > ub) = ub;
    I2 = np.clip(I2, lb, ub)
    
    # Subtract lower bound (baseline correction)
    # MATLAB: I2 = I2 - lb;
    I2 = I2 - lb
    
    # Final normalization to [0,1]
    # MATLAB: int2 = I2 ./ max(I2(:));
    int2 = I2 / np.max(I2) if np.max(I2) > 0 else I2
    
    print(f"  - Final intensity range: [{np.min(int2):.3f}, {np.max(int2):.3f}]")
    
    return int2, bblur

def subplot_cim(x, brightness=None, color_range=None, colormap=None, ax=None):
    """
    Create a cim-style display within a matplotlib subplot.
    
    This function replicates the cim functionality but works within subplots,
    creating an RGB image with color determined by x and brightness overlay.
    
    Parameters
    ----------
    x : ndarray
        Color data (2D array)
    brightness : ndarray, optional
        Brightness overlay data (same shape as x)
    color_range : tuple, optional
        (min, max) range for color scaling
    colormap : ndarray, optional
        Custom colormap (64×3 RGB values)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes
        
    Returns
    -------
    handle : matplotlib image handle
        Handle to the displayed image
    """
    
    if ax is None:
        ax = plt.gca()
    
    x = np.asarray(x, dtype=float)
    
    # Simple display if no brightness overlay
    if brightness is None:
        if color_range is not None:
            handle = ax.imshow(x, cmap='viridis', vmin=color_range[0], vmax=color_range[1])
        else:
            handle = ax.imshow(x, cmap='viridis')
        ax.axis('off')
        return handle
    
    # Create brightness overlay
    brightness = np.asarray(brightness, dtype=float)
    
    # Normalize brightness to [0, 1]
    valid_brightness = np.isfinite(brightness)
    if np.any(valid_brightness):
        b_min = np.min(brightness[valid_brightness])
        b_max = np.max(brightness[valid_brightness])
        if b_max > b_min:
            brightness = (brightness - b_min) / (b_max - b_min)
        else:
            brightness = np.zeros_like(brightness)
    brightness = np.clip(brightness, 0, 1)
    
    # Handle color scaling
    if color_range is not None:
        x_min, x_max = color_range
    else:
        valid_x = np.isfinite(x) & valid_brightness
        if np.any(valid_x):
            x_min = np.min(x[valid_x])
            x_max = np.max(x[valid_x])
        else:
            x_min, x_max = 0, 1
    
    # Scale x to colormap indices [0, 1]
    if x_max > x_min:
        x_scaled = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    else:
        x_scaled = np.full_like(x, 0.5)
    
    # Set up colormap
    if colormap is None:
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, 256))[:, :3]
    else:
        colors = np.asarray(colormap)
        if colors.shape[0] != 256:
            # Interpolate to 256 colors for better resolution
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, colors.shape[0])
            new_indices = np.linspace(0, 1, 256)
            colors_interp = []
            for i in range(3):
                f = interp1d(old_indices, colors[:, i], kind='linear')
                colors_interp.append(f(new_indices))
            colors = np.column_stack(colors_interp)
    
    # Create RGB image
    h, w = x.shape
    rgb_image = np.zeros((h, w, 3))
    
    # Handle NaN values
    x_scaled = np.nan_to_num(x_scaled, nan=0)
    brightness = np.nan_to_num(brightness, nan=0)
    
    # Map to colormap and apply brightness
    color_indices = (x_scaled * (colors.shape[0] - 1)).astype(int)
    color_indices = np.clip(color_indices, 0, colors.shape[0] - 1)
    
    for channel in range(3):
        rgb_image[:, :, channel] = colors[color_indices, channel] * brightness
    
    # Display the image
    handle = ax.imshow(rgb_image)
    ax.axis('off')
    
    # Add colorbar as scale bar
    if color_range is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Create a colorbar using the colormap
        if colormap is None:
            cmap = plt.cm.viridis
        else:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colormap)
        
        # Create a scalar mappable for the colorbar
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=x_min, vmax=x_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)
    
    return handle

def display_amplitude_maps_subplot(Amp, int2, taufit, flag_win=True, colormap=None, save_path=None):
    """
    Display all amplitude maps in a single figure with subplots using cim-style visualization.
    
    Parameters
    ----------
    Amp : ndarray
        Amplitude data (nx, ny, n_components)
    int2 : ndarray
        Vignette-corrected intensity for brightness overlay
    taufit : array-like
        Fitted lifetime values
    flag_win : bool, optional
        Flag indicating windowed vs pixel-wise analysis
    colormap : ndarray, optional
        Custom colormap (64×3 RGB values)
    save_path : str, optional
        Path to save the figure
    """
    
    print(f"Creating amplitude subplot display for {len(taufit)} lifetime components...")
    
    # Create custom colormap if not provided
    if colormap is None:
        import matplotlib.cm as cm
        colormap = cm.viridis(np.linspace(0, 1, 64))[:, :3]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Display background/offset amplitude (component 0)
    amp_range = [np.min(Amp[:, :, 0]), np.max(Amp[:, :, 0])]
    subplot_cim(Amp[:, :, 0], int2**0.5, amp_range, colormap, ax=axes[0])
    axes[0].set_title('Background/Offset', fontsize=12, pad=10)
    
    # Display amplitudes for each lifetime component
    for i, tau in enumerate(taufit):
        if i + 1 < len(axes):
            amp_data = Amp[:, :, i + 1]
            amp_range = [np.min(amp_data), np.max(amp_data)]
            subplot_cim(amp_data, int2**0.5, amp_range, colormap, ax=axes[i + 1])
            axes[i + 1].set_title(f'Component {i+1}: τ = {tau:.2f} ns', fontsize=12, pad=10)
    
    # Hide unused subplot if only 2 lifetimes
    if len(taufit) < 3:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'FLIM Amplitude Maps - {"Windowed" if flag_win else "Pixel-wise"} Analysis',
                 fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Amplitude maps saved to: {save_path}")
    
    plt.show()
    
    print(f" All {len(taufit)+1} amplitude components displayed in single subplot figure")

#%%
# ============================================================================
# CORE FLIM ANALYSIS FUNCTIONS (extracted from FlavMetaFLIM.py)
# ============================================================================

def load_or_process_ptu_data(filename: str, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Load PTU data from cache or process raw PTU file.
    
    Parameters
    ----------
    filename : str
        Path to PTU file
    force_reprocess : bool, optional
        Force reprocessing even if cache exists
        
    Returns
    -------
    dict
        Dictionary containing FLIM data arrays and header
    """
    
    res_file = filename[:-4] + '_FLIM_data.pkl'
    
    if os.path.exists(res_file) and not force_reprocess:
        print(f"Loading cached FLIM data from: {res_file}")
        with open(res_file, 'rb') as f:
            FLIM_data = pickle.load(f)
        
        # Extract data
        data = {
            'im_sync': FLIM_data['im_sync'],
            'im_tcspc': FLIM_data['im_tcspc'],
            'im_chan': FLIM_data['im_chan'],
            'im_line': FLIM_data['im_line'],
            'im_frame': FLIM_data['im_frame'],
            'im_col': FLIM_data['im_col'],
            'head': FLIM_data['head']
        }
        del FLIM_data
    else:
        print(f"Processing raw PTU file: {filename}")
        head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead(filename)
        
        data = {
            'im_sync': im_sync,
            'im_tcspc': im_tcspc,
            'im_chan': im_chan,
            'im_line': im_line,
            'im_frame': im_frame,
            'im_col': im_col,
            'head': head
        }
        del head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame
    return data

def analyze_flim_data(data: Dict[str, Any],
                     auto_det: int = 0,
                     auto_PIE: int = 1,
                     flag_win: bool = True,
                     resolution: float = 0.2,
                     tau0: np.ndarray = np.array([0.3, 1.7, 6.0]),
                     win_size: int = 8,
                     step: int = 2,
                     IRF_data = None) -> Dict[str, Any]:
    """
    Perform FLIM analysis on PTU data.
    
    Parameters
    ----------
    data : dict
        PTU data dictionary from load_or_process_ptu_data
    auto_det : int, optional
        Detector ID for autofluorescence channel detection
    auto_PIE : int, optional
        Laser pulse number for autofluorescence channel (PIE window)
    flag_win : bool, optional
        Flag for window based lifetime estimation
    resolution : float, optional
        Temporal resolution for TCSPC histogram binning (ns)
    tau0 : ndarray, optional
        Initial lifetime guesses
    win_size : int, optional
        Size of the sliding window (pixels)
    step : int, optional
        Step size for window movement
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    
    # Extract data
    # CRITICAL FIX: Use appropriate integer data types to save memory and ensure correct data handling
    im_sync = np.array(data['im_sync'], dtype=np.uint32)
    im_tcspc = np.array(data['im_tcspc'], dtype=np.uint16)
    im_chan = np.array(data['im_chan'], dtype=np.uint8)
    im_line = np.array(data['im_line'], dtype=np.uint16)
    im_col = np.array(data['im_col'], dtype=np.uint16)
    head = data['head']
    
    del data
    # PIE configuration
    cnum = 1
    if 'PIENumPIEWindows' in head:
        cnum = head['PIENumPIEWindows']
        print(f"PIE cycles detected: {cnum}")
    
    # Process frame
    print("Processing FLIM data...")
    tag, tau, tcspc_pix = Process_Frame(
        im_sync, im_col, im_line, im_chan,
        im_tcspc, head, cnum=cnum, resolution=resolution
    )
    
    nx, ny, ch, p = np.shape(tag)
    
    # Calculate parameters
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    #SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] *
                  (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'])
    
    # Extract channel data
    idx = im_chan == auto_det
    if np.sum(idx) > 0:
        tcspc_im = mHist((im_tcspc[idx] / chDiv).astype(np.int64) -
                        int((auto_PIE-1)*tmpCh/cnum/chDiv),
                        np.arange(Ngate))[0]
    else:
        print(f"Warning: No photons found for channel {auto_det} - skipping FLIM analysis")
        return {}
    
    # Calculate IRF and perform fitting
    if type(IRF_data) is dict:
        # Loading IRF provided as input
        tcspcIRF = np.array(IRF_data['tcspcIRF'], dtype=np.float64)
        # TODO: Check resolution of IRF file before accepting it
    else:
        print("Calculating IRF and performing lifetime fitting...")
        tcspcIRF = Calc_mIRF(head, tcspc_im[np.newaxis, :, np.newaxis])
        tmpi = np.where((tcspcIRF/np.max(tcspcIRF)) < (10**-4))[1]
        tcspcIRF[:, tmpi, :] = 0
        
    tmp_tau = tau0.copy()
    taufit, A, _, zfit, patterns, _, _, _, _ = FluoFit(
        np.squeeze(tcspcIRF), tcspc_im,
        np.floor(head['MeasDesc_GlobalResolution']*10**9/cnum + 0.5),
        resolution, tmp_tau, flag_ml=False
    )
    
    # Sort lifetime values and reorder corresponding patterns and amplitudes
    sort_idx = np.argsort(taufit)
    taufit = taufit[sort_idx]
    A = A[sort_idx]
    patterns = patterns[:, 1:]  # Remove background pattern (first column)
    patterns = patterns[:, sort_idx]  # Reorder patterns
    patterns = np.column_stack([np.ones(patterns.shape[0]), patterns])  # Add background back as first column
    patterns = patterns / np.sum(patterns, axis=0)  # normalized patterns
    
    print(f"Sorted lifetime values: {taufit}")
    
    # Pattern matching analysis
    if flag_win:
        print(f"Processing with sliding window: {win_size}x{win_size}, step={step}")
        
        # Calculate output dimensions
        n_win_x = (nx - win_size) // step + 1
        n_win_y = (ny - win_size) // step + 1
        
        print(f"Original image: {nx}x{ny}, Windows: {n_win_x}x{n_win_y}")
        
        # Initialize arrays
        tcspc_win = np.zeros((n_win_x, n_win_y, tcspc_pix.shape[2]))
        int_im = np.zeros((n_win_x, n_win_y))
        
        print("Aggregating TCSPC data with sliding windows...")
        for i in range(n_win_x):
            for j in range(n_win_y):
                x_start = i * step
                x_end = x_start + win_size
                y_start = j * step
                y_end = y_start + win_size
                
                tcspc_win[i, j, :] = np.sum(tcspc_pix[x_start:x_end, y_start:y_end, :, 0], axis=(0, 1))
                int_im[i, j] = np.sum(tcspc_win[i, j, :])
        
        print("Running pattern matching on windowed data...")
        Amp, Z = PatternMatchIm(tcspc_win, patterns, mode='PIRLS')
        print(f"Windowed analysis completed: {n_win_x}x{n_win_y} windows processed")
        
    else:
        # Pixel by pixel analysis
        print("Running pixel-by-pixel pattern matching...")
        Amp, Z = PatternMatchIm(tcspc_pix[:, :, :, 0], patterns, mode='PIRLS')
        int_im = np.sum(tcspc_pix[:, :, :, 0], axis=2)
    
    # Normalize amplitudes
    Amp = Amp / np.sum(Amp[:, :, 1:], axis=2, keepdims=True)
    
    # Apply vignette correction
    print("Applying vignette correction...")
    int2, bblur = vignette_correction(int_im)
    
    # Create average lifetime image
    tau_avg_int = np.zeros_like(int_im)
    for i, tau_val in enumerate(taufit):
        tau_avg_int += Amp[:, :, i + 1] * tau_val
    
    #print(np.sum(Amp[:, :, 1:],axis=2))
    #intensity_safe = np.where(int2 > 0, int2, 1)
    tau_avg_int = np.where(int2 > 0, tau_avg_int / np.sum(Amp[:, :, 1:],axis=2), 0) # this is intensity averaged
    
    
    # 2. Rate-averaged lifetime
    rate_const = 1.0/taufit
    tau_avg_rate = np.zeros_like(int_im)
    for i, rate in enumerate(rate_const):
        tau_avg_rate += Amp[:, :, i + 1] * rate
    
    # Handle division by zero and invalid values in rate averaging
    tau_avg_rate = np.where(tau_avg_rate > 0, np.sum(Amp[:, :, 1:],axis=2)/tau_avg_rate, 0)
    
    # Print statistics
    print("\nLifetime Statistics:")
    valid_int = tau_avg_int > 0
    valid_rate = tau_avg_rate > 0
    print(f"Intensity-averaged lifetime: min={np.min(tau_avg_int[valid_int]):.2f}ns, max={np.max(tau_avg_int[valid_int]):.2f}ns, mean={np.mean(tau_avg_int[valid_int]):.2f}ns")
    print(f"Rate-averaged lifetime: min={np.min(tau_avg_rate[valid_rate]):.2f}ns, max={np.max(tau_avg_rate[valid_rate]):.2f}ns, mean={np.mean(tau_avg_rate[valid_rate]):.2f}ns")
    
    print("\nAmplitude Statistics:")
    print(f"Background: min={np.min(Amp[:,:,0]):.3f}, max={np.max(Amp[:,:,0]):.3f}, mean={np.mean(Amp[:,:,0]):.3f}")
    for i, tau in enumerate(taufit):
        amp_comp = Amp[:, :, i + 1]
        print(f"Component {i+1} (τ={tau:.2f}ns): min={np.min(amp_comp):.3f}, max={np.max(amp_comp):.3f}, mean={np.mean(amp_comp):.3f}")
    
    return {
        'tcspcIRF': tcspcIRF,
        'taufit': taufit,
        'patterns': patterns,
        'Amp': Amp,
        'int2': int2,
        'int_im': int_im,
        'tau_avg_int': tau_avg_int,
        'tau_avg_rate': tau_avg_rate,
        'flag_win': flag_win,
        'head': head
    }

def save_flim_results(results: Dict[str, Any], output_dir: str, filename_base: str):
    """
    Save FLIM analysis results in multiple formats.
    
    Parameters
    ----------
    results : dict
        Results dictionary from analyze_flim_data
    output_dir : str
        Output directory path
    filename_base : str
        Base filename for output files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Save as pickle
    pkl_file = output_path / f"{filename_base}_FLIM_results.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f" Pickle file saved: {pkl_file}")
    
    # Save lifetime values as CSV
    csv_file = output_path / f"{filename_base}_lifetimes.csv"
    lifetime_data = {
        'Component': [f'τ{i+1}' for i in range(len(results['taufit']))],
        'Lifetime_ns': results['taufit']
    }
    df_lifetimes = pd.DataFrame(lifetime_data)
    df_lifetimes.to_csv(csv_file, index=False)
    print(f" Lifetime CSV saved: {csv_file}")
    
    # Save amplitude statistics as CSV
    stats_file = output_path / f"{filename_base}_amplitude_stats.csv"
    Amp = results['Amp']
    stats_data = []
    
    # Background statistics
    stats_data.append({
        'Component': 'Background',
        'Min': np.min(Amp[:, :, 0]),
        'Max': np.max(Amp[:, :, 0]),
        'Mean': np.mean(Amp[:, :, 0]),
        'Std': np.std(Amp[:, :, 0])
    })
    
    # Lifetime component statistics
    for i, tau in enumerate(results['taufit']):
        amp_comp = Amp[:, :, i + 1]
        stats_data.append({
            'Component': f'τ{i+1}_{tau:.2f}ns',
            'Min': np.min(amp_comp),
            'Max': np.max(amp_comp),
            'Mean': np.mean(amp_comp),
            'Std': np.std(amp_comp)
        })
    
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(stats_file, index=False)
    print(f" Amplitude statistics CSV saved: {stats_file}")
    
    # Save images as TIFF
    # Intensity image
    int_tiff = output_path / f"{filename_base}_intensity.tif"
    imwrite(int_tiff, results['int2'].astype(np.float32))
    print(f"Intensity TIFF saved: {int_tiff}")
    
    # Average lifetime images
    tau_int_tiff = output_path / f"{filename_base}_intensity_averaged_lifetime.tif"
    imwrite(tau_int_tiff, results['tau_avg_int'].astype(np.float32))
    print(f"Intensity-averaged lifetime TIFF saved: {tau_int_tiff}")
    
    tau_rate_tiff = output_path / f"{filename_base}_rate_averaged_lifetime.tif"
    imwrite(tau_rate_tiff, results['tau_avg_rate'].astype(np.float32))
    print(f"Rate-averaged lifetime TIFF saved: {tau_rate_tiff}")
    
    # Individual amplitude component TIFFs
    for i in range(results['Amp'].shape[2]):
        if i == 0:
            comp_name = "background"
        else:
            comp_name = f"component_{i}_tau_{results['taufit'][i-1]:.2f}ns"
        
        amp_tiff = output_path / f"{filename_base}_amplitude_{comp_name}.tif"
        imwrite(amp_tiff, results['Amp'][:, :, i].astype(np.float32))
        print(f"Amplitude TIFF saved: {amp_tiff}")
    
    # Display and save CIM lifetime images
    print("\nDisplaying FLIM images using subplot_cim function with vignette-corrected intensity...")
    
    # Create custom colormap (similar to MATLAB's jet)
    custom_jet_cmap = plt.cm.jet(np.linspace(0, 1, 64))[:, :3]
    
    # Create figure with subplots for both lifetime images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Intensity-averaged lifetime
    subplot_cim(results['tau_avg_int'], results['int2']**0.75, [1.5, 4], custom_jet_cmap, ax=ax1)
    ax1.set_title('Intensity-Averaged Lifetime', fontsize=12)
    
    # Rate-averaged lifetime
    subplot_cim(results['tau_avg_rate'], results['int2']**0.75, [1.5, 4], custom_jet_cmap, ax=ax2)
    ax2.set_title('Rate-Averaged Lifetime', fontsize=12)
    
    plt.suptitle('FLIM Images: Lifetime with Vignette-Corrected Intensity Overlay', fontsize=14)
    plt.tight_layout()
    
    # Save FLIM image
    flim_file = output_path / f"{filename_base}_FLIM_lifetime_comparison.png"
    plt.savefig(flim_file, dpi=300, bbox_inches='tight')
    print(f"FLIM lifetime comparison image saved: {flim_file}")
    plt.show()
    
    # Display and save amplitude visualization with cim-style subplots
    print("\nDisplaying amplitude maps...")
    display_amplitude_maps_subplot(
        results['Amp'], results['int2'], results['taufit'],
        results['flag_win'], custom_jet_cmap, save_path=None
    )

def process_single_file(ptu_file: str, **analysis_params) -> Dict[str, Any]:
    """
    Process a single PTU file with FLIM analysis.
    Results are saved in the same directory as the PTU file.
    
    Parameters
    ----------
    ptu_file : str
        Path to PTU file
    **analysis_params
        Additional parameters for analyze_flim_data
        
    Returns
    -------
    dict
        Analysis results
    """
    
    print(f"\n{'='*60}")
    print(f"Processing: {ptu_file}")
    print(f"{'='*60}")
    
    try:
        # Load/process data
        data = load_or_process_ptu_data(ptu_file)
        
        # Analyze
        results = analyze_flim_data(data, **analysis_params)
        
        if not results:
            print(f" Analysis failed for {ptu_file}")
            return {}
        
        # Save results in the same directory as the PTU file
        ptu_path = Path(ptu_file)
        output_dir = ptu_path.parent  # Same directory as PTU file
        filename_base = ptu_path.stem
        save_flim_results(results, str(output_dir), filename_base)
        
        print(f" Successfully processed: {ptu_file}")
        return results
        
    except Exception as e:
        print(f" Error processing {ptu_file}: {str(e)}")
        return {}

def process_multiple_folders(folder_paths: List[str], **analysis_params) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple folders containing PTU files.
    Results are saved in the same directories as the PTU files.
    
    Parameters
    ----------
    folder_paths : list of str
        List of folder paths containing PTU files
    **analysis_params
        Additional parameters for analyze_flim_data
        
    Returns
    -------
    dict
        Dictionary with folder names as keys and analysis results as values
    """
    
    all_results = {}
    
    for folder_path in folder_paths:
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        
        print(f"\n{'='*80}")
        print(f"Processing folder: {folder_path}")
        print(f"{'='*80}")
        
        # Find all PTU files in folder
        ptu_files = list(folder_path.glob("*.ptu"))
        
        if not ptu_files:
            print(f" No PTU files found in {folder_path}")
            continue
        
        print(f"Found {len(ptu_files)} PTU files")
        
        folder_results = {}
        
        for ptu_file in ptu_files:
            # Process file and save results next to the PTU file
            file_results = process_single_file(str(ptu_file), **analysis_params)
            if file_results:
                folder_results[ptu_file.stem] = file_results
        
        all_results[folder_name] = folder_results
        
        print(f"\n Completed folder: {folder_name} ({len(folder_results)}/{len(ptu_files)} files processed)")
    
    return all_results
