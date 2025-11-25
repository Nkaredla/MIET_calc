# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:28:38 2025

@author: narai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j0, j1, jn
from typing import Optional, Tuple, Dict
import warnings

#%%
# ---------- Fresnel + coverslip ATF ----------
def _snell_theta(theta, n1, n2):
    s = (n1 / n2) * np.sin(theta)
    s = np.clip(s, -1.0, 1.0)
    return np.arcsin(s)

def _fresnel_ts_tp(theta1, n1, n2):
    """Transmission amplitudes from medium n1->n2, incident angle theta1 (in n1)."""
    # s- and p- polarization
    theta2 = _snell_theta(theta1, n1, n2)
    cos1 = np.cos(theta1)
    cos2 = np.cos(theta2)
    # avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ts = 2*n1*cos1 / (n1*cos1 + n2*cos2)
        tp = 2*n1*cos1 / (n2*cos1 + n1*cos2)
    ts = np.nan_to_num(ts)
    tp = np.nan_to_num(tp)
    return ts, tp, theta2

def _aberration_phase(theta_s, n_s, n_cs, dt, k):
    """OPL mismatch phase for thickness mismatch dt (same length units as lambda)."""
    if dt == 0 or n_cs is None:
        return 0.0*theta_s
    theta_cs = _snell_theta(theta_s, n_s, n_cs)
    opl = n_s*dt/np.cos(theta_s) - n_cs*dt/np.cos(theta_cs)
    return k * opl

# ---------- Vectorial Hankel integrals (Richards–Wolf, cylindrical) ----------
def excitation_psf_hankel(
    r_um: np.ndarray, z_um: np.ndarray,
    NA: float, n_medium: float, wavelength_um: float,
    pupil: str = "aplanatic",           # "aplanatic" or "gaussian:<w0_over_aperture>"
    overfill: float = 0.0,              # unused unless you craft a custom pupil
    atf: Optional[Tuple[float, float]] = None,   # (n_coverslip, delta_t_um) or None
    n_capside: Optional[float] = None,  # side of interface for Fresnel (defaults to n_medium)
    n_other: Optional[float] = None,    # the other side index (defaults to n_coverslip)
    n_for_fresnel: str = "focus"        # "focus" uses Fresnel at the focusing interface
) -> np.ndarray:
    """
    Returns I_exc(r,z) on a (Nr x Nz) mesh via Hankel integrals with J0,J1,J2 (x-pol).
    - If atf is None: no Fresnel / aberration.
    - If atf=(n_cs, dt): apply amplitude factor 0.5*(|t_s|+|t_p|) and aberration phase from thickness mismatch dt.
    """
    r = np.atleast_1d(r_um).astype(float)
    z = np.atleast_1d(z_um).astype(float)
    Nr, Nz = r.size, z.size

    n = float(n_medium)
    k = 2*np.pi*n / float(wavelength_um)
    alpha = np.arcsin(min(NA/n, 0.999999))  # cap

    # theta sampling (Gauss-Legendre is nice; uniform is OK if dense)
    n_th = 800
    th = np.linspace(0.0, alpha, n_th)
    dth = th[1]-th[0]

    # pupil apodization
    if pupil == "aplanatic":
        A = np.sqrt(np.cos(th))
    elif pupil.startswith("gaussian"):
        # gaussian in sin(theta) with width parameter after colon, e.g., "gaussian:0.8"
        try:
            w = float(pupil.split(":")[1])
        except Exception:
            w = 0.8
        A = np.exp(-(np.sin(th)/np.sin(alpha))**2 / (2*w*w))
    else:
        A = np.ones_like(th)

    # ATF amplitude + aberration
    if atf is None:
        amp_T = np.ones_like(th)
        phi_ab = 0.0*th
    else:
        n_cs, dt = float(atf[0]), float(atf[1])
        n1 = n if (n_for_fresnel=="focus") else (n_capside or n)
        n2 = n_cs if (n_for_fresnel=="focus") else (n_other or n_cs)
        ts, tp, _ = _fresnel_ts_tp(th, n1, n2)
        amp_T = 0.5*(np.abs(ts) + np.abs(tp))
        phi_ab = _aberration_phase(th, n, n_cs, dt, k)

    W = A * amp_T

    # Precompute sin/cos and the radial Bessel arguments on an (Nr x n_th) grid
    sth, cth = np.sin(th), np.cos(th)
    kr_sin = (k * np.outer(r, sth))    # shape (Nr, n_th)
    J0 = j0(kr_sin)
    J1 = j1(kr_sin)
    # J2 from scipy.special.jn
    J2 = jn(2, kr_sin)

    # Axial phase per z
    # shape (n_th, Nz)
    phase_z = np.exp(1j*(np.outer(cth, k*z) + phi_ab[:, None]))

    # Build the three integrals
    # I0:  W * cos(th) * J0 * phase * sin(th) dth   (broadcast to Nr x Nz)
    w0 = (W * cth * sth)[None, :, None]      # (1, n_th, 1)
    w2 = (W * cth * sth)[None, :, None]      # same weight for J2 branch
    w1 = (W * sth * sth)[None, :, None]

    # expand Bessel (Nr, n_th) -> (Nr, n_th, 1) to matmul with phase (n_th, Nz)
    I0 = (J0[:, :, None] * w0 * phase_z[None, :, :]).sum(axis=1) * dth
    I2 = (J2[:, :, None] * w2 * phase_z[None, :, :]).sum(axis=1) * dth
    I1 = (J1[:, :, None] * w1 * phase_z[None, :, :]).sum(axis=1) * dth

    I_exc = (np.abs(I0)**2 + 0.5*np.abs(I2)**2 + np.abs(I1)**2).real
    return I_exc  # (Nr x Nz)

# ---------- Confocal MDF builder ----------
def confocal_mdf(
    r_um: np.ndarray, z_um: np.ndarray,
    NA: float, n_medium: float,
    lam_exc_um: float, lam_em_um: float,
    pupil_exc: str = "aplanatic", pupil_det: str = "aplanatic",
    atf_exc: Optional[Tuple[float, float]] = None,
    atf_det: Optional[Tuple[float, float]] = None,
    pinhole_um_image: float = 0.0,      # radius in the IMAGE plane (um)
    magnification: float = 60.0
) -> Dict[str, np.ndarray]:
    """
    Returns dict with 'Exc', 'Det', 'MDF' on (Nr x Nz) grids.
    If pinhole_um_image==0 => infinitesimal pinhole: MDF = Exc*Det.
    If >0 => FFT-based convolution of Det with a disk of radius pinhole_um_image, then MDF = Exc * Det_conv.
    """
    r = np.atleast_1d(r_um)
    z = np.atleast_1d(z_um)

    Exc = excitation_psf_hankel(r, z, NA, n_medium, lam_exc_um, pupil_exc, atf=atf_exc)
    Det = excitation_psf_hankel(r, z, NA, n_medium, lam_em_um, pupil_det, atf=atf_det)

    if pinhole_um_image <= 0:
        MDF = Exc * Det
        return {"r": r, "z": z, "Exc": Exc, "Det": Det, "MDF": MDF}

    # finite pinhole: do a small 2D FFT convolution on a Cartesian patch
    # Build tiny Cartesian grid (covers ~ max r) and revolve the radial Det at each z
    try:
        import numpy.fft as fft

        Rmax = r[-1]
        Nx = 256
        L = float(Rmax)
        x = np.linspace(-L, L, Nx)
        X, Y = np.meshgrid(x, x, indexing='xy')
        Rxy = np.hypot(X, Y)

        # detection PSF on 2D by radial lookup
        Det2D = np.empty((Nx, Nx, z.size), dtype=float)
        for k, zk in enumerate(z):
            # 1D radial profile at this z
            dprof = Det[:, k]
            # interpolate to Rxy
            Det2D[..., k] = np.interp(Rxy, r, dprof, left=dprof[0], right=dprof[-1])

        # build image-plane disk kernel (radius a_v), map to object space radius = a_v / M
        r_disk_obj = pinhole_um_image / float(magnification)
        disk = (Rxy <= r_disk_obj).astype(float)
        # normalize kernel area to 1 so energy is preserved (optional)
        disk /= disk.sum()

        # convolve Det2D(:,:,z) with disk via FFT
        Det2D_conv = np.empty_like(Det2D)
        H = fft.fftn(disk)
        for k in range(z.size):
            Det2D_conv[..., k] = fft.ifftn(fft.fftn(Det2D[..., k]) * H).real

        # sample Det_conv back on the radial axis (azimuthal avg)
        # do a quick radial average using bins
        MDF = np.empty_like(Exc)
        for k in range(z.size):
            img = Det2D_conv[..., k]
            # radial average
            rbins = r
            dr = rbins[1]-rbins[0] if rbins.size>1 else Rmax
            vals = np.zeros_like(rbins)
            for i, rv in enumerate(rbins):
                mask = (Rxy >= rv-dr/2) & (Rxy < rv+dr/2)
                s = img[mask]
                vals[i] = s.mean() if s.size>0 else img[Nx//2, Nx//2]
            MDF[:, k] = Exc[:, k] * vals
    except Exception as e:
        warnings.warn(f"Finite pinhole convolution fell back to infinitesimal (reason: {e})")
        MDF = Exc * Det

    return {"r": r, "z": z, "Exc": Exc, "Det": Det, "MDF": MDF}

#%%

# radial/axial grids in micrometers
r = np.linspace(0.0, 1.0, 201)   # 0..2 µm
z = np.linspace(0, 1.0, 201)  # -2..2 µm

NA, n_med = 1.2, 1.333
lam_exc, lam_em = 0.5, 0.55     # µm

# optional coverslip ATF: (n_coverslip, delta_t_um)
atf = (1.52, 0.15)   # example: #1.5 glass with +150 nm mismatch

out = confocal_mdf(
    r, z, NA, n_med, lam_exc, lam_em,
    pupil_exc="aplanatic", pupil_det="aplanatic",
    atf_exc=atf, atf_det=atf,
    pinhole_um_image=50.0,  # set >0 for finite pinhole (image-plane radius)
    magnification=60.0
)

Exc, Det, MDF = out["Exc"], out["Det"], out["MDF"]
# Now display with your FocusImage2D/3D helpers


extent = [z.min(), z.max(), r.min(), r.max()]  # x=z, y=r
fig, axs = plt.subplots(1,3, figsize=(12,3.5), constrained_layout=True)

im0 = axs[0].imshow(Exc, origin='lower', aspect='equal', extent=extent)
axs[0].set_title('Excitation'); axs[0].set_xlabel('z (µm)'); axs[0].set_ylabel('r (µm)')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(Det, origin='lower', aspect='equal', extent=extent)
axs[1].set_title('Detection'); axs[1].set_xlabel('z (µm)'); axs[1].set_ylabel('r (µm)')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(MDF, origin='lower', aspect='equal', extent=extent)
axs[2].set_title('Confocal MDF'); axs[2].set_xlabel('z (µm)'); axs[2].set_ylabel('r (µm)')
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.show()


# 2D coordinate grids shaped (Nr, Nz)
R = np.tile(r[:,None], (1, z.size))
Z = np.tile(z[None,:], (r.size, 1))

# DC-only “stack” so FocusImage2D/3D can read it
vol_exc = Exc[..., None]
vol_det = Det[..., None]
vol_mdf = MDF[..., None]

from utilities import FocusImage2D, FocusImage3D  # your existing funcs

# 2D mirrored plots
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
ax1 = plt.subplot(1,2,1)
FocusImage2D(R, Z, vol_exc, phi=0.0, flag="vertical", ax=ax1, cmap="hot")
ax1.set_title('Excitation (Hankel)'); ax1.set_xlabel('r (µm)'); ax1.set_ylabel('z (µm)')

ax2 = plt.subplot(1,2,2)
FocusImage2D(R, Z, vol_mdf, phi=0.0, flag="vertical", ax=ax2, cmap="hot")
ax2.set_title('Confocal MDF'); ax2.set_xlabel('r (µm)'); ax2.set_ylabel('z (µm)')
plt.tight_layout(); plt.show()

# 3D isosurfaces (pick thresholds)
tsh = 1.0 / np.exp(np.array([1.0, 2.0, 3.0]))
L = len(tsh)
colors = [(1.0, j/L, 0.0) for j in range(L-1, -1, -1)]  # like your MATLAB flipud([1 y 0])

plt.figure(figsize=(10,4))
ax = plt.subplot(1,2,1, projection='3d')
FocusImage3D(R, Z, vol_exc, tsh=tsh, maxangle=2*np.pi, ax=ax,
             plot_isosurfaces=True, colors=colors, alphas=[0.5, 0.35, 0.25])
ax.set_title('Excitation — isosurfaces')

ax = plt.subplot(1,2,2, projection='3d')
FocusImage3D(R, Z, vol_mdf, tsh=tsh, maxangle=2*np.pi, ax=ax,
             plot_isosurfaces=True, colors=colors, alphas=[0.5, 0.35, 0.25])
ax.set_title('Confocal MDF — isosurfaces')
plt.tight_layout(); plt.show()
