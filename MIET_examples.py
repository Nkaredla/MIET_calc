# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 23:53:06 2025

@author: narai
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from MIET_main import MetalsDB, plot_dipole_adr, miet_calc, brightness_dipole
from Optics_main import PatternGeneration, radialpattern

#%% Examples

 
# Calculate Dipole Angular Distribution of Radiation

HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))

# 2) Choose wavelength (nm) and fetch the metal index from the DB
wavelength_nm = 690.0
n_gold  = db.get_index(20, wavelength_nm)   # 20 -> 'gold' (from your MetalsDB mapping)
n_spacer = 1.46                              # polymer/spacer (dielectric); keep as is or fetch if you store it

# 3) Call your existing plotter exactly as before, but pass the DB-derived n for the metal
plot_dipole_adr(
    wavelength_nm=wavelength_nm,
    n_glass=1.52,
    n_top=1.33,
    n_polymer=1.33,                          # your dipole layer index (can also be 1.46 if that’s your polymer)
    stack_metal=(n_gold, n_spacer),        # <- metal from metals.mat, spacer as dielectric
    d_metal_nm=(10.0, 10.0),               # metal thickness (nm), spacer thickness (nm)
    dip_z_nm=200.0,
    al_list=np.linspace(0, np.pi/2, 5),
    be = np.pi,
    lw = 2.0
)


#%% Calculate MIET curve for a single wavelength for a randomly oriented dipole

HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))
n0 = [1.52, 20, 1.46]   # 20 means "gold" (from the .mat), sandwiched with glass & spacer
d0 = [10, 5]            # thicknesses between those 3 indices (nm)
n  = 1.33               # dipole layer index
n1 = [1.33]             # top half-space
d1 = []                 # none
z, life = miet_calc(
    al_res=np.nan,          # average orientation (use None or np.nan)
    lamem=690.0,            # single wavelength, nm
    n0=n0, n=n, n1=n1, d0=d0, d=200.0, d1=d1,
    qyield=0.8, tau_free=3.5,
    fig=True,
    curveType='maximum',
    metals_db=db            # <<— uses the .mat
)

#%% Calculate MIET curve for a single wavelength, but for parallel and vertical dipole orientations

HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))    # must contain fields: wavelength, gold, etc.

# Substrate stack: glass | Au | spacer (polymer) | dipole layer | top half-space
n0 = [1.52, 20, 1.46]           # 20 => 'gold' in metals.mat; glass and spacer are real indices
d0 = [10, 5]                    # nm: Au thickness=10, spacer (e.g., polymer) thickness=5
n  = 1.33                       # dipole layer index (e.g., water)
n1 = [1.33]                     # top half-space
d1 = []                         # no extra layers above

lam_em = 690.0                  # nm (single-wavelength MIET)
tau_free = 3.5                  # ns
qyield   = 0.80                 # intrinsic QY (fraction)

# --- Ask for vertical (0°) and parallel (90°) dipoles  ---
z_nm, life_oriented = miet_calc(
                            al_res=[0.0, 90.0],         # theta list in degrees → vertical & parallel
                            lamem=lam_em,
                            n0=n0, n=n, n1=n1, d0=d0, d=200.0, d1=d1,   # compute up to 200 nm layer
                            qyield=qyield, tau_free=tau_free,
                            fig=False,                  # we'll plot below
                            curveType='maximum',        # mimic MATLAB trimming behavior
                            metals_db=db
                            )
# life_oriented has shape (len(z_nm), 2); columns correspond to 0° and 90°
life_vert_ns    = life_oriented[:, 0]
life_parallel_ns= life_oriented[:, 1]

# --- Plot ---
plt.figure(figsize=(6,4))
plt.plot(z_nm, life_vert_ns,    label="vertical dipole (0°)",   lw=2)
plt.plot(z_nm, life_parallel_ns,label="parallel dipole (90°)",  lw=2)
plt.plot(z_nm, (2*life_parallel_ns + life_vert_ns)/3,label="randomly oriented dipole",  lw=2)
plt.xlabel("distance from metal (nm)")
plt.ylabel("lifetime (ns)")
plt.title(f"MIET @ {lam_em:.0f} nm, QY={qyield:.2f}, τ₀={tau_free:.2f} ns")
plt.legend()
plt.tight_layout()
plt.show()


#%% 
########################################################################################
#     Fig. 1(c) - 3D membrane dynamics paper.    #
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
try:
    # Spyder sets this when you run a file
    here = Path(__file__).resolve().parent
except NameError:
    # In interactive console / cell execution
    import inspect
    frame = inspect.currentframe()
    # Walk back to the file from which this code was run
    here = Path(inspect.getfile(frame)).resolve().parent

# Add to sys.path
if str(here) not in sys.path:
    sys.path.insert(0, str(here))
from MIET_main import MetalsDB, miet_calc, brightness_dipole

HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))    # must contain fields: wavelength, gold, etc.
for code, name in db.code_to_field.items():
    print(code, "→", name)

#
# Substrate stack: glass | Au | spacer (polymer) | dipole layer | top half-space
lam_em = 700.0                  # nm (single-wavelength MIET)
n_gold       = db.get_index(20, lam_em) # 20 -> gold
n_titanium   = db.get_index(80, lam_em) # 20 -> gold
n0 = [1.52, n_titanium, n_gold, n_titanium, 1.46]   # 20 => 'gold',  80=> titanium in metals.mat; glass and spacer are real indices
d0 = [2, 10, 1, 10]                    # nm: Au thickness=10, spacer (e.g., polymer) thickness=5
n  = 1.33                       # dipole layer index (e.g., water)
n1 = [1.33]                     # top half-space
d1 = []                         # no extra layers above
d  = 100.0                      # dipole layer - sufficiently large


tau_free = 1.0                  # ns
qyield   = 0.80                 # intrinsic QY (fraction)

# --- Ask for parallel (90°) dipole  ---
z_nm, life_parallel = miet_calc(
                            al_res=[90.0],         # theta list in degrees → vertical & parallel
                            lamem=lam_em,
                            n0=n0, n=n, n1=n1, d0=d0, d=d, d1=d1,   # compute up to 200 nm layer
                            qyield=qyield, tau_free=tau_free,
                            fig=False,                  # we'll plot below
                            curveType='maximum',        # mimic MATLAB trimming behavior
                            metals_db=db
                            )

# -----------------------------
# Unit conversion: nm -> "k0-units"
# brightness_dipole() expects z, d0, d, d1 in k0-units (same convention as the ADR code)
# -----------------------------
k0 = 2*np.pi / (lam_em * 1e-3)      # wavelength in µm → k0 [1/µm]
to_k = lambda nm: (nm * 1e-3) * k0

z_k   = to_k(z_nm)
d0_k  = to_k(np.asarray(d0, float))
d_k   = to_k(d)
d1_k  = to_k(np.asarray(d1, float)) if len(d1) else np.array([])
NA = 1.49
# -----------------------------
# Compute brightness curves
# -----------------------------
bv, bp, br, bf = brightness_dipole(
    z=z_k,
    n0=n0,
    n=n,
    n1=n1,
    d0=d0_k,
    d=d_k,
    d1=d1_k,
    NA=NA,
    QY=qyield,
    curves=False
)


#%% --- Plot ---
fig, ax1 = plt.subplots(figsize=(6, 4))

# First axis (left)
fsize = 20
ax1.plot(z_nm, life_parallel, color="tab:brown", lw=3)
ax1.set_xlabel("height z (nm)", fontsize=fsize, fontname="Calibri")
ax1.set_ylabel(r"relative lifetime ($\tau_f / \tau_0$)",  fontsize=fsize, fontname="Calibri",color="tab:brown")
ax1.tick_params(axis="y", labelcolor="tab:brown", labelsize = fsize)
ax1.tick_params(axis="x",labelsize=fsize)

for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname("Calibri")
# Second axis (right)
ax2 = ax1.twinx()
ax2.plot(z_nm, np.real(bp), label="relative brightness", color="tab:orange", lw=3)
ax2.set_ylabel("Relative brightness", fontsize=fsize, fontname="Calibri", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange",labelsize=fsize)
ax2.tick_params(axis="x",labelsize=fsize)

for label in ax2.get_yticklabels():
    label.set_fontname("Calibri")
    
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

# Title and layout
#fig.suptitle("MIET lifetime & brightness vs. emitter height (Au film @ 690 nm)")
fig.tight_layout()
plt.show()


#######################################################################################

#%% Calculate MIET curve for a spectrum
    
lam_spec = {
"SpectrumFile": "my_dye_spectrum.txt",  # two columns: wavelength, intensity
"Wavel_Small": 650.0,                   # nm
"Wavel_Large": 750.0,                   # nm
}

z, life = miet_calc(np.nan, lam_spec, n0 = n0, 
                    n = n, n1 = n1, d0 = d0, d = 200.0, 
                    d1 = d1,  qyield=0.8, tau_free=3.5, 
                    fig=True, metals_db=db)


#%% Defocused Pattern calculations on glass


# --- Parameters ---
z      = 0.0
NA     = 1.45
n0     = 1.52      # glass (below)
n      = 1.0       # dipole layer
n1     = 1.0       # top half-space (air)
d0     = []        # no extra layers below
d      = 0.01      # dipole layer thickness (same length units as lamem)
d1     = []        # no extra layers above
lamem  = 0.52      # emission wavelength (same units as d, e.g., µm)
mag    = 100.0
focus  = 0.9
atf    = None
ring   = None      # MATLAB had '0' — here just pass None/0
pixel  = 1.0
pic    = 1         # show tiled montage of masks
be_res = 15.0      # degrees
al_res = 15.0      # degrees
nn     = 100       # PSF stamp half-size (produces a (201 x 201) patch)

# --- Run ---
plt.figure()
model = PatternGeneration(
    z, NA, n0, n, n1, d0, d, d1,
    lamem, mag, focus,
    atf, ring,
    pixel, nn,
    be_res, al_res,
    pic
)

# `model` now holds:
# model['rho'], model['theta'], model['phi'], model['mask'],
# model['fxx0'], model['fxx2'], model['fxz'], model['byx0'], model['byx2'], model['byz']

print(f"mask stack shape: {model['mask'].shape}  (Ny, Nx, Norient)")


#%% generate radial/azimuthal patterns


pixel = 0.06                              # µm per pixel
nn = int(np.ceil(0.6 / pixel))            # stamp half-size (pixels)

lamex = 0.640                             # µm
resolution = np.array([lamex/0.02, lamex/0.001], float)
rhofield = np.array([-lamex/resolution[0]/2, nn*pixel*1.1], float)
zfield   = np.array([0.0, 1.0], float)

NA = 1.49
fd = 3e3                                  # µm (focal length)

n0 = 1.52                                 # below
n  = 1.33                                 # dipole layer
n1 = 1.33                                 # above

d0 = []                                   # µm
d  = 0.0                                  # µm
d1 = []                                   # µm

over   = np.inf
focpos = 0.0
atf    = None

theta_min = np.arcsin(0.3 * NA / n0)
theta_max = np.arcsin(0.9 * NA / n0)

# Option A: simple ring
ring = 'np.cos(psi)*rad'

# Option B: band-limited ring (uncomment to use)
# ring = f'np.cos(psi) * ((rad < np.sin({theta_max})) & (rad > np.sin({theta_min})))'

maxm = 3

# Angular sampling resolution (degrees)
be_res = 10.0
al_res = 30.0

# --- Run (pattern can be 'azimuthal' to apply the x/y swap+zero as in MATLAB) ---
fxc, fxs, fyc, fys, fzc, fzs, rho, rr, psi, mask, imtheo, theta, phi = radialpattern(
    pattern='radial',          # or any other string (no swap), e.g. 'radial'
    pixel=pixel,
    NA=NA,
    n=(n, n1),                    # pass (n, n1)
    nn=nn,
    lamex=lamex,
    resolution=resolution,
    rhofield=rhofield,
    zfield=zfield,
    fd=fd,
    n0=n0,
    d0=np.asarray(d0, float),
    d=d,
    d1=np.asarray(d1, float),
    over=over,
    focpos=focpos,
    atf=atf,
    ring=ring,
    maxm=maxm,
    theta_min=theta_min,
    be_res=be_res,
    al_res=al_res,
    pic=1                          # show the tiled montage
)


#%%

from Optics_main import mdf_confocal_microscopy_py

exc_lin, exc_rad, mdf_lin, mdf_rad = mdf_confocal_microscopy_py(
    rhofield=[0., 0.5],
    zfield=[0, 0.5],
    NA=1.49,
    fd=3e3,
    n0=1.52,
    n=1.333,
    n1=1.333,
    d0=[],              # no layers below
    d=0.0,              # dipole layer thickness
    d1=[],              # no layers above
    lamex=0.645,
    over=5e3,           # beam overfilling
    focpos=0.,
    defoc=0.0,
    av= 50/2,           # unused in this proxy MDF
    lamem=0.650,        # unused in this proxy MDF
    mag=60,             # unused in this proxy MDF
    zpin=0.0,           # unused in this proxy MDF
    atf=[],             # no coverslip correction
    resolution=50,      # sampling (Nrho_per_lambda, Nz_samples)
    ring=None,          # linear pol (function also computes radial internally)
    maxm=2,
    bild=True           # show the 2D plots via FocusImage2D
)

plt.show()


#%% dipole brightness calculation

wavelength_nm = 690.0                      # emission wavelength (nm)
HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))

n_glass  = 1.52
n_spacer = 1.46
n_dip    = 1.33                            # dipole layer index (e.g., water/polymer)
n_top    = 1.33
n_gold   = db.get_index(20, wavelength_nm) # 20 -> gold

# MIET stack BELOW the dipole layer (order = bottom -> top, as in your examples):
# glass | gold | spacer  (closest to dipole is the spacer)
n0 = [n_glass, n_gold, n_spacer]
d0_nm = [10.0, 5.0]                        # thicknesses between n0 entries (nm): Au=10 nm, spacer=5 nm

# Dipole layer thickness (any sufficiently large number; units converted below)
d_nm = 200.0

# Stack ABOVE the dipole layer (top half-space only here)
n1 = [n_top]
d1_nm = []

# Detection / emitter parameters
NA = 1.45                                  # high-NA oil objective
QY = 0.8                                   # far-from-surface quantum yield (0..1)

# Height range (nm) for brightness curves
z_nm = np.linspace(2.0, 150.0, 300)

# -----------------------------
# Unit conversion: nm -> "k0-units"
# brightness_dipole() expects z, d0, d, d1 in k0-units (same convention as the ADR code)
# -----------------------------
k0 = 2*np.pi / (wavelength_nm * 1e-3)      # wavelength in µm → k0 [1/µm]
to_k = lambda nm: (nm * 1e-3) * k0

z_k   = to_k(z_nm)
d0_k  = to_k(np.asarray(d0_nm, float))
d_k   = to_k(d_nm)
d1_k  = to_k(np.asarray(d1_nm, float)) if len(d1_nm) else np.array([])

# -----------------------------
# Compute brightness curves
# -----------------------------
bv, bp, br, bf = brightness_dipole(
    z=z_k,
    n0=n0,
    n=n_dip,
    n1=n1,
    d0=d0_k,
    d=d_k,
    d1=d1_k,
    NA=NA,
    QY=QY,
    curves=False
)

# -----------------------------
# Plot vs. physical height (nm)
# -----------------------------
plt.figure(figsize=(6.5, 4.5))
plt.plot(z_nm, np.real(bv), label='Vertical dipole (bv)', lw=2)
plt.plot(z_nm, np.real(bp), label='Parallel dipole (bp)', lw=2)
plt.plot(z_nm, np.real(bf), label='Random (fixed-orientation ensemble, bf)', lw=2)
# (Optional) also show fast-rotating ensemble:
# plt.plot(z_nm, np.real(br), label='Fast-rotating (br)', lw=2, ls='--')

plt.xlabel('Emitter height z (nm)')
plt.ylabel('Brightness (arb. units)')
plt.title('MIET brightness vs. height — thin Au film @ 690 nm')
plt.legend()
plt.tight_layout()
plt.show()

#%%
"""

Effective photon emission rate maps on a MIET stack using gauss_exc.
- Stack: glass | gold | silica | water (dipoles live in water, z=0 at silica|water).
- Excitation: linear (x-pol) and radial (cos ψ + π/2).
- Outputs: 2x3 subplots: (parallel, vertical, random) × (linear, radial).
"""
import numpy as np
import matplotlib.pyplot as plt

from Optics_main import gauss_exc, RotateEMField
from MIET_main import brightness_dipole   # <-- use this

# -----------------------------
# helpers
# -----------------------------
def reconstruct_fields(fxc, fxs, fyc, fys, fzc, fzs, phi):
    """
    Reconstruct (Ex,Ey,Ez) at a given azimuth 'phi' from Fourier harmonics
    returned by gauss_exc. All arrays are (Nrho, Nz, ...).
    Returns Ex,Ey,Ez of shape (Nrho, Nz) (complex).
    """
    Ex = fxc[..., 0].copy()
    Ey = fyc[..., 0].copy()
    Ez = fzc[..., 0].copy()
    maxm = fxs.shape[2] if (fxs is not None and fxs.ndim == 3) else 0
    if maxm > 0:
        for j in range(1, maxm + 1):
            c = np.cos(j * phi); s = np.sin(j * phi)
            Ex += fxc[..., j] * c + fxs[..., j-1] * s
            Ey += fyc[..., j] * c + fys[..., j-1] * s
            Ez += fzc[..., j] * c + fzs[..., j-1] * s
    return Ex, Ey, Ez


def brightness_miet_vs_z(z_vec_um, n0, n_dip, n1, d0_um, d_um, d1_um, lam_em_um, NA, QY):
    """
    Call brightness_dipole with z and layer thicknesses in k-units (k0=1).
    Returns bp, bv, bf arrays over z (parallel, vertical, fixed-random).
    """
    # convert to k-units (k0 = 2π/λ  →  k0*length, but brightness_dipole assumes k0=1)
    z_k  = 2*np.pi*np.asarray(z_vec_um, float) / lam_em_um
    d0_k = 2*np.pi*np.atleast_1d(np.asarray(d0_um, float)) / lam_em_um
    d1_k = 2*np.pi*np.atleast_1d(np.asarray(d1_um, float)) / lam_em_um
    d_k  = 2*np.pi*float(d_um) / lam_em_um

    # brightness_dipole(z_k, n0, n, n1, d0_k, d_k, d1_k, NA, QY)
    bv, bp, br, bf = brightness_dipole(
        z_k, np.atleast_1d(n0), complex(n_dip), np.atleast_1d(n1),
        d0_k, d_k, d1_k, float(NA), float(QY), curves=False
    )
    # return as 1D arrays in z order
    return np.asarray(bp, float), np.asarray(bv, float), np.asarray(bf, float)


# -----------------------------
# configuration
# -----------------------------
# Materials
n_glass  = 1.52
n_silica = 1.46
n_water  = 1.333

# Objective/beam
NA     = 1.45
fd     = 1.8e3
over   = 5e3
maxm   = 2
resolution = (100, 100)  # (Nrho per λ, Nz samples)

# Wavelengths (µm)
lamex_um = 0.645
lamem_um = 0.690

n_gold_ex = db.get_index(20, lamex_um*1e3)  # Au @ excitation
n_gold_em = db.get_index(20, lamem_um*1e3)  # Au @ emission

# MIET geometry (µm)
t_gold_um   = 0.010   # 10 nm
t_spacer_um = 0.005   # 5  nm
d_dip_um    = 0.30    # thickness covering z-range for phasing

# Grids (ρ and z in µm); z=0 at silica|water interface (focus here)
rho_min, rho_max = 0.0, 0.5
z_min,   z_max   = 0.0, 0.3
focpos = 0.0

# Build stacks for gauss_exc (EXCITATION: below dipole = glass|gold|silica; dipole layer = water; above = water)
n0_exc = [n_glass, n_gold_ex, n_silica]
d0_exc = [t_gold_um, t_spacer_um]
n_exc  = n_water
n1_exc = [n_water]
d1_exc = []

# Build stacks for MIET emission (same geometry but Au @ emission λ)
n0_em = [n_glass, n_gold_em, n_silica]
d0_em = [t_gold_um, t_spacer_um]
n_em  = n_water
n1_em = [n_water]
d1_em = []

# Detection quantum yield in bulk emitter medium
QY_bulk = 0.9

# -----------------------------
# excitation fields (gauss_exc)
# -----------------------------
# Linear (x-polarized pupil)
fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho, z = gauss_exc(
    rhofield=[rho_min, rho_max],
    zfield=[z_min, z_max],
    NA=NA, fd=fd,
    n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring=None, maxm=maxm
)

# Radial (cosψ) + π/2 rotated coherent sum
fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, _, _ = gauss_exc(
    rhofield=[rho_min, rho_max],
    zfield=[z_min, z_max],
    NA=NA, fd=fd,
    n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring='np.cos(psi)', maxm=maxm
)
fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r = RotateEMField(
    fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, np.pi/2
)
fxc_rad = fxc_c + fxc_r
fxs_rad = fxs_c + fxs_r
fyc_rad = fyc_c + fyc_r
fys_rad = fys_c + fys_r
fzc_rad = fzc_c + fzc_r
fzs_rad = fzs_c + fzs_r

# -----------------------------
# reconstruct fields at φ = 0 (ρ–z plane x-axis)
# -----------------------------
phi_sample = 0.0
Ex_lin, Ey_lin, Ez_lin = reconstruct_fields(fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, phi_sample)
Ex_rad, Ey_rad, Ez_rad = reconstruct_fields(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, phi_sample)

# Intensities
Ix_lin = np.abs(Ex_lin)**2
Iy_lin = np.abs(Ey_lin)**2
Iz_lin = np.abs(Ez_lin)**2
It_lin = Ix_lin + Iy_lin + Iz_lin

Ix_rad = np.abs(Ex_rad)**2
Iy_rad = np.abs(Ey_rad)**2
Iz_rad = np.abs(Ez_rad)**2
It_rad = Ix_rad + Iy_rad + Iz_rad

# -----------------------------
# MIET brightness factors vs z (k-units for brightness_dipole)
# -----------------------------
z_vec = z[0, :]  # (Nz,)
bp, bv, bf = brightness_miet_vs_z(
    z_vec, n0_em, n_em, n1_em, d0_em, d_dip_um, d1_em, lamem_um, NA, QY_bulk
)
# Broadcast to (Nrho, Nz)
bp_2d = np.ones_like(Ix_lin) * bp[None, :]
bv_2d = np.ones_like(Ix_lin) * bv[None, :]
bf_2d = np.ones_like(Ix_lin) * bf[None, :]

# -----------------------------
# Effective photon emission rates
# -----------------------------
# Parallel (μ∥x):    R = |Ex|² · bp(z)
R_par_lin = Ix_lin * bp_2d
R_par_rad = Ix_rad * bp_2d

# Vertical (μ∥z):    R = |Ez|² · bv(z)
R_ver_lin = Iz_lin * bv_2d
R_ver_rad = Iz_rad * bv_2d

# Random (fixed):    R = (|E|²/3) · bf(z)
R_rnd_lin = (It_lin / 3.0) * bf_2d
R_rnd_rad = (It_rad / 3.0) * bf_2d

# -----------------------------
# Plot (2 × 3) with consistent axes
# -----------------------------
def imshow_rz(ax, rho_grid, z_grid, data, title):
    im = ax.imshow(
        data.T, origin='lower',
        extent=[rho_grid[0,0], rho_grid[-1,0], z_grid[0,0], z_grid[0,-1]],
        aspect='auto'
    )
    ax.set_xlabel(r'$\rho$  [$\mu$m]')
    ax.set_ylabel('z  [$\mu$m]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

plt.figure(figsize=(14, 8))

# Linear row
ax = plt.subplot(2,3,1); imshow_rz(ax, rho, z, R_par_lin, 'Linear — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,2); imshow_rz(ax, rho, z, R_ver_lin, 'Linear — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,3); imshow_rz(ax, rho, z, R_rnd_lin, 'Linear — Random    ((|E|²/3) · bf)')

# Radial row
ax = plt.subplot(2,3,4); imshow_rz(ax, rho, z, R_par_rad, 'Radial — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,5); imshow_rz(ax, rho, z, R_ver_rad, 'Radial — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,6); imshow_rz(ax, rho, z, R_rnd_rad, 'Radial — Random    ((|E|²/3) · bf)')

plt.tight_layout()
plt.show()


#%%
"""
Effective photon emission rate maps on a MIET stack using gauss_exc.
- Stack: glass | gold | silica | water (dipoles live in water, z=0 at silica|water).
- Excitation: linear (x-pol) and radial (cos ψ + π/2).
- Outputs:
    • 2D maps (6 subplots): (parallel, vertical, random) × (linear, radial)
    • 3D Plotly isosurfaces (2×3 scenes) with a custom colorscale
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RectBivariateSpline, interp1d

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"   

# local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Optics_main import gauss_exc, RotateEMField
from MIET_main import MetalsDB, brightness_dipole


# -----------------------------
# Colors: Plotly-style -> Matplotlib colormap
# -----------------------------
PLOTLY_SCALE = [
    (0.00, 'rgb(12,51,131)'),
    (0.25, 'rgb(10,136,186)'),
    (0.50, 'rgb(242,211,56)'),
    (0.75, 'rgb(242,143,56)'),
    (1.00, 'rgb(217,30,30)'),
]
def _to_mpl_color(s):
    if isinstance(s, str) and s.startswith('rgb'):
        r, g, b = [int(v)/255.0 for v in s[s.find('(')+1:s.find(')')].split(',')]
        return (r, g, b)
    return s
CMAP_CONT = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_cont', [(t, _to_mpl_color(c)) for t, c in PLOTLY_SCALE], N=256
)


# -----------------------------
# Physics helpers
# -----------------------------
def brightness_miet_vs_z(z_vec_um, n0, n_dip, n1, d0_um, d_um, d1_um, lam_em_um, NA, QY):
    """Wrap brightness_dipole (expects k0=1 units). Return bp, bv, bf vs z."""
    z_k  = 2*np.pi*np.asarray(z_vec_um, float) / lam_em_um
    d0_k = 2*np.pi*np.atleast_1d(np.asarray(d0_um, float)) / lam_em_um
    d1_k = 2*np.pi*np.atleast_1d(np.asarray(d1_um, float)) / lam_em_um
    d_k  = 2*np.pi*float(d_um) / lam_em_um
    bv, bp, br, bf = brightness_dipole(
        z_k, np.atleast_1d(n0), complex(n_dip), np.atleast_1d(n1),
        d0_k, d_k, d1_k, float(NA), float(QY), curves=False
    )
    return np.asarray(bp, float), np.asarray(bv, float), np.asarray(bf, float)


def _fill_axis_cubic(rho_1d, F_rz, nfit=6):
    """
    Replace the ρ=0 row in a (Nr,Nz) array with a cubic extrapolation from the
    first rings (per z). F_rz may be complex. Returns a copy.
    """
    F = np.array(F_rz, copy=True)
    r = np.asarray(rho_1d, float).ravel()
    i0 = np.argmin(np.abs(r))
    for iz in range(F.shape[1]):
        col = F[:, iz]
        m = np.isfinite(col) & (r > 0)
        idx = np.nonzero(m)[0]
        if idx.size == 0:
            F[i0, iz] = 0.0
            continue
        use = idx[:min(nfit, idx.size)]
        rr = r[use]
        ff = col[use]
        if np.iscomplexobj(ff):
            cre = np.polyfit(rr, np.real(ff), 3); cim = np.polyfit(rr, np.imag(ff), 3)
            f0 = np.polyval(cre, 0.0) + 1j*np.polyval(cim, 0.0)
        else:
            c = np.polyfit(rr, ff, 3); f0 = np.polyval(c, 0.0)
        F[i0, iz] = f0
    return F


def sanitize_harmonics_for_spline(fxc, fxs, fyc, fys, fzc, fzs, rho):
    """
    Prepare GaussExc harmonics for cubic (rho,z) splines:
    - cos slices: fill ρ=0 row via cubic extrapolation
    - sin slices: force 0 at ρ=0 (odd in azimuth → vanishes on axis)
    Returns copies.
    """
    def fix_pack(fcos, fsin):
        fcos = np.array(fcos, copy=True)
        for k in range(fcos.shape[2]):  # DC + cos j
            fcos[:, :, k] = _fill_axis_cubic(rho[:, 0], fcos[:, :, k])
        if fsin is not None and fsin.size:
            fsin = np.array(fsin, copy=True)
            i0 = np.argmin(np.abs(rho[:, 0]))
            fsin[i0, :, :] = 0.0
        else:
            fsin = fsin
        return fcos, fsin

    fxc2, fxs2 = fix_pack(fxc, fxs)
    fyc2, fys2 = fix_pack(fyc, fys)
    fzc2, fzs2 = fix_pack(fzc, fzs)
    return fxc2, fxs2, fyc2, fys2, fzc2, fzs2


def fields_on_xyz_cubic(fxc, fxs, fyc, fys, fzc, fzs, rho, z_grid, X, Y, Z):
    """
    Sample Ex,Ey,Ez on Cartesian grids X,Y,Z using cubic (rho,z) splines
    for each harmonic (MATLAB-like interp2(...,'cubic')).
    f*c: (Nr,Nz,M+1), f*s: (Nr,Nz,M); rho:(Nr,1), z_grid:(1,Nz).
    Returns Ex,Ey,Ez with same shape as X.
    """
    r1d = np.asarray(rho)[:, 0]
    z1d = np.asarray(z_grid)[0, :]

    Nr, Nz, M1 = fxc.shape
    M = M1 - 1

    # Build real/imag splines per harmonic
    def make_splines(F):
        s_re, s_im = [], []
        for k in range(F.shape[2]):
            s_re.append(RectBivariateSpline(r1d, z1d, np.real(F[:, :, k]), kx=3, ky=3))
            s_im.append(RectBivariateSpline(r1d, z1d, np.imag(F[:, :, k]), kx=3, ky=3))
        return s_re, s_im

    cEx_re, cEx_im = make_splines(fxc)
    cEy_re, cEy_im = make_splines(fyc)
    cEz_re, cEz_im = make_splines(fzc)

    sEx_re = sEx_im = sEy_re = sEy_im = sEz_re = sEz_im = []
    if fxs is not None and fxs.size:
        sEx_re, sEx_im = make_splines(fxs)
    if fys is not None and fys.size:
        sEy_re, sEy_im = make_splines(fys)
    if fzs is not None and fzs.size:
        sEz_re, sEz_im = make_splines(fzs)

    # Cylindrical coords on target grid
    R = np.sqrt(X**2 + Y**2)
    P = np.arctan2(Y, X)
    r = R.ravel(); z = Z.ravel()

    def eval_pair(sr, si, k):
        return (sr[k].ev(r, z) + 1j * si[k].ev(r, z)).reshape(X.shape)

    # DC
    Ex = eval_pair(cEx_re, cEx_im, 0)
    Ey = eval_pair(cEy_re, cEy_im, 0)
    Ez = eval_pair(cEz_re, cEz_im, 0)

    # higher harmonics
    for j in range(1, M+1):
        cj = np.cos(j * P); sj = np.sin(j * P)
        Ex += cj * eval_pair(cEx_re, cEx_im, j)
        Ey += cj * eval_pair(cEy_re, cEy_im, j)
        Ez += cj * eval_pair(cEz_re, cEz_im, j)
        if sEx_re: Ex += sj * eval_pair(sEx_re, sEx_im, j-1)
        if sEy_re: Ey += sj * eval_pair(sEy_re, sEy_im, j-1)
        if sEz_re: Ez += sj * eval_pair(sEz_re, sEz_im, j-1)

    return Ex, Ey, Ez


# -----------------------------
# 2D plotting helper
# -----------------------------
def imshow_rz_sym(ax, rho_grid, z_grid, data, title):
    """Mirror the ρ-axis so the map shows both +x and −x sides."""
    r = rho_grid[:, 0]
    rho_sym_1d = np.concatenate([-r[::-1], r])
    data_sym = np.concatenate([data[::-1, :], data], axis=0)
    im = ax.imshow(
        data_sym.T, origin='lower',
        extent=[rho_sym_1d[0], rho_sym_1d[-1], z_grid[0,0], z_grid[0,-1]],
        aspect='auto', cmap=CMAP_CONT
    )
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]'); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)



# -----------------------------
# 3D plotting helper (one window per volume)
# -----------------------------
def get_iso_color(value, vmin, vmax, colorscale):
    """
    Map a scalar 'value' in [vmin, vmax] to an RGB string using the given
    Plotly colorscale [(t, 'rgb(...)'), ...], with the same normalization
    that Plotly uses when cmin=vmin, cmax=vmax.
    """
    # normalize to [0,1]
    if vmax == vmin:
        t_norm = 0.0
    else:
        t_norm = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    t_norm = float(np.clip(t_norm, 0.0, 1.0))

    # find the segment in the colorscale
    for i in range(len(colorscale) - 1):
        t0, c0 = colorscale[i]
        t1, c1 = colorscale[i+1]
        if t0 <= t_norm <= t1:
            # local interpolation factor
            if t1 == t0:
                alpha = 0.0
            else:
                alpha = (t_norm - t0) / (t1 - t0)

            def parse_rgb(s):
                vals = s[s.find('(')+1:s.find(')')].split(',')
                return np.array([int(v) for v in vals], float)

            rgb0 = parse_rgb(c0)
            rgb1 = parse_rgb(c1)
            rgb = (1.0 - alpha)*rgb0 + alpha*rgb1
            rgb = np.clip(np.round(rgb), 0, 255).astype(int)
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    # if somehow out of range after clip, fall back to last color
    return colorscale[-1][1]


# def show_iso(X, Y, Z, V, title, colorscale, *,
#              keep_halfspace=True, vmin_frac=0.25, vmax_frac=0.9, surface_count=10):
#     V = V.astype(np.float32)
#     # mask halfspace by moving values outside below isomin
#     if keep_halfspace:
#         mask = (Y < 0)
#         vmax = np.nanmax(V[~mask]) if np.isfinite(V[~mask]).any() else None
#         if vmax is None or vmax <= 0:
#             print(f"[warn] '{title}': no finite data in kept region.")
#             return
#         Vn = V / vmax
#         Vn[mask] = -1.0                                 # finite sentinel below isomin
#     else:
#         vmax = np.nanmax(V) if np.isfinite(V).any() else None
#         if vmax is None or vmax <= 0:
#             print(f"[warn] '{title}': no finite data.")
#             return
#         Vn = V / vmax

#     fig = go.Figure(go.Isosurface(
#         x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=Vn.ravel(),
#         isomin=float(vmin_frac), isomax=float(vmax_frac),
#         opacity=0.2,
#         surface_count=int(surface_count),
#         colorscale=[(t, c) for (t, c) in colorscale],
#         caps=dict(x_show=False, y_show=False, z_show=False),
#         showscale=False,
#     ))
#     fig.update_layout(
#         title=title, template="plotly_white",
#         scene=dict(
#             aspectmode="data",
#             xaxis=dict(
#                 title="x [μm]",
#                 showgrid=True, gridcolor="lightgray", zeroline=False,
#                 showline=True, linecolor="black", mirror=True
#                 ),
#             yaxis=dict(
#                 title="y [μm]",
#                 showgrid=True, gridcolor="lightgray", zeroline=False,
#                 showline=True, linecolor="black", mirror=True
#                 ),
#             zaxis=dict(
#                 title="z [μm]",
#                 showgrid=True, gridcolor="lightgray", zeroline=False,
#                 showline=True, linecolor="black", mirror=True
#                 ),
#             # set background box color
#             xaxis_showbackground=True, xaxis_backgroundcolor="white",
#             yaxis_showbackground=True, yaxis_backgroundcolor="white",
#             zaxis_showbackground=True, zaxis_backgroundcolor="white",
#             camera=dict(eye=dict(x=0, y=-2.0, z=0.75))
#             )   
#     )
#     fig.show()

def show_iso(X, Y, Z, V, title, colorscale, *,
             keep_halfspace=True, vmin_frac=0.25, vmax_frac=0.9,
             surface_count=10, show_z_plane_cuts=True):
    """
    Show a 3D isosurface volume and (optionally) dashed intersection lines
    where the isosurfaces cut the z-min and z-max planes.

    Dashed lines use the same solid RGB colors as the isosurfaces.
    """
    V = V.astype(np.float32)

    # mask halfspace by moving values outside below isomin
    if keep_halfspace:
        mask = (Y < 0)
        vmax = np.nanmax(V[~mask]) if np.isfinite(V[~mask]).any() else None
        if vmax is None or vmax <= 0:
            print(f"[warn] '{title}': no finite data in kept region.")
            return
        Vn = V / vmax
        Vn[mask] = -1.0  # sentinel below isomin
    else:
        vmax = np.nanmax(V) if np.isfinite(V).any() else None
        if vmax is None or vmax <= 0:
            print(f"[warn] '{title}': no finite data.")
            return
        Vn = V / vmax

    # Main isosurface: force color mapping to [vmin_frac, vmax_frac]
    fig = go.Figure(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=Vn.ravel(),
        isomin=float(vmin_frac), isomax=float(vmax_frac),
        cmin=float(vmin_frac), cmax=float(vmax_frac),  # << crucial
        opacity=0.2,
        surface_count=int(surface_count),
        colorscale=[(t, c) for (t, c) in colorscale],
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
    ))

    # --------------------------------------------------
    # Dashed intersection curves on z-min / z-max planes
    # --------------------------------------------------
    if show_z_plane_cuts:
        z_axis = np.unique(Z[0, 0, :])
        zmin, zmax = float(z_axis[0]), float(z_axis[-1])

        idx_min = int(np.argmin(np.abs(z_axis - zmin)))
        idx_max = int(np.argmin(np.abs(z_axis - zmax)))

        levels = np.linspace(vmin_frac, vmax_frac, int(surface_count))

        def add_plane_cuts(k, plane_label):
            z_plane = z_axis[k]
            Xz = X[:, :, k]
            Yz = Y[:, :, k]
            Vz = Vn[:, :, k]

            # Use matplotlib contour to get iso-lines
            fig_tmp, ax_tmp = plt.subplots()
            for lvl in levels:
                # skip levels outside this plane’s range
                if not (np.nanmin(Vz) <= lvl <= np.nanmax(Vz)):
                    continue

                cs = ax_tmp.contour(Xz, Yz, Vz, levels=[lvl])

                # exact same color mapping as the isosurface
                color = get_iso_color(lvl, vmin_frac, vmax_frac, colorscale)

                for coll in cs.collections:
                    for path in coll.get_paths():
                        verts = path.vertices
                        if verts.shape[0] < 2:
                            continue
                        xs = verts[:, 0]
                        ys = verts[:, 1]
                        zs = np.full_like(xs, z_plane, dtype=float)

                        fig.add_trace(go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode="lines",
                            line=dict(color=color, width=3, dash="dash"),
                            showlegend=False,
                            name=f"{plane_label} (iso={lvl:.2f})"
                        ))
            plt.close(fig_tmp)

        add_plane_cuts(idx_min, "z_min")
        add_plane_cuts(idx_max, "z_max")

    fig.update_layout(
        title=title, template="plotly_white",
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                title="x [μm]",
                showgrid=True, gridcolor="lightgray", zeroline=False,
                showline=True, linecolor="black", mirror=True
            ),
            yaxis=dict(
                title="y [μm]",
                showgrid=True, gridcolor="lightgray", zeroline=False,
                showline=True, linecolor="black", mirror=True
            ),
            zaxis=dict(
                title="z [μm]",
                showgrid=True, gridcolor="lightgray", zeroline=False,
                showline=True, linecolor="black", mirror=True
            ),
            xaxis_showbackground=True, xaxis_backgroundcolor="white",
            yaxis_showbackground=True, yaxis_backgroundcolor="white",
            zaxis_showbackground=True, zaxis_backgroundcolor="white",
            camera=dict(eye=dict(x=0, y=-2.0, z=0.75))
        )
    )
    fig.show()



# =============================
# Configuration
# =============================
HERE = Path(__file__).resolve().parent
db = MetalsDB(str(HERE / "metals.mat"))

# Materials
n_glass, n_silica, n_water = 1.52, 1.46, 1.333

# Objective/beam
NA, fd, over, maxm = 1.45, 1.8e3, 5e3, 2
resolution = (100, 100)  # (Nrho, Nz)

# Wavelengths (µm)
lamex_um, lamem_um = 0.645, 0.690
n_gold_ex = db.get_index(20, lamex_um*1e3)  # Au @ excitation
n_gold_em = db.get_index(20, lamem_um*1e3)  # Au @ emission

# MIET geometry (µm)
t_gold_um, t_spacer_um, d_dip_um = 0.010, 0.005, 0.30

# Grids (ρ and z in µm); z=0 at silica|water interface (focus here)
rho_min, rho_max = 0.0, 0.25
z_min,   z_max   = 0.0, 0.1
focpos = 0.0

# Excitation stack
n0_exc = [n_glass, n_gold_ex, n_silica]; d0_exc = [t_gold_um, t_spacer_um]
n_exc = n_water; n1_exc = [n_water]; d1_exc = []

# Emission (MIET) stack
n0_em = [n_glass, n_gold_em, n_silica]; d0_em = [t_gold_um, t_spacer_um]
n_em = n_water; n1_em = [n_water]; d1_em = []

QY_bulk = 0.9


# =============================
# Compute harmonics (gauss_exc)
# =============================
# Linear
fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho, z = gauss_exc(
    rhofield=[rho_min, rho_max], zfield=[z_min, z_max],
    NA=NA, fd=fd, n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring=None, maxm=maxm
)
# Radial (cosψ) + π/2 rotation, coherent sum
fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, _, _ = gauss_exc(
    rhofield=[rho_min, rho_max], zfield=[z_min, z_max],
    NA=NA, fd=fd, n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring='np.cos(psi)', maxm=maxm
)
fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r = RotateEMField(fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, np.pi/2)
fxc_rad, fxs_rad = fxc_c + fxc_r, fxs_c + fxs_r
fyc_rad, fys_rad = fyc_c + fyc_r, fys_c + fys_r
fzc_rad, fzs_rad = fzc_c + fzc_r, fzs_c + fzs_r

# === sanitize harmonics at rho=0 before building splines ===
fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin = sanitize_harmonics_for_spline(
    fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho
)
fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad = sanitize_harmonics_for_spline(
    fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, rho
)


# =============================
# 2D φ=0 maps (fast, uses DC+harmonics directly on (ρ,z))
# =============================
def reconstruct_fields(fxc, fxs, fyc, fys, fzc, fzs, phi):
    Ex = fxc[..., 0].copy(); Ey = fyc[..., 0].copy(); Ez = fzc[..., 0].copy()
    M = fxc.shape[2] - 1
    if M > 0:
        for j in range(1, M+1):
            c = np.cos(j * phi); s = np.sin(j * phi)
            Ex += fxc[..., j] * c + (fxs[..., j-1] if fxs is not None and fxs.size else 0) * s
            Ey += fyc[..., j] * c + (fys[..., j-1] if fys is not None and fys.size else 0) * s
            Ez += fzc[..., j] * c + (fzs[..., j-1] if fzs is not None and fzs.size else 0) * s
    return Ex, Ey, Ez

phi_sample = 0.0
Ex_lin, Ey_lin, Ez_lin = reconstruct_fields(fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, phi_sample)
Ex_rad, Ey_rad, Ez_rad = reconstruct_fields(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, phi_sample)

Ix_lin, Iy_lin, Iz_lin = np.abs(Ex_lin)**2, np.abs(Ey_lin)**2, np.abs(Ez_lin)**2
It_lin = Ix_lin + Iy_lin + Iz_lin
Ix_rad, Iy_rad, Iz_rad = np.abs(Ex_rad)**2, np.abs(Ey_rad)**2, np.abs(Ez_rad)**2
It_rad = Ix_rad + Iy_rad + Iz_rad

# MIET brightness factors vs z
z_vec = z[0, :]
bp, bv, bf = brightness_miet_vs_z(z_vec, n0_em, n_em, n1_em, d0_em, d_dip_um, d1_em, lamem_um, NA, QY_bulk)
bp_2d = np.ones_like(Ix_lin) * bp[None, :]
bv_2d = np.ones_like(Ix_lin) * bv[None, :]
bf_2d = np.ones_like(Ix_lin) * bf[None, :]

# Effective rates (2D)
R_par_lin = Ix_lin * bp_2d
R_par_rad = Ix_rad * bp_2d
R_ver_lin = Iz_lin * bv_2d
R_ver_rad = Iz_rad * bv_2d
R_rnd_lin = (It_lin / 3.0) * bf_2d
R_rnd_rad = (It_rad / 3.0) * bf_2d

# Show 2D maps (±x symmetry)
plt.figure(figsize=(14, 8))
ax = plt.subplot(2,3,1); imshow_rz_sym(ax, rho, z, R_par_lin, 'Linear — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,2); imshow_rz_sym(ax, rho, z, R_ver_lin, 'Linear — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,3); imshow_rz_sym(ax, rho, z, R_rnd_lin, 'Linear — Random    ((|E|²/3) · bf)')
ax = plt.subplot(2,3,4); imshow_rz_sym(ax, rho, z, R_par_rad, 'Radial — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,5); imshow_rz_sym(ax, rho, z, R_ver_rad, 'Radial — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,6); imshow_rz_sym(ax, rho, z, R_rnd_rad, 'Radial — Random    ((|E|²/3) · bf)')
plt.tight_layout(); plt.show()


# =============================
# 3D volumes via cubic (rho,z) splines → Cartesian mgrid
# =============================
# Cartesian grid for 3D (adjust for resolution / speed)
Nx, Ny, Nz_cart = 96, 96, 80
rmax = float(rho[-1, 0])
x = np.linspace(-rmax, rmax, Nx)
y = np.linspace(-rmax, rmax, Ny)
z_cart = np.linspace(z[0, 0], z[0, -1], Nz_cart)
Xg, Yg, Zg = np.meshgrid(x, y, z_cart, indexing='ij')

# Build fields on Cartesian grid (cubic splines), then intensities
ExL, EyL, EzL = fields_on_xyz_cubic(fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho, z, Xg, Yg, Zg)
ExR, EyR, EzR = fields_on_xyz_cubic(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, rho, z, Xg, Yg, Zg)

IxL, IyL, IzL = np.abs(ExL)**2, np.abs(EyL)**2, np.abs(EzL)**2
IL = IxL + IyL + IzL
IxR, IyR, IzR = np.abs(ExR)**2, np.abs(EyR)**2, np.abs(EzR)**2
IR = IxR + IyR + IzR

# Interpolate brightness factors from native z to Cartesian z
bp_w = interp1d(z_vec, bp, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]
bv_w = interp1d(z_vec, bv, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]
bf_w = interp1d(z_vec, bf, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]

# Emission volumes (Cartesian)
Rpar_L = IxL * bp_w
Rver_L = IzL * bv_w
Rrnd_L = (IL / 3.0) * bf_w

Rpar_R = IxR * bp_w
Rver_R = IzR * bv_w
Rrnd_R = (IR / 3.0) * bf_w

# ---- one Plotly window per volume (half-space y>=0) ----
show_iso(Xg, Yg, Zg, Rpar_L, "Linear — Parallel",  PLOTLY_SCALE, keep_halfspace=False)
show_iso(Xg, Yg, Zg, Rver_L, "Linear — Vertical",  PLOTLY_SCALE, keep_halfspace=False)
show_iso(Xg, Yg, Zg, Rrnd_L, "Linear — Random",    PLOTLY_SCALE, keep_halfspace=False)

# show_iso(Xg, Yg, Zg, Rpar_R, "Radial — Parallel",  PLOTLY_SCALE, keep_halfspace=False)
# show_iso(Xg, Yg, Zg, Rver_R, "Radial — Vertical",  PLOTLY_SCALE, keep_halfspace=False)
# show_iso(Xg, Yg, Zg, Rrnd_R, "Radial — Random",    PLOTLY_SCALE, keep_halfspace=False)

# =============================
# Glass substrate calculations 
# =============================


# Excitation stack
n0_exc = [n_glass]; d0_exc = []
n_exc = n_water; n1_exc = [n_water]; d1_exc = []

# Emission (MIET) stack
n0_em = [n_glass]; d0_em = []
n_em = n_water; n1_em = [n_water]; d1_em = []


# =============================
# Compute harmonics (gauss_exc)
# =============================
# Linear
fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho, z = gauss_exc(
    rhofield=[rho_min, rho_max], zfield=[z_min, z_max],
    NA=NA, fd=fd, n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring=None, maxm=maxm
)
# Radial (cosψ) + π/2 rotation, coherent sum
fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, _, _ = gauss_exc(
    rhofield=[rho_min, rho_max], zfield=[z_min, z_max],
    NA=NA, fd=fd, n0=n0_exc, n=n_exc, n1=n1_exc,
    d0=d0_exc, d=d_dip_um, d1=d1_exc,
    lamex=lamex_um, over=over, focpos=focpos,
    atf=None, resolution=resolution, ring='np.cos(psi)', maxm=maxm
)
fxc_r, fxs_r, fyc_r, fys_r, fzc_r, fzs_r = RotateEMField(fxc_c, fxs_c, fyc_c, fys_c, fzc_c, fzs_c, np.pi/2)
fxc_rad, fxs_rad = fxc_c + fxc_r, fxs_c + fxs_r
fyc_rad, fys_rad = fyc_c + fyc_r, fys_c + fys_r
fzc_rad, fzs_rad = fzc_c + fzc_r, fzs_c + fzs_r

# === sanitize harmonics at rho=0 before building splines ===
fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin = sanitize_harmonics_for_spline(
    fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho
)
fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad = sanitize_harmonics_for_spline(
    fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, rho
)


# =============================
# 2D φ=0 maps (fast, uses DC+harmonics directly on (ρ,z))
# =============================
def reconstruct_fields(fxc, fxs, fyc, fys, fzc, fzs, phi):
    Ex = fxc[..., 0].copy(); Ey = fyc[..., 0].copy(); Ez = fzc[..., 0].copy()
    M = fxc.shape[2] - 1
    if M > 0:
        for j in range(1, M+1):
            c = np.cos(j * phi); s = np.sin(j * phi)
            Ex += fxc[..., j] * c + (fxs[..., j-1] if fxs is not None and fxs.size else 0) * s
            Ey += fyc[..., j] * c + (fys[..., j-1] if fys is not None and fys.size else 0) * s
            Ez += fzc[..., j] * c + (fzs[..., j-1] if fzs is not None and fzs.size else 0) * s
    return Ex, Ey, Ez

phi_sample = 0.0
Ex_lin, Ey_lin, Ez_lin = reconstruct_fields(fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, phi_sample)
Ex_rad, Ey_rad, Ez_rad = reconstruct_fields(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, phi_sample)

Ix_lin, Iy_lin, Iz_lin = np.abs(Ex_lin)**2, np.abs(Ey_lin)**2, np.abs(Ez_lin)**2
It_lin = Ix_lin + Iy_lin + Iz_lin
Ix_rad, Iy_rad, Iz_rad = np.abs(Ex_rad)**2, np.abs(Ey_rad)**2, np.abs(Ez_rad)**2
It_rad = Ix_rad + Iy_rad + Iz_rad

# MIET brightness factors vs z
z_vec = z[0, :]
bp, bv, bf = brightness_miet_vs_z(z_vec, n0_em, n_em, n1_em, d0_em, d_dip_um, d1_em, lamem_um, NA, QY_bulk)
bp_2d = np.ones_like(Ix_lin) * bp[None, :]
bv_2d = np.ones_like(Ix_lin) * bv[None, :]
bf_2d = np.ones_like(Ix_lin) * bf[None, :]

# Effective rates (2D)
R_par_lin = Ix_lin * bp_2d
R_par_rad = Ix_rad * bp_2d
R_ver_lin = Iz_lin * bv_2d
R_ver_rad = Iz_rad * bv_2d
R_rnd_lin = (It_lin / 3.0) * bf_2d
R_rnd_rad = (It_rad / 3.0) * bf_2d

# Show 2D maps (±x symmetry)
plt.figure(figsize=(14, 8))
ax = plt.subplot(2,3,1); imshow_rz_sym(ax, rho, z, R_par_lin, 'Linear — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,2); imshow_rz_sym(ax, rho, z, R_ver_lin, 'Linear — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,3); imshow_rz_sym(ax, rho, z, R_rnd_lin, 'Linear — Random    ((|E|²/3) · bf)')
ax = plt.subplot(2,3,4); imshow_rz_sym(ax, rho, z, R_par_rad, 'Radial — Parallel  (|Ex|² · bp)')
ax = plt.subplot(2,3,5); imshow_rz_sym(ax, rho, z, R_ver_rad, 'Radial — Vertical  (|Ez|² · bv)')
ax = plt.subplot(2,3,6); imshow_rz_sym(ax, rho, z, R_rnd_rad, 'Radial — Random    ((|E|²/3) · bf)')
plt.tight_layout(); plt.show()


# =============================
# 3D volumes via cubic (rho,z) splines → Cartesian mgrid
# =============================
# Cartesian grid for 3D (adjust for resolution / speed)
Nx, Ny, Nz_cart = 96, 96, 80
rmax = float(rho[-1, 0])
x = np.linspace(-rmax, rmax, Nx)
y = np.linspace(-rmax, rmax, Ny)
z_cart = np.linspace(z[0, 0], z[0, -1], Nz_cart)
Xg, Yg, Zg = np.meshgrid(x, y, z_cart, indexing='ij')

# Build fields on Cartesian grid (cubic splines), then intensities
ExL, EyL, EzL = fields_on_xyz_cubic(fxc_lin, fxs_lin, fyc_lin, fys_lin, fzc_lin, fzs_lin, rho, z, Xg, Yg, Zg)
ExR, EyR, EzR = fields_on_xyz_cubic(fxc_rad, fxs_rad, fyc_rad, fys_rad, fzc_rad, fzs_rad, rho, z, Xg, Yg, Zg)

IxL, IyL, IzL = np.abs(ExL)**2, np.abs(EyL)**2, np.abs(EzL)**2
IL = IxL + IyL + IzL
IxR, IyR, IzR = np.abs(ExR)**2, np.abs(EyR)**2, np.abs(EzR)**2
IR = IxR + IyR + IzR

# Interpolate brightness factors from native z to Cartesian z
bp_w = interp1d(z_vec, bp, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]
bv_w = interp1d(z_vec, bv, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]
bf_w = interp1d(z_vec, bf, kind='cubic', bounds_error=False, fill_value="extrapolate")(z_cart)[None, None, :]

# Emission volumes (Cartesian)
Rpar_L = IxL * bp_w
Rver_L = IzL * bv_w
Rrnd_L = (IL / 3.0) * bf_w

Rpar_R = IxR * bp_w
Rver_R = IzR * bv_w
Rrnd_R = (IR / 3.0) * bf_w

# ---- one Plotly window per volume (half-space y>=0) ----
show_iso(Xg, Yg, Zg, Rpar_L, "Linear — Parallel",  PLOTLY_SCALE, keep_halfspace=False)
show_iso(Xg, Yg, Zg, Rver_L, "Linear — Vertical",  PLOTLY_SCALE, keep_halfspace=False)
show_iso(Xg, Yg, Zg, Rrnd_L, "Linear — Random",    PLOTLY_SCALE, keep_halfspace=False)

#%%
