# MIET and Confocal Microscopy Simulation Package

## Overview

This Python package provides a collection of tools for simulating phenomena related to fluorescence microscopy, specifically focusing on Metal-Induced Energy Transfer (MIET) and the optics of confocal microscopy. It is a Python port of a MATLAB-based calculations, designed for researchers and students in optics and biophysics. A majority of the functions were written in Matlab originally by Joerg Enderlein, University of Goettingen (https://www.joerg-enderlein.de/).

The code can calculate fluorescence lifetime and brightness of dipoles near layered metallic or dielectric structures and simulate Point Spread Functions (PSFs) using vectorial and scalar models.

## Features

*   **Metal-Induced Energy Transfer (MIET):**
    *   Calculate fluorescence lifetime (`miet_calc`) and brightness (`brightness_dipole`) for emitters near multi-layered substrates.
    *   Supports metallic layers by loading complex refractive index data from a `.mat` file.
    *   Calculates the angular distribution of radiation for dipoles in stratified media.

*   **Confocal Microscopy Optics:**
    *   Vectorial PSF simulation based on the Richards-Wolf model (`gauss_exc` in `Optics_main.py`).
    *   Scalar PSF simulation, including coverslip aberration and finite pinhole effects (`Scalar_Confocal.py`).
    *   Generation of defocused patterns and single-emitter PSFs (`PatternGeneration`).

## Core Modules

*   `MIET_main.py`: Contains the core physics functions for MIET calculations, including Fresnel coefficients, dipole radiation models, and lifetime/brightness calculations.
*   `Optics_main.py`: Implements the vectorial model for focused Gaussian beams and PSF generation in confocal microscopy.
*   `Scalar_Confocal.py`: Provides a scalar model for calculating excitation and detection PSFs, including aberrations.
*   `utilities.py`: A collection of helper functions for plotting and data visualization, including 2D and 3D rendering of simulation results.
*   `MIET_examples.py`: A script containing various examples demonstrating how to use the functions in this package.

## Key Functions

This section provides a brief overview of the most important functions in the core modules.

### `MIET_main.py`

*   [`fresnel(w1, n1, n2)`](MIET_main.py:20): Calculates the Fresnel reflection and transmission coefficients (rp, rs, tp, ts) for p- and s-polarized light. It can handle both a single interface between two media and a multi-layered stack.

*   [`dipoleL(theta, z, n0, n, n1, d0, d, d1)`](MIET_main.py:189): Computes the electromagnetic field components radiated by a dipole embedded in a stratified medium. This is a core function for determining how the environment modifies the dipole's emission.

*   [`lifetimeL(...)`](MIET_main.py:479): Calculates the normalized decay rates for a dipole emitter. It computes the contributions from propagating waves (far-field radiation), evanescent waves (near-field quenching), and guided modes. This is the fundamental calculation underlying the MIET effect.

*   [`miet_calc(...)`](MIET_main.py:1349): A high-level function to compute the fluorescence lifetime calibration curve (lifetime vs. distance `z`). It integrates the results from `lifetimeL` over the emission spectrum of a dye and considers the quantum yield and free-space lifetime.

*   [`brightness_dipole(...)`](MIET_main.py:1517): Calculates the apparent brightness of a dipole by considering both the collection efficiency (how much light is collected by the objective) and the local quantum yield. It provides results for different dipole orientations (vertical, parallel, rotating, and fixed-random).

*   [`MetalsDB(mat_path)`](MIET_main.py:798): A helper class to load and interpolate wavelength-dependent complex refractive indices for various metals from a `.mat` file.

### `Optics_main.py`

*   [`gauss_exc(...)`](Optics_main.py:35): Implements the Richards-Wolf vectorial focusing model to calculate the electromagnetic field of a tightly focused Gaussian beam in a stratified medium. It returns the field components as a series of cylindrical harmonics, which can be used to reconstruct the 3D Point Spread Function (PSF).

*   [`PulsedExcitation(...)`](Optics_main.py:388): Models the effects of fluorescence saturation under pulsed laser excitation. It calculates the effective excitation probability, which can be used to correct for saturation artifacts in fluorescence correlation spectroscopy (FCS) or lifetime measurements.

*   [`SEPDipole(...)`](Optics_main.py:1004): Calculates the image of a single point-like dipole emitter (Single Emitter PSF) as it would appear on a camera. It takes into account the layered structure, objective NA, and dipole orientation. The output is a set of radial harmonics that can be used to reconstruct the 2D image.

*   [`PatternGeneration(...)`](Optics_main.py:1500): Generates a basis set of theoretical PSF patterns for a range of different dipole orientations. This is useful for single-molecule orientation imaging techniques, where the shape of a molecule's PSF is used to determine its 3D orientation. The function can also generate **defocused patterns** by adjusting the `focus` parameter. The function signature is as follows:
    ```python
    model = PatternGeneration(
        z, NA, n0, n, n1, d0, d, d1,
        lamem, mag, focus,
        atf, ring,
        pixel, nn,
        be_res, al_res,
        pic
    )
    ```

*   [`mdf_confocal_microscopy_py(...)`](Optics_main.py:2176): A high-level function that demonstrates how to combine the excitation fields from `gauss_exc` and the detection fields to generate a Molecular Detection Function (MDF) for a confocal microscope.

## Getting Started

### Installation

To install the package and set up the environment, please follow these steps. It is recommended to use Conda for environment management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MIET-confocal-simulation.git
    cd MIET-confocal-simulation
    ```

2.  **Create the Conda environment from the `MIET_calc_env.yml` file:**
    This file contains all the necessary dependencies for the project. Run the following command in your terminal:
    ```bash
    conda env create -f MIET_calc_env.yml
    ```
    This will create a new Conda environment named `MIET_calc_env`.

3.  **Activate the Conda environment:**
    Before running the scripts, you need to activate the newly created environment:
    ```bash
    conda activate MIET_calc_env
    ```
    You should see `(MIET_calc_env)` at the beginning of your terminal prompt.

### Running the Examples

Once the environment is set up and activated, you can run the example scripts to see the package in action. The recommended IDE for this project is Spyder, which is included in the Conda environment.

1.  **Launch Spyder:**
    ```bash
    spyder
    ```

2.  **Run the `MIET_examples.py` script:**
    Inside Spyder, open the `MIET_examples.py` file and run it. This will execute various simulations and generate plots, demonstrating the core functionalities of the package.

## Dependencies

All the required dependencies for this project are listed in the `MIET_calc_env.yml` file. This includes Python 3 and libraries such as:

*   **numpy**: For numerical operations.
*   **scipy**: For scientific computing, including special functions and interpolation.
*   **matplotlib**: For 2D plotting.
*   **scikit-image**: For 3D visualization of isosurfaces (`marching_cubes`).
*   **plotly**: For interactive 3D visualizations.
*   **spyder**: The recommended IDE for running the code.

The environment can be set up using the command provided in the "Getting Started" section.

**Note on `.mat` files:** The `scipy.io.loadmat` function is used to load the `metals.mat` file. The environment file includes the necessary libraries to handle this.

## Data

*   `metals.mat`: A MATLAB data file containing wavelength-dependent complex refractive indices for various metals. This file is required for any MIET calculations involving metallic layers. It is loaded by the `MetalsDB` class in `MIET_main.py`.

## Usage Example

Here is a simple example from `MIET_examples.py` to calculate and plot a MIET lifetime curve.

```python
import numpy as np
from MIET_main import MetalsDB, miet_calc

# Load the metals database
db = MetalsDB('metals.mat')

# Define the layer stack
# Here: glass | gold (10nm) | spacer (5nm) | dipole layer (water) | top (water)
n0 = [1.52, 20, 1.46]   # 20 is the ID for 'gold' in the .mat file
d0 = [10, 5]            # thicknesses in nm
n  = 1.33               # dipole layer index
n1 = [1.33]             # top half-space
d1 = []

# Calculate and plot the lifetime curve
z, life = miet_calc(
    al_res=np.nan,          # Use np.nan for orientation averaging
    lamem=690.0,            # Emission wavelength in nm
    n0=n0, n=n, n1=n1, d0=d0, d=200.0, d1=d1,
    qyield=0.8, tau_free=3.5,
    fig=True,               # Set to True to generate a plot
    metals_db=db            # Pass the database object
)
## License
Copyright Narain Karedla 2025, GPLv3.

