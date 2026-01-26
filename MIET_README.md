# MIET (Metal-Induced Energy Transfer) Analysis Pipeline

This repository provides a comprehensive Python implementation for MIET (Metal-Induced Energy Transfer) analysis, enabling quantitative determination of molecular heights from fluorescence lifetime measurements.

## Overview

MIET is a powerful technique that exploits the distance-dependent interaction between fluorescent molecules and metal surfaces to determine molecular heights with nanometer precision. This implementation provides tools for:

- **MIET calibration curve generation** - Calculate theoretical lifetime vs. height relationships
- **Brightness enhancement modeling** - Account for metal-induced fluorescence enhancement
- **Multi-layer substrate support** - Handle complex optical structures (glass/metal/polymer stacks)
- **Multiple fluorophore orientations** - Random, vertical, and horizontal dipole orientations

## Core Modules

### [`MIET_main.py`](MIET_main.py)
The main MIET calculation engine containing:

- **`miet_calc()`** - Calculate MIET lifetime calibration curves
- **`brightness_dipole()`** - Compute brightness enhancement factors
- **`MetalsDB`** - Database of metal optical properties
- **Fresnel coefficients** and electromagnetic field calculations

### [`MIET_FLIM_Analysis.py`](MIET_FLIM_Analysis.py)
Complete MIET-FLIM analysis pipeline:

- **`MIETConfig`** - Configuration class for analysis parameters
- **`analyze_miet_flim_data()`** - Full analysis pipeline for PTU files
- **`create_miet_calibration()`** - Generate MIET calibration curves
- **Visualization and export** functions

### [`MIET_examples.py`](MIET_examples.py)
Comprehensive examples demonstrating:

- Single and multi-wavelength MIET calculations
- Different substrate configurations
- Orientation-dependent measurements
- Brightness vs. lifetime analysis

## Quick Start

### Basic MIET Analysis

```python
from MIET_FLIM_Analysis import MIETConfig, analyze_miet_flim_data

# Create configuration
config = MIETConfig()
config.wavelength_nm = 690.0  # Emission wavelength
config.quantum_yield = 0.6    # Fluorophore quantum yield
config.tau_free = 2.5         # Free-space lifetime (ns)

# Analyze PTU file
results = analyze_miet_flim_data("data.ptu", config)

# Access results
height_map = results['height_map_nm']  # Height in nanometers
lifetime_map = results['lifetime_map_ns']  # Measured lifetimes
```

### Custom Substrate Configuration

```python
# Glass-gold-polymer substrate
config = MIETConfig()
config.substrate_layers = [1.52, 20, 1.46]  # glass, gold (code), polymer
config.layer_thicknesses_nm = [80, 20, 100]  # Ti, Au, Ti thicknesses
config.spacer_thickness_nm = 5  # Distance from metal to fluorophore
```

### Generate MIET Calibration

```python
from MIET_main import miet_calc, MetalsDB

# Load metal database
metals_db = MetalsDB("metals.mat")

# Calculate calibration curve
z_nm, lifetime_ns = miet_calc(
    al_res=100,           # Angular resolution
    lam_nm=690.0,         # Wavelength
    n0=[1.52, n_gold, 1.46],  # Layer refractive indices
    n=1.33,               # Medium RI
    n1=1.33,              # Top medium RI
    d0_nm=[80, 20],       # Layer thicknesses
    d_nm=5,               # Spacer thickness
    d1_nm=[],             # Top layers
    qy=0.6,               # Quantum yield
    tau_free_ns=2.5,      # Free lifetime
    pol=1,                # Random orientation
    curveType=2           # Curve type
)
```

## Substrate Configurations

The pipeline supports various substrate configurations:

### Standard Glass-Gold-Polymer
```python
config.substrate_layers = [1.52, 20, 1.46]  # glass, gold, polymer
config.layer_thicknesses_nm = [80, 20, 80]  # Ti/Au/Ti stack
```

### Thick Gold Layer
```python
config.substrate_layers = [1.52, 20]        # glass, gold
config.layer_thicknesses_nm = [80, 50]      # Ti/thick Au
```

### Silver Substrate
```python
config.substrate_layers = [1.52, 10]        # glass, silver
config.layer_thicknesses_nm = [10]          # thin silver
```

### No Metal Control
```python
config.substrate_layers = [1.52, 1.46]      # glass, polymer only
config.layer_thicknesses_nm = []            # no metal layers
```

## Metal Database

The system uses a MATLAB `.mat` file containing wavelength-dependent refractive indices:

```python
from MIET_main import MetalsDB

metals_db = MetalsDB("metals.mat")

# Get refractive index at specific wavelength
n_gold = metals_db.get_index(20, 690)  # Code 20 = gold at 690 nm
n_silver = metals_db.get_index(10, 690)  # Code 10 = silver at 690 nm
```

### Supported Metals
- **10**: Silver
- **20**: Gold  
- **22**: Titanium
- **30**: Platinum
- **40**: Palladium
- **50**: Copper
- **60**: Aluminum
- **70**: Chromium
- **80**: Titanium (alternate)
- **90**: Tungsten

## Fluorophore Orientations

The pipeline supports different dipole orientations:

```python
config.orientation = 'random'     # Isotropic orientation (default)
config.orientation = 'vertical'   # Vertical dipoles
config.orientation = 'horizontal' # Horizontal dipoles
```

## Output Data

MIET analysis provides comprehensive results:

```python
results = {
    'height_map_nm': ...,          # 2D height map
    'lifetime_map_ns': ...,        # 2D lifetime map
    'intensity_map': ...,          # 2D intensity map
    'miet_calibration': {          # Calibration data
        'z_nm': ...,               # Height points
        'lifetime_ns': ...,        # Theoretical lifetimes
        'brightness': ...          # Enhancement factors
    },
    'flim_results': ...,           # Raw FLIM fitting results
    'statistics': ...              # Analysis statistics
}
```

## Visualization

Generate comprehensive plots:

```python
from MIET_FLIM_Analysis import visualize_miet_flim_results

visualize_miet_flim_results(results, 'analysis_summary.png')
```

This creates a multi-panel figure showing:
- MIET calibration curve
- Height and lifetime maps
- Statistical distributions
- Lifetime vs. height correlation

## Dependencies

- **numpy** - Numerical computing
- **scipy** - Scientific computing and optimization
- **matplotlib** - Plotting and visualization
- **pandas** - Data handling and export
- **pickle** - Results serialization

## Advanced Features

### Multi-Component Fitting
```python
config.tau0 = np.array([0.5, 2.5])  # Two-component lifetime
config.fit_components = 2
```

### High-Resolution Analysis
```python
config.flag_win = False    # Pixel-by-pixel analysis
config.resolution_ns = 0.1 # Fine time resolution
```

### Batch Processing
```python
from MIET_FLIM_Analysis import process_miet_flim_batch

file_list = ["file1.ptu", "file2.ptu", "file3.ptu"]
batch_results = process_miet_flim_batch(file_list, config)
```

## References

1. Chizhik, A. I. et al. Metal-induced energy transfer for live cell nanoscopy. *Nat. Photonics* **8**, 124–129 (2014).
2. Baronsky, T. et al. Cell–substrate dynamics of the epithelial-to-mesenchymal transition. *Nano Lett.* **17**, 3320–3326 (2017).
3. Karedla, N. et al. Metal-induced energy transfer. *Phys. Rev. Lett.* **115**, 173002 (2015).

## Contact

For questions and support, please refer to the main repository documentation or contact the development team.