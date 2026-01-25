# MIET-FLIM Analysis Pipeline

This repository provides a complete pipeline for analyzing PTU FLIM (Fluorescence Lifetime Imaging Microscopy) data and determining molecular heights using MIET (Metal-Induced Energy Transfer) calculations.

## Overview

The pipeline combines:
1. **FLIM Analysis**: Processing PTU files to extract fluorescence lifetime maps
2. **MIET Calibration**: Theoretical calculation of lifetime vs. height curves
3. **Height Determination**: Converting measured lifetimes to molecular heights

## Key Features

- **Complete PTU Processing**: Reads raw PTU files from PicoQuant TCSPC systems
- **Advanced FLIM Fitting**: Multi-exponential decay analysis with IRF deconvolution
- **MIET Theory**: Full electromagnetic modeling of metal-induced energy transfer
- **Height Mapping**: Pixel-by-pixel conversion from lifetime to height
- **Batch Processing**: Automated analysis of multiple files
- **Comprehensive Output**: TIFF images, CSV statistics, and publication-ready plots

## Installation

### Required Dependencies

```bash
# Core scientific computing
pip install numpy scipy matplotlib pandas tifffile

# Image processing and progress bars
pip install scikit-image tqdm

# Optional: GPU acceleration (recommended for large datasets)
pip install scikit-learn

# Optional: Enhanced interpolation
pip install scipy
```

### File Structure
```
MIET_calc/
├── MIET_FLIM_Analysis.py      # Main analysis pipeline
├── PTU_ScanRead.py            # PTU file reader
├── FLIM_fitter.py             # FLIM analysis functions
├── FlavMetaFLIM.py            # FLIM processing utilities
├── MIET_main.py               # MIET calculations
├── metals.mat                 # Metal optical constants database
└── examples/                  # Example scripts
```

## Quick Start

### Basic Single File Analysis

```python
from MIET_FLIM_Analysis import MIETConfig, analyze_miet_flim_data, visualize_miet_flim_results

# Create configuration
config = MIETConfig()

# Analyze single PTU file
results = analyze_miet_flim_data(
    ptu_file="your_data.ptu",
    config=config,
    metals_db_path="metals.mat",
    orientation='random',
    save_results=True
)

# Visualize results
if results:
    visualize_miet_flim_results(results, "analysis_results.png")
```

### Batch Processing

```python
from MIET_FLIM_Analysis import process_miet_flim_batch, create_batch_summary
import glob

# Find all PTU files
ptu_files = glob.glob("data_folder/*.ptu")

# Process batch
batch_results = process_miet_flim_batch(
    file_list=ptu_files,
    config=config,
    metals_db_path="metals.mat",
    orientation='random'
)

# Create summary
create_batch_summary(batch_results, "batch_summary.csv")
```

## Configuration

The `MIETConfig` class controls all analysis parameters:

```python
config = MIETConfig()

# Substrate parameters
config.wavelength_nm = 690.0       # Emission wavelength
config.n_glass = 1.52              # Glass substrate index
config.n_polymer = 1.46            # Sample medium index  
config.n_water = 1.33              # Top medium index

# Metal stack (material codes from metals.mat)
config.metal_stack = [80, 20, 80]  # Ti, Au, Ti
config.metal_thickness_nm = [2, 10, 1]  # nm

# Sample layer
config.polymer_thickness_nm = 50.0  # nm
config.quantum_yield = 0.6          # Quantum yield
config.tau_free_ns = 4.0           # Free-space lifetime

# FLIM analysis
config.tau0 = np.array([0.3, 1.7, 6.0])  # Initial lifetime guesses
config.resolution_ns = 0.2               # Time resolution
config.win_size = 8                      # Spatial binning window
```

## Substrate Configurations

### Standard Glass-Gold-Polymer Stack
```python
config = MIETConfig()
# Default configuration is glass + Ti/Au/Ti + polymer + water
```

### Custom Metal Stack
```python
config = MIETConfig()
config.metal_stack = [10, 20]  # Silver + Gold (see metals.mat for codes)
config.metal_thickness_nm = [5, 15]
```

### No Metal (Control)
```python
config = MIETConfig()
config.metal_stack = []
config.metal_thickness_nm = []
# Just glass + polymer + water
```

## Output Files

For each analyzed PTU file, the pipeline generates:

- `*_height_map_nm.tif` - Height map (nanometers)
- `*_MIET_FLIM_results.pkl` - Complete results (Python pickle)
- `*_height_statistics.csv` - Summary statistics
- `*_MIET_calibration.csv` - Calibration curve data
- `*_FLIM_results.pkl` - Detailed FLIM fitting results

## Analysis Workflow

### 1. MIET Calibration
The pipeline first calculates a theoretical calibration curve relating fluorescence lifetime to distance from the metal surface using electromagnetic theory.

### 2. PTU Data Processing
- Reads raw photon arrival times from PTU files
- Builds TCSPC histograms for each pixel/region
- Performs background subtraction and temporal binning

### 3. FLIM Analysis
- Calculates instrumental response function (IRF)
- Fits multi-exponential decay models
- Extracts lifetime maps with statistical analysis

### 4. Height Calculation
- Applies intensity thresholding to avoid noise
- Interpolates measured lifetimes using MIET calibration
- Generates height maps with uncertainty estimates

## Advanced Features

### Dipole Orientation
The analysis supports different molecular orientation models:

```python
# Random orientation (isotropic)
results = analyze_miet_flim_data(..., orientation='random')

# Vertical orientation (perpendicular to surface)
results = analyze_miet_flim_data(..., orientation='vertical')

# Horizontal orientation (parallel to surface)  
results = analyze_miet_flim_data(..., orientation='horizontal')
```

### Custom Fitting Models
```python
config = MIETConfig()

# Two-component fit
config.tau0 = np.array([0.5, 2.5])
config.lifetime_bounds = [0.1, 1.0, 1.0, 5.0]

# Three-component fit (default)
config.tau0 = np.array([0.3, 1.7, 6.0])
config.lifetime_bounds = [0.1, 0.5, 2.0, 2.0, 4.0, 10.0]
```

### Spatial Binning Options
```python
config = MIETConfig()

# Pixel-by-pixel analysis (high resolution, slower)
config.flag_win = False

# Windowed analysis (lower resolution, faster)
config.flag_win = True
config.win_size = 8    # 8x8 pixel windows
config.step = 2        # 2-pixel step size
```

## Troubleshooting

### Common Issues

1. **"No photons found"** - Check detector channel (`config.auto_det`)
2. **"IRF calculation failed"** - May need custom IRF file
3. **"Height interpolation out of bounds"** - Adjust MIET parameters
4. **Memory errors** - Use windowed analysis or reduce file size

### Performance Optimization

- Use windowed analysis for large images (`config.flag_win = True`)
- Enable GPU acceleration if available
- Process files in smaller batches for large datasets

### Quality Control

Monitor these metrics for good results:
- Photon counts: >100 per pixel/window
- Lifetime fitting χ² < 2.0
- MIET calibration covers measured lifetime range

## Example Data Analysis

### Membrane Distance Measurements
```python
# Configuration for membrane labeling
config = MIETConfig()
config.polymer_thickness_nm = 20.0  # Thin sample layer
config.metal_stack = [80, 20]       # Ti/Au bilayer
config.tau0 = np.array([0.5, 2.0])  # Two-component model

results = analyze_miet_flim_data("membrane_sample.ptu", config)
```

### Protein Height Mapping
```python
# Configuration for protein complexes
config = MIETConfig()
config.wavelength_nm = 680.0        # Different fluorophore
config.quantum_yield = 0.4          # Lower QY protein
config.tau_free_ns = 3.5           # Different free lifetime

results = analyze_miet_flim_data("protein_sample.ptu", config)
```

## References

1. Chizhik, A. et al. "Metal-induced energy transfer for live cell nanoscopy." Nature Photonics (2014)
2. Baronsky, T. et al. "Cell-substrate dynamics of the epithelial-to-mesenchymal transition." Nano Letters (2017)
3. Karedla, N. et al. "Single-molecule metal-induced energy transfer (smMIET)." ChemPhysChem (2014)

## Citation

If you use this code in your research, please cite:
```
[Your publication details here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.