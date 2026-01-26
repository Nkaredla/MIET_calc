# MIET-pFCS: Membrane Fluctuation Analysis via PTU Files

This module provides comprehensive analysis of membrane fluctuation dynamics using MIET (Metal-Induced Energy Transfer) combined with pseudo-FCS (fluorescence correlation spectroscopy) from PicoQuant PTU files.

## Overview

The MIET-pFCS pipeline analyzes time-tagged time-resolved (TTTR) photon data to study membrane fluctuation dynamics by correlating fluorescence lifetime changes with molecular height variations. This approach enables quantitative measurement of membrane dynamics with nanometer height resolution and microsecond temporal resolution.

## Key Features

- **PTU file processing** - Direct reading of PicoQuant TTTR data
- **MIET height calibration** - Convert lifetime measurements to molecular heights
- **Multiple lifetime estimators** - MLE, variance-based, and intensity-weighted methods
- **Auto-correlation analysis** - Extract fluctuation dynamics from height traces
- **Photobleaching correction** - Exponential drift correction
- **Real-time streaming** - Process large PTU files in chunks

## Module: [`MemFluc_pFCS.py`](MemFluc_pFCS.py)

### Main Function

```python
from MemFluc_pFCS import run_miet_ptu_pipeline

results = run_miet_ptu_pipeline(
    ptu_path="membrane_dynamics.ptu",
    # Fluorophore properties
    tau0=2.9,           # Free-space lifetime (ns)
    qy0=0.6,            # Quantum yield
    tau1=2.2,           # Quenched lifetime (ns)
    # Optical parameters
    lamex_um=0.640,     # Excitation wavelength (μm)
    lamem_um=0.690,     # Emission wavelength (μm)
    NA=1.49,            # Numerical aperture
    # Substrate geometry
    d_um=3e-1,          # Spacer thickness (μm)
    # Analysis parameters
    tbin_s=1e-4,        # Macro time binning (s)
    cutoff_ns=10.0,     # TCSPC gate width (ns)
    nbunches=10         # Number of correlation segments
)
```

### Input Parameters

#### Fluorophore Properties
- **`tau0`** - Free-space fluorescence lifetime (ns)
- **`qy0`** - Quantum yield in free space
- **`tau1`** - Quenched lifetime near metal (ns)

#### Optical Configuration
- **`lamex_um`** - Excitation wavelength (μm)
- **`lamem_um`** - Emission wavelength (μm)
- **`NA`** - Objective numerical aperture

#### Substrate Geometry
```python
# Layer structure (bottom to top)
glass_n = 1.52      # Glass refractive index
n1 = 1.33           # Buffer refractive index
n = 1.33            # Medium refractive index
top_n = 1.46        # Top layer refractive index

# Layer thicknesses in μm
d0_um = (2e-3, 10e-3, 1e-3, 10e-3)  # Ti, Au, Ti, spacer
d_um = 3e-1         # Fluorophore layer thickness
d1_um = ()          # Top layers (empty for standard setup)
```

#### Analysis Parameters
- **`tbin_s`** - Macro time bin size (s)
- **`cutoff_ns`** - TCSPC time gate width (ns)
- **`shift_ns`** - Gate shift from peak (ns)
- **`micro_rebin`** - Microtime rebinning factor
- **`nbunches`** - Number of segments for correlation analysis

## Analysis Workflow

### 1. MIET Calibration
The pipeline first generates theoretical calibration curves relating fluorescence lifetime to height:

```python
# Automatic calibration based on substrate parameters
z_nm, lifecurve = MIET_calc_fn(...)  # Height vs. lifetime
brightness_factors = BrightnessDipole_fn(...)  # Brightness enhancement
```

### 2. PTU Data Processing
Stream-processes large PTU files in chunks:

```python
# Read PTU file header
head = PTU_Read_Head(ptu_path)

# Stream photon data in chunks
while flag:
    sync, tcspc, chan, special, num, loc, _ = PTU_Read(ptu_path, [cnts, chunk_size], head)
    # Process chunk...
```

### 3. Lifetime Estimation
Three complementary lifetime estimation methods:

#### Intensity-Weighted Lifetime
```python
# Simple intensity-weighted average lifetime
htrace_int = interp1_nan(norm_br, z_nm, normtrace)
```

#### Variance-Based Lifetime  
```python
# Statistical variance of decay curves
Et = np.sum(counts * delay_ns, axis=0) / denom
Et2 = np.sum(counts * (delay_ns**2), axis=0) / denom
meantau = np.sqrt(Et2 - Et**2)
htrace_var = interp1_nan(lifecurve, z_nm, meantau)
```

#### Maximum Likelihood Estimation
```python
# Grid-based MLE for mono-exponential + background
grid_lts = np.linspace(0.1, 3.0, 200)  # Lifetime grid
grid_bs = np.linspace(0.0, 0.2, 60)    # Background grid
# Minimize negative log-likelihood...
htrace_mle = interp1_nan(lifecurve, z_nm, gridMLE_tau)
```

### 4. Correlation Analysis
Extract fluctuation dynamics via auto-correlation:

```python
# Height auto-correlation functions
auto_height, autotime = tttr2xfcs(timepoints, height_trace, Ncasc, Nsub)

# Intensity auto-correlation for comparison  
auto_intensity, _ = tttr2xfcs(timepoints, intensity_trace, Ncasc, Nsub)
```

## Output Data Structure

```python
results = {
    # MIET calibration
    'z_nm': z_nm,                    # Height calibration points (nm)
    'lifecurve': lifecurve,          # Theoretical lifetimes (ns)
    'br': br,                        # Brightness factors
    
    # Raw traces
    'ttrace': ttrace,                # Intensity trace (counts)
    'normtrace': normtrace,          # Normalized intensity
    'tmptcspc': tmptcspc,            # Accumulated TCSPC histogram
    'tmptau': tmptau,                # Time-resolved TCSPC matrix
    
    # Height traces
    'htrace_int': htrace_int,        # Height from intensity
    'htrace_var': htrace_var,        # Height from variance lifetime
    'htrace_mle': htrace_mle,        # Height from MLE lifetime
    
    # Correlation functions
    'auto': auto,                    # Height ACF (intensity-based)
    'auto2': auto2,                  # Height ACF (variance-based)
    'auto3': auto3,                  # Height ACF (MLE-based)
    'autoi': autoi,                  # Intensity ACF
    'autotime': autotime,            # Correlation time axis
    'tau_s': tau_s,                  # Time axis in seconds
    
    # Calibration parameters
    'z_avg_nm': z_avg,               # Average height (nm)
    'tau_avg_ns': tau_avg            # Average lifetime (ns)
}
```

## Visualization

The pipeline generates comprehensive correlation function plots:

```python
import matplotlib.pyplot as plt

# Differential correlation functions
plt.figure()
plt.semilogx(tau_s[:-1], np.mean(auto3_n[:-1,:], axis=1), label="dACF_MLE")
plt.semilogx(tau_s[:-1], np.mean(auto2_n[:-1,:], axis=1), label="dACF_var") 
plt.semilogx(tau_s[:-1], np.mean(auto_n[:-1,:], axis=1), label="dACF_b(h)")
plt.semilogx(tau_s[:-1], np.mean(autoi_n[:-1,:], axis=1)*bhmean, label="ACF (scaled)")
plt.xlabel("t / (s)")
plt.ylabel("g(t)")
plt.legend()

# Normalized correlation functions
plt.figure()
plt.semilogx(tau_s, np.mean(auto3/auto3[0:1,:], axis=1), label="dACF_MLE")
plt.semilogx(tau_s, np.mean(auto2/auto2[0:1,:], axis=1), label="dACF_var")
plt.semilogx(tau_s, np.mean(auto/auto[0:1,:], axis=1), label="dACF_b(h)")
plt.semilogx(tau_s, np.mean(autoi/autoi[0:1,:], axis=1), label="ACF")
plt.xlabel("t / (s)")
plt.ylabel("g(t) / g(0)")
plt.legend()
```

## Example Usage

### Basic Membrane Fluctuation Analysis

```python
from MemFluc_pFCS import run_miet_ptu_pipeline

# Analyze membrane dynamics on gold substrate
results = run_miet_ptu_pipeline(
    "membrane_fluctuations.ptu",
    tau0=2.9,           # Alexa647-like dye
    lamem_um=0.690,     # 690 nm emission
    d_um=0.005,         # 5 nm spacer layer
    tbin_s=100e-6,      # 100 μs time bins
    nbunches=20         # 20 correlation segments
)

# Extract height fluctuation amplitude
height_trace = results['htrace_mle']
fluctuation_amplitude = np.std(height_trace[np.isfinite(height_trace)])
print(f"Height fluctuation amplitude: {fluctuation_amplitude:.2f} nm")

# Show plots
import matplotlib.pyplot as plt
plt.show()
```

### Time-Series Analysis

```python
# Extract temporal dynamics
tau_s = results['tau_s']
height_acf = results['auto3']  # MLE-based height correlation

# Fit exponential decay to extract characteristic time
def exp_decay(t, A, tau_decay, offset):
    return A * np.exp(-t/tau_decay) + offset

from scipy.optimize import curve_fit

# Fit first decade of correlation data
fit_mask = (tau_s > 1e-4) & (tau_s < 1e-2)
popt, _ = curve_fit(exp_decay, 
                   tau_s[fit_mask], 
                   np.mean(height_acf[fit_mask,:], axis=1))

characteristic_time = popt[1]
print(f"Characteristic fluctuation time: {characteristic_time*1000:.1f} ms")
```

## Experimental Considerations

### Optimal Measurement Conditions

1. **Substrate Preparation**
   - Clean glass coverslips
   - Uniform metal deposition (Ti/Au/Ti stack)
   - Controlled spacer layer thickness

2. **Sample Conditions**
   - Low fluorophore concentration to avoid aggregation
   - Appropriate buffer ionic strength
   - Temperature control for reproducible dynamics

3. **Acquisition Parameters**
   - Sufficient photon statistics (>10⁶ photons)
   - Appropriate time binning for target dynamics
   - Multiple measurement repeats for statistics

### Data Quality Criteria

```python
# Check data quality metrics
total_photons = np.sum(results['ttrace'])
signal_to_noise = np.mean(results['ttrace']) / np.std(results['ttrace'])
bleaching_rate = np.polyfit(np.arange(len(results['normtrace'])), 
                           results['normtrace'], 1)[0]

print(f"Total photons: {total_photons:,}")
print(f"Signal-to-noise ratio: {signal_to_noise:.2f}")
print(f"Bleaching rate: {bleaching_rate*100:.3f}%/bin")
```

## Related Modules

- **[`PTU_utils.py`](PTU_utils.py)** - PTU file reading and TCSPC processing
- **[`MIET_main.py`](MIET_main.py)** - MIET calibration calculations
- **[`Membrane_fluctuations.py`](Membrane_fluctuations.py)** - Theoretical fluctuation models

## References

1. Chizhik, A. I. et al. Metal-induced energy transfer for live cell nanoscopy. *Nat. Photonics* **8**, 124–129 (2014).
2. Baronsky, T. et al. Cell–substrate dynamics of the epithelial-to-mesenchymal transition. *Nano Lett.* **17**, 3320–3326 (2017).
3. Karedla, N. et al. Three-dimensional nanoscopy of whole cells by using metal-induced energy transfer imaging. *Nat. Methods* **14**, 1145–1148 (2017).
4. Ghosh, A. et al. Graphene-based metal-induced energy transfer for sub-nanometre optical localization. *Nat. Photonics* **13**, 860–865 (2019).

## Dependencies

- **numpy** - Numerical arrays and computing
- **scipy** - Scientific computing and optimization  
- **matplotlib** - Plotting and visualization
- **pathlib** - File path handling

## Environment Setup

The analysis requires the MIET_calc conda environment:

```bash
conda env create -f MIET_calc_env.yml
conda activate MIET_calc_env
python MemFluc_pFCS.py
```

## Contact

For questions about membrane fluctuation analysis or PTU data processing, please refer to the main repository documentation.