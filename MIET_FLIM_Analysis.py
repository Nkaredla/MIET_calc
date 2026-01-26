# -*- coding: utf-8 -*-
"""
MIET-FLIM Analysis Pipeline

This module combines FLIM (Fluorescence Lifetime Imaging Microscopy) data processing
with MIET (Metal-Induced Energy Transfer) calculations to determine molecular heights
from lifetime measurements.

The pipeline:
1. Reads PTU FLIM data using PTU_ScanRead
2. Processes lifetime data using FLIM_fitter
3. Calculates MIET calibration curves using MIET_main
4. Determines molecular heights from measured lifetimes

Created on: Jan 23, 2025
Author: Combined analysis pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import required modules
from PTU_ScanRead import PTU_ScanRead, Process_Frame
from FLIM_fitter import Calc_mIRF, FluoFit, PatternMatchIm, PIRLSnonneg_batch_gpu
from utilities import (
    load_or_process_ptu_data,
    analyze_flim_data,
    vignette_correction,
    display_amplitude_maps_subplot
)
from MIET_main import (
    miet_calc, 
    MetalsDB, 
    hash_waveguide_mode,
    brightness_dipole
)

#%%
# ============================================================================
# MIET-FLIM CONFIGURATION AND SETUP
# ============================================================================

class MIETConfig:
    """Configuration class for MIET-FLIM analysis."""
    
    def __init__(self):
        # MIET substrate parameters (default: glass + gold + polymer)
        self.wavelength_nm = 690.0  # Emission wavelength
        self.n_glass = 1.52         # Glass substrate
        self.n_polymer = 1.46       # Polymer layer (sample medium)
        self.n_water = 1.33         # Aqueous medium on top
        
        # Metal stack (gold mirror with titanium adhesion layers)
        self.metal_stack = [80, 20, 80]  # Material codes: Ti, Au, Ti
        self.metal_thickness_nm = [2, 10, 1]  # nm
        self.polymer_thickness_nm = 50.0  # nm
        
        # MIET calculation parameters
        self.quantum_yield = 0.6    # Quantum yield
        self.tau_free_ns = 4.0      # Free-space lifetime (ns)
        self.z_range_nm = (0.5, 49.5)  # Height range to calculate
        self.z_step_nm = 0.5        # Height step
        
        # FLIM analysis parameters
        self.auto_det = 0           # Detector channel
        self.auto_PIE = 1           # PIE window
        self.flag_win = True        # Use windowed analysis
        self.resolution_ns = 0.2    # Time resolution
        self.win_size = 8           # Window size for binning
        self.step = 2               # Step size for windows
        
        # Lifetime fitting parameters
        self.tau0 = np.array([0.3, 1.7, 6.0])  # Initial lifetime guesses (ns)
        self.lifetime_bounds = [0.1, 0.5, 2.0, 2.0, 4.0, 10.0]  # [min1,min2,min3,max1,max2,max3]
        
    def get_z_grid_nm(self) -> np.ndarray:
        """Get the height grid for MIET calculations."""
        return np.arange(self.z_range_nm[0], self.z_range_nm[1], self.z_step_nm)

#%%
# ============================================================================
# MIET CALIBRATION FUNCTIONS
# ============================================================================

def create_miet_calibration(config: MIETConfig, 
                           metals_db_path: Optional[str] = None,
                           orientation: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create MIET calibration curve relating lifetime to height.
    
    Parameters
    ----------
    config : MIETConfig
        Configuration parameters
    metals_db_path : str, optional
        Path to metals database .mat file
    orientation : str
        Dipole orientation: 'random', 'vertical', or 'horizontal'
        
    Returns
    -------
    z_nm : ndarray
        Height values in nanometers
    lifetime_ns : ndarray
        Corresponding lifetime values in nanoseconds
    """
    
    print("Creating MIET calibration curve...")
    print(f"  - Wavelength: {config.wavelength_nm} nm")
    print(f"  - Metal stack: {config.metal_stack} (codes)")
    print(f"  - Metal thicknesses: {config.metal_thickness_nm} nm")
    print(f"  - Polymer thickness: {config.polymer_thickness_nm} nm")
    print(f"  - Orientation: {orientation}")
    
    # Load metals database if provided
    metals_db = None
    if metals_db_path and Path(metals_db_path).exists():
        try:
            metals_db = MetalsDB(metals_db_path)
            print(f"  - Loaded metals database: {metals_db_path}")
        except Exception as e:
            print(f"  - Warning: Could not load metals database: {e}")
            metals_db = None
    
    # Set up layer structure for MIET calculation
    # Bottom to top: glass | Ti | Au | Ti | polymer | water
    n0 = [config.n_glass]  # Bottom substrate
    n1 = config.metal_stack + [config.n_polymer]  # Metal stack + polymer
    n_top = [config.n_water]  # Top medium
    
    # Corresponding thicknesses (exclude semi-infinite media)
    d0 = []  # No layers below glass
    d1 = config.metal_thickness_nm + [config.polymer_thickness_nm]  # Metal + polymer thicknesses
    d_top = []  # No layers above water
    
    # Dipole layer is the polymer
    n_dipole = config.n_polymer
    d_dipole = config.polymer_thickness_nm
    
    # Set orientation
    if orientation.lower() == 'random':
        al_res = None  # Random orientation
    elif orientation.lower() == 'vertical':
        al_res = np.array([0.0])  # Vertical (0°)
    elif orientation.lower() == 'horizontal':
        al_res = np.array([np.pi/2])  # Horizontal (90°)
    else:
        al_res = None  # Default to random
    
    # Calculate MIET curve
    try:
        z_nm, lifetime_ns = miet_calc(
            al_res=al_res,
            lamem=config.wavelength_nm,
            n0=n0,
            n=n_dipole,
            n1=n1 + n_top,
            d0=d0,
            d=d_dipole,
            d1=d1 + [],
            qyield=config.quantum_yield,
            tau_free=config.tau_free_ns,
            fig=False,
            metals_db=metals_db
        )
        
        print(f"  - Generated {len(z_nm)} height points")
        print(f"  - Height range: {np.min(z_nm):.1f} - {np.max(z_nm):.1f} nm")
        print(f"  - Lifetime range: {np.nanmin(lifetime_ns):.2f} - {np.nanmax(lifetime_ns):.2f} ns")
        
        return z_nm, lifetime_ns
        
    except Exception as e:
        print(f"Error creating MIET calibration: {e}")
        raise

def interpolate_height_from_lifetime(lifetime_measured: np.ndarray,
                                   z_calibration: np.ndarray,
                                   lifetime_calibration: np.ndarray,
                                   method: str = 'linear') -> np.ndarray:
    """
    Convert measured lifetimes to heights using MIET calibration.
    
    Parameters
    ----------
    lifetime_measured : ndarray
        Measured lifetime values (ns)
    z_calibration : ndarray
        Calibration height values (nm)
    lifetime_calibration : ndarray
        Calibration lifetime values (ns)
    method : str
        Interpolation method ('linear', 'cubic')
        
    Returns
    -------
    height_nm : ndarray
        Estimated heights in nanometers
    """
    
    # Remove NaN values from calibration
    valid_mask = np.isfinite(lifetime_calibration) & np.isfinite(z_calibration)
    z_cal = z_calibration[valid_mask]
    lt_cal = lifetime_calibration[valid_mask]
    
    if len(z_cal) < 2:
        warnings.warn("Insufficient valid calibration points")
        return np.full_like(lifetime_measured, np.nan)
    
    # Sort calibration data by lifetime
    sort_idx = np.argsort(lt_cal)
    z_cal = z_cal[sort_idx]
    lt_cal = lt_cal[sort_idx]
    
    # Create interpolation function (lifetime -> height)
    if method == 'cubic' and len(z_cal) >= 4:
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(lt_cal, z_cal, bc_type='natural', extrapolate=False)
    else:
        interp_func = interp1d(lt_cal, z_cal, kind='linear', 
                              bounds_error=False, fill_value=np.nan)
    
    # Interpolate heights
    height_nm = interp_func(lifetime_measured)
    
    return height_nm

#%%
# ============================================================================
# COMBINED MIET-FLIM ANALYSIS FUNCTIONS
# ============================================================================

def analyze_miet_flim_data(ptu_file: str,
                          config: MIETConfig,
                          metals_db_path: Optional[str] = None,
                          orientation: str = 'random',
                          save_results: bool = True) -> Dict:
    """
    Complete MIET-FLIM analysis pipeline.
    
    Parameters
    ----------
    ptu_file : str
        Path to PTU file
    config : MIETConfig
        Configuration parameters
    metals_db_path : str, optional
        Path to metals database
    orientation : str
        Dipole orientation assumption
    save_results : bool
        Whether to save results
        
    Returns
    -------
    dict
        Complete analysis results
    """
    
    print(f"\n{'='*60}")
    print(f"MIET-FLIM Analysis: {Path(ptu_file).name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Step 1: Create MIET calibration
    print("\n1. Creating MIET calibration curve...")
    try:
        z_calibration, lifetime_calibration = create_miet_calibration(
            config, metals_db_path, orientation
        )
        results['miet_calibration'] = {
            'z_nm': z_calibration,
            'lifetime_ns': lifetime_calibration,
            'orientation': orientation
        }
    except Exception as e:
        print(f"Error creating MIET calibration: {e}")
        return {}
    
    # Step 2: Load and process PTU data
    print("\n2. Loading PTU data...")
    try:
        ptu_data = load_or_process_ptu_data(ptu_file, force_reprocess=False)
        if not ptu_data:
            print("Failed to load PTU data")
            return {}
    except Exception as e:
        print(f"Error loading PTU data: {e}")
        return {}
    
    # Step 3: Perform FLIM analysis
    print("\n3. Performing FLIM analysis...")
    try:
        flim_results = analyze_flim_data(
            ptu_data,
            auto_det=config.auto_det,
            auto_PIE=config.auto_PIE,
            flag_win=config.flag_win,
            resolution=config.resolution_ns,
            tau0=config.tau0,
            win_size=config.win_size,
            step=config.step,
            IRF_data=None
        )
        
        if not flim_results:
            print("FLIM analysis failed")
            return {}
            
        results['flim_results'] = flim_results
        
    except Exception as e:
        print(f"Error in FLIM analysis: {e}")
        return {}
    
    # Step 4: Convert lifetimes to heights using MIET
    print("\n4. Converting lifetimes to heights using MIET...")
    try:
        # Use intensity-averaged lifetime for MIET analysis
        lifetime_map = flim_results['tau_avg_int']  # Shape: (nx, ny)
        intensity_map = flim_results['int2']  # Vignette-corrected intensity
        
        # Apply intensity threshold to avoid noise
        intensity_threshold = np.percentile(intensity_map[intensity_map > 0], 10)
        mask = intensity_map > intensity_threshold
        
        # Convert lifetimes to heights
        height_map = np.full_like(lifetime_map, np.nan)
        valid_pixels = mask & np.isfinite(lifetime_map) & (lifetime_map > 0)
        
        if np.any(valid_pixels):
            height_map[valid_pixels] = interpolate_height_from_lifetime(
                lifetime_map[valid_pixels],
                z_calibration,
                lifetime_calibration,
                method='linear'
            )
            
            print(f"  - Processed {np.sum(valid_pixels)} valid pixels")
            valid_heights = height_map[np.isfinite(height_map)]
            if len(valid_heights) > 0:
                print(f"  - Height range: {np.min(valid_heights):.1f} - {np.max(valid_heights):.1f} nm")
                print(f"  - Mean height: {np.mean(valid_heights):.1f} ± {np.std(valid_heights):.1f} nm")
        else:
            print("  - No valid pixels found for height calculation")
        
        results['height_analysis'] = {
            'height_map': height_map,
            'lifetime_map': lifetime_map,
            'intensity_map': intensity_map,
            'valid_mask': valid_pixels,
            'statistics': {
                'n_valid_pixels': np.sum(valid_pixels),
                'height_mean': np.nanmean(height_map),
                'height_std': np.nanstd(height_map),
                'height_min': np.nanmin(height_map),
                'height_max': np.nanmax(height_map)
            }
        }
        
    except Exception as e:
        print(f"Error converting lifetimes to heights: {e}")
        return {}
    
    # Step 5: Save results if requested
    if save_results:
        print("\n5. Saving results...")
        try:
            save_miet_flim_results(results, ptu_file, config)
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
    
    print(f"\nMIET-FLIM analysis completed successfully!")
    return results

def save_miet_flim_results(results: Dict, ptu_file: str, config: MIETConfig):
    """Save MIET-FLIM analysis results."""
    
    ptu_path = Path(ptu_file)
    output_dir = ptu_path.parent
    filename_base = ptu_path.stem
    
    print(f"Saving results to: {output_dir}")
    
    # Save complete results as pickle
    pkl_file = output_dir / f"{filename_base}_MIET_FLIM_results.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"  - Complete results: {pkl_file}")
    
    # Save height map as TIFF
    if 'height_analysis' in results:
        from tifffile import imwrite
        height_map = results['height_analysis']['height_map']
        
        height_tiff = output_dir / f"{filename_base}_height_map_nm.tif"
        imwrite(height_tiff, height_map.astype(np.float32))
        print(f"  - Height map TIFF: {height_tiff}")
        
        # Save statistics as CSV
        stats = results['height_analysis']['statistics']
        stats_file = output_dir / f"{filename_base}_height_statistics.csv"
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(stats_file, index=False)
        print(f"  - Height statistics: {stats_file}")
    
    # Save MIET calibration
    if 'miet_calibration' in results:
        cal_file = output_dir / f"{filename_base}_MIET_calibration.csv"
        cal_data = pd.DataFrame({
            'height_nm': results['miet_calibration']['z_nm'],
            'lifetime_ns': results['miet_calibration']['lifetime_ns'],
            'orientation': results['miet_calibration']['orientation']
        })
        cal_data.to_csv(cal_file, index=False)
        print(f"  - MIET calibration: {cal_file}")

def visualize_miet_flim_results(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization of MIET-FLIM results.
    
    Parameters
    ----------
    results : dict
        Analysis results from analyze_miet_flim_data
    save_path : str, optional
        Path to save the figure
    """
    
    if not results or 'miet_calibration' not in results or 'height_analysis' not in results:
        print("Insufficient data for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MIET calibration curve
    z_cal = results['miet_calibration']['z_nm']
    lt_cal = results['miet_calibration']['lifetime_ns']
    orientation = results['miet_calibration']['orientation']
    
    axes[0, 0].plot(z_cal, lt_cal, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Height (nm)')
    axes[0, 0].set_ylabel('Lifetime (ns)')
    axes[0, 0].set_title(f'MIET Calibration\n({orientation} orientation)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height map
    height_map = results['height_analysis']['height_map']
    im1 = axes[0, 1].imshow(height_map, cmap='viridis', origin='lower')
    axes[0, 1].set_title('Height Map')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 1], label='Height (nm)')
    
    # Lifetime map
    lifetime_map = results['height_analysis']['lifetime_map']
    im2 = axes[0, 2].imshow(lifetime_map, cmap='plasma', origin='lower')
    axes[0, 2].set_title('Lifetime Map')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 2], label='Lifetime (ns)')
    
    # Intensity map
    intensity_map = results['height_analysis']['intensity_map']
    im3 = axes[1, 0].imshow(intensity_map, cmap='gray', origin='lower')
    axes[1, 0].set_title('Intensity Map')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=axes[1, 0], label='Intensity')
    
    # Height histogram
    valid_heights = height_map[np.isfinite(height_map)]
    if len(valid_heights) > 0:
        axes[1, 1].hist(valid_heights, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(valid_heights), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(valid_heights):.1f} nm')
        axes[1, 1].set_xlabel('Height (nm)')
        axes[1, 1].set_ylabel('Pixel Count')
        axes[1, 1].set_title('Height Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Lifetime vs Height scatter plot
    valid_mask = results['height_analysis']['valid_mask']
    if np.any(valid_mask):
        lt_valid = lifetime_map[valid_mask]
        h_valid = height_map[valid_mask]
        
        axes[1, 2].scatter(lt_valid, h_valid, alpha=0.5, s=1)
        axes[1, 2].plot(lt_cal, z_cal, 'r-', linewidth=2, label='MIET calibration')
        axes[1, 2].set_xlabel('Measured Lifetime (ns)')
        axes[1, 2].set_ylabel('Calculated Height (nm)')
        axes[1, 2].set_title('Lifetime vs Height')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()

#%%
# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_miet_flim_batch(file_list: List[str],
                           config: MIETConfig,
                           metals_db_path: Optional[str] = None,
                           orientation: str = 'random') -> Dict[str, Dict]:
    """
    Process multiple PTU files with MIET-FLIM analysis.
    
    Parameters
    ----------
    file_list : list of str
        List of PTU file paths
    config : MIETConfig
        Configuration parameters
    metals_db_path : str, optional
        Path to metals database
    orientation : str
        Dipole orientation assumption
        
    Returns
    -------
    dict
        Results for each file
    """
    
    print(f"\n{'='*80}")
    print(f"BATCH MIET-FLIM ANALYSIS")
    print(f"{'='*80}")
    print(f"Processing {len(file_list)} files...")
    print(f"Orientation: {orientation}")
    
    all_results = {}
    
    for i, ptu_file in enumerate(file_list):
        print(f"\n[{i+1}/{len(file_list)}] Processing: {Path(ptu_file).name}")
        
        try:
            results = analyze_miet_flim_data(
                ptu_file, config, metals_db_path, orientation, save_results=True
            )
            
            if results:
                all_results[Path(ptu_file).stem] = results
                print(f"✓ Successfully processed: {Path(ptu_file).name}")
            else:
                print(f"✗ Failed to process: {Path(ptu_file).name}")
                
        except Exception as e:
            print(f"✗ Error processing {Path(ptu_file).name}: {e}")
    
    print(f"\nBatch processing complete: {len(all_results)}/{len(file_list)} files processed")
    return all_results

def create_batch_summary(batch_results: Dict[str, Dict], 
                        output_path: str):
    """Create summary statistics from batch processing results."""
    
    summary_data = []
    
    for filename, results in batch_results.items():
        if 'height_analysis' not in results:
            continue
            
        stats = results['height_analysis']['statistics']
        
        summary_data.append({
            'filename': filename,
            'n_valid_pixels': stats['n_valid_pixels'],
            'height_mean_nm': stats['height_mean'],
            'height_std_nm': stats['height_std'],
            'height_min_nm': stats['height_min'],
            'height_max_nm': stats['height_max'],
            'orientation': results.get('miet_calibration', {}).get('orientation', 'unknown')
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        print(f"Batch summary saved: {output_path}")
        
        # Print overview
        print(f"\nBatch Summary:")
        print(f"  - Files processed: {len(summary_data)}")
        print(f"  - Mean height range: {summary_df['height_mean_nm'].min():.1f} - {summary_df['height_mean_nm'].max():.1f} nm")
        print(f"  - Overall mean: {summary_df['height_mean_nm'].mean():.1f} ± {summary_df['height_mean_nm'].std():.1f} nm")
    else:
        print("No valid results for summary")

#%%
# ============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Configuration
    config = MIETConfig()
    
    # Optional: Path to metals database (set to None for simplified analysis)
    metals_db_path = "metals.mat"  # Update path as needed
    
    # Example 1: Single file analysis
    single_file_example = False
    if single_file_example:
        ptu_file = "path/to/your/file.ptu"  # Replace with actual path
        
        results = analyze_miet_flim_data(
            ptu_file=ptu_file,
            config=config,
            metals_db_path=metals_db_path,
            orientation='random',  # 'random', 'vertical', or 'horizontal'
            save_results=True
        )
        
        if results:
            # Create visualization
            save_path = ptu_file.replace('.ptu', '_MIET_FLIM_analysis.png')
            visualize_miet_flim_results(results, save_path)
    
    # Example 2: Batch processing
    batch_example = True
    if batch_example:
        # List of PTU files to process
        ptu_files = [
            "path/to/file1.ptu",
            "path/to/file2.ptu",
            # Add more files as needed
        ]
        
        # Process all files
        batch_results = process_miet_flim_batch(
            file_list=ptu_files,
            config=config,
            metals_db_path=metals_db_path,
            orientation='random'
        )
        
        # Create summary
        if batch_results:
            create_batch_summary(
                batch_results, 
                "MIET_FLIM_batch_summary.csv"
            )
    
    # Example 3: Custom configuration
    custom_config_example = False
    if custom_config_example:
        # Create custom configuration
        custom_config = MIETConfig()
        
        # Modify parameters
        custom_config.wavelength_nm = 680.0
        custom_config.metal_stack = [80, 20]  # Ti, Au (no top Ti layer)
        custom_config.metal_thickness_nm = [2, 15]
        custom_config.polymer_thickness_nm = 100.0
        custom_config.tau0 = np.array([0.5, 2.5])  # Two-component fit
        
        # Process with custom config
        # results = analyze_miet_flim_data(ptu_file, custom_config, ...)
    
    print("\nMIET-FLIM analysis module loaded.")
    print("Use analyze_miet_flim_data() for single file analysis")
    print("Use process_miet_flim_batch() for batch processing")