#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIET-FLIM Analysis Examples

This script demonstrates various use cases for the MIET-FLIM analysis pipeline,
including different substrate configurations, analysis options, and data processing scenarios.

Examples included:
1. Basic single file analysis
2. Batch processing multiple files
3. Custom substrate configurations
4. Different orientation models
5. Quality control and validation
6. Advanced analysis options

Author: MIET-FLIM Pipeline
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pandas as pd

# Import the MIET-FLIM analysis pipeline
from MIET_FLIM_Analysis import (
    MIETConfig, 
    analyze_miet_flim_data, 
    visualize_miet_flim_results,
    process_miet_flim_batch,
    create_batch_summary,
    create_miet_calibration,
    interpolate_height_from_lifetime
)

#%%
# ============================================================================
# EXAMPLE 1: BASIC SINGLE FILE ANALYSIS
# ============================================================================

def example_1_basic_analysis():
    """Basic MIET-FLIM analysis of a single PTU file."""
    
    print("Example 1: Basic Single File Analysis")
    print("="*50)
    
    # Create default configuration
    config = MIETConfig()
    
    # File to analyze (replace with your actual file)
    ptu_file = "sample_data.ptu"  # Update this path
    
    if not Path(ptu_file).exists():
        print(f"File not found: {ptu_file}")
        print("Please update the path to point to your PTU file")
        return
    
    # Perform analysis
    print("Analyzing PTU file...")
    results = analyze_miet_flim_data(
        ptu_file=ptu_file,
        config=config,
        metals_db_path="metals.mat",
        orientation='random',
        save_results=True
    )
    
    if results:
        print("Analysis successful!")
        
        # Print summary statistics
        stats = results['height_analysis']['statistics']
        print(f"\nHeight Analysis Results:")
        print(f"  Valid pixels: {stats['n_valid_pixels']}")
        print(f"  Mean height: {stats['height_mean']:.1f} ± {stats['height_std']:.1f} nm")
        print(f"  Height range: {stats['height_min']:.1f} - {stats['height_max']:.1f} nm")
        
        # Create visualization
        save_path = ptu_file.replace('.ptu', '_analysis_results.png')
        visualize_miet_flim_results(results, save_path)
        
    else:
        print("Analysis failed!")

#%%
# ============================================================================
# EXAMPLE 2: BATCH PROCESSING
# ============================================================================

def example_2_batch_processing():
    """Process multiple PTU files in batch mode."""
    
    print("Example 2: Batch Processing")
    print("="*50)
    
    # Configuration
    config = MIETConfig()
    
    # Find all PTU files in a directory
    data_directory = "path/to/your/data"  # Update this path
    ptu_files = glob.glob(f"{data_directory}/*.ptu")
    
    if not ptu_files:
        print(f"No PTU files found in: {data_directory}")
        print("Please update the data_directory path")
        # Create some example file paths for demonstration
        ptu_files = [
            "sample1.ptu",
            "sample2.ptu", 
            "sample3.ptu"
        ]
        print(f"Using example files: {ptu_files}")
    
    print(f"Found {len(ptu_files)} PTU files to process")
    
    # Process batch
    batch_results = process_miet_flim_batch(
        file_list=ptu_files,
        config=config,
        metals_db_path="metals.mat",
        orientation='random'
    )
    
    if batch_results:
        # Create summary report
        create_batch_summary(batch_results, "batch_analysis_summary.csv")
        
        print(f"\nBatch processing completed:")
        print(f"  Successfully processed: {len(batch_results)} files")
        
        # Summary statistics
        all_heights = []
        for filename, results in batch_results.items():
            if 'height_analysis' in results:
                height_map = results['height_analysis']['height_map']
                valid_heights = height_map[np.isfinite(height_map)]
                if len(valid_heights) > 0:
                    all_heights.extend(valid_heights)
        
        if all_heights:
            all_heights = np.array(all_heights)
            print(f"  Overall height statistics:")
            print(f"    Mean: {np.mean(all_heights):.1f} nm")
            print(f"    Std:  {np.std(all_heights):.1f} nm")
            print(f"    Range: {np.min(all_heights):.1f} - {np.max(all_heights):.1f} nm")

#%%
# ============================================================================
# EXAMPLE 3: CUSTOM SUBSTRATE CONFIGURATIONS
# ============================================================================

def example_3_custom_substrates():
    """Demonstrate different substrate configurations."""
    
    print("Example 3: Custom Substrate Configurations")
    print("="*50)
    
    # Example file (replace with actual)
    ptu_file = "sample_data.ptu"
    
    # Configuration 1: Standard glass-gold-polymer
    print("\n1. Standard glass-gold-polymer substrate:")
    config1 = MIETConfig()
    config1.metal_stack = [80, 20, 80]  # Ti, Au, Ti
    config1.metal_thickness_nm = [2, 10, 1]
    config1.polymer_thickness_nm = 50.0
    
    # Configuration 2: Thick gold layer
    print("\n2. Thick gold layer substrate:")
    config2 = MIETConfig()
    config2.metal_stack = [80, 20]  # Ti, Au (no top Ti)
    config2.metal_thickness_nm = [3, 25]  # Thicker Au
    config2.polymer_thickness_nm = 100.0
    
    # Configuration 3: Silver substrate
    print("\n3. Silver substrate:")
    config3 = MIETConfig()
    config3.metal_stack = [10]  # Silver only (code 10)
    config3.metal_thickness_nm = [15]
    config3.polymer_thickness_nm = 30.0
    
    # Configuration 4: No metal (control)
    print("\n4. No metal control:")
    config4 = MIETConfig()
    config4.metal_stack = []
    config4.metal_thickness_nm = []
    config4.polymer_thickness_nm = 200.0
    
    # Generate and compare MIET calibration curves
    configs = [config1, config2, config3, config4]
    labels = ['Glass-Au-Polymer', 'Thick Au', 'Silver', 'No Metal']
    
    plt.figure(figsize=(12, 8))
    
    for i, (config, label) in enumerate(zip(configs, labels)):
        try:
            z_cal, lt_cal = create_miet_calibration(config, "metals.mat", 'random')
            plt.subplot(2, 2, i+1)
            plt.plot(z_cal, lt_cal, 'b-', linewidth=2)
            plt.xlabel('Height (nm)')
            plt.ylabel('Lifetime (ns)')
            plt.title(f'{label}')
            plt.grid(True, alpha=0.3)
            
            print(f"  {label}: {len(z_cal)} points, "
                  f"lifetime range: {np.nanmin(lt_cal):.2f}-{np.nanmax(lt_cal):.2f} ns")
                  
        except Exception as e:
            print(f"  {label}: Error - {e}")
    
    plt.tight_layout()
    plt.savefig('substrate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#%%
# ============================================================================
# EXAMPLE 4: ORIENTATION COMPARISON
# ============================================================================

def example_4_orientation_comparison():
    """Compare different molecular orientation models."""
    
    print("Example 4: Molecular Orientation Comparison")
    print("="*50)
    
    config = MIETConfig()
    orientations = ['random', 'vertical', 'horizontal']
    
    plt.figure(figsize=(15, 5))
    
    for i, orientation in enumerate(orientations):
        print(f"\nCalculating MIET curve for {orientation} orientation...")
        
        try:
            z_cal, lt_cal = create_miet_calibration(config, "metals.mat", orientation)
            
            plt.subplot(1, 3, i+1)
            plt.plot(z_cal, lt_cal, 'b-', linewidth=2)
            plt.xlabel('Height (nm)')
            plt.ylabel('Lifetime (ns)')
            plt.title(f'{orientation.capitalize()} Orientation')
            plt.grid(True, alpha=0.3)
            
            print(f"  Range: {np.nanmin(lt_cal):.2f} - {np.nanmax(lt_cal):.2f} ns")
            
        except Exception as e:
            print(f"  Error for {orientation}: {e}")
    
    plt.tight_layout()
    plt.savefig('orientation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#%%
# ============================================================================
# EXAMPLE 5: QUALITY CONTROL AND VALIDATION
# ============================================================================

def example_5_quality_control():
    """Demonstrate quality control measures and validation."""
    
    print("Example 5: Quality Control and Validation")
    print("="*50)
    
    # Simulate some analysis results for demonstration
    # In real use, this would come from actual PTU file analysis
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    nx, ny = 64, 64
    
    # Simulate lifetime map with some realistic values
    lifetime_map = np.random.normal(2.5, 0.3, (nx, ny))
    lifetime_map[lifetime_map < 0.5] = np.nan  # Remove unphysical values
    
    # Simulate intensity map
    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    intensity_map = np.exp(-(x**2 + y**2)/0.5) + 0.1*np.random.random((nx, ny))
    
    # Quality control metrics
    print("\n1. Data Quality Assessment:")
    
    # Check 1: Intensity distribution
    valid_pixels = intensity_map > np.percentile(intensity_map, 10)
    print(f"   Valid pixels (>10th percentile): {np.sum(valid_pixels)}/{nx*ny} ({100*np.sum(valid_pixels)/(nx*ny):.1f}%)")
    
    # Check 2: Lifetime range
    valid_lifetimes = lifetime_map[np.isfinite(lifetime_map)]
    print(f"   Lifetime range: {np.min(valid_lifetimes):.2f} - {np.max(valid_lifetimes):.2f} ns")
    print(f"   Lifetime CV: {100*np.std(valid_lifetimes)/np.mean(valid_lifetimes):.1f}%")
    
    # Check 3: Spatial correlation
    # Simple edge detection to check for structure
    from scipy import ndimage
    edges = ndimage.sobel(lifetime_map)
    edge_strength = np.nanmean(np.abs(edges))
    print(f"   Spatial structure (edge strength): {edge_strength:.3f}")
    
    # Create quality control plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Intensity map
    im1 = axes[0,0].imshow(intensity_map, cmap='gray')
    axes[0,0].set_title('Intensity Map')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Lifetime map
    im2 = axes[0,1].imshow(lifetime_map, cmap='viridis')
    axes[0,1].set_title('Lifetime Map')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Valid pixel mask
    axes[0,2].imshow(valid_pixels, cmap='RdYlBu')
    axes[0,2].set_title('Valid Pixel Mask')
    
    # Intensity histogram
    axes[1,0].hist(intensity_map.ravel(), bins=50, alpha=0.7)
    axes[1,0].axvline(np.percentile(intensity_map, 10), color='red', linestyle='--')
    axes[1,0].set_xlabel('Intensity')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Intensity Distribution')
    
    # Lifetime histogram
    axes[1,1].hist(valid_lifetimes, bins=50, alpha=0.7)
    axes[1,1].set_xlabel('Lifetime (ns)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Lifetime Distribution')
    
    # Lifetime vs intensity scatter
    valid_mask = valid_pixels & np.isfinite(lifetime_map)
    if np.any(valid_mask):
        axes[1,2].scatter(intensity_map[valid_mask], lifetime_map[valid_mask], alpha=0.5, s=1)
        axes[1,2].set_xlabel('Intensity')
        axes[1,2].set_ylabel('Lifetime (ns)')
        axes[1,2].set_title('Lifetime vs Intensity')
    
    plt.tight_layout()
    plt.savefig('quality_control.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Validation recommendations
    print("\n2. Quality Control Recommendations:")
    
    if np.sum(valid_pixels) < 0.1 * nx * ny:
        print("   ⚠️  Low number of valid pixels - check sample preparation")
    else:
        print("   ✓  Sufficient valid pixels for analysis")
    
    if np.std(valid_lifetimes)/np.mean(valid_lifetimes) > 0.5:
        print("   ⚠️  High lifetime variability - check fitting quality")
    else:
        print("   ✓  Reasonable lifetime variability")
    
    if edge_strength < 0.01:
        print("   ⚠️  Low spatial structure - may indicate noise dominance")
    else:
        print("   ✓  Good spatial structure detected")

#%%
# ============================================================================
# EXAMPLE 6: ADVANCED ANALYSIS OPTIONS
# ============================================================================

def example_6_advanced_options():
    """Demonstrate advanced analysis options and customizations."""
    
    print("Example 6: Advanced Analysis Options")
    print("="*50)
    
    # Advanced configuration
    config = MIETConfig()
    
    print("\n1. Custom Fitting Parameters:")
    
    # Two-component fit for fast dynamics
    config_2comp = MIETConfig()
    config_2comp.tau0 = np.array([0.5, 2.5])  # Two components
    config_2comp.lifetime_bounds = [0.1, 1.0, 1.0, 5.0]  # [min1, min2, max1, max2]
    print("   Two-component model configured")
    
    # Four-component fit for complex samples
    config_4comp = MIETConfig()
    config_4comp.tau0 = np.array([0.2, 0.8, 2.5, 8.0])  # Four components
    config_4comp.lifetime_bounds = [0.05, 0.3, 1.0, 4.0, 0.5, 1.5, 5.0, 15.0]
    print("   Four-component model configured")
    
    print("\n2. Spatial Analysis Options:")
    
    # High-resolution pixel-by-pixel
    config_hires = MIETConfig()
    config_hires.flag_win = False  # Pixel-by-pixel
    print("   High-resolution pixel analysis configured")
    
    # Fast windowed analysis
    config_fast = MIETConfig()
    config_fast.flag_win = True
    config_fast.win_size = 16  # Larger windows
    config_fast.step = 8       # Bigger steps
    print("   Fast windowed analysis configured")
    
    print("\n3. Temporal Resolution Options:")
    
    # High temporal resolution
    config_fine_time = MIETConfig()
    config_fine_time.resolution_ns = 0.1  # Finer time bins
    print("   High temporal resolution configured")
    
    # Coarse temporal resolution for speed
    config_coarse_time = MIETConfig()
    config_coarse_time.resolution_ns = 0.5  # Coarser time bins
    print("   Coarse temporal resolution configured")
    
    print("\n4. Different Fluorophore Parameters:")
    
    # EGFP-like fluorophore
    config_egfp = MIETConfig()
    config_egfp.wavelength_nm = 510.0
    config_egfp.tau_free_ns = 2.8
    config_egfp.quantum_yield = 0.6
    print("   EGFP parameters configured")
    
    # Alexa647-like fluorophore  
    config_alexa = MIETConfig()
    config_alexa.wavelength_nm = 665.0
    config_alexa.tau_free_ns = 1.2
    config_alexa.quantum_yield = 0.3
    print("   Alexa647 parameters configured")
    
    # Compare MIET curves for different fluorophores
    configs = [config_egfp, config_alexa]
    labels = ['EGFP-like (510nm)', 'Alexa647-like (665nm)']
    
    plt.figure(figsize=(12, 5))
    
    for i, (config, label) in enumerate(zip(configs, labels)):
        try:
            z_cal, lt_cal = create_miet_calibration(config, "metals.mat", 'random')
            
            plt.subplot(1, 2, i+1)
            plt.plot(z_cal, lt_cal, 'b-', linewidth=2)
            plt.xlabel('Height (nm)')
            plt.ylabel('Lifetime (ns)')
            plt.title(f'{label}')
            plt.grid(True, alpha=0.3)
            
            print(f"   {label}: Dynamic range {np.nanmax(lt_cal)/np.nanmin(lt_cal):.1f}x")
            
        except Exception as e:
            print(f"   Error for {label}: {e}")
    
    plt.tight_layout()
    plt.savefig('fluorophore_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#%%
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    
    print("MIET-FLIM Analysis Examples")
    print("="*60)
    print("This script demonstrates various use cases for the MIET-FLIM pipeline.")
    print("Update file paths and run individual examples as needed.\n")
    
    # List of examples
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Batch Processing", example_2_batch_processing),
        ("Custom Substrates", example_3_custom_substrates),
        ("Orientation Comparison", example_4_orientation_comparison),
        ("Quality Control", example_5_quality_control),
        ("Advanced Options", example_6_advanced_options),
    ]
    
    # Run examples based on user choice
    print("Available examples:")
    for i, (name, func) in enumerate(examples):
        print(f"  {i+1}. {name}")
    
    print("\nTo run a specific example, call its function directly:")
    for i, (name, func) in enumerate(examples):
        print(f"  {func.__name__}()")
    
    # For demonstration, run the examples that don't require files
    print("\nRunning examples that don't require PTU files:")
    
    try:
        example_3_custom_substrates()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_orientation_comparison()  
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_quality_control()
    except Exception as e:
        print(f"Example 5 error: {e}")
        
    try:
        example_6_advanced_options()
    except Exception as e:
        print(f"Example 6 error: {e}")

if __name__ == "__main__":
    main()