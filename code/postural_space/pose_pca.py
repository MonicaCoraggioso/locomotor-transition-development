#!/usr/bin/env python3
"""
Fish Body Shape Principal Component Analysis

This script performs PCA on fish body shape time series to extract dominant
postural modes and analyze their structure across ages and individuals.

Input:
    - Preprocessed HDF5 files (INFO, KINE, POSE) from data processing pipeline
    - POSE contains body segment angles: ['ID', 't', 'theta_1', ..., 'theta_19']

Output:
    - Principal component scores (PCS): Time series of PC amplitudes per fish
    - Eigenvectors (EIGENTHETAS): Shape modes
    - Explained variance ratios: Variance captured by each PC
    - Visualizations: Cumulative variance, age trends, eigenshapes

Analysis modes:
    - Pooled: Single PCA across all fish (population-level shape space)
    - Per-fish: Separate PCA for each individual (captures individual variability)

Usage:
    python pose_pca.py

Author: Vito Dichio
Date: 2025-01-29
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# Configuration
# ============================================================================

# Plotting parameters
CM = 1 / 2.54  # Centimeters to inches
DPI = 300
FONTSIZE = 11
LABELSIZE = 10
COLS_PCS = ['#46211A', '#A02334', '#F1D3B2']  # Colors for PC1, PC2, PC3


# ============================================================================
# Utility Functions
# ============================================================================

def get_fish_ID(ID_string):
    """
    Extract fish identifier by removing run number from ID.
    
    Parameters
    ----------
    ID_string : str
        Full ID (e.g., '25_10Jun24_1_2')
        
    Returns
    -------
    str
        Fish ID without run (e.g., '25_10Jun24_1')
        
    Examples
    --------
    >>> get_fish_ID('25_10Jun24_1_2')
    '25_10Jun24_1'
    """
    return '_'.join(ID_string.split('_')[:-1])


# ============================================================================
# PCA Functions
# ============================================================================

def perform_pca(thetas, n_pcs, get_explained_variance=False, verbose=False):
    """
    Perform PCA on body shape angle time series.
    
    Parameters
    ----------
    thetas : pd.DataFrame
        Columns: ['ID', 't', 'theta_1', 'theta_2', ...]
    n_pcs : int
        Number of principal components to retain
    get_explained_variance : bool, default=False
        If True, return explained variance ratios
    verbose : bool, default=False
        Print PCA summary
        
    Returns
    -------
    tuple
        - pcs: DataFrame with ['ID', 't', 'PC1', 'PC2', ...]
        - eigenvectors: array (n_pcs, n_thetas) - shape modes
        - explained_variance: array (n_pcs,) - only if get_explained_variance=True
    """
    # Separate metadata and angle columns
    meta_cols = thetas[['ID', 't']].reset_index(drop=True)
    angle_cols = thetas.filter(like='theta').reset_index(drop=True)
    
    # Fit PCA
    pca = PCA(n_components=n_pcs)
    pca.fit(angle_cols)
    
    # Transform and combine with metadata
    pcs_data = pca.transform(angle_cols)
    pc_cols = [f'PC{i+1}' for i in range(n_pcs)]
    pcs_df = pd.DataFrame(pcs_data, columns=pc_cols)
    pcs = pd.concat([meta_cols, pcs_df], axis=1)
    
    if verbose:
        print(f"PCA: {n_pcs} components")
        print(f"Explained variance: {pca.explained_variance_ratio_}")
    
    if get_explained_variance:
        return pcs, pca.components_, pca.explained_variance_ratio_
    else:
        return pcs, pca.components_


def perform_PCA_exps(df, pooled=False, n_pcs=3):
    """
    Perform PCA on body shape angles across fish experiments.
    
    Parameters
    ----------
    df : pd.DataFrame
        Angular data with columns ['ID', 't', 'theta_1', 'theta_2', ...]
    pooled : bool, default=False
        If True, compute single PCA for all fish combined
        If False, compute separate PCA for each fish
    n_pcs : int, default=3
        Number of principal components to extract
        
    Returns
    -------
    tuple
        - PCS: DataFrame with principal component scores ['ID', 't', 'PC1', 'PC2', ...]
        - EIGENTHETAS: Eigenvectors (shape modes)
          * If pooled: array (n_pcs, n_thetas)
          * If not pooled: array (n_fish, n_pcs, n_thetas)
        - EXPL_VAR: Explained variance ratio
          * If pooled: array (n_pcs,)
          * If not pooled: array (n_fish, n_pcs)
    """
    df = df.copy()

    if pooled:
        # Single PCA for all fish
        print("🐟 Performing PCA for all fish pooled together")
        PCS, EIGENTHETAS, EXPL_VAR = perform_pca(
            df, n_pcs=n_pcs,
            get_explained_variance=True,
            verbose=False
        )

    else:
        # Separate PCA for each fish
        df['fish_ID'] = df['ID'].apply(get_fish_ID)

        PCS_list = []
        EIGENTHETAS = []
        EXPL_VAR = []

        for fish_id in df['fish_ID'].unique():
            print(f"🐟 Performing PCA for fish: {fish_id}")

            df_fish = df[df['fish_ID'] == fish_id].drop(columns='fish_ID')

            pcs_raw, eigenthetas, expl_var = perform_pca(
                df_fish, n_pcs=n_pcs,
                get_explained_variance=True,
                verbose=False
            )

            PCS_list.append(pcs_raw)
            EIGENTHETAS.append(eigenthetas)
            EXPL_VAR.append(expl_var)

        # Concatenate results
        PCS_raw = pd.concat(PCS_list, ignore_index=True)
        EIGENTHETAS = np.array(EIGENTHETAS)
        EXPL_VAR = np.array(EXPL_VAR)
    
    return PCS, EIGENTHETAS, EXPL_VAR


# ============================================================================
# Shape Reconstruction
# ============================================================================

def reconstruct_pose(angles, L=None):
    """
    Reconstruct body coordinates from segment angles.
    
    Parameters
    ----------
    angles : array-like or pd.DataFrame
        Body segment angles (n_segments,)
    L : float, optional
        Total body length for scaling. If None, normalize to unit length
        
    Returns
    -------
    np.ndarray
        Body point coordinates (n_segments+1, 2), starting at origin (0,0)
        
    Notes
    -----
    Integrates angles to reconstruct body shape, assuming equal segment lengths.
    """
    if isinstance(angles, pd.DataFrame):
        angles = angles.values
    
    n = len(angles)
    coords = np.zeros((n+1, 2))
    
    # Cumulative integration of angles
    for i in range(1, n+1):
        coords[i, 0] = coords[i-1, 0] + np.cos(angles[i-1])
        coords[i, 1] = coords[i-1, 1] + np.sin(angles[i-1])
    
    # Scale by body length or normalize to unit length
    scale = L / n if L is not None else 1.0 / n
    coords *= scale
    
    return coords


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_cumulative_explained_variance(EXPL_VAR, EXPL_VAR_pooled, plotpath=None):
    """
    Plot cumulative explained variance for each fish and pooled result.
    
    Parameters
    ----------
    EXPL_VAR : np.ndarray
        Per-fish explained variance ratios, shape (n_fish, n_pcs)
    EXPL_VAR_pooled : np.ndarray
        Pooled explained variance ratios, shape (n_pcs,)
    plotpath : str, optional
        Directory to save figure
    """
    # Validate inputs
    EXPL_VAR = np.asarray(EXPL_VAR)
    EXPL_VAR_pooled = np.asarray(EXPL_VAR_pooled)
    if EXPL_VAR.ndim != 2:
        raise ValueError("EXPL_VAR must be 2D array (n_fish, n_pcs)")
    if EXPL_VAR_pooled.ndim != 1:
        raise ValueError("EXPL_VAR_pooled must be 1D array (n_pcs,)")
    
    n_fish, n_pcs = EXPL_VAR.shape
    if n_pcs != EXPL_VAR_pooled.shape[0]:
        raise ValueError("PC count mismatch between EXPL_VAR and EXPL_VAR_pooled")
    
    x = np.arange(1, n_pcs + 1)
    
    # Compute cumulative explained variance
    cum_var = np.cumsum(EXPL_VAR, axis=1)
    cum_var_pooled = np.cumsum(EXPL_VAR_pooled)
    
    # Plot
    plt.figure(figsize=(4*CM, 6*CM), dpi=DPI)
    
    # Individual fish (semi-transparent)
    for i in range(n_fish):
        for pc in range(n_pcs):
            plt.scatter(x[pc], cum_var[i, pc],
                       s=10, linewidths=0, color=COLS_PCS[pc], alpha=0.5)
    
    # Pooled result (prominent)
    for pc in range(n_pcs):
        plt.scatter(x[pc], cum_var_pooled[pc],
                   marker='s', s=60, lw=0.75, edgecolor='w', color=COLS_PCS[pc])
    
    plt.ylabel('Cum. Expl. Var.', fontsize=FONTSIZE)
    plt.xlabel('PC', fontsize=FONTSIZE)
    plt.xticks(x, x, fontsize=LABELSIZE)
    plt.yticks(fontsize=LABELSIZE)
    plt.ylim(0.4, 1.05)
    plt.xlim(0.6, n_pcs + 0.4)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.gca().set_axisbelow(True)
    
    sns.despine()
    plt.tight_layout()
    
    if plotpath:
        os.makedirs(plotpath, exist_ok=True)
        plt.savefig(os.path.join(plotpath, 'cumulative_explained_variance.pdf'),
                   dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_explained_variance_vs_age(EXPL_VAR, EXPL_VAR_pooled, fish_IDs, 
                                   plotpath=None, smooth=True, sigma=1):
    """
    Plot explained variance trends across age for each PC.
    
    Parameters
    ----------
    EXPL_VAR : np.ndarray
        Per-fish explained variance, shape (n_fish, n_pcs)
    EXPL_VAR_pooled : np.ndarray
        Pooled explained variance, shape (n_pcs,)
    fish_IDs : array-like
        Fish identifiers (format: 'age_date_fish')
    plotpath : str, optional
        Directory to save figure
    smooth : bool, default=True
        Apply Gaussian smoothing to mean curves
    sigma : float, default=1
        Smoothing parameter
    """
    # Validate inputs
    EXPL_VAR = np.asarray(EXPL_VAR)
    EXPL_VAR_pooled = np.asarray(EXPL_VAR_pooled)
    if EXPL_VAR.ndim != 2:
        raise ValueError("EXPL_VAR must be 2D array")
    if EXPL_VAR_pooled.ndim != 1:
        raise ValueError("EXPL_VAR_pooled must be 1D array")
    
    n_fish, n_pcs = EXPL_VAR.shape
    if EXPL_VAR_pooled.shape[0] != n_pcs:
        raise ValueError("PC count mismatch")
    if len(fish_IDs) != n_fish:
        raise ValueError("fish_IDs length must match EXPL_VAR rows")
    
    # Extract ages from fish IDs
    try:
        ages = np.array([int(id_.split('_')[0]) for id_ in fish_IDs])
    except Exception as e:
        raise ValueError("Unable to parse ages from fish_IDs (expected 'age_...')") from e
    
    # Create long-form DataFrame
    columns = [f'PC{i+1}' for i in range(n_pcs)]
    df = pd.DataFrame(EXPL_VAR, columns=columns)
    df['age'] = ages
    df_melt = df.melt(id_vars='age', value_vars=columns,
                     var_name='PC', value_name='Explained Variance')
    
    # Compute mean per age and PC
    means = df_melt.groupby(['age', 'PC'])['Explained Variance'].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(9*CM, 6*CM), dpi=DPI)
    
    for i, pc in enumerate(columns):
        pc_data = df_melt[df_melt['PC'] == pc]
        pc_mean = means[means['PC'] == pc].sort_values('age')
        x_vals = pc_mean['age'].values
        y_vals = pc_mean['Explained Variance'].values
        
        # Individual points
        plt.scatter(pc_data['age'], pc_data['Explained Variance'],
                   alpha=0.8, color=COLS_PCS[i], edgecolors='w',
                   linewidths=0.3, s=20, label=pc)
        
        # Mean curve (smoothed if requested)
        if smooth and len(y_vals) >= 3:
            y_smooth = gaussian_filter1d(y_vals, sigma=sigma)
            plt.plot(x_vals, y_smooth, color=COLS_PCS[i], lw=0.75)
        else:
            plt.plot(x_vals, y_vals, color=COLS_PCS[i], lw=0.75)
        
        # Pooled baseline
        plt.axhline(y=EXPL_VAR_pooled[i], color=COLS_PCS[i],
                   linestyle='--', linewidth=0.75)
    
    plt.xlabel('Age (dpf)', fontsize=FONTSIZE)
    plt.ylabel('Expl. Var.', fontsize=FONTSIZE)
    plt.xticks(fontsize=LABELSIZE)
    plt.yticks(fontsize=LABELSIZE)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=LABELSIZE-1, ncol=min(3, n_pcs),
              framealpha=1, frameon=False, loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    
    if plotpath:
        os.makedirs(plotpath, exist_ok=True)
        plt.savefig(os.path.join(plotpath, 'explained_variance_vs_age.pdf'),
                   dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_eigenfish(EIGENTHETAS, EIGENTHETAS_pooled, plotpath=None):
    """
    Visualize eigenshapes (shape modes) for pooled and per-fish PCAs.
    
    Parameters
    ----------
    EIGENTHETAS : np.ndarray
        Per-fish eigenvectors, shape (n_fish, n_pcs, n_thetas)
    EIGENTHETAS_pooled : np.ndarray
        Pooled eigenvectors, shape (n_pcs, n_thetas)
    plotpath : str, optional
        Directory to save figure
    """
    # Validate inputs
    EIGENTHETAS = np.asarray(EIGENTHETAS)
    EIGENTHETAS_pooled = np.asarray(EIGENTHETAS_pooled)
    if EIGENTHETAS.ndim != 3:
        raise ValueError("EIGENTHETAS must be 3D (n_fish, n_pcs, n_thetas)")
    if EIGENTHETAS_pooled.ndim != 2:
        raise ValueError("EIGENTHETAS_pooled must be 2D (n_pcs, n_thetas)")
    if EIGENTHETAS.shape[1] != EIGENTHETAS_pooled.shape[0]:
        raise ValueError("PC count mismatch")
    
    # Reconstruct shapes from angles
    EIGENSHAPE_pooled = np.apply_along_axis(reconstruct_pose, 1, EIGENTHETAS_pooled)
    EIGENSHAPE = np.apply_along_axis(reconstruct_pose, 2, EIGENTHETAS)
    n_fish, n_pcs, n_points, _ = EIGENSHAPE.shape
    
    # Align per-fish signs with pooled (compare final y-coordinate)
    for pc in range(n_pcs):
        pooled_sign = np.sign(EIGENSHAPE_pooled[pc, -1, 1])
        for fish in range(n_fish):
            fish_sign = np.sign(EIGENSHAPE[fish, pc, -1, 1])
            if fish_sign != pooled_sign and fish_sign != 0:
                EIGENSHAPE[fish, pc, :, 1] *= -1
    
    # Compute mean and std across fish
    EIGENSHAPE_mean = np.mean(EIGENSHAPE, axis=0)
    EIGENSHAPE_std = np.std(EIGENSHAPE, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for pc in range(n_pcs):
        # Faint per-fish traces
        for fish in range(n_fish):
            ax.plot(EIGENSHAPE[fish, pc, :, 0], EIGENSHAPE[fish, pc, :, 1],
                   color=COLS_PCS[pc], alpha=0.4, lw=0.25)
        
        # Shaded uncertainty band
        ax.fill_between(EIGENSHAPE_pooled[pc, :, 0],
                       EIGENSHAPE_mean[pc, :, 1] - EIGENSHAPE_std[pc, :, 1],
                       EIGENSHAPE_mean[pc, :, 1] + EIGENSHAPE_std[pc, :, 1],
                       color=COLS_PCS[pc], alpha=0.3, lw=0)
        
        # Pooled central shape
        ax.plot(EIGENSHAPE_pooled[pc, :, 0], EIGENSHAPE_pooled[pc, :, 1],
               color=COLS_PCS[pc], lw=1.5, label=f'PC{pc+1} pooled')
    
    # Clean axes
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if plotpath:
        os.makedirs(plotpath, exist_ok=True)
        plt.savefig(os.path.join(plotpath, 'eigenshapes.svg'),
                   format='svg', bbox_inches='tight')
    
    plt.show()


# ============================================================================
# I/O Functions
# ============================================================================

def save_array(path, verbose=True, **arrays):
    """
    Save multiple numpy arrays to .npy files.
    
    Parameters
    ----------
    path : str
        Output directory
    verbose : bool, default=True
        Print confirmation messages
    **arrays
        Named arrays to save
    """
    os.makedirs(path, exist_ok=True)
    
    for name, arr in arrays.items():
        filepath = os.path.join(path, f"{name}.npy")
        np.save(filepath, arr)
        if verbose:
            print(f"✅ Saved {name}.npy")


def save_h5(datapath, verbose=True, **dataframes):
    """
    Save multiple DataFrames to HDF5 files.
    
    Parameters
    ----------
    datapath : str
        Output directory
    verbose : bool, default=True
        Print confirmation messages
    **dataframes
        Named DataFrames to save
    """
    os.makedirs(datapath, exist_ok=True)
    
    for name, df in dataframes.items():
        filepath = os.path.join(datapath, f'{name}.h5')
        df.to_hdf(filepath, key='df', mode='w', complevel=9, complib='blosc')
        if verbose:
            print(f"✅ Saved {name}.h5")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Execute PCA analysis pipeline."""
    
    print("=" * 60)
    print("FISH BODY SHAPE PCA ANALYSIS")
    print("=" * 60)
    
    # ================================
    # Setup paths
    # ================================
    
    base_path = os.path.join(os.getcwd(), 'coraggioso+_2026')
    input_path = os.path.join(base_path, 'data_processing', 'data_output')
    output_path = os.path.join(base_path, 'postural_space', 'pca_output')
    
    # ================================
    # Load data
    # ================================
    
    print("\n1. Loading preprocessed data...")
    
    INFO = pd.read_hdf(os.path.join(input_path, 'INFO.h5'), key='df')
    POSE = pd.read_hdf(os.path.join(input_path, 'POSE.h5'), key='df')
    
    IDs = INFO['ID'].unique()
    fish_IDs = INFO['ID'].str.rsplit('_', n=1).str[0].unique()
    
    print(f"  Loaded {len(IDs)} experiments from {len(fish_IDs)} fish")
    
    # ================================
    # Perform PCA
    # ================================
    
    print("\n2. Performing PCA...")
    
    n_pcs = 3
    
    # Per-fish PCA
    print("\n  Per-fish PCA:")
    PCS, EIGENTHETAS, EXPL_VAR = perform_PCA_exps(POSE, pooled=False, n_pcs=n_pcs)
    
    # Pooled PCA
    print("\n  Pooled PCA:")
    PCS_pooled, EIGENTHETAS_pooled, EXPL_VAR_pooled = perform_PCA_exps(
        POSE, pooled=True, n_pcs=n_pcs
    )
    
    # ================================
    # Generate plots
    # ================================
    
    print("\n3. Generating visualizations...")
    
    plot_cumulative_explained_variance(EXPL_VAR, EXPL_VAR_pooled, plotpath=output_path)
    plot_explained_variance_vs_age(EXPL_VAR, EXPL_VAR_pooled, fish_IDs, plotpath=output_path)
    plot_eigenfish(EIGENTHETAS, EIGENTHETAS_pooled, plotpath=output_path)
    
    # ================================
    # Save results
    # ================================
    
    print("\n4. Saving results...")
    
    # Save numpy arrays
    save_array(
        output_path,
        EIGENTHETAS=EIGENTHETAS,
        EXPL_VAR=EXPL_VAR,
        EIGENTHETAS_pooled=EIGENTHETAS_pooled,
        EXPL_VAR_pooled=EXPL_VAR_pooled
    )
    
    # Save DataFrames
    save_h5(
        output_path,
        PCS=PCS,
        PCS_pooled=PCS_pooled
    )
    
    print("\n" + "=" * 60)
    print("✅ PCA ANALYSIS COMPLETE")
    print(f"   Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
