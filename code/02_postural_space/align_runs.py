#!/usr/bin/env python3
"""
Fish Experiment Run Alignment and Concatenation

This script aligns and concatenates multiple experimental runs from the same fish
to create continuous time series suitable for longitudinal analysis.

Problem:
    - Multiple runs per fish have independent time values (each starts at t=0)
    - Different datasets (KINE, POSE, PCS) may have slightly different start/end times
    - Need synchronized, continuous time series for time-series analysis

Solution:
    1. Align: Find common time windows across all datasets for each run
    2. Concatenate: Join runs sequentially with continuous time (remove resets)

Input:
    - KINE, POSE: Kinematic and shape data from data processing pipeline
    - PCS, PCS_pooled: Principal components from PCA analysis

Output:
    - Same datasets with aligned runs and continuous time values

Usage:
    python align_runs.py

Author: Vito Dichio
Date: 2025-01-29
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# Helper Functions
# ============================================================================

def _ensure_boundary(seg, t0, t1):
    """
    Ensure segment includes exact boundary times by duplicating nearest rows.
    
    Parameters
    ----------
    seg : pd.DataFrame
        Data segment with 't' column
    t0 : float
        Start time boundary
    t1 : float
        End time boundary
        
    Returns
    -------
    pd.DataFrame
        Segment with guaranteed t0 and t1 entries
    """
    # Add t0 if missing
    if not np.any(np.isclose(seg['t'], t0)):
        row = seg.iloc[(seg['t'] - t0).abs().argmin()].copy()
        row['t'] = t0
        seg = pd.concat([pd.DataFrame([row]), seg], ignore_index=True)
    
    # Add t1 if missing
    if not np.any(np.isclose(seg['t'], t1)):
        row = seg.iloc[(seg['t'] - t1).abs().argmin()].copy()
        row['t'] = t1
        seg = pd.concat([seg, pd.DataFrame([row])], ignore_index=True)
    
    return seg


# ============================================================================
# Alignment Functions
# ============================================================================

def align_DFs_run(dfs, fish_IDs, verbose=False):
    """
    Align multiple DataFrames to common time windows for each run.
    
    For each fish and run, finds the overlapping time window across all
    DataFrames and trims data to this common interval. Handles runs where
    time resets (multiple trials per fish).
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrames to align, each with ['ID', 't'] columns
    fish_IDs : array-like
        Fish identifiers (without run suffixes)
    verbose : bool, default=False
        Print alignment details
        
    Returns
    -------
    list of pd.DataFrame
        Aligned DataFrames in same order as input
        
    Notes
    -----
    Alignment strategy:
    - Detect run boundaries (where t resets)
    - For each run, find max start time and min end time across DataFrames
    - Trim all DataFrames to this common window
    - Ensure exact boundaries by duplicating nearest rows if needed
    """
    n_dfs = len(dfs)
    aligned_dfs = [[] for _ in range(n_dfs)]
    
    for fish_ID in fish_IDs:
        # Extract data for this fish from each DataFrame
        subs = [df[df['ID'].str.startswith(fish_ID)].copy() for df in dfs]
        
        # Skip if any DataFrame is missing this fish
        if any(sub.empty for sub in subs):
            if verbose:
                print(f"⚠️ Skipping {fish_ID}: missing data in one or more DataFrames")
            continue
        
        # Detect run boundaries (where t decreases)
        t_vals = [sub['t'].values for sub in subs]
        resets = [np.where(np.diff(t) < 0)[0] + 1 for t in t_vals]
        bounds = [np.concatenate(([0], r, [len(t)])) for r, t in zip(resets, t_vals)]
        n_runs = min(len(b) - 1 for b in bounds)
        
        if verbose:
            print(f"\n🐟 Aligning fish: {fish_ID} — {n_runs} runs")
        
        # Process each run
        for i in range(n_runs):
            # Extract segments for this run from each DataFrame
            segs = [sub.iloc[b[i]:b[i+1]].copy() for sub, b in zip(subs, bounds)]
            
            # Find common time window
            t_starts = [seg['t'].iloc[0] for seg in segs]
            t_ends = [seg['t'].iloc[-1] for seg in segs]
            t0 = max(t_starts)  # Latest start
            t1 = min(t_ends)    # Earliest end
            
            if verbose:
                print(f"\n  ── Run {i + 1} ─────────────────────────────")
                for j, (start, end) in enumerate(zip(t_starts, t_ends)):
                    print(f"  df{j}: t_start = {start:.3f}, t_end = {end:.3f}")
                print(f"  ✂ Cutting to common window: [{t0:.3f}, {t1:.3f}]")
            
            # Trim each segment to common window
            for j, seg in enumerate(segs):
                seg = _ensure_boundary(seg, t0, t1)
                aligned = seg[(seg['t'] >= t0) & (seg['t'] <= t1)]
                aligned_dfs[j].append(aligned)
                
                if verbose:
                    print(f"    ✅ df{j}: kept {len(aligned)} points")
    
    return [pd.concat(parts, ignore_index=True) for parts in aligned_dfs]


# ============================================================================
# Concatenation Functions
# ============================================================================

def concatenate_run_tvals(df, return_reset_idxs=False, verbose=False):
    """
    Convert run-reset times to continuous time series.
    
    Detects time resets (where t decreases) and shifts subsequent runs
    to create continuous time values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 't' column containing time resets
    return_reset_idxs : bool, default=False
        If True, also return indices where resets occurred
    verbose : bool, default=False
        Print detailed concatenation info
        
    Returns
    -------
    pd.DataFrame
        Data with continuous time values
    np.ndarray, optional
        Reset indices (only if return_reset_idxs=True)
        
    Examples
    --------
    Original times with resets:
        [0, 1, 2, 3, 0, 1, 2]  # Reset at index 4
    
    After concatenation:
        [0, 1, 2, 3, 3, 4, 5]  # Continuous
    """
    t = np.array(df['t'])
    t_new = t.copy()
    
    # Find reset indices (where t decreases)
    reset_idxs = np.where(np.diff(t) < 0)[0] + 1
    run_boundaries = np.concatenate(([0], reset_idxs, [len(t)]))
    
    # Compute cumulative shifts
    last_vals = [t[run_boundaries[i+1] - 1] for i in range(len(run_boundaries) - 2)]
    shifts = np.cumsum([0] + last_vals)
    
    if verbose:
        print("=== CONCATENATE RUN T VALUES ===")
        print(f"→ Total points: {len(t)}")
        print(f"→ Detected {len(reset_idxs) + 1} segments")
        print(f"→ Reset indices: {reset_idxs.tolist()}")
        print(f"→ Cumulative shifts: {[f'{s:.3f}' for s in shifts]}")
        print("-" * 40)
    
    # Apply shifts to each segment
    for i in range(1, len(run_boundaries) - 1):
        start = run_boundaries[i]
        end = run_boundaries[i+1]
        shift = shifts[i]
        t_new[start:end] += shift
        
        if verbose:
            print(f"Segment {i}: rows {start}-{end-1}, shift={shift:.3f}")
            print(f"  Before: [{t[start]:.3f}, {t[end-1]:.3f}]")
            print(f"  After:  [{t_new[start]:.3f}, {t_new[end-1]:.3f}]")
    
    df_new = df.copy()
    df_new['t'] = t_new
    df_new = df_new.reset_index(drop=True)
    
    if verbose:
        print("✔ Concatenation complete.\n")
    
    if return_reset_idxs:
        return df_new, reset_idxs
    else:
        return df_new


def concatenate_run_tvals_DATA(DATA, return_reset_idxs=False, verbose=False):
    """
    Concatenate runs for all fish in a DataFrame.
    
    Groups data by fish (removing run suffix from ID) and applies
    time concatenation to each fish's runs.
    
    Parameters
    ----------
    DATA : pd.DataFrame
        Data with ['ID', 't'] columns
        ID format: 'age_date_fish_run' (e.g., '25_10Jun24_1_2')
    return_reset_idxs : bool, default=False
        Return reset indices per fish
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Data with continuous time per fish
    dict, optional
        Maps fish IDs to reset indices (only if return_reset_idxs=True)
    """
    processed_list = []
    reset_dict = {}
    
    # Group by fish ID (remove run suffix)
    def get_fish_id(s):
        parts = s.split('_')
        # Keep first 3 parts if 4+ parts and last is numeric
        if len(parts) > 3 and parts[-1].isdigit():
            return '_'.join(parts[:3])
        return s
    
    grouped = DATA.groupby(DATA['ID'].apply(get_fish_id))
    
    for fish_ID, group in grouped:
        if verbose:
            print(f"\nProcessing {fish_ID} ({len(group)} rows)")
        
        if return_reset_idxs:
            group_new, resets = concatenate_run_tvals(
                group, return_reset_idxs=True, verbose=verbose
            )
            reset_dict[fish_ID] = resets
        else:
            group_new = concatenate_run_tvals(group, verbose=verbose)
        
        processed_list.append(group_new)
    
    DATA_new = pd.concat(processed_list, ignore_index=True)
    
    if return_reset_idxs:
        return DATA_new, reset_dict
    else:
        return DATA_new


# ============================================================================
# I/O Functions
# ============================================================================

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
    """Execute run alignment and concatenation pipeline."""
    
    print("=" * 60)
    print("RUN ALIGNMENT AND CONCATENATION")
    print("=" * 60)
    
    # ================================
    # Setup paths
    # ================================
    
    base_path = os.path.join(os.getcwd(), 'coraggioso+_2026')
    
    datapath_kine = os.path.join(base_path, 'data_processing', 'data_output')
    datapath_pca = os.path.join(base_path, 'postural_space', 'pca_output')
    
    # ================================
    # Load data
    # ================================
    
    print("\n1. Loading data...")
    
    INFO = pd.read_hdf(os.path.join(datapath_kine, 'INFO.h5'), key='df')
    KINE = pd.read_hdf(os.path.join(datapath_kine, 'KINE.h5'), key='df')
    POSE = pd.read_hdf(os.path.join(datapath_kine, 'POSE.h5'), key='df')
    
    PCS = pd.read_hdf(os.path.join(datapath_pca, 'PCS.h5'), key='df')
    PCS_pooled = pd.read_hdf(os.path.join(datapath_pca, 'PCS_pooled.h5'), key='df')
    
    fish_IDs = INFO['ID'].str.rsplit('_', n=1).str[0].unique()
    
    print(f"  Loaded data for {len(fish_IDs)} fish")
    print(f"  Total experiments: {len(INFO)}")
    
    # ================================
    # Align runs
    # ================================
    
    print("\n2. Aligning runs across datasets...")
    print("  (Finding common time windows for each run)")
    
    aligned_dfs = align_DFs_run(
        [KINE, POSE, PCS, PCS_pooled],
        fish_IDs,
        verbose=True
    )
    
    KINE, POSE, PCS, PCS_pooled = aligned_dfs
    
    # ================================
    # Concatenate runs
    # ================================
    
    print("\n3. Concatenating runs into continuous time series...")
    
    KINE = concatenate_run_tvals_DATA(KINE, verbose=True)
    POSE = concatenate_run_tvals_DATA(POSE, verbose=True)
    PCS = concatenate_run_tvals_DATA(PCS, verbose=True)
    PCS_pooled = concatenate_run_tvals_DATA(PCS_pooled, verbose=True)
    
    # ================================
    # Save results
    # ================================
    
    print("\n4. Saving aligned and concatenated data...")
    
    save_h5(datapath_kine, KINE=KINE, POSE=POSE)
    save_h5(datapath_pca, PCS=PCS, PCS_pooled=PCS_pooled)
    
    print("\n" + "=" * 60)
    print("✅ ALIGNMENT AND CONCATENATION COMPLETE")
    print(f"   Kinematic data: {datapath_kine}")
    print(f"   PCA data: {datapath_pca}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
