import os
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from functions.utilities import *
from functions.visual_flow_behavior_analysis import *
from functions.brain_analysis import *
import pandas as pd
from skimage.filters import threshold_otsu,threshold_li

import numpy as np

def compute_tau_on_off(mask, dt=1/300):
    # Convert to numpy array in case it's not
    mask = np.asarray(mask)

    # Find change points (where value flips)
    change_points = np.where(np.diff(mask.astype(int)) != 0)[0] + 1

    # Add start and end
    segments = np.split(mask, change_points)

    tau_on = []
    tau_off = []

    for segment in segments:
        duration = len(segment) * dt
        if segment[0]:  # True
            tau_on.append(duration)
        else:           # False
            tau_off.append(duration)

    return np.array(tau_on), np.array(tau_off)

import numpy as np

def compute_peaks_on_segments(mask, signal, dt=1/300, peak_window=0.3):
    mask = np.asarray(mask)
    signal = np.asarray(signal)

    # Identify segment change points
    change_points = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
    segments = np.split(mask, change_points)
    signal_segments = np.split(signal, change_points)

    peaks_on = []
    peaks_on_indices = []
    peak_window_samples = int(peak_window / dt)

    signal_index = 0  # To track global index

    for segment, sig_segment in zip(segments, signal_segments):
        if segment[0]:  # Only for True segments
            window_end = min(peak_window_samples, len(sig_segment))
            if window_end > 0:
                window_signal = sig_segment[:window_end]
                local_peak_idx = np.argmax(window_signal)
                global_peak_idx = signal_index + local_peak_idx
                peaks_on.append(signal[global_peak_idx])
                peaks_on_indices.append(global_peak_idx)
        signal_index += len(sig_segment)

    return np.array(peaks_on), np.array(peaks_on_indices)


def optimal_bins_fd(data):
    # Compute the 75th and 25th percentiles of the data
    q75, q25 = np.percentile(data, [75, 25])

    # Calculate the interquartile range (IQR)
    iqr = q75 - q25

    # Compute the optimal bin width using the Freedman–Diaconis rule
    # The formula is: bin width = 2 * IQR / n^(1/3), where n is the number of data points
    bin_width = 2 * iqr * len(data) ** (-1/3)

    # Calculate the number of bins as the range of the data divided by the bin width
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width))

    # Return the computed number of bins
    return num_bins


def extract_speed_onsets(index_group, vfish_c_list, onset_indices_list, bout_duration_list,
                         dur_thr=0.8, dt=1/300, t_before=0.1, t_after=1):
    """
    Extract onset-centered velocity segments for a group of fish indices.
    Handles bounds and pads with NaNs as needed.
    
    Returns:
        List of (valid bouts × time) 2D arrays
    """
    i_before = round(t_before / dt)
    i_after = round(t_after / dt)
    total_window = i_before + i_after
    
    vfish_c_onsets_group = []

    for i in index_group:
        n_bouts = len(onset_indices_list[i])
        vfish_c_onsets = np.full((n_bouts, total_window), np.nan)

        for n in range(n_bouts):
            onset = onset_indices_list[i][n]
            start = max(0, onset - i_before)
            stop = onset + i_after

            if onset - i_before < 0:
                print('start before')

            if stop > len(vfish_c_list[i]):
                missing = stop - len(vfish_c_list[i])
                #print('stop after')
                vfish_c_list[i] = np.concatenate([vfish_c_list[i], np.full(missing, np.nan)])

            segment = np.copy(vfish_c_list[i][start:stop])
            segment[:i_before] = 0

            if bout_duration_list[i][n] < t_after:
                cut = i_before + round(bout_duration_list[i][n] / dt)
                segment[cut:] = 0

            vfish_c_onsets[n] = segment

        valid_mask = np.array(bout_duration_list[i]) > 0
        vfish_c_onsets_group.append(vfish_c_onsets[valid_mask])

    return vfish_c_onsets_group


def extract_speed_onsets_continuous(index_group, vfish_c_list, onset_indices_list, bout_duration_list, cont_bout_dur,
                         dur_thr=0.8, dt=1/300, t_before=0.1, t_after=1):
    """
    Extract onset-centered velocity segments for a group of fish indices.
    Handles bounds and pads with NaNs as needed.
    
    Returns:
        List of (valid bouts × time) 2D arrays
    """
    i_before = round(t_before / dt)
    i_after = round(t_after / dt)
    total_window = i_before + i_after
    
    vfish_c_onsets_group = []
    time_onsets_cont_group = []
    for i in index_group:
        n_bouts = len(onset_indices_list[i])
        vfish_c_onsets = np.full((n_bouts, total_window), np.nan)
        time_onsets = np.full((n_bouts, total_window), np.nan)

        for n in range(n_bouts):
            onset = onset_indices_list[i][n]
            start = max(0, onset - i_before)
            stop = onset + i_after

            if onset - i_before < 0:
                print('start before')

            if stop > len(vfish_c_list[i]):
                missing = stop - len(vfish_c_list[i])
                #print('stop after')
                vfish_c_list[i] = np.concatenate([vfish_c_list[i], np.full(missing, np.nan)])
                
    
            segment = np.copy(vfish_c_list[i][start:stop])
            segment[:i_before] = 0


            if bout_duration_list[i][n] < t_after:
                cut = i_before + round(bout_duration_list[i][n] / dt)
                segment[cut:] = 0

            vfish_c_onsets[n] = segment
            time_onsets[n] = np.linspace(-i_before * dt, (total_window - i_before - 1) * dt, total_window)

        valid_mask = np.array(bout_duration_list[i]) > cont_bout_dur
        vfish_c_onsets_group.append(vfish_c_onsets[valid_mask])
        time_onsets_cont_group.append(time_onsets[valid_mask])

    return time_onsets_cont_group, vfish_c_onsets_group




from scipy.stats import sem

def compute_pooledstats_vfis_onset(group):
    """
    Stack all bouts across fish and compute mean, SEM, std, and asymmetric IQR over time.
    """
    all_bouts = np.vstack(group)  # shape: (total_bouts, time_points)

    mean = np.nanmean(all_bouts, axis=0)
    std = np.nanstd(all_bouts, axis=0)
    sem_vals = sem(all_bouts, axis=0, nan_policy='omit')

    # Compute median and IQR
    median = np.nanmedian(all_bouts, axis=0)
    q25 = np.nanquantile(all_bouts, 0.25, axis=0)
    q75 = np.nanquantile(all_bouts, 0.75, axis=0)
    iqr_lower = median - q25
    iqr_upper = q75 - median

    return mean, sem_vals, std, median, iqr_lower, iqr_upper



def compute_stats_vfish_onset_across_fish(group):
    """
    Compute per-fish mean and median time series,
    then return the mean and SEM across fish for both.

    Parameters:
    - group: list of arrays (one per fish), each array of shape (n_bouts, time_points)

    Returns:
    - mean_of_means: average across fish of per-fish mean curves
    - sem_of_means: SEM across fish of per-fish mean curves
    - mean_of_medians: average across fish of per-fish median curves
    - sem_of_medians: SEM across fish of per-fish median curves
    """
    fish_means = []
    fish_medians = []

    for fish_data in group:
        if fish_data.size == 0:
            continue
        # fish_data: shape (n_bouts, time_points)
        fish_mean = np.nanmean(fish_data, axis=0)
        fish_median = np.nanmedian(fish_data, axis=0)

        fish_means.append(fish_mean)
        fish_medians.append(fish_median)

    fish_means = np.array(fish_means)
    fish_medians = np.array(fish_medians)

    # Compute group-level summary
    mean_of_means = np.nanmean(fish_means, axis=0)
    sem_of_means = sem(fish_means, axis=0, nan_policy='omit')

    mean_of_medians = np.nanmean(fish_medians, axis=0)
    sem_of_medians = sem(fish_medians, axis=0, nan_policy='omit')

    return mean_of_means, sem_of_means, mean_of_medians, sem_of_medians



from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def extract_first_peak_times_and_values(vfish_c_onsets_tot, sigma=10, dt=1/300, t_before=0.1):
    i_before = round(t_before / dt)
    total_points = vfish_c_onsets_tot[0].shape[1]
    time = np.linspace(-i_before * dt, (total_points - i_before - 1) * dt, total_points)

    peak_times_per_fish = []
    peak_values_per_fish = []

    for fish_curves in vfish_c_onsets_tot:
        fish_peak_times = []
        fish_peak_values = []

        for curve in fish_curves:
            if np.all(np.isnan(curve)):
                continue
            smoothed = gaussian_filter1d(curve, sigma=sigma)
            peaks, _ = find_peaks(smoothed)
            if len(peaks) > 0:
                first_peak_idx = peaks[0]
                fish_peak_times.append(time[first_peak_idx])
                fish_peak_values.append(smoothed[first_peak_idx])

        peak_times_per_fish.append(np.array(fish_peak_times))
        peak_values_per_fish.append(np.array(fish_peak_values))

    return peak_times_per_fish, peak_values_per_fish

def extract_first_peak_times_and_values_single_onset(vfish_c_onset, t_cont, sigma=10):
    smoothed = gaussian_filter1d(vfish_c_onset, sigma=sigma)
    peaks, _ = find_peaks(smoothed)
    if len(peaks) > 0:
        first_peak_idx = peaks[0]

    return t_cont[first_peak_idx], smoothed[first_peak_idx]