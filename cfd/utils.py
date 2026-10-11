from pathlib import Path
from typing import Union, Mapping
from types import MappingProxyType
import json
import logging

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_root_folder() -> Path:
    """
    Get the root folder from the `folder_path.json` file.
    """
    with open(Path.cwd() / "folder_path.json", "r") as f:
        folder_path = json.load(f)
    return Path(folder_path["main"])


def energetics_calculator(fname: Union[Path, str], start_frame: int = 0, end_frame: int = -1) -> float:
    """
    This function loads the CFD power expenditure outputs, computes the total power expenditure,
    and computes the total energy usage.

    Parameters:
    fname : str
        The path to the file containing the simulation power data.

    Returns:
    E_tot : float
        The total energy usage.
    """
    # Load the simulation power data
    ibamr_power = np.loadtxt(fname).T
    if start_frame and end_frame:
        ibamr_power = ibamr_power[:, start_frame:end_frame]

    power_spent_x = -1 * (ibamr_power[1] - ibamr_power[4])
    power_spent_y = -1 * (ibamr_power[2] - ibamr_power[5])

    # Compute energy using trapezoidal integration
    t_arr = ibamr_power[0]
    E_x = np.trapezoid(power_spent_x, t_arr)
    E_y = np.trapezoid(power_spent_y, t_arr)
    E_tot = E_x + E_y

    return E_tot


def load_data(fname: Path, interpolate: bool = True) -> Mapping[str, np.ndarray]:
    """
    Load tracking data from an HDF5 file.

    Opens the file at the specified path and extracts:
      - 'speed': post-process speed (converted from mm/s to cm/s),
      - 't': time values (in seconds),
      - 'fish_coords': tail points converted from pixels to cm using a calibration factor.

    The function adjusts the fish coordinates in-place and then creates a read-only mapping
    with keys 't', 'speed', and 'fish_coords'. Each value in the mapping is a NumPy array.

    Parameters:
        fname (Path): Path to the HDF5 file.
        interpolate (bool): If True, modifies the tail points in-place to replace values between some parts of the fish.

    Returns:
        Mapping[str, np.ndarray]: A read-only mapping containing:
            't'           - time values in seconds,
            'speed'       - velocity in cm/s,
            'fish_coords' - fish coordinates in cm.
    """
    with h5py.File(fname, "r") as f:
        # this seems to be stored in mm/s, so divide by 10 to get cm/s
        v2d_ds = f["post_process/speed"][:] / 10
        t_ds = f["post_process/tvals"][:]

        px2cm = f["post_process/mm_per_pix"][()] / 10
        fish_coords = f["post_process/tail_points"][:] * px2cm
        coords_raw = f["post_process/tail_sy0"][:] * px2cm

    if interpolate:
        for coords in (fish_coords, coords_raw):
            # Interpolate the tail points to fill in the gaps
            inplace_modify_tail(coords)

    data = MappingProxyType(
        {
            "t": t_ds,  # in seconds
            "speed": v2d_ds,  # in cm/s
            "fish_coords": fish_coords,  # in cm
            "coords_raw": coords_raw,  # in cm
        }
    )

    for v in data.values():
        v.setflags(write=False)

    return data


def calculate_fish_length(fish_coords):
    # calculate arc length of 2d line segments
    line_segments = np.diff(fish_coords, axis=1)
    # calculate the length of each segment
    line_segments_len = np.linalg.norm(line_segments, axis=2)
    # calculate the cumulative length of the segments at each time point
    length_t = line_segments_len.sum(axis=1)
    return np.nanmedian(length_t)


def calc_dist(out_folder: Path, return_path: bool = False) -> float:
    """
    Calculate the distance between the first and last point travelled by the fish
    """
    df = pd.read_csv(
        out_folder / "examplespline.txt",
        sep=" ",
        header=None,
        names=["x", "y", "z", "idx"],
    )
    df = df.query("idx == 2").drop(columns=["idx"])
    res = df[["x", "y"]].to_numpy()
    dist_sim = np.linalg.norm(res[-1] - res[0])
    if return_path:
        return dist_sim, res
    return dist_sim


def get_bout(cfg, age: str, fish_id: str, t0: int, t1: int):
    """
    Get the bout object for a given age, fish_id, and time range.
    """
    # Walk through each age/session
    rec = cfg["sessions"][age][fish_id]
    for bout in rec.bouts:
        if t0 == bout.start_frame and t1 == bout.end_frame:
            return bout

    return None


def get_bout_type(cfg, age: str, fish_id: str, t0: int, t1: int) -> str:
    """
    Get the bout type for a given age, fish_id, and time range.
    """
    rename_dict = {
        "burst-and-coast": "burst",
        "continuously swimming": "cont",
    }
    # Walk through each age/session
    rec = cfg["sessions"][age][fish_id]
    for bout in rec.bouts:
        if t0 == bout.start_frame and t1 == bout.end_frame:
            return rename_dict[bout.bout_type]

    return None

def check_simulation(fname: Union[Path, str], out_folder: Path, FPS: int):
    """
    Check if the simulation is valid and return the dpf and fishlen.
    Parameters:
        fname: Union[Path, str]
        out_folder: Path
        FPS: int
    Returns:
        dpf: int
        fishlen: float
    """

    # Load the simulation power data
    ibamr_power = np.loadtxt(fname).T
    t_arr = ibamr_power[0]

    dpf_folder = out_folder.parent.parent.parent
    dpf = int(dpf_folder.stem[:2])
    fishlen = float(dpf_folder.parent.name)

    if not np.allclose(np.diff(t_arr), 1 / FPS):
        # If the time array is constant, skip this simulation
        logging.info(f"Skipping {out_folder} due to inaccurate time array")
        return False

    if dpf > 25 and t_arr[-1] < 1.2 * fishlen:
        logging.info(
            f"This is a burst-and-coast simulation but the simulation time is too short: {out_folder}, likely CFD crashed"
        )
        return False
    elif t_arr[-1] < 0.5:
        # If the simulation time is less than 1 second, skip this simulation
        logging.info(
            f"Continously swimming bouts, skipping {out_folder} due to short simulation time: either not enough frames or CFD crashed"
        )
        return False
    
    return dpf, fishlen


def plot_speed(
    main_folder: Union[str, Path],
    fish_id: str,
    age: str,
    t0: int,
    t1: int,
    start_time: float,
    end_time: float,
    FPS: int,
    window_size: int = 50,
    ax=None,
):
    """
    Plot experimental and simulated speed traces for a fish.

    Parameters:
        fish_id (str): Fish identifier.
        age (str): Age/session identifier.
        t0 (int): Start frame.
        t1 (int): End frame.
        start_time (float): Start time (in seconds) for plotting window.
        end_time (float): End time (in seconds) for plotting window.
        main_folder (Path): Root data folder.
        FPS (int): Frames per second.
        window_size (int): Moving average window size.
        ax (matplotlib.axes.Axes): Optional matplotlib axis to plot on.

    Returns:
        fig, ax: Figure and axis objects.
    """
    out_folder = main_folder / age / "sim" / fish_id / f"frame_{t0}_{t1}"
    _, res = calc_dist(out_folder, return_path=True)

    with np.load(out_folder / "tmp.npz") as f:
        t_list = f["t_list"]
        t_all = f["t_all"]
        speed_all = f["speed_all"]

    power_fname = out_folder / "Eel3d" / "Eel3d_Power_spent_struct_no_0"

    distances = np.linalg.norm(np.diff(res, axis=0), axis=1)
    speed = distances * FPS
    t_arr = np.loadtxt(power_fname, usecols=0)

    df_speed = pd.DataFrame(speed, columns=["speed"])
    df_speed["t"] = t_arr[1:] + t_all[t0]
    df_speed_avg = df_speed.rolling(window=window_size).mean()

    assert t_list[0] == 0, "Time should start at 0"

    idx = np.searchsorted(t_list, t_arr[-1])
    assert idx <= t1 - t0, "idx should be less than or equal to t1 - t0"
    t_ds_clip = t_list[:idx] + t_all[t0]
    logging.info(f"simulation: {t_ds_clip[0]:.2f}--{t_ds_clip[-1]:.2f}s")
    v2d_ds = speed_all[t0 : t0 + idx]

    idx_downsample = np.searchsorted(df_speed["t"], t_ds_clip)
    df_speed_avg_downsampled = df_speed_avg.iloc[idx_downsample]

    mask = (df_speed_avg_downsampled["t"] > start_time) & (df_speed_avg_downsampled["t"] < end_time)
    mask_arr = (t_ds_clip > start_time) & (t_ds_clip < end_time)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 1))
    else:
        fig = ax.figure

    ax.plot((t_ds_clip[mask_arr] - start_time) * 1000, v2d_ds[mask_arr], label="exp.", color="gray", linewidth=1)
    ax.plot(
        (df_speed_avg_downsampled["t"][mask] - start_time) * 1000,
        df_speed_avg_downsampled["speed"][mask],
        label="sim.",
        color="black",
        linewidth=1,
    )
    ax.set(xlabel="time (ms)", ylabel="v (cm/s)")

    return fig, ax
