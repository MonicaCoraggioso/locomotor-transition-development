"""
Project path configuration for publication-ready analyses.

All paths are relative to the project root (parent of `code/`). Use these
constants instead of hardcoded paths so the repository runs on any machine.

Usage in notebooks (from project root, or from inside code/ or code/subdir/):
    import sys
    from pathlib import Path
    _code = Path.cwd() if (Path.cwd() / "paths.py").exists() else Path.cwd().parent
    if str(_code.resolve()) not in [str(Path(p).resolve()) for p in sys.path]:
        sys.path.insert(0, str(_code.resolve()))
    from paths import DATA_DIR, FIGURES_DIR, SPEED_SEGMENTED_DIR, media_dir, ...
"""

from pathlib import Path
import os as _os

# -----------------------------------------------------------------------------
# Project layout: PROJECT_ROOT contains "code/" and "data/"
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_CODE_DIR = _THIS_FILE.parent
PROJECT_ROOT = _CODE_DIR.parent

# Main output dirs (all under project root)
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
INPUT_DATA_DIR = PROJECT_ROOT / "input_data"

# Data subdirs used by analysis pipelines
SPEED_SEGMENTED_DIR = DATA_DIR / "speed_segmented"
SPEED_ONSETS_DIR = DATA_DIR / "speed_onsets"
VIRTUAL_REALITY_DIR = DATA_DIR / "virtual_reality"
TRACKING_DIR = DATA_DIR / "tracking"

# Manipulation-experiment data: data/manipulation_experiments/<experiment>/
# Override with env vars if your raw data lives elsewhere, e.g.:
#   export DATA_DARK_2WPF="/path/to/dark_2_wpf"
_DATA_MANIP = DATA_DIR / "manipulation_experiments"
DATA_DARK_2WPF = Path(_os.environ.get("DATA_DARK_2WPF", str(_DATA_MANIP / "dark_2_wpf")))
DATA_DARK_4WPF = Path(_os.environ.get("DATA_DARK_4WPF", str(_DATA_MANIP / "dark_4_wpf")))
DATA_VISCOSITY = Path(_os.environ.get("DATA_VISCOSITY", str(_DATA_MANIP / "viscosity")))
DATA_VISCOSITY_ZF = Path(_os.environ.get("DATA_VISCOSITY_ZF", str(_DATA_MANIP / "viscosity_ZF")))
DATA_LATERAL_LINE = Path(_os.environ.get("DATA_LATERAL_LINE", str(_DATA_MANIP / "ablation_lateral_line")))


def media_dir(base_dir, create=True):
    """
    Return the `media` subdirectory under `base_dir`, e.g. for saving
    pipeline figures. Creates the dir if `create` is True.
    """
    out = Path(base_dir) / "media"
    if create and not out.exists():
        _os.makedirs(out, exist_ok=True)
    return out


def notebook_media_dir(*subdirs, create=True):
    """
    Return the `media` folder in the same subfolder as the notebook.
    Pass the subfolder(s) of code/ that contain the notebook, e.g.:
      notebook_media_dir("bout_durations")  -> code/bout_durations/media/
      notebook_media_dir()                  -> code/media/
    Creates the directory if create is True.
    """
    d = _CODE_DIR.joinpath(*subdirs) / "media"
    if create and not d.exists():
        _os.makedirs(d, exist_ok=True)
    return d


def ensure_figures_dir(create=True):
    """Return FIGURES_DIR and create it if requested."""
    if create and not FIGURES_DIR.exists():
        _os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR
