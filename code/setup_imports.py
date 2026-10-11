"""
Shared setup for analysis notebooks: path configuration and code-directory discovery.

Add the `code` directory to sys.path so that `from paths import ...` and
`from functions.xxx import ...` work regardless of whether the notebook is
run from the project root, `code/`, or `code/subdir/`.

Usage in any notebook under `code/` or `code/*/`:
    import sys
    from pathlib import Path
    # Ensure code directory is on path (handles both code/ and code/subdir/)
    _cwd = Path.cwd()
    _code = _cwd if (_cwd / "paths.py").exists() else _cwd.parent
    if _code.parent != _code and (_code.parent / "data").is_dir():
        _code = _code.parent / "code" if (_code.parent / "code").is_dir() else _code
    if str(_code.resolve()) not in [str(Path(p).resolve()) for p in sys.path]:
        sys.path.insert(0, str(_code.resolve()))
    from paths import (
        PROJECT_ROOT, DATA_DIR, FIGURES_DIR, INPUT_DATA_DIR,
        SPEED_SEGMENTED_DIR, SPEED_ONSETS_DIR, VIRTUAL_REALITY_DIR, TRACKING_DIR,
        DATA_DARK_2WPF, DATA_DARK_4WPF, DATA_VISCOSITY, DATA_VISCOSITY_ZF, DATA_LATERAL_LINE,
        media_dir, ensure_figures_dir,
    )
"""

from pathlib import Path
import sys

def _find_code_dir():
    """Locate the project code directory (contains paths.py and functions/)."""
    cwd = Path.cwd().resolve()
    for d in [cwd, cwd.parent] + list(cwd.parents)[:8]:
        if (d / "paths.py").exists() and (d / "functions").is_dir():
            return d
        if (d / "code").is_dir() and ((d / "code") / "paths.py").exists():
            return d / "code"
    return cwd

_CODE_DIR = _find_code_dir()
if str(_CODE_DIR) not in [str(Path(p).resolve()) for p in sys.path]:
    sys.path.insert(0, str(_CODE_DIR))

# Re-export path constants and helpers for convenience
from paths import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    INPUT_DATA_DIR,
    SPEED_SEGMENTED_DIR,
    SPEED_ONSETS_DIR,
    VIRTUAL_REALITY_DIR,
    TRACKING_DIR,
    DATA_DARK_2WPF,
    DATA_DARK_4WPF,
    DATA_VISCOSITY,
    DATA_VISCOSITY_ZF,
    DATA_LATERAL_LINE,
    media_dir,
    ensure_figures_dir,
)
