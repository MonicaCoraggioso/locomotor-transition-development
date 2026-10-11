# check each simulation

import subprocess
import os
from pathlib import Path
import logging
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from utils import get_root_folder

# set up logging
logging.basicConfig(level=logging.INFO)
root_folder = get_root_folder()

cfg = OmegaConf.load(root_folder / "config.yaml")


db_file = root_folder / "results.db"
db_file.unlink(missing_ok=True)
os.system("python create_db.py")
assert db_file.is_file(), "Results database not found"


def validate_simulation(out_folder: Path):
    env = os.environ.copy()
    env["OUTPUT_FOLDER"] = str(out_folder)

    # call jupyter notebook
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "html",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            str(out_folder / "res.html"),
            "compare_res.ipynb",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.info("Error executing notebook")
        logging.error(result.stderr)
    else:
        logging.info("Notebook executed successfully")


fish_to_be_processed = set()
# Walk through each age/session
for age, recs in cfg.sessions.items():
    assert type(age) is str
    if age not in ("20dpf_exp1", "22dpf"):
        continue

    for fish_id, rec in recs.items():
        fish_id = str(fish_id)
        if rec.get("status") == "discard":
            continue
        for bout in rec.bouts:
            out_folder = (
                root_folder
                / age
                / "sim"
                / fish_id
                / f"frame_{bout.start_frame}_{bout.end_frame}"
            )
            fish_to_be_processed.add(out_folder)

# Use joblib to parallelize the validation of simulations
Parallel(n_jobs=5, verbose=10)(
    delayed(validate_simulation)(out_folder)
    for out_folder in tqdm(fish_to_be_processed)
)
