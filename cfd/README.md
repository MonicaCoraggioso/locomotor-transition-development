# CFD: swimming efficiency across body sizes and swimming styles

This project provides computational tools and analysis scripts for studying the swimming behavior and energetic efficiency of Danionella fish using Computational Fluido Dynamics (CFD) simulations. It includes workflows for data processing, simulation, and figure generation for scientific investigations.

## Figure generation from CFD simulations

Create the conda environment with `conda env create -f environment.yml` (environment name: `cfd`).

For investigating the energetic efficiency at different fish lengths and swimming styles, you can run [`compare_efficiency.ipynb`](compare_efficiency.ipynb) to create Fig. 2{d,e,f,g} and Fig. S6{e,f}

For investigating the swimming styles at the intermediate length (23dpf), you can run

- [`postprocess.py`](postprocess.py) to check simulations and create the database file.
- [`compare_bout_type.py`](compare_bout_type.py) to create Fig. S6{c,d}
- [`speed_plots.py`](speed_plots.py) to create Fig. S6b

## CFD simulation

The CFD simulations were conducted using the [ZFLOW](https://github.com/GoodhillLab/ZFLOW) repository based on [IBAMR](https://ibamr.github.io/) software.
The simulation inputs and outputs are on [Zenodo](https://doi.org/10.5281/zenodo.17650136) (`CFD_simulations.7z`). Download them with `python scripts/download_data.py cfd` from the repository root; they are unpacked to `data/cfd/`, which `folder_path.json` points to. Run the scripts from inside `cfd/`.
Using the [ZFLOW](https://github.com/GoodhillLab/ZFLOW) repository, you can rerun the simulations to reproduce the output files using the following command:
`./main3d input3d` where `input3d` is the name of the input file at each sub-folder in `data/cfd/`.
For detailed instructions and more information on the output files, please refer to the [ZFLOW](https://github.com/GoodhillLab/ZFLOW) repository.

Docker image for IBAMR is available on [Docker Hub](https://hub.docker.com/repository/docker/d0ckaaa/ibamr).

## Data layout

- `XXdpf/` folders: simulations of fish at the age of each video clip (alternative swimming styles in `swap_sim/`).
- `d14/` (young) and `d30/` (old) 3D models, scaled to 0.25, 0.5, 0.75 and 1 body length, swimming continuously or in burst-and-coast mode.
- `config.yaml`: bout start/end frames and global CFD parameters (e.g. FPS).

## Credits

CFD simulations and analysis code by Robert Wong and Geoffrey Goodhill (Washington University in St. Louis), using [ZFLOW](https://github.com/GoodhillLab/ZFLOW) and [IBAMR](https://ibamr.github.io/).
