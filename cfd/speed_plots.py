# %% This script plots panels in Fig. S6b.

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from utils import plot_speed, get_root_folder

main_folder = get_root_folder()
(main_folder / "figs").mkdir(exist_ok=True)
# use custom matplotlib style
plt.style.use(main_folder / "main.mplstyle")

# Load the YAML into a DictConfig
cfg = OmegaConf.load(main_folder / "config.yaml")
FPS: int = cfg.FPS

# %% Speed plots

fig, ax = plot_speed(
    main_folder, "02182025133316", "20dpf_exp1", 200, 1200, 575 / 250, 850 / 250, FPS
)
ax.set(xticks=(0, 500, 1000))

fig, ax = plot_speed(
    main_folder, "04112025160935", "22dpf", 700, 1015, 825 / 250, 1015 / 250, FPS
)
ax.set(xticks=(0, 250, 500, 750))

fig, ax = plot_speed(
    main_folder, "04112025170138", "22dpf", 50, 440, 145 / 250, 229 / 250, FPS
)

fig, ax = plot_speed(
    main_folder, "04112025170138", "22dpf", 50, 440, 229 / 250, 339 / 250, FPS
)
ax.set(xticks=(0, 200, 400))
