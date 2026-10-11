import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr



def plot_median_pdf_errorbars(pdf_list, fish_ids, index_groups, group_colors, ylabel=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=80)

    for group_name, indices in index_groups.items():
        color = group_colors[group_name]

        for j, i in enumerate(indices):
            data = pdf_list[i]
            median_val = np.nanquantile(data, 0.5)
            iqp_val = np.nanquantile(data, 0.75) - median_val
            iqm_val = median_val - np.nanquantile(data, 0.25)

            ax.errorbar(fish_ids[i], median_val,
                        yerr=[[iqm_val], [iqp_val]],
                        fmt='o', color=color, capsize=5,
                        label=group_name if j == 0 else None)

  
    ax.set_xticks([fish_ids[i] for group in index_groups.values() for i in group])
    ax.tick_params(axis='x', rotation=-90)
    ax.set_xlabel('Fish ID')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.axhline(0, color='lightgray')
    ax.axvline(0, color='lightgray')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    

def plot_pdf_errorbars_median_and_mean(pdf_list, fish_ids, index_groups, group_colors, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), dpi=100, sharex=True, sharey=True)

    # --- Subplot 1: Median ± IQR ---
    k = 0
    for group_name, indices in index_groups.items():
        color = group_colors[k]
        for j, i in enumerate(indices):
            data = pdf_list[i]
            median_val = np.nanquantile(data, 0.5)
            iqp_val = np.nanquantile(data, 0.75) - median_val
            iqm_val = median_val - np.nanquantile(data, 0.25)

            ax1.errorbar(fish_ids[i], median_val,
                         yerr=[[iqm_val], [iqp_val]],
                         fmt='o', color=color, capsize=5,
                         label=group_name if j == 0 else None)
        k = k + 1

    ax1.set_ylabel('Median ± IQR')
    ax1.axhline(0, color='lightgray')
    ax1.axvline(0, color='lightgray')
    ax1.legend()

    # --- Subplot 2: Mean ± SD ---
    k = 0
    for group_name, indices in index_groups.items():
        color = group_colors[k]
        for j, i in enumerate(indices):
            data = pdf_list[i]
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)

            ax2.errorbar(fish_ids[i], mean_val,
                         yerr=std_val,
                         fmt='s', color=color, capsize=5,
                         label=group_name if j == 0 else None)
        k = k + 1

    ax2.set_ylabel('Mean ± SD')
    ax2.set_xlabel('Fish ID')
    ax2.axhline(0, color='lightgray')
    ax2.axvline(0, color='lightgray')

    # Shared x-ticks
    xticks = [fish_ids[i] for group in index_groups.values() for i in group]
    ax2.set_xticks(xticks)
    ax2.tick_params(axis='x', rotation=90)
    
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
    


def plot_hist_dpf_three_subplots(pdf_list, index_groups, group_colors, xlabel=None, xlim=None, nbins=None, ylog = None):


    fig, axs = plt.subplots(3, 1, figsize=(6.4, 8.5), dpi=80, sharex=True)
    j = 0
    for ax, group in zip(axs, ['1-2 wpf', '3 wpf', '4-5 wpf']):
        indices = index_groups[group]
        color = group_colors[j]
        
        for i in indices:
            if nbins is not None:
                ax.hist(pdf_list[i], bins=nbins, density=True, color=color, alpha=0.3,  label=f"{group.capitalize()}" if i == indices[0] else None)
            else:
                ax.hist(pdf_list[i], bins='auto', density=True, color=color, alpha=0.3,  label=f"{group.capitalize()}" if i == indices[0] else None)

        ax.set_ylabel('PDF')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylog is not None:
            ax.set_xlim(xlim)
        if ylog:
            ax.set_yscale('log')
            
        ax.axhline(0, color='lightgray')
        ax.axvline(0, color='lightgray')
        ax.legend(loc="upper right")
        
        
        
        j = j + 1


    plt.show()