# immediately we can see that our datasets is extremely imbalanced
# vast majority of samples have max values close to 0, some samples have maximums of 700+

import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from src.util.plot.magic import OPERA_15M_QPE_PALLETE, OPERA_15M_QPE_BOUNDARIES


def plot_opera_16hr(X: torch.Tensor, dpi=100):
    """
    TODO: use `dpi`

    args
    ---
    :X: `(B, T=16, 1, H, W)`

    returns
    ---
    - `Figure`
    """

    X = X.clone().detach().cpu()

    # HACK: plot the first sample from the batch
    X = X[0, ...]

    # pallete from NSSL MRMS web viewer
    colors     = OPERA_15M_QPE_PALLETE
    boundaries = OPERA_15M_QPE_BOUNDARIES

    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, extend='max')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(13, 11), dpi=300)
    gs  = fig.add_gridspec(nrows=4, ncols=4, wspace=0)



    axs = []

    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j],)
            axs.append(ax)

    for i, ax in enumerate(axs):

        offset = (i)
        T = offset

        if (offset + 1) % 4 == 0:
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label("rain-rate (mm/hr)")

        # [252, 252]
        sli = X[T, ...].numpy().squeeze().clip(0)
        ax.imshow(
                sli,
                norm=norm,
                cmap=cmap,
            )

        ax.set_title(f"T={T}")
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xticklabels([""])
        # ax.set_axis_off()
        ax.minorticks_off()

    return fig