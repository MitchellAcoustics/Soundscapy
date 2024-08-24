"""
Plotting functions for visualising Likert scale data.
"""

from math import pi

from matplotlib import pyplot as plt

from soundscapy.surveys import PAQ_LABELS


def paq_radar_plot(data, ax=None, index=None):
    """Generate a radar/spider plot of PAQ values

    Parameters
    ----------
    data : pd.Dataframe
        dataframe of PAQ values
        recommended max number of values: 3
    ax : matplotlib.pyplot.Axes, optional
        existing subplot axes to plot to, by default None

    Returns
    -------
    plt.Axes
        matplotlib Axes with radar plot
    """
    # TODO: Resize the plot
    # TODO WARNING: Likely broken now
    if index:
        data = data.set_index(index)
    data = data[PAQ_LABELS]
    if ax is None:
        ax = plt.axes(polar=True)
    # ---------- Part 1: create background
    # Number of variables
    categories = [
        "          pleasant",
        "    vibrant",
        "eventful",
        "chaotic    ",
        "annoying          ",
        "monotonous            ",
        "uneventful",
        "calm",
    ]
    N = len(categories)

    # What will be the angle of each axis in the plot (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(1, 5)

    # -------- Part 2: Add plots

    # Plot each individual = each line of the data
    fill_col = ["b", "r", "g"]
    for i in range(len(data.index)):
        # Ind1
        values = data.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=data.index[i])
        ax.fill(angles, values, fill_col[i], alpha=0.25)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    return ax
