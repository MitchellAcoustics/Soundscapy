"""Plotting functions for visualizing Likert scale data."""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from soundscapy.surveys import rename_paqs, return_paqs
from soundscapy.surveys.survey_utils import EQUAL_ANGLES, PAQ_LABELS


def paq_radar_plot(
    data: pd.DataFrame,
    ax: Axes | None = None,
    index: str | None = None,
    angles: list[float] | tuple[float, ...] = EQUAL_ANGLES,
    *,
    figsize: tuple[float, float] = (8, 8),
    palette: str | Sequence[str] | None = "colorblind",
    alpha: float = 0.25,
    linewidth: float = 1.5,
    linestyle: str = "solid",
    ylim: tuple[int, int] = (1, 5),
    title: str | None = None,
    label_pad: float | None = 15,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.1, 0.1),
) -> Axes:
    """
    Generate a radar/spider plot of PAQ values.

    This function creates a radar plot showing PAQ (Perceived Affective Quality)
    values from a dataframe. The radar plot displays values for all 8 PAQ dimensions
    arranged in a circular layout.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing PAQ values. Must contain columns matching PAQ_LABELS
        or they will be filtered out.
    ax : matplotlib.pyplot.Axes, optional
        Existing polar subplot axes to plot to. If None, new axes will be created.
    index : str, optional
        Column(s) to set as index for the data. Useful for labeling in the legend.
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches, by default (8, 8).
        Only used when creating new axes.
    colors : Optional[Union[List[str], Dict[str, str], str, Colormap]], optional
        Colors for the plot lines and fills. Can be:
        - List of color names/values for each data row
        - Dictionary mapping index values to colors
        - Single color name/value to use for all data rows
        - A matplotlib colormap to generate colors from
        If None, a default colormap will be used.
    alpha : float, optional
        Transparency for the filled areas, by default 0.25
    linewidth : float, optional
        Width of the plot lines, by default 1.5
    linestyle : str, optional
        Style of the plot lines, by default "solid"
    ylim : Tuple[int, int], optional
        Y-axis limits (min, max), by default (1, 5) for standard Likert scale
    title : str, optional
        Plot title, by default None
    text_padding : Dict[str, int], optional
        Padding for category labels, by default auto-generated
    legend_loc : str, optional
        Legend location, by default "upper right"
    legend_bbox_to_anchor : Tuple[float, float], optional
        Legend bbox_to_anchor parameter, by default (0.1, 0.1)

    Returns
    -------
    plt.Axes
        Matplotlib Axes with radar plot

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from soundscapy.plotting.likert import paq_radar_plot
    >>>
    >>> # Sample data with PAQ values for two locations
    >>> data = pd.DataFrame({
    ...     "Location": ["Park", "Street"],
    ...     "pleasant": [4.2, 2.1],
    ...     "vibrant": [3.5, 4.2],
    ...     "eventful": [2.8, 4.5],
    ...     "chaotic": [1.5, 3.9],
    ...     "annoying": [1.2, 3.7],
    ...     "monotonous": [2.5, 1.8],
    ...     "uneventful": [3.1, 1.9],
    ...     "calm": [4.3, 1.4]
    ... })
    >>>
    >>> # Create radar plot with the "Location" column as index
    >>> ax = paq_radar_plot(data, index="Location", title="PAQ Comparison")
    >>> plt.show()

    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        msg = "The 'data' parameter must be a pandas DataFrame"
        raise TypeError(msg)

    # Set index if provided
    if index is not None:
        data = data.set_index(index)

    # Filter to only include columns that match PAQ_LABELS
    # This handles cases where the data might have extra columns
    data = rename_paqs(data, paq_aliases=PAQ_LABELS)
    data = return_paqs(data, incl_ids=False)

    # Create axes if needed
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

    # ---------- Part 1: Create background
    # Calculate angles for each axis
    rad_angles = np.deg2rad(angles)

    # Draw one axis per variable + add labels
    plt.xticks(rad_angles, PAQ_LABELS)
    ax.tick_params(axis="x", pad=label_pad)

    # Draw y-labels
    ax.set_rlabel_position(0)  # type: ignore[reportAttributeAccessIssues]
    y_ticks = list(range(ylim[0], ylim[1] + 1))
    plt.yticks(y_ticks, [str(y) for y in y_ticks], color="grey", size=8)
    plt.ylim(*ylim)

    # Add title if provided
    if title:
        ax.set_title(title, pad=2.5 * label_pad if label_pad else 20, fontsize=16)

    # -------- Part 2: Add plots

    # Need to add the first value to the end of the data to close the loop
    ext_angles = [*list(rad_angles), rad_angles[0]]
    # Plot each row of data
    with sns.color_palette(palette) as plot_colors:
        for i, (idx, row) in enumerate(data.iterrows()):
            if i == 4:  # noqa: PLR2004
                warnings.warn(
                    "More than 4 sets of data may not be visually clear.", stacklevel=2
                )

            # Extract values and duplicate the first value at the end to close the loop
            values = row.to_numpy().flatten().tolist()
            values += values[:1]

            # Get current color
            color = plot_colors[i]

            # Plot values
            ax.plot(
                ext_angles,
                values,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                label=idx,
            )
            ax.fill(ext_angles, values, color=color, alpha=alpha)

    # Add legend
    if legend_bbox_to_anchor:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        ax.legend(loc=legend_loc)

    plt.tight_layout()

    return ax
