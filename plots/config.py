# Ref: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
font_options = {
    "text.usetex": False,
    "font.family": "Arial",
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    'hatch.linewidth': 1,
    'axes.linewidth': 1,
    'xtick.major.width': 0.1,
    'xtick.minor.width': 0.1,
    'ytick.major.width': 0.1,
    'ytick.minor.width': 0.1,
    'xtick.major.size': 2,
    'ytick.major.size': 2
}
dpi = 1000
width = 1000


def set_size(width, height_div=2, fraction=1, subplots=(1, 1)):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / height_div

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def configure(mpl, plt):
    mpl.rcParams.update(font_options)
