from soundscapy.plotting.circumplex import _circumplex_grid
import matplotlib.pyplot as plt
import matplotlib
def test__circumplex_grid():
    """Test the circumplex grid."""
    fig, ax = plt.subplots()
    ax = _circumplex_grid(ax)
    assert type(ax) is plt.Subplot

#%%
