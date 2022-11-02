import matplotlib.pyplot as plt
import numpy as np
import pytest

from lick import lick_box_plot


@pytest.mark.mpl_image_compare()
def test_lick_img():
    fig, ax = plt.subplots()
    x = np.geomspace(0.1, 10, 128)
    y = np.geomspace(0.1, 5, 128)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    V1 = np.cos(XX)
    V2 = np.sin(YY)
    field = V1**2 + V2**2
    lick_box_plot(
        fig,
        ax,
        XX,
        YY,
        V1,
        V2,
        field,
        size_interpolated=256,
        xmin=1,
        xmax=9,
        ymin=1,
        ymax=4,
        niter_lic=5,
        kernel_length=64,
        cmap="inferno",
        stream_density=0.5,
    )
    return fig
