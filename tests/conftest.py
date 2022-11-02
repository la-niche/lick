import matplotlib.pyplot as plt
import pytest


@pytest.fixture()
def temp_figure_and_axis():
    fig, ax = plt.subplots()
    yield (fig, ax)
    plt.close(fig)
