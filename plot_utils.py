"""Some helper functions for plotting"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display


def display_dataframe(dataframe: pd.DataFrame):
    display.display(display.HTML(dataframe.to_html()))


def display_markdown(text: str):
    display.display_markdown(display.Markdown(text))


def save_pub(name, **kwargs):
    """Save the figure for publication"""
    experiment_dir = Path(os.environ["experiment_dir"])
    pub_dir = experiment_dir / "pub"
    if not pub_dir.exists():
        pub_dir.mkdir()
    plt.savefig(pub_dir / f"{name}.png", dpi=600, **kwargs)
    # plt.savefig(pub_dir / f"{name}.pdf", dpi=600, **kwargs)
    plt.show()
    plt.close()


def create_axes():
    """Create a regular grid so figures don't jump"""
    figure = plt.figure(figsize=(10.5, 6))
    # create the axes to get uniform figure
    axes_list = []
    for i, lbl in enumerate(["T2w", "b800", "ADC"]):
        ax_line = []

        line_spacing = 0.25
        col_spacing = 0.33
        for j in range(4):
            if j > 1:
                width = 0.23
                height = 0.3
                offset = 0
            else:
                width = 0.19
                height = 0.27
                offset = 0.025
            # [left, bottom, width, height]
            rect = [
                offset + j * line_spacing + (line_spacing - width) / 2,
                1 - i * col_spacing + (col_spacing - height),
                width,
                height,
            ]
            ax = plt.Axes(figure, rect)
            figure.add_axes(ax)
            ax_line.append(ax)

        ax_line[0].text(
            x=-0.5,
            y=0.5,
            s=lbl,
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_line[0].transAxes,
            bbox=dict(facecolor="grey", alpha=0.5, boxstyle="round"),
        )

        axes_list.append(ax_line)
    return np.array(axes_list)
