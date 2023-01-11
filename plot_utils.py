"""Some helper functions for plotting"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from scipy.stats import ttest_ind


def display_dataframe(dataframe: pd.DataFrame):
    display.display(display.HTML(dataframe.to_html()))


def display_markdown(text: str):
    display.display_markdown(display.Markdown(text))


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


def plot_metrics(data: pd.DataFrame, metrics: List[str]):
    """Plot the listed metrics including the validation

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot
    metrics : List[str]
        The metrics to plot
    """
    data = data.dropna(axis="columns")
    metrics_present = [t for t in metrics if f"val_{t}" in data]
    if len(metrics_present) == 0:
        return
    nrows = int(np.ceil(len(metrics_present) / 4))
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=4,
        sharex=True,
        sharey=False,
        figsize=(16, nrows * 3.5),
    )
    for metric, ax in zip(metrics_present, axes.flat):
        plot_with_ma(ax, data, metric)
        ax.set_ylabel(metric)
    for ax in axes.flat[3 : 4 : len(metrics_present)]:
        ax.legend()
    for ax in axes.flat[len(metrics_present) :]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_disc(res: pd.DataFrame, disc_type: str):
    """Plot the image or latent discriminators"""
    res = res.dropna(axis="columns")
    disc = [
        c[6 + len(disc_type) : -5]
        for c in res.columns
        if c.startswith(f"disc_{disc_type}/") and c.endswith("loss") and len(c) > 17
    ]
    if len(disc) == 0:
        return
    img_gen_list = [c for c in res.columns if c.startswith("disc_image_gen")]
    image_gen_pres = len(img_gen_list) > 0 and disc_type == "image"
    if image_gen_pres:
        ncols = 6
    else:
        ncols = 4
    _, axes_disc = plt.subplots(
        nrows=len(disc),
        ncols=ncols,
        sharex=True,
        sharey=False,
        figsize=(4 * ncols, len(disc) * 4),
    )
    for disc, axes_line in zip(disc, axes_disc):
        disc_start = f"disc_{disc_type}/{disc}_"
        disc_metric = [c for c in res.columns if c.startswith(disc_start)][0].partition(
            disc_start
        )[-1]
        if disc_metric == "RootMeanSquaredError":
            disc_metric_name = "RMSE"
        else:
            disc_metric_name = disc_metric
        last_present = 0
        fields = [
            f"disc_{disc_type}/{disc}/loss",
            f"disc_{disc_type}/{disc}_{disc_metric}",
            f"generator-{disc_type}/{disc}/loss",
            f"generator-{disc_type}/{disc}_{disc_metric}",
        ]
        names = [
            f"Disc-{disc_type.capitalize()} {disc} loss",
            f"Disc-{disc_type.capitalize()} {disc} {disc_metric_name}",
            f"Generator {disc} loss",
            f"Generator {disc} {disc_metric_name}",
        ]
        if image_gen_pres:
            fields = (
                fields[:2]
                + [
                    f"disc_image_gen/{disc}/loss",
                    f"disc_image_gen/{disc}_{disc_metric}",
                ]
                + fields[2:]
            )
            names = (
                names[:2]
                + [
                    f"Disc-Image-Gen {disc} loss",
                    f"Disc-Image-Gen {disc} {disc_metric_name}",
                ]
                + names[2:]
            )
        for num, (field, y_label) in enumerate(
            zip(
                fields,
                names,
            )
        ):
            if field in res.columns:
                plot_with_ma(axes_line[num], res, field)
                axes_line[num].set_ylabel(y_label)
                last_present = max(last_present, num)
        axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_disc_exp(res, experiments, prefixes, disc_type, channel=0):
    "Plot one channel over multiple experiments"
    res = res.query(f"channel=='{channel}'")
    if res.size == 0:
        return
    disc_list = [
        c[6 + len(disc_type) : -5]
        for c in res
        if c.startswith(f"disc_{disc_type}/") and c.endswith("loss") and len(c) > 17
    ]
    img_gen_list = [c for c in res.columns if c.startswith("disc_image_gen")]
    image_gen_pres = len(img_gen_list) > 0 and disc_type == "image"
    if image_gen_pres:
        ncols = 6
    else:
        ncols = 4
    _, axes_disc = plt.subplots(
        nrows=len(disc_list),
        ncols=ncols,
        sharex=True,
        sharey=False,
        figsize=(4 * ncols, len(disc_list) * 4),
    )

    for experiment, pre in zip(experiments, prefixes):
        res_exp = res.query(f"experiment == '{experiment}'")
        if res_exp.size == 0:
            continue
        if np.any(res_exp.epoch.value_counts() > 1):
            raise ValueError("Multiple data points for one epoch, check for duplicates")
        for disc, axes_line in zip(disc_list, axes_disc):
            disc_start = f"disc_{disc_type}/{disc}_"
            disc_metric = [c for c in res_exp.columns if c.startswith(disc_start)][
                0
            ].partition(disc_start)[-1]
            if disc_metric == "RootMeanSquaredError":
                disc_metric_name = "RMSE"
            else:
                disc_metric_name = disc_metric
            last_present = 0
            fields = [
                f"disc_{disc_type}/{disc}/loss",
                f"disc_{disc_type}/{disc}_{disc_metric}",
                f"generator-{disc_type}/{disc}/loss",
                f"generator-{disc_type}/{disc}_{disc_metric}",
            ]
            names = [
                f"Disc-{disc_type.capitalize()} {disc} loss",
                f"Disc-{disc_type.capitalize()} {disc} {disc_metric_name}",
                f"Generator {disc} loss",
                f"Generator {disc} {disc_metric_name}",
            ]
            if image_gen_pres:
                fields = (
                    fields[:2]
                    + [
                        f"disc_image_gen/{disc}/loss",
                        f"disc_image_gen/{disc}_{disc_metric}",
                    ]
                    + fields[2:]
                )
                names = (
                    names[:2]
                    + [
                        f"Disc-Image-Gen {disc} loss",
                        f"Disc-Image-Gen {disc} {disc_metric_name}",
                    ]
                    + names[2:]
                )
            for num, (field, y_label) in enumerate(
                zip(
                    fields,
                    names,
                )
            ):
                if field in res_exp.columns:
                    plot_with_ma(
                        axes_line[num], res_exp, field, label_prefix=pre, dash_val=True
                    )
                    axes_line[num].set_ylabel(y_label)
                    last_present = max(last_present, num)
            axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_with_ma(
    ax_ma: plt.Axes,
    data_frame: pd.DataFrame,
    field: str,
    window_size=5,
    label_prefix="",
    dash_val=False,
):
    """Plot a line with the value in the background and the moving average on top"""
    for val in ("", "val_"):
        for channel in range(3):
            df_channel = data_frame.query(f"channel == '{channel}'")
            if df_channel.size == 0:
                continue
            if dash_val and val == "val_":
                linestyle = "dashed"
            else:
                linestyle = "solid"
            plot = ax_ma.plot(
                df_channel.epoch,
                df_channel[val + field],
                alpha=0.4,
                linestyle=linestyle,
            )
            # plot with moving average
            color = plot[-1].get_color()
            if dash_val and val == "val_":
                color = color_not_val
                label = None
            else:
                color_not_val: str = color
                label = f"{label_prefix}{val}{channel}"
            ax_ma.plot(
                df_channel.epoch,
                np.convolve(
                    np.pad(
                        df_channel[val + field],
                        (window_size // 2 - 1 + window_size % 2, window_size // 2),
                        mode="edge",
                    ),
                    np.ones(window_size) / window_size,
                    mode="valid",
                ),
                label=label,
                linestyle=linestyle,
                color=color,
            )


def plot_significance(grouped_data, name: str, metric: str):
    """Plot the significance as heatmap

    Parameters
    ----------
    grouped_data : pd.GroupBy
        The grouped data to analyze
    name : str
        The name of the experiment
    metric : str
        The metric that is being analyzed
    """
    grouped_data_metric = grouped_data[metric]
    data = pd.DataFrame(grouped_data_metric.mean())
    print(f"{name}: Mean: {data[metric].mean():.2f}")
    significance = pd.DataFrame(index=data.index, columns=data.index)
    for first, first_data in grouped_data_metric:
        for second, second_data in grouped_data_metric:
            if np.allclose(first_data, 0) and np.allclose(second_data, 0):
                pvalue = 1
            else:
                _, pvalue = ttest_ind(
                    first_data.astype(float), second_data.astype(float), nan_policy="omit"
                )
            significance.loc[first, second] = pvalue

    if isinstance(significance.index, pd.MultiIndex):
        left_names = ["-".join([str(cc) for cc in c]) for c in significance.index]
    else:
        left_names = list(significance.index)
    bottom_labels = left_names
    left_labels = [f"{n} ({d:.2f})" for n, d in zip(left_names, data[metric])]
    colors = np.zeros(significance.shape + (3,), dtype=int)
    colors[:, :] = np.array([255, 34, 45])
    colors[significance < 0.05] = np.array([0, 170, 0])

    fig_size = len(grouped_data) * 0.25 + 1.5
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(colors)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(bottom_labels)), labels=bottom_labels)
    ax.set_yticks(np.arange(len(left_labels)), labels=left_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(left_labels)):
        for j in range(len(bottom_labels)):
            if np.isnan(significance.values[i, j]):
                text = "-"
            else:
                text = f"{significance.values[i, j]:0.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black")

    ax.set_title(f"{name} - P-values (green < 0.05, red > 0.05)")
    plt.grid(False)
    fig.tight_layout()
    plt.show()
    plt.close()
