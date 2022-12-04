import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots.config as config
import argparse
from matplotlib.lines import Line2D

config.configure(mpl, plt)
FPS = [15, 25]
SLOs = [75, 100, 150]
CLIENTS = [2, 4, 8]

colors = ['tab:blue',  'tab:red', 'tab:orange', 'tab:green']
styles = ['solid', 'dashed', (0, (5, 5)), (0, (3, 5, 1, 5))]

FRAME_SIZES = [192, 224, 256, 288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]

TOTAL_MODELS = 10


def get_median(df_data):
    samples = df_data.iloc[0:]
    return (samples).quantile(0.5)


def read_data(data_dir, fps):
    df_slo_dict = {}
    for slo in SLOs:
        df_slo_dict[slo] = []
        for clients in CLIENTS:
            df = df = pd.read_csv(
                f"{data_dir}/frame_stats_{clients}_{fps}_{slo}.csv", delimiter=",")
            df.loc[(df.used_model == -1), 'e2e_latency'] = None
            df_slo_dict[slo].append(df)

    return df_slo_dict


def plot_data(data_dir, output_dir):
    # Plot CDF here
    def plot_cdf(data, ax, label, color, n_bins=200):
        n, bins, patches = ax.hist(
            data, n_bins, density=True, histtype='step', cumulative=True, linewidth=1.5, color=color)
        patches[0].set_xy(patches[0].get_xy()[:-1])

    def plot_time_in_cdf(data, val_P99, ax, label, color, xlim=None):
        plot_cdf(data, ax, label, color, n_bins=1000)
        ax.plot(val_P99, 0.99, alpha=1.0,
                marker="x", markersize=5.0, color=color)
        max_lat = data.max()
        min_lat = data.min()
        if xlim is None:
            xlim = [min_lat, max_lat]
        ax.set_xlim(xlim)
        ax.grid(linestyle='dotted', linewidth=0.5)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=config.set_size(
        config.width, height_div=3, fraction=0.5))

    colors = ["tab:green", "tab:blue", "tab:pink", "tab:purple"]
    for k, fps in enumerate(FPS):
        df_slo_dict = read_data(data_dir, fps)
        for i, slo in enumerate(SLOs):
            _ax = ax[k][i]
            for j, df in enumerate(df_slo_dict[slo]):
                # 99th Percentile latency value
                samples = df["e2e_latency"].iloc[0:]
                lat_P99 = (samples).quantile(0.99)
                lat_median = (samples).quantile(0.50)
                print(
                    f"For {CLIENTS[j]}_{fps}_{slo}, lat_median = {lat_median}, P99 = {lat_P99}, Samples = {len(samples)}")
                plot_time_in_cdf(df["e2e_latency"], lat_P99, _ax,
                                 f"{CLIENTS[j]} Clients", color=colors[j], xlim=[0, slo])

            if i == 0 and k == 0:
                _ax.annotate("P99", xy=(lat_P99, 0.99-0.03), fontsize=14,
                             xycoords='data',
                             xytext=(lat_P99-10, 0.50),
                             textcoords='data',
                             arrowprops=dict(arrowstyle='-|>',
                                             color='tab:grey',
                                             lw=1,
                                             ls='-')
                             )
            if i == 0:
                _ax.set_ylabel(f"CDF\n({fps} FPS)", fontsize=14)

            yticks = [0.0, 0.20, 0.40, 0.60, 0.8, 1.0]
            yticklabels = yticks if i == 0 else []
            _ax.set_yticks(yticks)
            _ax.set_yticklabels(yticklabels, fontsize=12)

            xticks = list(np.linspace(0, slo, num=4).astype(int))
            _ax.set_xticks(xticks)
            if k == 0:
                _ax.set_title(f"{slo}ms SLO", fontsize=14)
                _ax.set_xticklabels([])
            elif k == 1:
                _ax.set_xlabel("Latency (ms)", fontsize=14)

                _ax.set_xticklabels(xticks, fontsize=12)
                _ax.tick_params(axis='both', which='major', labelsize=12)

    custom_lines = []
    labels = []
    for i, clients in enumerate(CLIENTS):
        custom_lines.append(Line2D([0], [0], color=colors[i], lw=1.5))
        labels.append(f"{clients}")

    fig.legend(custom_lines, labels, title="#Clients", bbox_to_anchor=[0.480, 0.36],
               labelspacing=0.25, handletextpad=0.5, columnspacing=0.5,
               loc='center', fontsize=12, ncol=len(labels)//3,
               borderpad=0.1, frameon=True)

    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f'{output_dir}/fig_9.pdf', format='pdf',
                dpi=config.dpi, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help="directory containing csv files",
                        default="./data/latency_cdf/")
    parser.add_argument("--output_dir",
                        help="Directory to save the output figure",
                        default="./")

    args = parser.parse_args()
    data_dir = args.data_dir

    plot_data(data_dir, output_dir=args.output_dir)
