import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots.config as config
import argparse

config.configure(mpl, plt)
SLOs = [75, 100, 150]
FPS = [15, 25]
LOCAL_MODEL_ACC = 0.334

colors = ['tab:blue',  'tab:red', 'tab:orange', 'tab:green']
styles = ['solid', 'dashed', (0, (5, 5)), (0, (3, 5, 1, 5))]


def read_data(data_dir, data_file):
    file = data_dir + "/" + data_file
    header_names = ["Clients", "FPS", "SLO", "Accuracy", "MissRate"]
    df_data = pd.read_csv(file, delimiter=",")

    return df_data


def plot_data(data_dir, output_dir):
    df_data = read_data(data_dir, "data.csv")

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=config.set_size(
        config.width, height_div=2, fraction=0.5))

    # First plot Accuracies
    row = 0
    marker = ['o', 'x']

    color = {15: 'tab:blue', 25: 'tab:orange'}
    for i, slo in enumerate(SLOs):
        for j, fps in enumerate(FPS):
            acc_lst = df_data.loc[(
                df_data["SLO"] == slo) & (df_data["FPS"] == fps), "F1"].values
            clients_count_lst = df_data.loc[(
                df_data["SLO"] == slo) & (df_data["FPS"] == fps), "Clients"].values

            # plot
            _ax = ax[row][i]
            _ax.plot(np.arange(1, len(clients_count_lst)+1),
                     acc_lst, marker=marker[j], linewidth=2, label=f"FPS = {fps}",
                     color=color[fps])
            _ax.set_ylim([0.0, 1])
            _ax.set_xlim([0, len(clients_count_lst)+1])
            _ax.grid(True, which='both', linestyle='dotted', linewidth=0.5)

            if i == 0:
                _ax.set_ylabel("Accuracy (F1 Score)", fontsize=14)
            else:
                _ax.set_yticklabels([])

            _ax.set_title(f"{slo}ms SLO", fontsize=14)
            _ax.set_xticks(np.arange(1, len(clients_count_lst)+1, 1))
            _ax.set_xticklabels([])
            _ax.set_yticks(np.arange(0, 1.1, 0.2))

            # Plot lower model accuracy line
            if j == 0:
                _ax.axhline(LOCAL_MODEL_ACC, xmin=0,
                            xmax=1, ls="--", lw=1, c="black", label=None)
                if i == 1:
                    _ax.text(0.10, LOCAL_MODEL_ACC - 0.100,
                             'Smallest DNN accuracy', fontsize=12)
    # Now plot deadline misses
    row = 1
    bar_width = 0.25
    bar_labels = []
    for i, slo in enumerate(SLOs):
        for j, fps in enumerate(FPS):
            miss_rate_lst = df_data.loc[(
                df_data["SLO"] == slo) & (df_data["FPS"] == fps), "MissRate"].values
            clients_count_lst = df_data.loc[(
                df_data["SLO"] == slo) & (df_data["FPS"] == fps), "Clients"].values

            _ax = ax[row][i]

            bar_indexes = np.arange(1, len(clients_count_lst)+1, dtype=float)
            bar_indexes += (bar_width * j - bar_width/2)
            _ax.bar(bar_indexes, miss_rate_lst,
                    width=bar_width,   lw=1.0, color=color[fps])

            _ax.set_ylim([0, 1.0])
            _ax.set_xlim([0, len(clients_count_lst)+1])
            _ax.grid(True, which='both', linestyle='dotted', linewidth=0.5)
            _ax.set_xlabel("Number of Clients", fontsize=14)
            if i == 0:
                _ax.set_ylabel("Miss Rate (%)", fontsize=14)
            else:
                _ax.set_yticklabels([])

            _ax.set_xticks(np.arange(1, len(clients_count_lst)+1, 1))
            _ax.set_xticklabels(clients_count_lst)
            _ax.set_yticks(np.arange(0, 1.1, 0.2))
            if i == 1:
                bar_labels.append(f'FPS = {fps}')

            # Show out-of-plot bar with a value
            if (slo == 100 and fps == 15) or \
                    (slo == 75 and fps == 25):
                value = round(miss_rate_lst[-1], 3)
                _ax.annotate(value, xy=(bar_indexes[-1], 1.0), fontsize=14,
                             xycoords='data',
                             xytext=(bar_indexes[-3], 0.80),
                             textcoords='data',
                             arrowprops=dict(arrowstyle='-|>',
                                             color='tab:grey',
                                             lw=1,
                                             ls='--')
                             )

    # Set legends
    handles, labels = ax[0][2].get_legend_handles_labels()

    ax[0][2].legend(handles, labels, fontsize=12, loc="lower center")
    ax[1][2].legend(bar_labels, fontsize=12, loc="upper center")

    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f'{output_dir}/fig_8.pdf', format='pdf',
                dpi=config.dpi, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help="directory containing csv files",
                        default="./data/")
    parser.add_argument("--output_dir",
                        help="Output directory to save figures")

    args = parser.parse_args()
    data_dir = args.data_dir

    plot_data(data_dir, output_dir=args.output_dir)
