import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots.config as config
import argparse
from datetime import datetime
from matplotlib.lines import Line2D

config.configure(mpl, plt)
FPS = [25, 25]

colors = ['tab:blue',  'tab:red', 'tab:orange', 'tab:green']
styles = ['solid', 'dashed', (0, (5, 5)), (0, (3, 5, 1, 5))]


def convert_bw_time_to_frames_2(data, data_bw, time_diff, interval_sec=20, fps=30):
    ''' shaping started later '''
    total_frames = data["frame_id"].count()
    total_bw_values = data_bw["Bandwidth"].count()

    frames_in_interval = interval_sec * fps  # 30FPS
    first_frame_with_bw = int((time_diff / 1000) * fps)
    bw_per_frame = [None for _ in range(first_frame_with_bw)]

    j = 0
    for _ in range(first_frame_with_bw, total_frames):
        index = (j // frames_in_interval) % total_bw_values
        bw_per_frame.append(data_bw["Bandwidth"][index])
        j += 1

    df = pd.DataFrame(bw_per_frame, columns=["Bandwidth"])
    df["Index"] = df.index
    return df


def convert_bw_time_to_frames_1(data, data_bw, time_diff, interval_sec=20, fps=30):
    ''' shaping started earlier than frame streaming '''
    total_frames = data["frame_id"].count()
    total_bw_values = data_bw["Bandwidth"].count()

    frames_in_interval = interval_sec * fps  # 30FPS
    time_diff_in_sec = time_diff / 1000  # time_diff is in ms
    total_shaping_change = (total_frames // frames_in_interval) + \
        1 + (time_diff_in_sec // interval_sec) + 1

    print(f"total_frames:{total_frames}, total_bw_values:{total_bw_values}, frames_in_interval:{frames_in_interval}",
          f" time_diff_in_sec:{time_diff_in_sec}, total_shaping_change:{total_shaping_change}")
    bw_per_frame = []
    for i in range(int(total_shaping_change)):
        index = (i % total_bw_values)
        bw_per_frame.extend([data_bw["Bandwidth"][index]] * frames_in_interval)

    # Remove first few values as shaping started earlier than the frames
    discard_count = int(time_diff_in_sec * fps)
    bw_per_frame = bw_per_frame[discard_count:(total_frames + discard_count)]
    assert len(bw_per_frame) == total_frames

    df = pd.DataFrame(bw_per_frame, columns=["Bandwidth"])
    df["Index"] = df.index
    return df


def time_diff_in_shaping(client_data, client_shaping_data):
    frame_start = datetime.strptime(client_data["send_ts"][0], '%H:%M:%S.%f')
    shaping_start = datetime.strptime(
        client_shaping_data["ts"][0], '%H:%M:%S.%f')
    print(f"Frame Start {frame_start}, Shaping Start {shaping_start}")

    if (frame_start > shaping_start):  # Frame started later than shaping
        diff = frame_start - shaping_start
        frame_started_early_flag = False
    else:
        diff = shaping_start - frame_start
        frame_started_early_flag = True

    elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))

    return elapsed_time, frame_started_early_flag


def read_data(data_dir):
    # Server
    df_server = pd.read_csv(f'{data_dir}/frame_path_server.csv', delimiter=",")

    # Host 0
    df_frame_stats_host_0 = pd.read_csv(
        f'{data_dir}/frame_stats_host_0.csv', delimiter=",")
    df_frame_stats_host_0 = df_frame_stats_host_0.sort_values(by='frame_id')
    client_id = df_frame_stats_host_0["client_id"][0]
    df_shaping_script_0 = pd.read_csv(
        f'{data_dir}/network_shaping_host_0.csv', delimiter=",")
    df_server_stats_host_0 = df_server[df_server['client_id'] == client_id]

    # Host 1
    df_frame_stats_host_1 = pd.read_csv(
        f'{data_dir}/frame_stats_host_1.csv', delimiter=",")
    df_frame_stats_host_1 = df_frame_stats_host_1.sort_values(by='frame_id')
    client_id = df_frame_stats_host_1["client_id"][0]
    df_shaping_script_1 = pd.read_csv(
        f'{data_dir}/network_shaping_host_1.csv', delimiter=",")
    df_server_stats_host_1 = df_server[df_server['client_id'] == client_id]

    frame_stats_lst = [df_frame_stats_host_0, df_frame_stats_host_1]
    shaping_script_lst = [df_shaping_script_0, df_shaping_script_1]
    server_stats_lst = [df_server_stats_host_0, df_server_stats_host_1]

    return frame_stats_lst, shaping_script_lst, server_stats_lst


def plot_data(data_dir, output_dir):

    frame_stats_lst, shaping_script_lst, server_stats_lst = read_data(data_dir)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=config.set_size(
        config.width, height_div=2.5, fraction=0.5))

    # First plot model adaptation
    row = 0
    marker = ['o', 'x']
    color = ['tab:blue', 'tab:green']
    # Color for False and True
    cmap = mpl.colors.ListedColormap(['blue', 'red'])

    for i, frame_stats in enumerate(frame_stats_lst):
        _ax = ax[row][i]
        _ax.scatter(frame_stats["frame_id"], frame_stats["used_model"],
                    s=1, c=frame_stats["dropped_frame"], cmap=cmap)

        _ax.set_ylim([-2, 16])
        _ax.set_xlim([0, 5000])
        _ax.set_title(f"Client {i}", fontsize=14)

        if i == 0:
            _ax.set_ylabel("DNN Index", fontsize=14)
        _ax.set_yticks(np.linspace(0, 16, num=5))
        if i == 1:
            _ax.set_yticklabels([], fontsize=14)
        _ax.tick_params(axis='both', which='major', labelsize=14)

        xticks = [0, 1000, 2000, 3000, 4000]
        xticklabels = []
        _ax.set_xticks(xticks)
        _ax.set_xticklabels(xticklabels, fontsize=14)

        _ax.grid(linestyle='dotted', linewidth=0.5)

    # Now plot bw shaping
    row = 1
    for i, shaping_data in enumerate(shaping_script_lst):
        df_bw = pd.DataFrame(
            shaping_data["bw"].to_list(), columns=['Bandwidth'])
        time_diff, frame_started_early = time_diff_in_shaping(
            frame_stats_lst[i], shaping_data)
        if frame_started_early:
            bw_data = convert_bw_time_to_frames_2(
                frame_stats_lst[i], df_bw, time_diff, interval_sec=20, fps=FPS[i])
        else:
            bw_data = convert_bw_time_to_frames_1(
                frame_stats_lst[i], df_bw, time_diff, interval_sec=20, fps=FPS[i])
        estimated_bw = server_stats_lst[i]["estimated_bw"] / 1024
        client_bw = server_stats_lst[i]["client_bw"] / 1024

        _ax = ax[row][i]
        _ax.scatter(bw_data["Index"], bw_data["Bandwidth"],
                    label="BW Limit", s=1)
        _ax.scatter(bw_data["Index"], estimated_bw, label="Estimated BW", s=1)
        _ax.grid(linestyle='dotted', linewidth=0.5)
        _ax.set_ylim([0, 25])
        _ax.set_xlim([0, 5000])
        _ax.set_xlabel("Frame Index", fontsize=14)
        if i == 0:
            _ax.set_ylabel("BW (Mbps)", fontsize=14)

        xticks = [0, 1000, 2000, 3000, 4000, 5000]
        if i == 0:
            xticks = xticks[:-1]
        xticklabels = xticks
        _ax.set_xticks(xticks)
        _ax.set_xticklabels(xticklabels, fontsize=14)

        yticks = [0, 10, 20]
        yticklabels = yticks if i == 0 else []
        _ax.set_yticks(yticks)
        _ax.set_yticklabels(yticklabels, fontsize=14)

        _ax.tick_params(axis='both', which='major', labelsize=14)
        if i == 1:
            # Custom legend
            legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='BW Limit'),
                               Line2D([0], [0], color='tab:orange',
                                      lw=2, label='Estimated BW'), ]
            _ax.legend(handles=legend_elements, fontsize=10, loc='lower left')

    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f'{output_dir}/fig_7.pdf', format='pdf',
                dpi=config.dpi, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help="directory containing csv files",
                        default="./data/")
    parser.add_argument("--output_dir",
                        help="Directory to save the output figure",
                        default="./")

    args = parser.parse_args()
    data_dir = args.data_dir

    plot_data(data_dir, output_dir=args.output_dir)
